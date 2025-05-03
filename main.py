# -*- coding: utf-8 -*-
import os
import logging
import re
import json
import requests
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")
    # Considera un manejo más robusto aquí
openai_client = AsyncOpenAI(api_key=openai_api_key)

app = FastAPI()

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
# --- FIN CORS Middleware ---

# Modelo para la solicitud POST
class ChatRequest(BaseModel):
    history: list # Recibe el historial completo de mensajes

# --- INSTRUCCIONES DEL SISTEMA CON INFERENCIA CONDICIONAL DE 'MENSAJE' ---
SYSTEM_MESSAGE = """
1. **Rol y objetivo:**
   - Actúa como agente de servicio al cliente de *Blimark Tech* (Agencia de Marketing e IA).
   - Tu objetivo principal es captar leads y agendar reuniones.

2. **Saludo, presentación y flujo:**
   - Saluda amablemente, preséntate y pregunta cómo ayudar.
   - Responde consultas sobre *Blimark Tech* con frases claras/cortas (usa vector store).
   - Ignora preguntas no relacionadas.

3. **Consulta datos de contacto (de Blimark Tech):**
   - Si el usuario pregunta por datos de la empresa, consúltalos en el vector store y dalos.

4. **Consulta servicios y precios:**
   - Responde *únicamente* con info del vector store. Si no está, sugiere agendar reunión.

5. **Programación de reuniones y Captura de Leads (CON INFERENCIA O PREGUNTA DE MENSAJE):**
   - Si el usuario muestra interés en contratar o pregunta precios:
     - Resuelve sus dudas (usa vector store) y confirma que no tenga más.
     - Sugiérele agendar reunión y espera confirmación.
   - **Si el usuario ACEPTA agendar la reunión:**
     - Explica que necesitas datos para el enlace de agendamiento.
     - **Paso 1: Intenta Inferir el 'Mensaje'.** Revisa el historial completo de la conversación. ¿Puedes identificar con claridad y certeza cuál es el servicio específico o la necesidad concreta que el usuario ha mencionado (ej: "chatbot para dentistas", "ayuda con SEO", "campañas SEM")?
     - **Paso 2: Pide los Datos.**
        - **Si pudiste inferir el `mensaje` claramente:** Pide explícitamente SÓLO nombre, apellidos, email, teléfono y país. NO preguntes por el mensaje.
        - **Si NO pudiste inferir el `mensaje` o no está claro en la conversación:** Pide explícitamente nombre, apellidos, email, teléfono, país **Y TAMBIÉN pregunta por el `mensaje`** (ej: "Para entender mejor cómo podemos ayudarte, ¿podrías indicarme brevemente qué servicio o necesidad tenías en mente?").
     - **Condición para llamar a la función:** Tan pronto como tengas el **nombre**, el **email** y el **`mensaje`** (ya sea el que inferiste o el que el usuario te dio después de preguntarle), **DEBES** llamar inmediatamente a la función `recolectarInformacionContacto` usando la tool call. Pasa todos los datos recopilados. **NO** respondas con texto normal en este punto, solo realiza la llamada a la función.
     - **Si faltan datos mínimos (nombre/email/mensaje si se preguntó):** Si el usuario responde pero no proporciona nombre y email (y el mensaje, si lo preguntaste), pídele *específicamente* los datos mínimos que falten para poder continuar.
   - **Después de la llamada a función exitosa:**
     - Una vez ejecutada la función `recolectarInformacionContacto` con éxito (confirmado por su resultado):
        - Agradécele por los datos.
        - Obtén la URL de agendamiento del vector store.
        - Envía la URL al usuario usando el placeholder `[MEETING_URL]`.
   - **Si el usuario RECHAZA compartir datos:**
     - Insiste *una sola vez*.
     - Si sigue negándose, no insistas más. Envía la URL (`[MEETING_URL]`) directamente.

6. **Resolución de dudas (General):**
   - Usa el vector store para resolver dudas sobre *Blimark Tech*.

### **Restricciones**
1. **Uso exclusivo del vector store:** Toda info de la empresa viene de ahí. Si no está, indícalo. No inventes. No menciones nombres de archivo.
2. **Preguntas no relacionadas:** No las respondas.
3. **Transparencia y límites:** Frases cortas (<500 chars). Sé claro sobre lo que no sabes.
4. **Placeholder para enlace:** Usa siempre `[MEETING_URL]` para el enlace de agendamiento.
"""

# --- DEFINICIÓN DE TOOLS ---
tools = [
    {
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"] # Asegúrate que este ID es correcto
    },
    {
        # Herramienta de función (índice 1)
        "type": "function",
        "name": "recolectarInformacionContacto",
        "description": "Recolecta información de contacto de un lead (nombre, email obligatorios; opcionales: apellidos, teléfono, país) y un mensaje sobre sus necesidades (idealmente inferido de la conversación, o preguntado si no estaba claro). Envía estos datos a un sistema externo (webhook).",
        "strict": True, # Mantenemos el modo estricto
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del lead."},
                "apellidos": {"type": "string", "description": "Apellidos del lead (opcional)."},
                "email": {"type": "string", "description": "Correo electrónico del lead."},
                "telefono": {"type": "string", "description": "Número de teléfono del lead (opcional)."},
                "pais": {"type": "string", "description": "País de residencia del lead (opcional)."},
                "mensaje": {
                    "type": "string",
                    "description": "Breve descripción del servicio o necesidad del lead (ej: chatbot IA, SEO). **IMPORTANTE: Intenta inferir este valor del historial. Si no es posible determinarlo claramente, debes preguntárselo explícitamente al usuario junto con los otros datos de contacto.**"
                }
            },
            "required": ["nombre", "email", "mensaje"],
            "additionalProperties": False  # <-- AÑADIR ESTA LÍNEA
        }
    }
]

# Webhook para leads
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://hook.eu2.make.com/9dmymw72hxvg4tlh412q7m5g7vgfpqo9")
if not WEBHOOK_URL:
    logger.warning("La URL del Webhook no está configurada. La exportación de leads no funcionará.")

async def handle_function_call(function_call):
    """Ejecuta la llamada a función (enviar datos al webhook)."""
    function_name = function_call.name
    if function_name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función desconocida: {function_name}"}

    logger.info(f"Ejecutando función: {function_name}")
    try:
        args = json.loads(function_call.arguments)
        logger.info(f"Argumentos recibidos para webhook (inferidos/preguntados por el modelo): {args}")

        # Verificación mínima
        if not all(key in args and args[key] for key in ["nombre", "email", "mensaje"]):
             logger.warning("La llamada a función no contenía todos los argumentos requeridos (nombre, email, mensaje).")
             # Decide cómo manejar esto: error o enviar datos parciales
             # return {"success": False, "error": "Faltan argumentos requeridos"}

        complete_args = {
            "nombre": args.get("nombre"),
            "apellidos": args.get("apellidos"),
            "email": args.get("email"),
            "telefono": args.get("telefono"),
            "pais": args.get("pais"),
            "mensaje": args.get("mensaje")
        }

        if WEBHOOK_URL:
             response = requests.post(WEBHOOK_URL, json=complete_args)
             response.raise_for_status()
             result = {"success": True, "status_code": response.status_code, "response": response.text[:500]}
             logger.info(f"Webhook enviado exitosamente. Respuesta: {result}")
        else:
             logger.warning("Webhook URL no configurada. Simulando éxito.")
             result = {"success": True, "status_code": 200, "response": "Simulated success (no webhook URL)"}

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando argumentos JSON: {e}")
        return {"success": False, "error": f"Argumentos JSON inválidos: {str(e)}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error enviando al webhook: {e}")
        return {"success": False, "error": f"Error de conexión con el webhook: {str(e)}"}
    except Exception as e:
        logger.error(f"Error inesperado en handle_function_call: {e}", exc_info=True)
        return {"success": False, "error": f"Error interno procesando la función: {str(e)}"}

@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    logger.info(f"Historial recibido: {request.history}")

    if not request.history:
        return JSONResponse(status_code=400, content={"error": "No se recibió historial de conversación."})

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages.extend(request.history)

    # --- BUFFER DE HISTORIAL ---
    MAX_HISTORY = 100
    if len(messages) > (MAX_HISTORY * 2 + 1):
         messages = [messages[0]] + messages[-(MAX_HISTORY * 2):]
         logger.info(f"Historial truncado a los últimos {MAX_HISTORY} intercambios.")

    final_response_text = ""
    try:
        # --- PRIMERA LLAMADA A OPENAI ---
        logger.info("Realizando primera llamada a OpenAI...")
        response = await openai_client.responses.create(
            model="gpt-4.1", # O considera gpt-4o / gpt-4-turbo
            input=messages,
            tools=tools,
            # tool_choice="auto"
        )
        logger.info(f"Respuesta inicial de OpenAI recibida. Output: {response.output}")

        tool_call_output = None
        for output_item in response.output:
            if hasattr(output_item, "function_call") and output_item.type == "function_call":
                logger.info(f"Llamada a función detectada: {output_item.function_call.name}")
                function_result = await handle_function_call(output_item.function_call)
                logger.info(f"Resultado de la ejecución de la función: {function_result}")

                tool_call_output = {
                     "type": "function_call_output",
                     "call_id": output_item.call_id,
                     "output": json.dumps(function_result)
                }
                if not function_result.get("success", False):
                    logger.warning("La ejecución de la función falló. Se informará al modelo.")
                break

            elif hasattr(output_item, "content"):
                 if output_item.content and hasattr(output_item.content[0], 'text'):
                      final_response_text = output_item.content[0].text
                      logger.info(f"Respuesta de texto directa recibida: {final_response_text}")

        # --- SI HUBO FUNCTION CALL, HACER SEGUNDA LLAMADA ---
        if tool_call_output:
            logger.info("Realizando segunda llamada a OpenAI con el resultado de la función...")
            messages.append(response.output[0])
            messages.append(tool_call_output)

            response2 = await openai_client.responses.create(
                model="gpt-4.1", # Usar el mismo modelo
                input=messages,
                tools=tools
            )
            logger.info(f"Respuesta final de OpenAI recibida. Output: {response2.output}")

            for out2 in response2.output:
                 if hasattr(out2, "content") and out2.content:
                      if hasattr(out2.content[0], 'text'):
                           final_response_text = out2.content[0].text
                           logger.info(f"Respuesta de texto final recibida después de function call: {final_response_text}")
                           break

        # --- MANEJO DE RESPUESTA VACÍA / FALLO ---
        if not final_response_text:
             logger.warning("No se generó una respuesta de texto final válida desde la API.")
             final_response_text = "Lo siento, hubo un problema procesando tu solicitud. Por favor, intenta de nuevo." # Mensaje genérico

        # --- REEMPLAZAR PLACEHOLDER CON ENLACE REAL ---
        # !!! NECESITAS IMPLEMENTAR ESTO: Consulta tu Vector Store aquí !!!
        meeting_url_from_vector_store = "https://tu-calendly-real.com/reunion" # Reemplaza con tu lógica
        if "[MEETING_URL]" in final_response_text:
            logger.info("Reemplazando placeholder [MEETING_URL]")
            final_response_text = final_response_text.replace("[MEETING_URL]", meeting_url_from_vector_store)
        # Fallback si olvida el placeholder pero la intención es clara
        elif ("agenda" in final_response_text.lower() or "enlace" in final_response_text.lower()) and "http" not in final_response_text:
             logger.warning("El modelo no usó [MEETING_URL]. Añadiendo URL manualmente.")
             final_response_text += f"\nPuedes agendar aquí: {meeting_url_from_vector_store}"


        logger.info(f"Enviando respuesta final al usuario: {final_response_text}")
        return JSONResponse(content={"response": final_response_text}, media_type="application/json; charset=utf-8")

    except Exception as e:
        logger.error(f"Error grave en /chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno inesperado procesando la solicitud."},
            media_type="application/json; charset=utf-8"
        )

# Descomenta para correr localmente
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
