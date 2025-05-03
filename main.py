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
from openai import AsyncOpenAI, BadRequestError # Importar BadRequestError

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")
    # Considera un manejo más robusto aquí, como salir o lanzar una excepción
    # exit()
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

# --- INSTRUCCIONES DEL SISTEMA CON INFERENCIA CONDICIONAL DE 'MENSAJE' Y RESPUESTA POST-FC ---
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
     - **Paso 1: Intenta Inferir el 'Mensaje'.** Revisa el historial. ¿Puedes identificar la necesidad específica del usuario (ej: "chatbot", "SEO")?
     - **Paso 2: Pide los Datos.**
        - **Si inferiste el `mensaje`:** Pide SÓLO nombre, apellidos, email, teléfono, país.
        - **Si NO inferiste el `mensaje`:** Pide nombre, apellidos, email, teléfono, país **Y TAMBIÉN** pregunta por el `mensaje`.
     - **Condición para llamar a la función:** Tan pronto como tengas el **nombre**, el **email** y el **`mensaje`** (inferido, preguntado, o "" si no se pudo determinar), y hayas intentado obtener los otros datos, **DEBES** llamar a la función `recolectarInformacionContacto`. Pasa todos los datos recopilados (usa "" para los opcionales no obtenidos). **NO respondas con texto normal**, solo llama a la función.
     - **Si faltan datos mínimos (nombre/email):** Pídele *específicamente* los datos mínimos que falten.
   - **Después de la llamada a función exitosa (MUY IMPORTANTE):**
     - Una vez que la función `recolectarInformacionContacto` se haya ejecutado y recibas la confirmación de éxito, **TU SIGUIENTE RESPUESTA AL USUARIO DEBE SER**:
       1.  Agradecerle explícitamente por compartir sus datos (ej: "¡Muchas gracias por tus datos, [Nombre]!").
       2.  Indicar que ya puedes enviarle el enlace.
       3.  Incluir el placeholder `[MEETING_URL]` para que el backend inserte el enlace real (ej: "Puedes agendar tu reunión directamente en el siguiente enlace: [MEETING_URL]").
       4.  NO hagas más preguntas en esta respuesta. Asegúrate de generar este texto.
   - **Si el usuario RECHAZA compartir datos:**
     - Insiste *una sola vez*.
     - Si sigue negándose, no insistas más. Obtén la URL del vector store y envíasela directamente usando el placeholder `[MEETING_URL]`.

6. **Resolución de dudas (General):**
   - Usa el vector store para resolver dudas sobre *Blimark Tech*.

### **Restricciones**
1. **Uso exclusivo del vector store:** Toda info de la empresa (contacto, servicios, URL agendamiento) DEBE venir de ahí. Si no está, indícalo y sugiere la reunión. No inventes información. No menciones nombres de archivo ni detalles técnicos del vector store.
2. **Preguntas no relacionadas:** No las respondas. Indica que no puedes ayudar y, si insiste, finaliza cortésmente.
3. **Transparencia y límites:** Usa frases cortas (<500 caracteres). Sé claro sobre lo que no sabes ("No tengo acceso a esa información...").
4. **Placeholder para enlace:** Usa siempre `[MEETING_URL]` para el enlace de agendamiento en tu respuesta final al usuario. El backend lo reemplazará.
"""

# --- DEFINICIÓN DE TOOLS CORREGIDA (required incluye todas las properties) ---
tools = [
    {
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"] # Asegúrate que este ID es correcto
    },
    {
        # Herramienta de función (índice 1)
        "type": "function",
        "name": "recolectarInformacionContacto",
        "description": "Recolecta información de contacto de un lead (nombre, email obligatorios) y también apellidos, teléfono, país, y un mensaje sobre sus necesidades (idealmente inferido o preguntado). Envía estos datos a un sistema externo (webhook). Si los campos opcionales (apellidos, teléfono, país, mensaje si no se infiere/obtiene) no están disponibles, se enviarán como strings vacíos.",
        "strict": True, # Mantenemos el modo estricto
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del lead."},
                "apellidos": {"type": "string", "description": "Apellidos del lead. Enviar '' (string vacío) si no se obtiene."},
                "email": {"type": "string", "description": "Correo electrónico del lead."},
                "telefono": {"type": "string", "description": "Número de teléfono del lead. Enviar '' (string vacío) si no se obtiene."},
                "pais": {"type": "string", "description": "País de residencia del lead. Enviar '' (string vacío) si no se obtiene."},
                "mensaje": {
                    "type": "string",
                    "description": "Breve descripción del servicio/necesidad (inferido o preguntado). **IMPORTANTE: Intenta inferir del historial. Si no es claro, pregunta. Si aun así no se obtiene, enviar '' (string vacío).**"
                }
            },
            # --- CORRECCIÓN CLAVE: Listar TODAS las propiedades definidas arriba ---
            "required": ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"],
            "additionalProperties": False  # Mantenemos esto por el error anterior
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
        logger.info(f"Argumentos recibidos para webhook (modelo DEBE proveer todos): {args}")

        # Verificación opcional: Asegurar que nombre y email no estén vacíos
        if not args.get("nombre") or not args.get("email"):
             logger.warning("La llamada a función tiene nombre o email vacíos, aunque el schema los requiere. Procediendo de todos modos.")

        complete_args = args

        if WEBHOOK_URL:
             logger.info(f"Enviando al webhook: {complete_args}")
             response = requests.post(WEBHOOK_URL, json=complete_args)
             response.raise_for_status() # Lanza excepción para errores HTTP 4xx/5xx
             result = {"success": True, "status_code": response.status_code, "response": response.text[:500]}
             logger.info(f"Webhook enviado exitosamente. Respuesta: {result}")
        else:
             logger.warning("Webhook URL no configurada. Simulando éxito.")
             result = {"success": True, "status_code": 200, "response": "Simulated success (no webhook URL)"}

        return result # Devuelve el resultado (éxito o fallo)

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

    final_response_text = "" # Texto final a enviar al usuario
    initial_text_content = "" # Texto recibido en la primera respuesta (si lo hay)
    tool_call_output_dict = None # Resultado de nuestra ejecución de función
    original_function_call_message = None # Mensaje original de la API con la FC
    function_call_detected = False # Flag para indicar si se detectó FC

    try:
        # --- PRIMERA LLAMADA A OPENAI ---
        logger.info("Realizando primera llamada a OpenAI...")
        response = await openai_client.responses.create(
            model="gpt-4.1", # O gpt-4o, gpt-4-turbo
            input=messages,
            tools=tools,
            # tool_choice="auto"
        )
        logger.info(f"Respuesta inicial de OpenAI recibida. Output: {response.output}")

        # Procesar la respuesta inicial
        for output_item in response.output:
            if hasattr(output_item, "function_call") and output_item.type == "function_call":
                logger.info(f"Llamada a función detectada: {output_item.function_call.name}")
                function_call_detected = True
                original_function_call_message = output_item # Guardar el mensaje original
                # Ejecutar la función AHORA
                function_result = await handle_function_call(output_item.function_call)
                logger.info(f"Resultado de la ejecución de la función: {function_result}")

                # Preparar el output para la posible segunda llamada
                tool_call_output_dict = {
                     "type": "function_call_output",
                     "call_id": output_item.call_id, # Usar el call_id correcto
                     "output": json.dumps(function_result) # El output debe ser un string JSON
                }
                # No hacer break, por si hay más items (aunque improbable con FC)

            elif hasattr(output_item, "content"):
                 # Capturar texto si la API lo envía (respuesta directa)
                 if output_item.content and hasattr(output_item.content[0], 'text'):
                      initial_text_content = output_item.content[0].text
                      logger.info(f"Respuesta de texto inicial recibida: {initial_text_content}")

        # --- DECISIÓN BASADA EN LA PRIMERA RESPUESTA ---

        if function_call_detected:
            # Si se detectó una llamada a función, necesitamos la segunda llamada para la respuesta textual
            if not tool_call_output_dict or not original_function_call_message:
                 # Error lógico interno si no se guardaron los detalles de la FC
                 logger.error("Error lógico: FC detectada pero falta tool_call_output_dict u original_function_call_message.")
                 return JSONResponse(status_code=500, content={"error": "Error interno procesando llamada a función."})

            logger.info("Realizando segunda llamada a OpenAI con el resultado de la función...")
            # Construir historial para la segunda llamada
            messages_for_second_call = list(messages) # Copiar historial original
            messages_for_second_call.append(original_function_call_message) # Añadir el mensaje con 'function_call'
            messages_for_second_call.append(tool_call_output_dict) # Añadir el resultado de la función

            response2 = await openai_client.responses.create(
                model="gpt-4.1", # Usar el mismo modelo
                input=messages_for_second_call,
                tools=tools # Volver a pasar las tools
            )
            logger.info(f"Respuesta final de OpenAI recibida tras FC. Output: {response2.output}")

            # Extraer la respuesta final de texto de la segunda llamada
            for out2 in response2.output:
                 if hasattr(out2, "content") and out2.content:
                      if hasattr(out2.content[0], 'text'):
                           final_response_text = out2.content[0].text
                           logger.info(f"Respuesta de texto final recibida después de function call: {final_response_text}")
                           break # Tomar la primera respuesta de texto

            # Fallback si la segunda llamada no dio texto
            if not final_response_text:
                logger.warning("La segunda llamada a OpenAI (después de FC) no generó texto.")
                function_success = json.loads(tool_call_output_dict["output"]).get("success", False)
                if function_success:
                    logger.info("Función exitosa, pero sin texto del modelo. Enviando confirmación genérica + URL.")
                    # Usar el nombre si está disponible en los args de la función original
                    try:
                        original_args = json.loads(original_function_call_message.function_call.arguments)
                        user_name = original_args.get("nombre", "usuario")
                    except:
                        user_name = "usuario"
                    final_response_text = f"¡Muchas gracias por tus datos, {user_name}! Puedes agendar tu reunión aquí: [MEETING_URL]"
                else:
                    logger.error("La función falló y el modelo no generó texto de error.")
                    final_response_text = "Lo siento, hubo un problema al procesar la información que proporcionaste. Por favor, inténtalo de nuevo."

        elif initial_text_content:
            # Si no hubo FC pero sí texto inicial, usar ese texto
            logger.info("No hubo llamada a función. Usando el texto de la respuesta inicial.")
            final_response_text = initial_text_content

        else:
            # Si no hubo ni FC ni texto inicial (respuesta vacía o error inesperado)
            logger.warning("Respuesta inicial de OpenAI no contenía ni llamada a función ni texto.")
            final_response_text = "Lo siento, no pude obtener una respuesta válida en este momento. Por favor, intenta de nuevo."


        # --- REEMPLAZAR PLACEHOLDER CON ENLACE REAL ---
        # !!! NECESITAS IMPLEMENTAR ESTO: Consulta tu Vector Store aquí !!!
        try:
            # Simulación de consulta a Vector Store (REEMPLAZAR CON TU LÓGICA REAL)
            meeting_url_from_vector_store = "https://calendly.com/tu-enlace-real" # ¡¡¡REEMPLAZA ESTO!!!
            if not meeting_url_from_vector_store:
                logger.error("¡No se pudo obtener la URL de agendamiento desde el Vector Store!")
                final_response_text = final_response_text.replace("[MEETING_URL]", "(enlace no disponible)")
            elif "[MEETING_URL]" in final_response_text:
                logger.info(f"Reemplazando placeholder [MEETING_URL] con: {meeting_url_from_vector_store}")
                final_response_text = final_response_text.replace("[MEETING_URL]", meeting_url_from_vector_store)

        except Exception as url_error:
            logger.error(f"Error al obtener/reemplazar URL de agendamiento: {url_error}", exc_info=True)
            final_response_text = final_response_text.replace("[MEETING_URL]", "(enlace no disponible)")


        # --- Enviar la respuesta final ---
        if not final_response_text:
             # Último recurso si todo lo demás falla y final_response_text sigue vacío
             logger.error("Error crítico: final_response_text quedó vacío al final del proceso.")
             final_response_text = "Lo siento, ocurrió un error inesperado. Intenta de nuevo más tarde."

        logger.info(f"Enviando respuesta final al usuario: {final_response_text}")
        return JSONResponse(content={"response": final_response_text}, media_type="application/json; charset=utf-8")

    except BadRequestError as e: # Capturar específicamente BadRequestError
        logger.error(f"Error de OpenAI (BadRequestError): {e}", exc_info=True)
        try:
            error_content = e.response.json()
            error_message = error_content.get('error', {}).get('message', str(e))
        except: # Fallback si no se puede parsear el JSON
             error_message = str(e)
        logger.error(f"Detalle del BadRequestError: {error_message}")
        # Devolver el error 400 con el detalle de OpenAI
        return JSONResponse(
            status_code=400,
            content={"error": f"Error de validación de OpenAI: {error_message}"},
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error grave en /chat endpoint: {e}", exc_info=True)
        # Error genérico 500 para otros casos
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno inesperado procesando la solicitud."},
            media_type="application/json; charset=utf-8"
        )

# Descomenta para correr localmente con uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     # Asegúrate de tener las variables de entorno OPENAI_API_KEY y opcionalmente WEBHOOK_URL configuradas
#     port = int(os.getenv("PORT", 8000)) # Render usa la variable PORT
#     uvicorn.run(app, host="0.0.0.0", port=port)
