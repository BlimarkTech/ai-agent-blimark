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
       1.  Agradecerle explícitamente por compartir sus datos (ej: "¡Muchas gracias por tus datos, Enrique!").
       2.  Indicar que ya puedes enviarle el enlace.
       3.  Incluir el placeholder `[MEETING_URL]` para que el backend inserte el enlace real (ej: "Puedes agendar tu reunión directamente en el siguiente enlace: [MEETING_URL]").
       4.  NO hagas más preguntas en esta respuesta.
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
             # Considera si esto debería ser un error para tu lógica de negocio
             # return {"success": False, "error": "Nombre o email están vacíos."}

        # El modelo debe haber llenado todos los campos (posiblemente con "")
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

    final_response_text = ""
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

        tool_call_output_dict = None # Para guardar el resultado de nuestra función
        original_function_call_message = None # Para guardar el mensaje original de la API con la FC

        # Procesar la respuesta inicial
        for output_item in response.output:
            if hasattr(output_item, "function_call") and output_item.type == "function_call":
                logger.info(f"Llamada a función detectada: {output_item.function_call.name}")
                original_function_call_message = output_item # Guardar el mensaje original
                function_result = await handle_function_call(output_item.function_call)
                logger.info(f"Resultado de la ejecución de la función: {function_result}")

                # Preparar el output para la segunda llamada
                tool_call_output_dict = {
                     "type": "function_call_output",
                     "call_id": output_item.call_id, # Usar el call_id correcto
                     "output": json.dumps(function_result) # El output debe ser un string JSON
                }
                # No hacer break, por si hay texto también (aunque no debería según el prompt)

            elif hasattr(output_item, "content"):
                 # Capturar texto si la API lo envía (respuesta directa o junto a FC)
                 if output_item.content and hasattr(output_item.content[0], 'text'):
                      # Si ya tenemos una llamada a función, ignoramos este texto inicial
                      if not original_function_call_message:
                          final_response_text = output_item.content[0].text
                          logger.info(f"Respuesta de texto directa recibida: {final_response_text}")
                      else:
                          logger.info(f"Texto recibido junto a FC (ignorado por ahora): {output_item.content[0].text}")


        # --- SI HUBO FUNCTION CALL, REALIZAR SEGUNDA LLAMADA ---
        if tool_call_output_dict:
            if not original_function_call_message:
                 # Sanity check: esto no debería ocurrir si tool_call_output_dict está seteado
                 logger.error("Error lógico: tool_call_output_dict existe pero original_function_call_message no.")
                 return JSONResponse(status_code=500, content={"error": "Error interno procesando llamada a función."})

            logger.info("Realizando segunda llamada a OpenAI con el resultado de la función...")
            # Añadir el mensaje original con la function_call y el resultado de nuestra función
            messages.append(original_function_call_message)
            messages.append(tool_call_output_dict)

            response2 = await openai_client.responses.create(
                model="gpt-4.1", # Usar el mismo modelo
                input=messages,
                tools=tools # Volver a pasar las tools por si acaso
            )
            logger.info(f"Respuesta final de OpenAI recibida tras FC. Output: {response2.output}")

            # Extraer la respuesta final de texto de la segunda llamada
            for out2 in response2.output:
                 if hasattr(out2, "content") and out2.content:
                      if hasattr(out2.content[0], 'text'):
                           final_response_text = out2.content[0].text
                           logger.info(f"Respuesta de texto final recibida después de function call: {final_response_text}")
                           break # Tomar la primera respuesta de texto

            # --- MANEJO SI LA SEGUNDA LLAMADA NO DIO TEXTO ---
            if not final_response_text:
                logger.warning("La segunda llamada a OpenAI (después de FC) no generó texto.")
                # Verificar si la función original tuvo éxito
                function_success = json.loads(tool_call_output_dict["output"]).get("success", False)
                if function_success:
                    # Si la función tuvo éxito pero no hay texto, enviar un mensaje genérico de éxito + enlace (último recurso)
                    logger.info("Función exitosa, pero sin texto del modelo. Enviando confirmación genérica + URL.")
                    final_response_text = "¡Listo! He procesado tus datos. Puedes agendar tu reunión aquí: [MEETING_URL]" # Incluir placeholder
                else:
                    # Si la función falló Y no hay texto, indicar el problema
                    logger.error("La función falló y el modelo no generó texto de error.")
                    final_response_text = "Lo siento, hubo un problema al procesar la información que proporcionaste. Por favor, inténtalo de nuevo."

        # --- SI NO HUBO FUNCTION CALL (Respuesta directa de texto) ---
        # `final_response_text` ya tendría el texto de la primera llamada si lo hubo.
        # Verificar si está vacío incluso sin FC (caso raro de error o respuesta vacía)
        elif not final_response_text:
             logger.warning("No hubo llamada a función y tampoco se recibió texto en la respuesta inicial.")
             final_response_text = "Lo siento, no pude procesar tu solicitud en este momento. Por favor, intenta de nuevo."


        # --- REEMPLAZAR PLACEHOLDER CON ENLACE REAL ---
        # !!! NECESITAS IMPLEMENTAR ESTO: Consulta tu Vector Store aquí !!!
        try:
            # Simulación de consulta a Vector Store (REEMPLAZAR CON TU LÓGICA REAL)
            # Ejemplo: meeting_url_from_vector_store = query_vector_store("meeting_link")
            meeting_url_from_vector_store = "https://calendly.com/tu-enlace-real" # ¡¡¡REEMPLAZA ESTO!!!
            if not meeting_url_from_vector_store:
                logger.error("¡No se pudo obtener la URL de agendamiento desde el Vector Store!")
                # Decide cómo manejarlo: ¿enviar respuesta sin enlace? ¿error?
                # Por ahora, eliminamos el placeholder si no hay URL
                final_response_text = final_response_text.replace("[MEETING_URL]", "(enlace no disponible)")
            elif "[MEETING_URL]" in final_response_text:
                logger.info(f"Reemplazando placeholder [MEETING_URL] con: {meeting_url_from_vector_store}")
                final_response_text = final_response_text.replace("[MEETING_URL]", meeting_url_from_vector_store)
            # Considerar el fallback si es necesario (añadido en versión anterior, opcional)

        except Exception as url_error:
            logger.error(f"Error al obtener/reemplazar URL de agendamiento: {url_error}", exc_info=True)
            # Decide cómo manejarlo, por ahora eliminamos el placeholder
            final_response_text = final_response_text.replace("[MEETING_URL]", "(enlace no disponible)")


        logger.info(f"Enviando respuesta final al usuario: {final_response_text}")
        return JSONResponse(content={"response": final_response_text}, media_type="application/json; charset=utf-8")

    except BadRequestError as e: # Capturar específicamente BadRequestError
        logger.error(f"Error de OpenAI (BadRequestError): {e}", exc_info=True)
        error_content = e.response.json() if hasattr(e, 'response') and hasattr(e.response, 'json') else str(e)
        logger.error(f"Detalle del BadRequestError: {error_content}")
        # Devolver el error 400 con el detalle de OpenAI
        return JSONResponse(
            status_code=400,
            content={"error": f"Error de validación de OpenAI: {error_content}"},
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
