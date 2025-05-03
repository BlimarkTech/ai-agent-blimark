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
from openai import AsyncOpenAI, BadRequestError

# ----------------------------
# Configuración de logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Cliente OpenAI
# ----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")
openai_client = AsyncOpenAI(api_key=openai_api_key)

# ----------------------------
# FastAPI app y CORS
# ----------------------------
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Modelo de petición
# ----------------------------
class ChatRequest(BaseModel):
    history: list  # Historial completo de mensajes

# ----------------------------
# Mensaje del sistema
# ----------------------------
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
  - Sugiere agendar reunión y espera confirmación.
- **Si el usuario ACEPTA agendar la reunión:**
  - Explica que necesitas datos para el enlace de agendamiento.
  - **Paso 1: Intenta Inferir el 'mensaje'.** Revisa el historial. ¿Puedes identificar la necesidad específica del usuario (ej: "chatbot", "SEO")?
  - **Paso 2: Pide los Datos.**
    - **Si inferiste el `mensaje`:** Pide SÓLO nombre, apellidos, email, teléfono, país.
    - **Si NO inferiste el `mensaje`:** Pide nombre, apellidos, email, teléfono, país **Y TAMBIÉN** pregunta por el `mensaje`.
- **Condición para llamar a la función:** Tan pronto como tengas el **nombre**, el **email** y el **`mensaje`** (inferido, preguntado, o "" si no se pudo determinar), y hayas intentado obtener los otros datos, **DEBES** llamar a la función `recolectarInformacionContacto`. Pasa todos los datos recopilados (usa "" para los opcionales no obtenidos). **NO respondas con texto normal**, solo llama a la función.
- **Si faltan datos mínimos (nombre/email):** Pídele *específicamente* los datos mínimos que falten.
- **Después de la llamada a función exitosa (MUY IMPORTANTE):**
  1. Agradecerle explícitamente por compartir sus datos (ej: "¡Muchas gracias por tus datos, [Nombre]!").
  2. Indicar que ya puedes enviarle el enlace.
  3. Incluir el placeholder `[MEETING_URL]` para que el backend inserte el enlace real.
  4. NO hagas más preguntas en esta respuesta. Asegúrate de generar este texto.
- **Si el usuario RECHAZA compartir datos:**
  - Insiste *una sola vez*.
  - Si sigue negándose, no insistas más y envía directamente el placeholder `[MEETING_URL]`.

6. **Resolución de dudas (General):**
- Usa el vector store para resolver dudas sobre *Blimark Tech*.

### **Restricciones**
1. **Uso exclusivo del vector store:** Toda info de la empresa (contacto, servicios, URL agendamiento) DEBE venir de ahí. No inventes datos.
2. **Preguntas no relacionadas:** No las respondas. Indica que no puedes ayudar y, si insiste, finaliza cortésmente.
3. **Transparencia y límites:** Usa frases cortas (<500 caracteres). Sé claro sobre lo que no sabes.
4. **Placeholder para enlace:** Usa siempre `[MEETING_URL]` para el enlace de agendamiento en tu respuesta final al usuario.
"""

# ----------------------------
# Definición de tools
# ----------------------------
tools = [
    {
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"]
    },
    {
        "type": "function",
        "name": "recolectarInformacionContacto",
        "description": (
            "Recolecta información de contacto de un lead (nombre, apellidos, email, teléfono, país, mensaje) "
            "y la envía a un webhook. Si un campo opcional no está disponible, enviar ''."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string"},
                "apellidos": {"type": "string"},
                "email": {"type": "string"},
                "telefono": {"type": "string"},
                "pais": {"type": "string"},
                "mensaje": {"type": "string"},
            },
            "required": ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"],
            "additionalProperties": False
        }
    }
]

WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL",
    "https://hook.eu2.make.com/9dmymw72hxvg4tlh412q7m5g7vgfpqo9"
)
if not WEBHOOK_URL:
    logger.warning("La URL del Webhook no está configurada. La exportación de leads no funcionará.")

# ----------------------------
# Función para ejecutar la llamada
# ----------------------------
async def handle_function_call(function_call):
    if function_call.name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función desconocida: {function_call.name}"}
    try:
        args = json.loads(function_call.arguments)
        if WEBHOOK_URL:
            resp = requests.post(WEBHOOK_URL, json=args)
            resp.raise_for_status()
            return {"success": True, "status_code": resp.status_code}
        return {"success": True, "status_code": 200}
    except Exception as e:
        logger.error(f"Error en handle_function_call: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ----------------------------
# Generador de fallback si falta texto tras FC
# ----------------------------
def generate_fallback_post_fc_message(result: dict, fc_msg) -> str:
    if result.get("success"):
        try:
            name = json.loads(fc_msg.arguments).get("nombre", "")
        except:
            name = ""
        greeting = f"¡Muchas gracias por tus datos, {name}!" if name else "¡Muchas gracias por tus datos!"
        return f"{greeting} Puedes agendar tu reunión aquí: [MEETING_URL]"
    return "Lo siento, hubo un problema al procesar la información. Por favor, inténtalo de nuevo."

# ----------------------------
# Endpoint /chat
# ----------------------------
@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    logger.info(f"Historial recibido: {request.history}")
    if not request.history:
        return JSONResponse(status_code=400, content={"error": "No se recibió historial de conversación."})

    # Armar mensajes de entrada
    current_messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + request.history
    if len(current_messages) > 201:
        current_messages = [current_messages[0]] + current_messages[-200:]

    try:
        # Primera llamada a OpenAI
        logger.info("Realizando primera llamada a OpenAI...")
        response = await openai_client.responses.create(
            model="gpt-4.1",
            input=current_messages,
            tools=tools
        )
        logger.info(f"Output primera llamada: {response.output}")

        # Detectar función o texto
        detected_fc = None
        initial_text = ""
        for item in response.output:
            # Nuevo: detectar ResponseFunctionToolCall por su tipo
            if getattr(item, "type", "") == "function_call":
                detected_fc = item
                break
            # Capturar todo el texto de salida
            if getattr(item, "content", None):
                for chunk in item.content:
                    if hasattr(chunk, "text"):
                        initial_text += chunk.text

        # Si hay llamada a función
        if detected_fc:
            logger.info(f"Función detectada: {detected_fc.name}")
            fc_result = await handle_function_call(detected_fc)

            # Segunda llamada a OpenAI con resultado de la función
            messages2 = list(current_messages)
            # 1) Mensaje de función
            messages2.append({
                "role": "assistant",
                "function_call": {
                    "name": detected_fc.name,
                    "arguments": detected_fc.arguments,
                    "call_id": detected_fc.call_id
                }
            })
            # 2) Resultado de la función
            messages2.append({
                "type": "function_call_output",
                "call_id": detected_fc.call_id,
                "output": json.dumps(fc_result)
            })

            logger.info("Realizando segunda llamada a OpenAI...")
            response2 = await openai_client.responses.create(
                model="gpt-4.1",
                input=messages2,
                tools=tools
            )
            logger.info(f"Output segunda llamada: {response2.output}")

            # Extraer texto final
            final_text = ""
            for out in response2.output:
                if getattr(out, "content", None):
                    for chunk in out.content:
                        if hasattr(chunk, "text"):
                            final_text += chunk.text
            if not final_text:
                final_text = generate_fallback_post_fc_message(fc_result, detected_fc)
        else:
            final_text = initial_text or "Lo siento, no pude generar una respuesta válida."

        # Reemplazar placeholder
        if "[MEETING_URL]" in final_text:
            meeting_url = "https://calendly.com/tu-enlace-real"
            final_text = final_text.replace("[MEETING_URL]", meeting_url)

        logger.info(f"Respuesta final: {final_text}")
        return JSONResponse(content={"response": final_text}, media_type="application/json; charset=utf-8")

    except BadRequestError as e:
        logger.error(f"BadRequestError: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Error interno inesperado."})

# Uncomment to run locally
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)