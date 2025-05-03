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

# ----------------------------
# Webhook externo para leads
# ----------------------------
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
        response = requests.post(WEBHOOK_URL, json=args)
        response.raise_for_status()
        return {"success": True, "status_code": response.status_code}
    except Exception as e:
        logger.error(f"Error en handle_function_call: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ----------------------------
# Generador de fallback post-función
# ----------------------------
def generate_fallback_post_fc_message(result: dict, fc_msg) -> str:
    if result.get("success"):
        try:
            name = json.loads(fc_msg.arguments).get("nombre", "")
        except:
            name = ""
        saludo = f"¡Muchas gracias por tus datos, {name}!" if name else "¡Muchas gracias por tus datos!"
        return f"{saludo} Puedes agendar tu reunión aquí: [MEETING_URL]"
    return "Lo siento, hubo un problema al procesar la información. Por favor, inténtalo de nuevo."

# ----------------------------
# Endpoint /chat
# ----------------------------
@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    history = request.history or []
    logger.info(f"Historial recibido: {history}")

    # Construir mensajes de entrada
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history
    if len(messages) > 201:
        messages = [messages[0]] + messages[-200:]

    try:
        # Primera llamada a la Responses API
        logger.info("Realizando primera llamada a OpenAI...")
        response1 = await openai_client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
        logger.info(f"Output primera llamada: {response1.output}")

        # Detectar function_call o texto normal
        detected_fc = None
        initial_text = ""
        for item in response1.output:
            if getattr(item, "type", "") == "function_call":
                detected_fc = item
                break
            if getattr(item, "content", None):
                for chunk in item.content:
                    initial_text += getattr(chunk, "text", "")

        # Si hay llamada a función
        if detected_fc:
            logger.info(f"Función detectada: {detected_fc.name}")
            fc_result = await handle_function_call(detected_fc)

            # Reinsertar mensajes en el contexto
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": detected_fc.name,
                    "arguments": detected_fc.arguments
                }
            })
            messages.append({
                "role": "function",
                "name": detected_fc.name,
                "content": json.dumps(fc_result)  # Asegúrate de que sea un string JSON válido
            })

            # Segunda llamada a la Responses API
            logger.info("Realizando segunda llamada a OpenAI...")
            response2 = await openai_client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=tools
            )
            logger.info(f"Output segunda llamada: {response2.output}")

            # Extraer texto final
            final_text = ""
            for out in response2.output:
                if getattr(out, "content", None):
                    for chunk in out.content:
                        final_text += getattr(chunk, "text", "")
            if not final_text:
                final_text = generate_fallback_post_fc_message(fc_result, detected_fc)

        else:
            # Sólo texto sin función
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

# Descomenta para ejecutar localmente con uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
