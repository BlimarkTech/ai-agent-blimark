# -*- coding: utf-8 -*-

import os
import logging
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
- Actúa como agente de servicio al cliente de la empresa *Blimark Tech* (Agencia de inteligencia artificial aplicada al marketing).
- Tu objetivo principal es responder consultas, captar leads y agendar reuniones.
2. **Saludo, presentación y flujo:**
- Saluda amablemente, preséntate y pregunta cómo ayudar.
- Responde consultas sobre la empresa con frases claras/cortas (usa vector store).
- Ignora preguntas no relacionadas.
3. **Consulta datos de contacto de la empresa:**
- Si el usuario pregunta por datos de la empresa, consúltalos en el vector store y dalos.
4. **Consulta servicios y precios:**
- Responde *únicamente* con información del vector store. Si no está, sugiere agendar reunión.
5. **Programación de reuniones y Captura de Leads:**
- Si el usuario muestra interés en contratar servicios, pregunta por precios o pide presupuesto:
    - Antes de sugerir al usuario agendar reunión resuelve todas sus dudas sobre los servicios que desea contratar (usa vector store) y asegúrate de que no tenga más dudas.
    - Sugiere agendar reunión y espera confirmación.
    - **Si el usuario ACEPTA agendar la reunión:**
        - Explica que necesitas datos para enviarle el enlace de agendamiento de citas.
        - **Paso 1:** Intenta Inferir el 'mensaje' revisando el historial.
        - **Paso 2:** Consulta en el vector store la lista de campos requeridos y pide exactamente esos datos.
        - **Condición para llamar a la función:** Tan pronto como tengas los datos requeridos y el mensaje, llama a la función `recolectarInformacionContacto`. Pasa todos los datos recopilados (usa "" para los opcionales no obtenidos).
        - **Después de la llamada a función exitosa:**
            1. Agradece explícitamente al usuario por compartir sus datos.
            2. Indica que ya puedes enviarle el enlace.
            3. **Incluye el enlace real de agendamiento de reuniones, recuperado del vector store, en tu respuesta final al usuario.**
            4. Si no encuentras el enlace, indícalo con transparencia.
            5. NO inventes enlaces de reunión, obtén el enlace real para agendar reuniones del vector store.
            6. No hagas más preguntas en esta respuesta.
    - **Si el usuario RECHAZA compartir datos:**
        - Insiste *una sola vez*.
        - Si sigue negándose, finaliza cortésmente.
6. **Resolución de dudas (General):**
- Usa SIEMPRE el vector store para resolver cualquier duda sobre la empresa y sus servicios.

---

**IMPORTANTE:**
Siempre que debas solicitar datos al usuario para cualquier proceso (agendar, cotizar, etc.), consulta en el vector store la lista de campos requeridos para ese proceso y pide exactamente esos datos, usando los nombres y el orden en que aparecen.
Nunca inventes ni omitas campos. Si la lista cambia en el vector store, debes adaptarte automáticamente.

---

### **Restricciones**

1. **Uso exclusivo del vector store:** Toda información de la empresa (contacto, servicios, URL agendamiento) DEBE venir de ahí. No inventes datos ni enlaces. Si no encuentras un dato en el vector store, responde con transparencia que no dispones de esa información.
2. **Preguntas no relacionadas:** No las respondas. Indica que no puedes ayudar y, si insiste, finaliza cortésmente.
3. **Transparencia y límites:** Usa frases cortas (<500 caracteres). Sé claro sobre lo que no sabes.
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
            args = json.loads(detected_fc.arguments)
            campos_obligatorios = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
            faltan = [campo for campo in campos_obligatorios if not args.get(campo)]
            if faltan:
                texto = "Para agendar la reunión necesito: " + ", ".join(faltan) + ". ¿Podrías indicármelos?"
                logger.info(f"Faltan datos: {faltan}. Solicitando al usuario.")
                return JSONResponse(content={"response": texto}, media_type="application/json; charset=utf-8")
            # Ejecuta la función normalmente
            fc_result = await handle_function_call(detected_fc)
            # Añade al historial el mensaje de function_call
            messages.append({
                "type": "function_call",
                "call_id": detected_fc.call_id,
                "name": detected_fc.name,
                "arguments": detected_fc.arguments
            })
            # Añade el output de la función
            messages.append({
                "type": "function_call_output",
                "call_id": detected_fc.call_id,
                "output": json.dumps(fc_result)
            })
            # Segunda llamada a la API para que el modelo genere la respuesta final
            response2 = await openai_client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=tools
            )
            final_text = ""
            for item in response2.output:
                if getattr(item, "content", None):
                    for chunk in item.content:
                        final_text += getattr(chunk, "text", "")
            logger.info(f"Respuesta final: {final_text}")
            return JSONResponse(content={"response": final_text}, media_type="application/json; charset=utf-8")
        else:
            # Sólo texto sin función
            final_text = initial_text or "Lo siento, no pude generar una respuesta válida."

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
