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
from supabase import create_client, Client

# ----------------------------
# Configuración de logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Variables de entorno y clientes
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")  # Bucket donde están los archivos

if not OPENAI_API_KEY:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("¡Las variables de entorno SUPABASE_URL y SUPABASE_KEY no están configuradas!")
if not WEBHOOK_URL:
    logger.warning("La URL del Webhook no está configurada. La exportación de leads no funcionará.")
if not VECTOR_STORE_ID:
    logger.error("¡La variable de entorno VECTOR_STORE_ID no está configurada!")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Utilidades para cargar archivos de Supabase Storage
# ----------------------------
def load_supabase_file(bucket: str, file_path: str, as_text: bool = True):
    """
    Descarga un archivo de Supabase Storage y lo retorna como texto (utf-8) o bytes.
    Lanza RuntimeError si hay error o el archivo no existe.
    """
    response = supabase.storage.from_(bucket).download(file_path)
    # Manejo robusto según versión del SDK
    if hasattr(response, "error") and response.error:
        raise RuntimeError(f"Error descargando archivo: {response.error}")
    if hasattr(response, "data"):
        data = response.data
    elif isinstance(response, bytes):
        data = response
    else:
        raise RuntimeError(f"No se pudo descargar {file_path} desde el bucket {bucket}")
    return data.decode("utf-8") if as_text else data

def load_system_message():
    return load_supabase_file(SUPABASE_BUCKET, "system_message.md", as_text=True)

def load_tools():
    tools_json = load_supabase_file(SUPABASE_BUCKET, "functions.json", as_text=True)
    return json.loads(tools_json)

# ----------------------------
# Carga dinámica de instrucciones y tools
# ----------------------------
SYSTEM_MESSAGE = load_system_message()
custom_tools = load_tools()
tools = [
    {
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID]
    }
] + custom_tools

# ----------------------------
# Función para obtener el enlace de agendamiento
# ----------------------------
def get_enlace_agenda():
    try:
        response = supabase.table("datos_criticos").select("valor").eq("clave", "enlace_agenda").execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['valor']
        return None
    except Exception as e:
        logger.error(f"Error consultando base estructurada: {e}")
        return None

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
            # Recupera el enlace real de la base estructurada
            enlace_agenda = get_enlace_agenda()
            final_text = ""
            for item in response2.output:
                if getattr(item, "content", None):
                    for chunk in item.content:
                        final_text += getattr(chunk, "text", "")
            # Si el modelo inventa o no incluye el enlace, lo agregamos seguro al final
            if enlace_agenda and enlace_agenda not in final_text:
                final_text += f"\n\nEnlace oficial para agendar tu cita: {enlace_agenda}"
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
