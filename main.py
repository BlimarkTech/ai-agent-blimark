# -*- coding: utf-8 -*-

import os
import logging
import json
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI, Body, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

from pydantic import BaseModel, EmailStr, validator
from openai import AsyncOpenAI, BadRequestError
from supabase import create_client, Client
from jose import JWTError, jwt

from core.hashing import Hasher

# ----------------------------
# Configuración de logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Variables de entorno (definidas en Render)
# ----------------------------
OPENAI_API_KEY               = os.getenv("OPENAI_API_KEY")
SUPABASE_URL                 = os.getenv("SUPABASE_URL")
SUPABASE_KEY                 = os.getenv("SUPABASE_KEY")
WEBHOOK_URL                  = os.getenv("WEBHOOK_URL")
VECTOR_STORE_ID              = os.getenv("VECTOR_STORE_ID")
SUPABASE_BUCKET              = os.getenv("SUPABASE_BUCKET", "config")
SECRET_KEY                   = os.getenv("SECRET_KEY", "clave-secreta-segura")
ALGORITHM                    = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Validaciones básicas
if not OPENAI_API_KEY:
    logger.error("La variable de entorno OPENAI_API_KEY no está configurada")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Las variables de entorno SUPABASE_URL/SUPABASE_KEY no están configuradas")
if not VECTOR_STORE_ID:
    logger.error("La variable de entorno VECTOR_STORE_ID no está configurada")
if not WEBHOOK_URL:
    logger.warning("La variable de entorno WEBHOOK_URL no está configurada. Las exportaciones de leads no funcionarán")

# ----------------------------
# Singletons: instanciar clientes una sola vez
# ----------------------------
openai_client: AsyncOpenAI = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client         = create_client(SUPABASE_URL, SUPABASE_KEY)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# ----------------------------
# Inicialización de FastAPI y recursos
# ----------------------------
app = FastAPI()

# Middleware CORS
origins = [
    "https://blimark.tech",
    "https://api.blimark.tech"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Evento de arranque (startup)
@app.on_event("startup")
async def startup_event():
    # Aquí podrías reconectar o asignar recursos a app.state si lo deseas
    logger.info("Aplicación iniciada: clientes de OpenAI y Supabase listos")

# Health check
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ----------------------------
# Funciones de JWT
# ----------------------------

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise credentials_exception

# ----------------------------
# Consulta de usuario en Supabase
# ----------------------------

def get_user(username: str) -> dict | None:
    resp = (
        supabase.table("users")
                .select("username, hashed_password")
                .eq("username", username)
                .single()
                .execute()
    )
    logger.info(f"Supabase get_user resp: {resp.data}")
    return resp.data

# ----------------------------
# Carga de archivos desde Supabase Storage
# ----------------------------

def load_supabase_file(bucket: str, file_path: str, as_text: bool = True):
    resp = supabase.storage.from_(bucket).download(file_path)
    if hasattr(resp, "error") and resp.error:
        raise RuntimeError(f"Error descargando {file_path}: {resp.error}")
    data = resp.data if hasattr(resp, "data") else resp
    return data.decode("utf-8") if as_text else data


def load_system_message() -> str:
    return load_supabase_file(SUPABASE_BUCKET, "system_message.md", as_text=True)


def load_tools() -> list:
    tools_json = load_supabase_file(SUPABASE_BUCKET, "functions.json", as_text=True)
    return json.loads(tools_json)

SYSTEM_MESSAGE = load_system_message()
custom_tools   = load_tools()
tools = [{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}] + custom_tools

# ----------------------------
# Envío de datos a través de webhook
# ----------------------------

async def handle_function_call(function_call) -> dict:
    if function_call.name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función desconocida: {function_call.name}"}
    try:
        args = json.loads(function_call.arguments)
        if "email" in args:
            EmailStr.validate(args["email"])
        resp = requests.post(WEBHOOK_URL, json=args, timeout=5)
        resp.raise_for_status()
        return {"success": True, "status_code": resp.status_code}
    except Exception as e:
        logger.error(f"Error en handle_function_call: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ----------------------------
# Modelos de datos
# ----------------------------

class ChatRequest(BaseModel):
    history: list

    @validator("history", pre=True, each_item=False)
    def max_history_length(cls, v):
        if not isinstance(v, list):
            raise ValueError("history debe ser una lista de mensajes")
        return v[-200:] if len(v) > 200 else v

# ----------------------------
# Endpoints
# ----------------------------

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado"
        )
    if not Hasher.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña incorrecta"
        )
    token = create_access_token(data={"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat", dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest = Body(...)):
    history = request.history or []
    logger.info(f"Historial recibido ({len(history)} ítems)")
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history

    try:
        response1 = await openai_client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
        detected_fc = None
        initial_text = ""
        for item in response1.output:
            if getattr(item, "type", "") == "function_call":
                detected_fc = item
                break
            if getattr(item, "content", None):
                for chunk in item.content:
                    initial_text += getattr(chunk, "text", "")

        if detected_fc:
            args = json.loads(detected_fc.arguments)
            required = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
            missing = [f for f in required if not args.get(f)]
            if missing:
                texto = f"Para agendar necesito: {', '.join(missing)}. ¿Podrías proporcionarlos?"
                return JSONResponse(content={"response": texto}, media_type="application/json; charset=utf-8")

            fc_result = await handle_function_call(detected_fc)
            messages.append({"type": "function_call", "call_id": detected_fc.call_id,
                             "name": detected_fc.name, "arguments": detected_fc.arguments})
            messages.append({"type": "function_call_output", "call_id": detected_fc.call_id,
                             "output": json.dumps(fc_result)})

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

            enlace = get_enlace_agenda()
            if enlace and enlace not in final_text:
                final_text += f"\n\nEnlace para agendar tu cita: {enlace}"

            return JSONResponse(content={"response": final_text}, media_type="application/json; charset=utf-8")

        return JSONResponse(content={"response": initial_text or "No pude generar una respuesta."},
                            media_type="application/json; charset=utf-8")

    except BadRequestError as e:
        logger.error(f"BadRequestError: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Error interno inesperado."})
