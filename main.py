# -*- coding: utf-8 -*-

import os
import logging
import json
import requests
from datetime import datetime, timedelta

from dotenv import load_dotenv

from fastapi import FastAPI, Body, HTTPException, Depends, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, validator

from openai import AsyncOpenAI, BadRequestError
from supabase import create_client, Client
from jose import JWTError, jwt

from core.hashing import Hasher

# ----------------------------
# Carga de variables de entorno
# ----------------------------
load_dotenv()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuración desde env
# ----------------------------
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY")
SUPABASE_URL            = os.getenv("SUPABASE_URL")
SUPABASE_KEY            = os.getenv("SUPABASE_KEY")
WEBHOOK_URL             = os.getenv("WEBHOOK_URL", "")
VECTOR_STORE_ID         = os.getenv("VECTOR_STORE_ID")
SUPABASE_BUCKET         = os.getenv("SUPABASE_BUCKET", "config")
SECRET_KEY              = os.getenv("SECRET_KEY", "clave-secreta-segura")
ALGORITHM               = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

if not OPENAI_API_KEY:
    logger.error("Falta OPENAI_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Faltan SUPABASE_URL/SUPABASE_KEY")
if not VECTOR_STORE_ID:
    logger.error("Falta VECTOR_STORE_ID")

# ----------------------------
# Clientes de API
# ----------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Configuración JWT y seguridad
# ----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise cred_exc

# ----------------------------
# Consulta de usuario en Supabase
# ----------------------------
def get_user(username: str) -> dict | None:
    resp = supabase.table("users") \
                   .select("username, hashed_password") \
                   .eq("username", username) \
                   .single().execute()
    logger.info(f"Supabase get_user resp: {resp.data}")
    return resp.data

# ----------------------------
# Carga de archivos desde Supabase
# ----------------------------
def load_supabase_file(bucket: str, file_path: str, as_text: bool = True):
    resp = supabase.storage.from_(bucket).download(file_path)
    if hasattr(resp, "error") and resp.error:
        raise RuntimeError(f"Error descargando {file_path}: {resp.error}")
    data = getattr(resp, "data", resp)
    return data.decode("utf-8") if as_text else data

def load_system_message() -> str:
    return load_supabase_file(SUPABASE_BUCKET, "system_message.md", as_text=True)

def load_tools() -> list:
    tools_json = load_supabase_file(SUPABASE_BUCKET, "functions.json", as_text=True)
    return json.loads(tools_json)

SYSTEM_MESSAGE = load_system_message()
custom_tools   = load_tools()
tools          = [{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}] + custom_tools

# ----------------------------
# Obtención de enlace de agenda
# ----------------------------
def get_enlace_agenda() -> str | None:
    try:
        resp = supabase.table("datos_criticos") \
                       .select("valor") \
                       .eq("clave", "enlace_agenda") \
                       .execute()
        if resp.data:
            return resp.data[0]["valor"]
    except Exception as e:
        logger.error(f"Error consultando enlace_agenda: {e}")
    return None

# ----------------------------
# Manejo de llamadas a funciones externas
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
# FastAPI app y CORS
# ----------------------------
app = FastAPI()
origins = ["https://blimark.tech", "https://api.blimark.tech"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ----------------------------
# Modelo de datos para /chat
# ----------------------------
class ChatRequest(BaseModel):
    history: list

    @validator("history", pre=True, each_item=False)
    def max_history_length(cls, v):
        if not isinstance(v, list):
            raise ValueError("history debe ser una lista")
        return v[-200:] if len(v) > 200 else v

# ----------------------------
# Endpoint de login (/token)
# ----------------------------
@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not Hasher.verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Usuario o contraseña incorrectos")
    token = create_access_token(data={"sub": user["username"]})
    return {"response": {"access_token": token, "token_type": "bearer"}}

# ----------------------------
# Endpoint protegido (/chat)
# ----------------------------
@app.post("/chat", dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest = Body(...)):
    history = request.history or []
    logger.info(f"Historial recibido: {len(history)} ítems")
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history

    try:
        # Primera llamada a GPT
        response1 = await openai_client.responses.create(
            model="gpt-4.1", input=messages, tools=tools
        )
        detected_fc, initial_text = None, ""
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
                return JSONResponse(content={"response": {"type": "text", "text": texto}})

            fc_result = await handle_function_call(detected_fc)
            messages += [
                {"type": "function_call", "call_id": detected_fc.call_id,
                 "name": detected_fc.name, "arguments": detected_fc.arguments},
                {"type": "function_call_output", "call_id": detected_fc.call_id,
                 "output": json.dumps(fc_result)}
            ]
            response2 = await openai_client.responses.create(
                model="gpt-4.1", input=messages, tools=tools
            )
            final_text = "".join(
                chunk.text
                for itm in response2.output if getattr(itm, "content", None)
                for chunk in itm.content
            )
            enlace = get_enlace_agenda()
            if enlace and enlace not in final_text:
                final_text += f"\n\nEnlace para agendar tu cita: {enlace}"

            return JSONResponse(content={"response": {"type": "text", "text": final_text}})

        # Si no hay llamada a función
        return JSONResponse(content={
            "response": {"type": "text", "text": initial_text or "No pude generar una respuesta."}
        })

    except BadRequestError as e:
        logger.error(f"BadRequestError: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Error interno inesperado."})
