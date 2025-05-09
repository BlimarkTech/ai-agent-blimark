# -*- coding: utf-8 -*-
import os
import logging
import json
import requests
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, validator
from openai import AsyncOpenAI, BadRequestError
from supabase import create_client, Client
from jose import JWTError, jwt

from core.hashing import Hasher

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse as StarletteJSONResponse

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")
SECRET_KEY = os.getenv("SECRET_KEY", "clave-secreta-segura")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not OPENAI_API_KEY:
    logger.error("Falta OPENAI_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Faltan SUPABASE_URL/SUPABASE_KEY")
if not VECTOR_STORE_ID:
    logger.error("Falta VECTOR_STORE_ID")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def get_user(username: str) -> dict | None:
    try:
        resp = supabase.table("users").select("username, hashed_password").eq("username", username).single().execute()
        return resp.data
    except Exception as e:
        logger.error(f"Error en get_user para {username}: {e}")
        return None

def load_supabase_file(bucket: str, file_path: str, as_text: bool = True):
    try:
        resp = supabase.storage.from_(bucket).download(file_path)
        data = getattr(resp, "data", resp)
        return data.decode("utf-8") if as_text else data
    except Exception as e:
        logger.error(f"Error descargando {file_path} de Supabase bucket {bucket}: {e}")
        raise RuntimeError(f"Error descargando {file_path}: {e}")

def load_system_message() -> str:
    return load_supabase_file(SUPABASE_BUCKET, "system_message.md", as_text=True)

def load_tools() -> list:
    tools_json = load_supabase_file(SUPABASE_BUCKET, "functions.json", as_text=True)
    return json.loads(tools_json)

SYSTEM_MESSAGE = load_system_message()
custom_tools = load_tools()
tools = [{"type": "file_search"}] + custom_tools

def get_enlace_agenda() -> str | None:
    try:
        resp = supabase.table("datos_criticos").select("valor").eq("clave", "enlace_agenda").single().execute()
        return resp.data["valor"] if resp.data else None
    except Exception as e:
        logger.error(f"Error consultando enlace_agenda: {e}")
    return None

async def handle_function_call(function_call) -> dict:
    function_name = getattr(function_call, "name", None)

    if function_name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función desconocida: {function_name}"}
    try:
        args = json.loads(getattr(function_call, "arguments", "{}"))
        if "email" in args:
            EmailStr.validate(args["email"])
        if not WEBHOOK_URL:
            logger.error("WEBHOOK_URL no está configurado.")
            return {"success": False, "error": "Configuración de webhook faltante."}

        resp = requests.post(WEBHOOK_URL, json=args, timeout=10)
        resp.raise_for_status()
        return {"success": True, "status_code": resp.status_code, "response_data": resp.json()}
    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando argumentos JSON para {function_name}: {e}", exc_info=True)
        return {"success": False, "error": f"Argumentos JSON inválidos para {function_name}."}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la llamada HTTP a WEBHOOK_URL para {function_name}: {e}", exc_info=True)
        return {"success": False, "error": f"Error de comunicación con el servicio externo: {e}"}
    except Exception as e:
        logger.error(f"Error inesperado en handle_function_call para {function_name}: {e}", exc_info=True)
        return {"success": False, "error": f"Error interno procesando la función: {str(e)}"}

app = FastAPI()

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    try:
        retry_after_str = exc.detail.split('Retry after ')[-1].split(' second')[0]
        retry_after_seconds = int(retry_after_str)
    except (IndexError, ValueError, AttributeError):
        retry_after_seconds = 60
        
    return StarletteJSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": f"Límite de solicitudes excedido. {exc.detail}",
            "retry_after_seconds": retry_after_seconds
        },
        headers={"Retry-After": str(retry_after_seconds)}
    )

origins = [
    "https://blimark.tech",
    "https://www.blimark.tech",
    "https://api.blimark.tech",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    history: list[dict]

    @validator("history", pre=True, each_item=False)
    def max_history_length(cls, v):
        if not isinstance(v, list):
            raise ValueError("El historial debe ser una lista.")
        for item in v:
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                raise ValueError("Cada ítem del historial debe ser un diccionario con 'role' y 'content'.")
        return v[-20:] if len(v) > 20 else v

@app.post("/token")
@limiter.limit("10/minute")
async def login(request: Request):
    username = None
    password = None
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON mal formado.")
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Tipo de contenido no soportado. Usar application/json o application/x-www-form-urlencoded.")

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Faltan 'username' o 'password'.")

    user = get_user(username)

    if not user or not Hasher.verify_password(password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user["username"]})
    return JSONResponse(content={"access_token": token, "token_type": "bearer"})

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest = Body(...),
    current_user: dict = Depends(verify_token)
):
    history = chat_request.history or []
    logger.info(f"Historial recibido para usuario '{current_user.get('sub')}': {len(history)} ítems")

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history

    try:
        response1 = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=custom_tools if custom_tools else None,
            tool_choice="auto" if custom_tools else None
        )
        
        response_message = response1.choices[0].message
        initial_text = response_message.content or ""
        tool_calls = response_message.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_to_call = tool_call.function
            
            logger.info(f"Llamada a función detectada: {function_to_call.name}")
            
            if function_to_call.name == "recolectarInformacionContacto":
                try:
                    args = json.loads(function_to_call.arguments)
                    required = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
                    missing = [f for f in required if not args.get(f)]
                    if missing:
                        texto = f"Para procesar tu solicitud de contacto, necesito los siguientes datos adicionales: {', '.join(missing)}. ¿Podrías proporcionarlos, por favor?"
                        return JSONResponse(content={"response": {"type": "text", "text": texto}})
                except json.JSONDecodeError:
                    logger.error(f"Argumentos de función mal formados: {function_to_call.arguments}")
                    pass

            fc_result = await handle_function_call(function_to_call)

            messages.append(response_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_to_call.name,
                "content": json.dumps(fc_result),
            })

            response2 = await openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                tools=custom_tools if custom_tools else None,
                tool_choice="auto" if custom_tools else None
            )
            final_text = response2.choices[0].message.content or ""

            enlace = get_enlace_agenda()
            if enlace and enlace not in final_text and "recolectarInformacionContacto" == function_to_call.name and fc_result.get("success"):
                final_text += f"\n\nHe recolectado tu información. Si deseas, puedes agendar una cita directamente aquí: {enlace}"
            
            return JSONResponse(content={"response": {"type": "text", "text": final_text}})

        return JSONResponse(content={
            "response": {"type": "text", "text": initial_text or "No pude generar una respuesta en este momento."}
        })

    except BadRequestError as e:
        logger.error(f"OpenAI BadRequestError en /chat para usuario '{current_user.get('sub')}': {e.message}", exc_info=True)
        error_detail = e.message
        if e.body and "message" in e.body:
             error_detail = e.body["message"]
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": f"Error de OpenAI: {error_detail}"})
    except Exception as e:
        logger.error(f"Error inesperado en /chat para usuario '{current_user.get('sub')}': {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Error interno inesperado en el servidor."})

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
