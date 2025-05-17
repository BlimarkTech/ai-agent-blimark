# -*- coding: utf-8 -*-
import os
import logging
import json
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Depends, status, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, EmailStr, ValidationError, field_validator
from openai import AsyncOpenAI, BadRequestError
from supabase import create_client, Client
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet

# ----------------------------
# Carga y validación de variables de entorno
# ----------------------------
load_dotenv()

REQUIRED_VARS = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "JWT_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY"
]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Faltan variables de entorno obligatorias: {', '.join(missing)}")

SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET       = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY        = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM         = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi-tenant-agent")

# ----------------------------
# Clientes de servicio
# ----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Conexión a Supabase establecida con service_role key.")
except Exception as e:
    logger.error(f"Error conectando a Supabase: {e}", exc_info=True)
    supabase = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cipher = Fernet(ENCRYPTION_MASTER_KEY)

# ----------------------------
# FastAPI app y CORS
# ----------------------------
app = FastAPI(title="Agente IA Multi-Tenant (Strict)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handler de validación para ver detalles de errores 422
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"RequestValidationError: {exc.errors()} – Body: {body.decode('utf-8')}")
    return await request_validation_exception_handler(request, exc)

# ----------------------------
# Configuración de JWT Multi-Tenant
# ----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class TokenData(BaseModel):
    tenant_id: str
    identifier: str

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant(token: str = Depends(oauth2_scheme)) -> TokenData:
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tid = payload.get("tenant_id")
        ident = payload.get("identifier")
        if not tid or not ident:
            raise creds_exc
        return TokenData(tenant_id=tid, identifier=ident)
    except JWTError:
        raise creds_exc

# ----------------------------
# Helpers
# ----------------------------
def load_supabase_file(bucket: str, path: str, as_text: bool = True) -> str | bytes:
    if not supabase:
        raise RuntimeError("Supabase no disponible para cargar archivo.")
    if ".." in path or path.startswith("/"):
        raise ValueError(f"Ruta inválida: {path}")
    logger.info(f"Descargando archivo {bucket}/{path}")
    try:
        file_bytes = supabase.storage.from_(bucket).download(path)
        return file_bytes.decode("utf-8") if as_text else file_bytes
    except Exception as e:
        logger.error(f"Error descargando {path}: {e}")
        raise RuntimeError(f"No se pudo cargar archivo: {path}")

def get_tenant_openai_key(tenant_id: str, tenant_identifier: str) -> str:
    if not supabase:
        raise RuntimeError("Supabase no disponible para obtener API Key.")
    resp = supabase.table("tenants").select("openai_api_key_encrypted")\
        .eq("id", tenant_id).single().execute()
    if not resp.data:
        raise RuntimeError(f"Inquilino {tenant_identifier} no encontrado.")
    encrypted = resp.data.get("openai_api_key_encrypted")
    if not encrypted or not encrypted.strip():
        logger.error(f"API Key encriptada faltante para {tenant_identifier}")
        raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except Exception as e:
        logger.error(f"Error desencriptando API Key para {tenant_identifier}: {e}")
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}.")

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None:
    if not supabase:
        return None
    resp = supabase.table("datos_criticos").select("valor")\
        .eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
    if resp.data:
        return resp.data["valor"]
    return None

class EmailCheckModel(BaseModel):
    email: EmailStr

async def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    name = call_obj.name
    if name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función no soporte: {name}"}
    # Obtener webhook específico
    resp = supabase.table("tenants").select("recoleccion_webhook_url")\
        .eq("id", tenant_id).single().execute()
    if not resp.data or not resp.data.get("recoleccion_webhook_url"):
        return {"success": False, "error": "Webhook no configurado para este tenant."}
    url = resp.data["recoleccion_webhook_url"]
    try:
        args = json.loads(call_obj.arguments)
        if "email" in args:
            try:
                EmailCheckModel(email=args["email"])
            except ValidationError as ve:
                return {"success": False, "error": f"Email inválido: {ve.errors()}"}
        payload = {**args, "_tenant": {"id": tenant_id, "ident": tenant_identifier}}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return {"success": True, "status_code": r.status_code,
                "response_data": r.json() if r.content else None}
    except Exception as e:
        logger.error(f"Error webhook {tenant_identifier}: {e}")
        return {"success": False, "error": str(e)}

# ----------------------------
# Modelos Pydantic
# ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: list[ChatMessage]
    conversation_id: str | None = None
    user_id_external: str | None = None

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list):
            raise ValueError("history debe ser una lista")
        return v[-200:] if len(v) > 200 else v

class ChatResponseData(BaseModel):
    type: str = "text"
    text: str | None

class ChatApiResponse(BaseModel):
    response: ChatResponseData

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/token", summary="Obtener JWT para tenant")
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase:
        raise HTTPException(503, "Servicio BD no disponible")
    resp = supabase.table("tenants").select("id,identifier,api_key_hash")\
        .eq("is_active", True).execute()
    if not resp.data:
        raise HTTPException(401, "No hay tenants activos")
    for row in resp.data:
        if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
            token = create_access_token({
                "tenant_id": str(row["id"]),
                "identifier": row["identifier"]
            })
            return {"response": {"access_token": token, "token_type": "bearer"}}
    raise HTTPException(401, "API Key inválida")

@app.post("/chat", response_model=ChatApiResponse, summary="Chat multi-tenant")
async def chat_endpoint(
    data: ChatRequest = Body(...),
    tenant: TokenData = Depends(get_current_tenant)
):
    if not supabase:
        raise HTTPException(503, "Servicio BD no disponible")
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"/chat para tenant {ident} (ID: {tid})")

    # Carga de configuración
    system_md = load_supabase_file(SUPABASE_BUCKET, f"{ident}/system_message_{ident}.md")
    tools_json = load_supabase_file(SUPABASE_BUCKET, f"{ident}/functions_{ident}.json")
    custom_tools = json.loads(tools_json)

    vs_conf = supabase.table("tenants")\
        .select("openai_vector_store_id").eq("id", tid).single().execute()
    if not vs_conf.data or not vs_conf.data.get("openai_vector_store_id"):
        raise HTTPException(500, f"Vector Store ID no configurado para {ident}")
    vsid = vs_conf.data["openai_vector_store_id"]

    # OpenAI client
    key = get_tenant_openai_key(tid, ident)
    client = AsyncOpenAI(api_key=key)

    # Prepara mensajes y tools
    messages = [{"role": "system", "content": system_md}] + [m.model_dump() for m in data.history]
    tools = [{"type": "file_search", "vector_store_ids": [vsid]}] + (custom_tools or [])

    # Primera llamada
    try:
        resp1 = await client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
    except BadRequestError as e:
        msg = getattr(e, "body", {}).get("message", str(e))
        raise HTTPException(400, f"Error OpenAI: {msg}")

    text, fn_call = "", None
    for o in resp1.output:
        if getattr(o, "type", "") == "function_call":
            fn_call = o
            break
        if getattr(o, "type", "") == "text":
            for chunk in o.content:
                text += getattr(chunk, "text", "")

    # Si función
    if fn_call:
        args = json.loads(fn_call.arguments)
        required = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
        missing = [f for f in required if not args.get(f)]
        if missing:
            text = f"Faltan: {', '.join(missing)}"
        else:
            fn_res = await handle_function_call(fn_call, tid, ident)
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": fn_call.call_id,
                    "type": "function",
                    "function": {"name": fn_call.name, "arguments": fn_call.arguments}
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": fn_call.call_id,
                "name": fn_call.name,
                "content": json.dumps(fn_res)
            })
            resp2 = await client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=tools
            )
            text = ""
            for o2 in resp2.output:
                if getattr(o2, "type", "") == "text":
                    for c2 in o2.content:
                        text += getattr(c2, "text", "")

    # Guardar historial
    conv_id = data.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
    uid = data.user_id_external or "unknown"
    if data.history:
        last = data.history[-1]
        supabase.table("chat_history").insert({
            "tenant_id": tid,
            "conversation_id": conv_id,
            "user_id_external": uid,
            "role": last.role,
            "content": last.content
        }).execute()
    supabase.table("chat_history").insert({
        "tenant_id": tid,
        "conversation_id": conv_id,
        "user_id_external": uid,
        "role": "assistant",
        "content": text
    }).execute()

    return ChatApiResponse(response=ChatResponseData(text=text))

@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
