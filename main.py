# -*- coding: utf-8 -*-
import os
import logging
import json
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Depends, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
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
    "SUPABASE_SERVICE_KEY",   # service_role key
    "JWT_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY"
]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

OPENAI_API_KEY_GLOBAL    = os.getenv("OPENAI_API_KEY", None)
SUPABASE_URL             = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY     = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET          = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY           = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM            = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
WEBHOOK_URL_GLOBAL       = os.getenv("WEBHOOK_URL", "").strip()
ENCRYPTION_MASTER_KEY    = os.getenv("ENCRYPTION_MASTER_KEY").encode()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multi-tenant-agent")

# ----------------------------
# Servicios externos
# ----------------------------
# Supabase client
supabase: Client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Conectado a Supabase con service_role key")
except Exception as e:
    logger.error("No se pudo inicializar Supabase", exc_info=True)
    supabase = None

# Passlib para validar API keys de Botpress
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fernet para desencriptar OpenAI keys
cipher = Fernet(ENCRYPTION_MASTER_KEY)

# ----------------------------
# JWT multi-tenant
# ----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class TokenData(BaseModel):
    tenant_id: str
    identifier: str

def create_access_token(data: dict, expires_delta: timedelta|None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant(token: str = Depends(oauth2_scheme)) -> TokenData:
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate":"Bearer"}
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
def load_supabase_file(bucket: str, path: str, as_text: bool=True) -> str|bytes:
    if not supabase:
        raise RuntimeError("Supabase no disponible")
    if ".." in path or path.startswith("/"):
        raise ValueError(f"Ruta inválida: {path}")
    data = supabase.storage.from_(bucket).download(path)
    if isinstance(data, bytes):
        return data.decode("utf-8") if as_text else data
    # v1 response object
    if data.status_code != 200:
        raise RuntimeError(f"Error descargando {path}: {data.status_code}")
    return data.content.decode("utf-8") if as_text else data.content

def get_tenant_openai_key(tenant_id: str) -> str:
    resp = supabase.table("tenants")\
        .select("openai_api_key_encrypted")\
        .eq("id", tenant_id).single().execute()
    enc = resp.data.get("openai_api_key_encrypted")
    if enc:
        return cipher.decrypt(enc.encode()).decode()
    if OPENAI_API_KEY_GLOBAL:
        return OPENAI_API_KEY_GLOBAL
    raise RuntimeError("No existe API Key de OpenAI para tenant y no hay fallback")

def get_data_critico(tenant_id: str, clave: str) -> str|None:
    # intenta específico
    res1 = supabase.table("datos_criticos")\
        .select("valor")\
        .eq("clave", clave).eq("tenant_id", tenant_id)\
        .maybe_single().execute()
    if res1.data:
        return res1.data["valor"]
    # intenta global
    res2 = supabase.table("datos_criticos")\
        .select("valor")\
        .eq("clave", clave).is_("tenant_id", None)\
        .maybe_single().execute()
    return res2.data["valor"] if res2.data else None

async def handle_function_call(call, tenant_id: str) -> dict:
    name = call.name
    # solo recolectarInformacionContacto
    if name != "recolectarInformacionContacto":
        return {"success":False,"error":f"Función no soportada: {name}"}
    # determinar webhook
    wh = WEBHOOK_URL_GLOBAL
    r = supabase.table("tenants").select("recoleccion_webhook_url")\
        .eq("id", tenant_id).single().execute()
    if r.data and r.data.get("recoleccion_webhook_url"):
        wh = r.data["recoleccion_webhook_url"]
    if not wh:
        return {"success":False,"error":"Webhook no configurado"}
    args = json.loads(call.arguments)
    if "email" in args:
        try:
            EmailStr.validate(args["email"])
        except ValidationError as e:
            return {"success":False,"error":"Email inválido"}
    try:
        r2 = requests.post(wh, json=args, timeout=10)
        r2.raise_for_status()
        return {"success":True,"status":r2.status_code,"data":r2.json() if r2.content else None}
    except Exception as e:
        return {"success":False,"error":str(e)}

# ----------------------------
# FastAPI & CORS
# ----------------------------
app = FastAPI(title="Agente IA Multi-Tenant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Modelos Pydantic
# ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: list[ChatMessage]
    conversation_id: str|None = None
    user_id_external: str|None = None

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list):
            raise ValueError("history debe ser lista")
        return v[-200:] if len(v)>200 else v

class ChatResponseData(BaseModel):
    type: str = "text"
    text: str

class ChatApiResponse(BaseModel):
    response: ChatResponseData

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/token")
async def login(api_key_botpress: str = Form(...)):
    if not supabase:
        raise HTTPException(503,"DB no disponible")
    tb = supabase.table("tenants")\
         .select("id,identifier,api_key_hash")\
         .eq("is_active",True).execute()
    for rec in tb.data:
        if pwd_context.verify(api_key_botpress, rec["api_key_hash"]):
            token = create_access_token({"tenant_id":rec["id"],"identifier":rec["identifier"]})
            return {"response":{"access_token":token,"token_type":"bearer"}}
    raise HTTPException(401,"API Key inválida")

@app.post("/chat", response_model=ChatApiResponse)
async def chat(req: ChatRequest, tenant: TokenData = Depends(get_current_tenant)):
    if not supabase:
        raise HTTPException(503,"DB no disponible")
    tid = tenant.tenant_id
    ident = tenant.identifier

    # carga system_message y functions
    sm = load_supabase_file(SUPABASE_BUCKET, f"{ident}/system_message_{ident}.md")
    fm = load_supabase_file(SUPABASE_BUCKET, f"{ident}/functions_{ident}.json")
    tools = json.loads(fm)
    vsid = supabase.table("tenants")\
        .select("openai_vector_store_id")\
        .eq("id",tid).single().execute().data["openai_vector_store_id"]

    # API Key OpenAI
    nokey = get_tenant_openai_key(tid)
    client = AsyncOpenAI(api_key=nokey)

    # mensajes y herramientas
    msgs = [{"role":"system","content":sm}] + [m.model_dump() for m in req.history]
    rag_tool = {"type":"file_search","vector_store_ids":[vsid]}
    tools = [rag_tool]+tools if tools else [rag_tool]

    # primera llamada RAG
    resp1 = await client.responses.create(
        model="gpt-4-turbo",
        input=msgs,
        tools=tools
    )
    text="",fc=None
    for o in resp1.output:
        if getattr(o,"type",None)=="function_call":
            fc = o; break
        if getattr(o,"type",None)=="text":
            for c in o.content: text+=c.text

    # si función
    if fc:
        textparts=[]
        args=json.loads(fc.arguments)
        missing=[f for f in ["nombre","apellidos","email","telefono","pais","mensaje"] if not args.get(f)]
        if missing:
            text=f"Faltan: {', '.join(missing)}"
        else:
            resfn = await handle_function_call(fc, tid)
            msgs.append({"role":"assistant","content":None,"tool_calls":[{"id":fc.call_id,"type":"function","function":{"name":fc.name,"arguments":fc.arguments}}]})
            msgs.append({"role":"tool","tool_call_id":fc.call_id,"name":fc.name,"content":json.dumps(resfn)})
            resp2 = await client.responses.create(model="gpt-4-turbo", input=msgs, tools=tools)
            text=""
            for o in resp2.output:
                if getattr(o,"type",None)=="text":
                    for c in o.content: text+=c.text

    # agrega enlace
    enlace = get_data_critico(tid,"enlace_agenda")
    if enlace and enlace not in text:
        text+=f"\n\nAgendar: {enlace}"

    # guarda historial
    cid=req.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
    uid=req.user_id_external or "unknown"
    if req.history:
        supabase.table("chat_history").insert({"tenant_id":tid,"conversation_id":cid,"user_id_external":uid,"role":req.history[-1].role,"content":req.history[-1].content}).execute()
    supabase.table("chat_history").insert({"tenant_id":tid,"conversation_id":cid,"user_id_external":uid,"role":"assistant","content":text}).execute()

    return {"response":{"text":text}}

@app.get("/health")
async def health():
    return {"status":"ok","time":datetime.now(timezone.utc).isoformat()}
