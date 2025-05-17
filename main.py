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
    "SUPABASE_SERVICE_KEY",
    "JWT_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY"
    # OPENAI_API_KEY y WEBHOOK_URL ya no son fallbacks obligatorios si no los quieres
]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Faltan variables de entorno obligatorias: {', '.join(missing)}")

# Estas son leídas pero ahora su ausencia se maneja explícitamente si no quieres fallbacks
OPENAI_API_KEY_FALLBACK  = os.getenv("OPENAI_API_KEY", None) # Leído, pero la lógica decidirá si usarlo
WEBHOOK_URL_FALLBACK     = os.getenv("WEBHOOK_URL", "").strip() # Leído, pero la lógica decidirá

SUPABASE_URL             = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY     = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET          = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY           = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM            = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY    = os.getenv("ENCRYPTION_MASTER_KEY").encode()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("multi-tenant-agent-aic-strict")

# ----------------------------
# Clientes de servicio
# ----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Conexión a Supabase establecida con service_role key.")
except Exception as e:
    logger.error(f"Error crítico conectando a Supabase: {e}", exc_info=True)
    supabase = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cipher = Fernet(ENCRYPTION_MASTER_KEY)

# ----------------------------
# Configuración de JWT Multi-Tenant
# ----------------------------
oauth2_scheme_docs_only = OAuth2PasswordBearer(tokenUrl="/token")
class TokenData(BaseModel): tenant_id: str; identifier: str

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    effective_delta = expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN)
    expire = datetime.now(timezone.utc) + effective_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant_from_token(token: str = Depends(oauth2_scheme_docs_only)) -> TokenData:
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tenant_id: str | None = payload.get("tenant_id"); identifier: str | None = payload.get("identifier")
        if tenant_id is None or identifier is None: raise credentials_exception
        return TokenData(tenant_id=tenant_id, identifier=identifier)
    except JWTError: raise credentials_exception
    except Exception: raise credentials_exception

# ----------------------------
# Funciones Helper
# ----------------------------
def load_supabase_file(bucket: str, file_path: str, as_text: bool = True) -> str | bytes:
    if not supabase: raise RuntimeError("Supabase no disponible para cargar archivo.")
    if ".." in file_path or file_path.startswith("/"): raise ValueError(f"Ruta de archivo no válida: {file_path}")
    logger.info(f"Descargando de Supabase Storage: {bucket}/{file_path}")
    try:
        file_bytes = supabase.storage.from_(bucket).download(file_path)
        return file_bytes.decode("utf-8") if as_text else file_bytes
    except Exception as e:
        logger.error(f"Excepción descargando {file_path}: {str(e)}", exc_info=False)
        raise RuntimeError(f"No se pudo cargar archivo de configuración: {file_path}")

def get_tenant_openai_api_key(tenant_id: str, tenant_identifier: str) -> str:
    if not supabase: raise RuntimeError(f"Supabase no disponible para obtener API key de {tenant_identifier}.")
    try:
        resp = supabase.table("tenants").select("openai_api_key_encrypted").eq("id", tenant_id).single().execute()
        if not resp.data: raise RuntimeError(f"Registro del inquilino {tenant_identifier} no encontrado.")
        encrypted_key = resp.data.get("openai_api_key_encrypted")
        if encrypted_key and encrypted_key.strip() != "":
            return cipher.decrypt(encrypted_key.encode()).decode()
        
        # Si NO quieres fallback y la clave encriptada es obligatoria:
        logger.error(f"CRÍTICO: No hay 'openai_api_key_encrypted' para {tenant_identifier}.")
        raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
        # Si quisieras fallback (lo hemos quitado de la lógica activa):
        # if OPENAI_API_KEY_FALLBACK:
        #     logger.warning(f"Usando API Key global como fallback para {tenant_identifier}.")
        #     return OPENAI_API_KEY_FALLBACK
        # raise RuntimeError(f"API Key no configurada para {tenant_identifier} y no hay fallback global.")

    except Exception as e:
        logger.error(f"Excepción obteniendo/desencriptando API Key para {tenant_identifier}: {e}", exc_info=True)
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}.")

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None: # Añadido identifier para log
    if not supabase: return None
    try:
        resp_tenant = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        if resp_tenant.data: return resp_tenant.data["valor"]
        
        # Si NO quieres fallback global para datos_criticos:
        # logger.info(f"Dato crítico '{clave}' no encontrado específicamente para {tenant_identifier}, y no se usa fallback global.")
        # return None 
        # Si quisieras fallback global (lo hemos quitado de la lógica activa):
        # resp_global = supabase.table("datos_criticos").select("valor").eq("clave", clave).is_("tenant_id", None).maybe_single().execute()
        # if resp_global.data: return resp_global.data["valor"]
    except Exception as e:
        logger.error(f"Error consultando dato crítico '{clave}' ({tenant_identifier}): {e}", exc_info=True)
    return None

class EmailCheckModel(BaseModel): email: EmailStr
async def handle_function_call(function_call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    function_name = function_call_obj.name
    if function_name != "recolectarInformacionContacto":
        return {"success": False, "error": f"Función desconocida: {function_name}"}

    webhook_url_to_use = None
    if supabase:
        try:
            resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
            if not resp_webhook.data: raise RuntimeError(f"Registro del inquilino {tenant_identifier} no encontrado para webhook.")
            
            url_from_db = resp_webhook.data.get("recoleccion_webhook_url")
            if url_from_db and url_from_db.strip() != "":
                webhook_url_to_use = url_from_db
            else: # No hay webhook URL para este tenant, es un error porque no hay fallback
                logger.error(f"CRÍTICO: Función '{function_name}' requiere 'recoleccion_webhook_url', no configurada para {tenant_identifier}.")
                return {"success": False, "error": f"Función '{function_name}' no configurada para {tenant_identifier} (falta URL webhook)."}
        except Exception as e:
            logger.error(f"Error obteniendo 'recoleccion_webhook_url' para {tenant_identifier}: {e}", exc_info=True)
            return {"success": False, "error": f"Error obteniendo config. webhook para {tenant_identifier}."}
    else: return {"success": False, "error": "Servicio de configuración no disponible."}

    try:
        args = json.loads(function_call_obj.arguments)
        if "email" in args and args["email"]:
            try: EmailCheckModel(email=args["email"])
            except ValidationError as ve: return {"success": False, "error": f"Email inválido: {ve.errors()}"}
        
        args_to_send = {**args, "_tenant_info": {"id": tenant_id, "identifier": tenant_identifier}}
        resp_wh = requests.post(webhook_url_to_use, json=args_to_send, timeout=10)
        resp_wh.raise_for_status()
        return {"success": True, "status_code": resp_wh.status_code, "response_data": resp_wh.json() if resp_wh.content else None}
    except requests.exceptions.HTTPError as http_err:
        return {"success": False, "error": f"Error comunicación webhook ({http_err.response.status_code if hasattr(http_err,'response') else 'N/A'})."}
    except Exception as e: return {"success": False, "error": f"Error procesando función: {e}"}

# ----------------------------
# FastAPI app y CORS
# ----------------------------
app = FastAPI(title="Agente IA Multi-Tenant AIC (Strict)", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ----------------------------
# Modelos Pydantic
# ----------------------------
class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel):
    history: list[ChatMessage]; conversation_id: str | None = None; user_id_external: str | None = None
    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list): raise ValueError("history debe ser lista")
        for item in v:
            if not (isinstance(item, dict) and "role" in item and "content" in item):
                 raise ValueError("Cada ítem en history debe ser un objeto con 'role' y 'content'")
        return v[-200:] if len(v) > 200 else v
class ChatResponseData(BaseModel): type: str = "text"; text: str | None = None
class ChatApiResponse(BaseModel): response: ChatResponseData

# ----------------------------
# Endpoints API
# ----------------------------
@app.post("/token", summary="Obtener Token de Acceso para Inquilino")
async def tenant_login(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(status_code=503, detail="Servicio BD no disponible.")
    try:
        db_resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active",True).execute()
        if not db_resp.data: raise HTTPException(status_code=401, detail="Autenticación fallida: Config. inquilinos no encontrada.")
        for rec in db_resp.data:
            if pwd_context.verify(api_key_botpress, rec["api_key_hash"]):
                token = create_access_token(data={"tenant_id": str(rec["id"]), "identifier": rec["identifier"]})
                return {"response": {"access_token": token, "token_type": "bearer"}}
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key de Botpress inválida o inquilino inactivo.")
    except Exception as e:
        logger.error(f"Error en /token: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno en autenticación.")

@app.post("/chat", response_model=ChatApiResponse, summary="Procesar chat para inquilino")
async def chat_endpoint(req: ChatRequest = Body(...), tenant: TokenData = Depends(get_current_tenant_from_token)):
    if not supabase: raise HTTPException(status_code=503, detail="Servicio BD no disponible.")
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"Solicitud /chat para tenant: {ident} (ID: {tid})")

    try:
        path_prefix = f"{ident}/"
        system_msg = load_supabase_file(SUPABASE_BUCKET, f"{path_prefix}system_message_{ident}.md")
        custom_tools = json.loads(load_supabase_file(SUPABASE_BUCKET, f"{path_prefix}functions_{ident}.json"))
        
        vs_conf = supabase.table("tenants").select("openai_vector_store_id").eq("id", tid).single().execute()
        if not vs_conf.data or not vs_conf.data.get("openai_vector_store_id"):
            raise RuntimeError(f"Vector Store ID no configurado para {ident}")
        tenant_vsid = vs_conf.data["openai_vector_store_id"]

        openai_key = get_tenant_openai_api_key(tid, ident)
        oai_client = AsyncOpenAI(api_key=openai_key)

        llm_msgs = [{"role": "system", "content": system_msg}] + [m.model_dump() for m in req.history]
        llm_tools = []
        if tenant_vsid: llm_tools.append({"type": "file_search", "vector_store_ids": [tenant_vsid]})
        if custom_tools: llm_tools.extend(custom_tools)
        
        logger.info(f"Llamando a OpenAI para {ident}. Modelo: gpt-4.1. VS: {tenant_vsid or 'N/A'}")
        resp1 = await oai_client.responses.create(model="gpt-4.1", input=llm_msgs, tools=llm_tools if llm_tools else [])
        
        txt_resp, fn_call = "", None
        for item in resp1.output:
            if getattr(item,"type",None) == "function_call": fn_call = item; break
            if getattr(item,"type",None) == "text":
                for c_part in item.content: txt_resp += getattr(c_part,"text","")
        
        final_txt = txt_resp
        if fn_call:
            args_fn = json.loads(fn_call.arguments)
            if fn_call.name == "recolectarInformacionContacto":
                missing = [f for f in ["nombre","apellidos","email","telefono","pais","mensaje"] if not args_fn.get(f,"")]
                if missing: final_txt = f"Necesito: {', '.join(missing)}."
                else:
                    fn_res = await handle_function_call(fn_call, tid, ident)
                    llm_msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": fn_call.call_id, "type": "function", "function": {"name": fn_call.name, "arguments": fn_call.arguments}}]})
                    llm_msgs.append({"role": "tool", "tool_call_id": fn_call.call_id, "name": fn_call.name, "content": json.dumps(fn_res)})
                    resp2 = await oai_client.responses.create(model="gpt-4.1", input=llm_msgs, tools=llm_tools if llm_tools else [])
                    txt_resp2 = ""
                    for item2 in resp2.output:
                        if getattr(item2,"type",None) == "text":
                            for c_part2 in item2.content: txt_resp2 += getattr(c_part2,"text","")
                    final_txt = txt_resp2 or "Información procesada."
            else: final_txt = f"Acción '{fn_call.name}' procesada."

        agenda_link = get_data_critico(tid, "enlace_agenda", ident)
        if agenda_link and agenda_link not in final_txt: final_txt += f"\n\nAgenda: {agenda_link}"

        cid, uid = req.conversation_id or f"c_{datetime.now(timezone.utc).isoformat()}", req.user_id_external or "unk"
        if req.history: supabase.table("chat_history").insert({"tenant_id":tid,"conversation_id":cid,"user_id_external":uid,"role":req.history[-1].role,"content":req.history[-1].content}).execute()
        if final_txt: supabase.table("chat_history").insert({"tenant_id":tid,"conversation_id":cid,"user_id_external":uid,"role":"assistant","content":final_txt}).execute()
        
        return ChatApiResponse(response=ChatResponseData(text=final_txt if final_txt else "No pude generar respuesta."))

    except BadRequestError as e:
        err_detail = str(e.body.get("message","")) if hasattr(e,'body') and isinstance(e.body,dict) else str(e)
        logger.error(f"BadRequestError /chat {ident}: {err_detail}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error OpenAI: {err_detail}")
    except RuntimeError as e:
        logger.error(f"RuntimeError /chat {ident}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error config. servidor: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado /chat {ident}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno inesperado.")

@app.get("/health", summary="Verificar estado API", tags=["Utilities"])
async def health_check(): return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
