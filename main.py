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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi-tenant-agent-responses-api")

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
app = FastAPI(title="Agente IA Multi-Tenant (Responses API)", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"RequestValidationError: {exc.errors()} – Body: {body.decode('utf-8')}")
    return await request_validation_exception_handler(request, exc)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class TokenData(BaseModel):
    tenant_id: str; identifier: str

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant(token: str = Depends(oauth2_scheme)) -> TokenData:
    creds_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tid, ident = payload.get("tenant_id"), payload.get("identifier")
        if not tid or not ident: raise creds_exc
        return TokenData(tenant_id=tid, identifier=ident)
    except JWTError: raise creds_exc

def load_supabase_file(bucket: str, path: str, as_text: bool = True) -> str | bytes:
    if not supabase: raise RuntimeError("Supabase no disponible para cargar archivo.")
    if ".." in path or path.startswith("/"): raise ValueError(f"Ruta inválida: {path}")
    logger.info(f"Descargando archivo {bucket}/{path}")
    try:
        data = supabase.storage.from_(bucket).download(path)
        return data.decode("utf-8") if as_text else data
    except Exception as e:
        logger.error(f"Error descargando {path}: {e}")
        raise RuntimeError(f"No se pudo cargar archivo: {path}")

def get_tenant_openai_key(tenant_id: str, tenant_identifier: str) -> str:
    if not supabase: raise RuntimeError("Supabase no disponible para obtener API Key.")
    resp = supabase.table("tenants").select("openai_api_key_encrypted").eq("id", tenant_id).single().execute()
    if not resp.data: raise RuntimeError(f"Inquilino {tenant_identifier} no encontrado.")
    encrypted = resp.data.get("openai_api_key_encrypted")
    if not encrypted or not encrypted.strip():
        logger.error(f"API Key encriptada faltante para {tenant_identifier}")
        raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
    try: return cipher.decrypt(encrypted.encode()).decode()
    except Exception as e:
        logger.error(f"Error desencriptando API Key para {tenant_identifier}: {e}")
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}.")

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None:
    if not supabase:
        logger.error(f"Supabase no disponible al buscar '{clave}' para {tenant_identifier}")
        return None
    try:
        logger.info(f"Buscando dato crítico '{clave}' para tenant {tenant_identifier} (ID: {tenant_id})")
        resp = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        if resp and hasattr(resp, 'data') and resp.data and isinstance(resp.data, dict):
            valor = resp.data.get("valor")
            if valor:
                logger.info(f"Dato '{clave}' encontrado para {tenant_identifier}: '{valor[:50]}...'")
                return valor
        logger.info(f"Dato crítico '{clave}' no encontrado o valor vacío para {tenant_identifier}.")
        return None
    except Exception as e:
        logger.error(f"Excepción al obtener dato crítico '{clave}' para {tenant_identifier}: {e}", exc_info=True)
        return None

class EmailCheckModel(BaseModel): email: EmailStr

async def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    name = call_obj.name
    
    # ✅ BACKEND NEUTRAL: Acepta CUALQUIER función definida en el JSON del tenant
    logger.info(f"Ejecutando función '{name}' para tenant {tenant_identifier}")
    
    if not supabase: 
        return {"success": False, "error": "Servicio de configuración no disponible."}
    
    # Obtener webhook del tenant
    resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
    if not resp_webhook.data: 
        return {"success": False, "error": f"Configuración webhook no encontrada para {tenant_identifier}."}
    
    url = resp_webhook.data.get("recoleccion_webhook_url")
    if not url or not url.strip(): 
        return {"success": False, "error": f"Función '{name}' no configurada (falta URL webhook)."}

    try:
        args = json.loads(call_obj.arguments)
        
        # ✅ VALIDACIÓN GENÉRICA: Solo valida email si existe en los argumentos
        if "email" in args and args["email"]: 
            try: 
                EmailCheckModel(email=args["email"])
            except ValidationError as ve: 
                logger.warning(f"Validación de email fallida para '{args['email']}' en tenant {tenant_identifier}: {ve.errors()}")
                return {"success": False, "error": f"Email inválido: {ve.errors()}"}
        
        # ✅ PAYLOAD GENÉRICO: Envía todos los argumentos + info del tenant
        payload = {**args, "_tenant_info": {"id": tenant_id, "identifier": tenant_identifier, "function_name": name}}
        logger.info(f"Enviando función '{name}' a webhook para {tenant_identifier}")
        
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info(f"Webhook para {tenant_identifier} respondió con status {r.status_code}")
        
        return {
            "success": True, 
            "status_code": r.status_code, 
            "function_name": name,
            "response_data": r.json() if r.content and 'application/json' in r.headers.get('Content-Type','').lower() else {"raw_response": r.text[:200]}
        }
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"Error HTTP en webhook para {tenant_identifier} (Status: {getattr(http_err.response, 'status_code', 'N/A')})")
        return {"success": False, "error": f"Error webhook ({getattr(http_err.response, 'status_code', 'N/A')})."}
    except Exception as e: 
        logger.error(f"Error general en función '{name}' para {tenant_identifier}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel):
    history: list[ChatMessage]; conversation_id: str | None = None; user_id_external: str | None = None
    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list): raise ValueError("history debe ser una lista")
        return v[-200:] if len(v) > 200 else v
class ChatResponseData(BaseModel): type: str = "text"; text: str | None
class ChatApiResponse(BaseModel): response: ChatResponseData

@app.post("/token", summary="Obtener JWT para tenant")
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(503, "Servicio BD no disponible")
    resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active", True).execute()
    if not resp.data: raise HTTPException(401, "No hay tenants activos o configuración incorrecta")
    for row in resp.data:
        if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
            token = create_access_token({"tenant_id": str(row["id"]), "identifier": row["identifier"]})
            return {"response": {"access_token": token, "token_type": "bearer"}}
    raise HTTPException(401, "API Key inválida o inquilino inactivo")

@app.post("/chat", response_model=ChatApiResponse, summary="Chat multi-tenant con Responses API")
async def chat_endpoint(data: ChatRequest = Body(...), tenant: TokenData = Depends(get_current_tenant)):
    if not supabase: raise HTTPException(503, "Servicio BD no disponible")
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"/chat para tenant {ident} (ID: {tid})")

    try:
        system_md = load_supabase_file(SUPABASE_BUCKET, f"{ident}/system_message_{ident}.md")
        tools_json_str = load_supabase_file(SUPABASE_BUCKET, f"{ident}/functions_{ident}.json")
        custom_tools = json.loads(tools_json_str)

        vs_conf_resp = supabase.table("tenants").select("openai_vector_store_id").eq("id", tid).single().execute()
        vsid = vs_conf_resp.data.get("openai_vector_store_id") if vs_conf_resp.data and vs_conf_resp.data.get("openai_vector_store_id") else None
        if vsid: logger.info(f"Vector Store ID para {ident}: {vsid}")
        else: logger.info(f"No se utilizará Vector Store (file_search) para {ident} (no configurado o vacío).")

        key = get_tenant_openai_key(tid, ident)
        client = AsyncOpenAI(api_key=key)
        messages_for_api = [{"role": "system", "content": system_md}] + [m.model_dump() for m in data.history]
        
        tools_for_api = []
        if vsid: tools_for_api.append({"type": "file_search", "vector_store_ids": [vsid]})
        if custom_tools: tools_for_api.extend(custom_tools)
        effective_tools = tools_for_api if tools_for_api else []

        logger.info(f"Llamando a OpenAI Responses API para {ident}. Modelo: gpt-4.1. Tools: {len(effective_tools)}")
        response1 = await client.responses.create(model="gpt-4.1", input=messages_for_api, tools=effective_tools)

        detected_fc, initial_text = None, ""
        for item in response1.output:
            if getattr(item, "type", "") == "function_call":
                detected_fc = item; logger.info(f"Función detectada: {detected_fc.name} para {ident}"); break
            if hasattr(item, "content") and item.content:
                for chunk in item.content:
                    if hasattr(chunk, "text"): initial_text += chunk.text
        final_text_to_return = initial_text

        if detected_fc:
            logger.info(f"Procesando función '{detected_fc.name}' para {ident}")
            args = json.loads(detected_fc.arguments)
            fc_result = None
            
            # ✅ LÓGICA GENÉRICA: Solo validación especial para recolectarInformacionContacto de Blimark Tech
            if detected_fc.name == "recolectarInformacionContacto" and ident == "blimark_tech":
                required_fields = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
                missing_fields = [f for f in required_fields if not args.get(f,"").strip()]
                if missing_fields:
                    final_text_to_return = f"Para procesar tu solicitud de contacto, necesito: {', '.join(missing_fields)}. ¿Podrías proporcionarlos?"
                    logger.info(f"Respondiendo con solicitud de campos faltantes para {ident}.")
                else:
                    fc_result = await handle_function_call(detected_fc, tid, ident)
                    logger.info(f"Resultado de {detected_fc.name} para {ident}: {fc_result}")
            else:
                # ✅ PARA TODAS LAS DEMÁS FUNCIONES: Ejecución directa sin validación de campos
                fc_result = await handle_function_call(detected_fc, tid, ident)
                logger.info(f"Resultado de {detected_fc.name} para {ident}: {fc_result}")
            
            if fc_result is not None:
                messages_for_api.append({"type": "function_call", "call_id": detected_fc.call_id, "name": detected_fc.name, "arguments": detected_fc.arguments})
                messages_for_api.append({"type": "function_call_output", "call_id": detected_fc.call_id, "output": json.dumps(fc_result)})
                logger.info(f"Realizando segunda llamada a Responses API para {ident} con resultado de función.")
                response2 = await client.responses.create(model="gpt-4.1", input=messages_for_api, tools=effective_tools)
                text_from_second_call = ""
                for item2 in response2.output:
                    if hasattr(item2, "content") and item2.content:
                        for chunk2 in item2.content:
                            if hasattr(chunk2, "text"): text_from_second_call += chunk2.text
                final_text_to_return = text_from_second_call or "Información procesada."
                logger.info(f"Texto de segunda llamada para {ident}: '{final_text_to_return}'")

                # ✅ LÓGICA GENÉRICA: Busca texto post-función para CUALQUIER función exitosa
                if fc_result.get("success", False):
                    clave_texto_post_funcion = f"post_text_{detected_fc.name}"
                    texto_personalizado = get_data_critico(tid, clave_texto_post_funcion, ident)
                    if texto_personalizado and texto_personalizado not in final_text_to_return:
                        final_text_to_return += f"\n\n{texto_personalizado}"
                        logger.info(f"Texto personalizado para '{clave_texto_post_funcion}' añadido para {ident}.")
                    else:
                        logger.info(f"Función '{detected_fc.name}' exitosa, pero no se encontró texto personalizado con clave '{clave_texto_post_funcion}' para {ident}.")
        else:
            final_text_to_return = initial_text or "No pude generar una respuesta en este momento."
            logger.info(f"Texto de primera llamada (sin función) para {ident}: '{final_text_to_return}'")

        conv_id = data.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
        user_ext_id = data.user_id_external or "unknown_user"
        if data.history:
            last_user_msg = data.history[-1]
            supabase.table("chat_history").insert({"tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id, "role": last_user_msg.role, "content": last_user_msg.content}).execute()
        if final_text_to_return:
            supabase.table("chat_history").insert({"tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id, "role": "assistant", "content": final_text_to_return}).execute()
        return ChatApiResponse(response=ChatResponseData(text=final_text_to_return))

    except BadRequestError as e:
        error_detail = str(e); logger.error(f"BadRequestError en /chat para {ident}: {error_detail}", exc_info=True)
        if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and "message" in e.body: error_detail = e.body["message"]
        elif hasattr(e, 'message'): error_detail = e.message
        raise HTTPException(status_code=400, detail=f"Error de OpenAI: {error_detail}")
    except RuntimeError as e:
        logger.error(f"RuntimeError en /chat para {ident}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error de configuración del servidor: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado en /chat para {ident}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno inesperado procesando su solicitud.")

@app.get("/health", summary="Health check")
async def health_check(): return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)