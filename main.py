# -*- coding: utf-8 -*-

import os
import logging
import json
import asyncio
import requests
from typing import Dict, Any, AsyncGenerator, List, Optional
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Depends, status, Form, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, EmailStr, ValidationError, field_validator

# OpenAI y Supabase
from openai import AsyncOpenAI, OpenAIError
from supabase import create_client, Client

# JWT y Criptografía
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet

# --- IMPORTACIÓN PARA PINECONE ---
from pinecone import Pinecone, PineconeException

# --- IMPORTACIÓN PARA DETECCIÓN DE IDIOMA ---
from langdetect import detect as detect_language, LangDetectException, DetectorFactory
DetectorFactory.seed = 0

# ----------------------------
# Carga y validación de variables de entorno
# ----------------------------
load_dotenv()
REQUIRED_VARS = [
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "JWT_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY", "PINECONE_API_KEY"
]
missing_vars = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing_vars:
    raise RuntimeError(f"Faltan variables de entorno obligatorias: {', '.join(missing_vars)}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_tenant_agent_v2")

# ----------------------------
# Lista de idiomas soportados
# ----------------------------
SUPPORTED_LANG_CODES = {"es", "it", "pt", "en", "fr", "de"}

# ----------------------------
# Global Tenant Configuration Cache
# ----------------------------
TENANT_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
logger.info("Caché de configuración de tenants inicializada.")

# ----------------------------
# Clientes de servicio
# ----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Conexión a Supabase establecida.")
except Exception as e:
    logger.error(f"Error conectando a Supabase: {e}", exc_info=True)
    supabase = None

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Cliente de Pinecone inicializado.")
except Exception as e:
    logger.error(f"Error inicializando cliente de Pinecone: {e}", exc_info=True)
    pc = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cipher = Fernet(ENCRYPTION_MASTER_KEY)

# ----------------------------
# FastAPI app y CORS
# ----------------------------
app = FastAPI(title="Agente IA Multi-Tenant v2 (Streaming + Cache)", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body_bytes = await request.body()
    logger.error(f"RequestValidationError: {exc.errors()} – Body: {body_bytes.decode('utf-8', errors='replace')}")
    return await request_validation_exception_handler(request, exc)

# --- Modelos Pydantic y JWT ---
class TokenData(BaseModel):
    tenant_id: str
    identifier: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant(
    token_header: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="bp-access-token")
) -> TokenData:
    
    token = token_header or token_query

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se proporcionó token de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )

    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tid, ident = payload.get("tenant_id"), payload.get("identifier")
        if not tid or not ident:
            raise creds_exc
        return TokenData(tenant_id=tid, identifier=ident)
    except JWTError:
        raise creds_exc

# --- Helpers ---
def load_supabase_file(bucket: str, path: str) -> str:
    if not supabase: raise RuntimeError("Supabase no disponible.")
    if ".." in path or path.startswith("/"): raise ValueError(f"Ruta inválida: {path}")
    logger.info(f"Descargando archivo de Supabase Storage: {bucket}/{path}")
    try:
        data_bytes = supabase.storage.from_(bucket).download(path)
        logger.info(f"Archivo {path} descargado exitosamente ({len(data_bytes)} bytes).")
        return data_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Error descargando {path} de Supabase: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo cargar archivo de configuración: {path}")

def get_tenant_config_from_db(tenant_id: str, tenant_identifier: str) -> dict:
    if not supabase: raise RuntimeError("Servicio de base de datos no disponible.")
    logger.info(f"Obteniendo configuración de DB para tenant: {tenant_identifier}")
    select_fields = "openai_api_key_encrypted, pinecone_index_name, pinecone_namespace, vector_store_provider, default_language_code, priority_greetings"
    resp = supabase.table("tenants").select(select_fields).eq("id", tenant_id).single().execute()
    if not resp.data: raise RuntimeError(f"Configuración no encontrada para tenant {tenant_identifier}")
    
    encrypted_key = resp.data.get("openai_api_key_encrypted")
    if not encrypted_key: raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
    
    try:
        resp.data["openai_api_key_decrypted"] = cipher.decrypt(encrypted_key.encode()).decode()
        return resp.data
    except Exception as e:
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}: {e}")

async def get_or_load_tenant_config(tenant_id: str, tenant_identifier: str) -> Dict[str, Any]:
    if tenant_id in TENANT_CONFIG_CACHE:
        logger.info(f"✅ Configuración para tenant '{tenant_identifier}' encontrada en caché.")
        return TENANT_CONFIG_CACHE[tenant_id]

    logger.info(f"⏳ Configuración para tenant '{tenant_identifier}' no encontrada en caché. Cargando...")
    try:
        config_from_db = get_tenant_config_from_db(tenant_id, tenant_identifier)
        system_md_template = load_supabase_file(SUPABASE_BUCKET, f"{tenant_identifier}/system_message_{tenant_identifier}.md")
        tools_json_str = load_supabase_file(SUPABASE_BUCKET, f"{tenant_identifier}/functions_{tenant_identifier}.json")
        custom_tools = json.loads(tools_json_str)

        full_config = {**config_from_db, "system_md_template": system_md_template, "custom_tools": custom_tools}
        TENANT_CONFIG_CACHE[tenant_id] = full_config
        logger.info(f"✅ Configuración para tenant '{tenant_identifier}' cargada y guardada en caché.")
        return full_config
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ Error crítico al cargar la configuración para '{tenant_identifier}': {e}", exc_info=True)
        raise RuntimeError(f"No se pudo cargar la configuración completa para el tenant {tenant_identifier}.") from e

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None:
    if not supabase: return None
    try:
        resp = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        return resp.data.get("valor") if resp and resp.data else None
    except Exception: return None

class EmailCheckModel(BaseModel): email: EmailStr

async def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    name, args_str = call_obj.name, call_obj.arguments
    logger.info(f"Procesando webhook '{name}' para {tenant_identifier}")
    if not supabase: return {"success": False, "error": "Servicio de configuración no disponible."}
    
    resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
    url = resp_webhook.data.get("recoleccion_webhook_url") if resp_webhook.data else None
    if not url: return {"success": False, "error": "Falta URL de webhook."}

    try:
        args_dict = json.loads(args_str)
        if "email" in args_dict: EmailCheckModel(email=args_dict["email"])
        
        payload = {**args_dict, "_tenant_info": {"id": tenant_id, "identifier": tenant_identifier, "function_name": name}}
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        
        return {"success": True, "status_code": r.status_code, "function_name": name, "response_data": r.json()}
    except ValidationError as ve: return {"success": False, "error": f"Email inválido: {args_dict.get('email')}"}
    except requests.RequestException as e: return {"success": False, "error": f"Error de comunicación con webhook: {e}"}
    except Exception as e: return {"success": False, "error": f"Error interno procesando función: {e}"}

def save_chat_history_to_db(tenant_id: str, conversation_id: str, user_id_external: str, user_message: str, assistant_response: str):
    if not supabase:
        logger.error("Supabase no disponible. No se puede guardar historial de chat.")
        return
    try:
        records = [
            {"tenant_id": tenant_id, "conversation_id": conversation_id, "user_id_external": user_id_external, "role": "user", "content": user_message},
            {"tenant_id": tenant_id, "conversation_id": conversation_id, "user_id_external": user_id_external, "role": "assistant", "content": assistant_response}
        ]
        supabase.table("chat_history").insert(records).execute()
        logger.info(f"📚 Historial de chat guardado en background para conv_id: {conversation_id}")
    except Exception as e:
        logger.error(f"Error guardando historial en background para conv_id {conversation_id}: {e}", exc_info=True)

# --- Modelos Pydantic para Chat ---
class ChatMessage(BaseModel): role: str; content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]
    conversation_id: str | None = None
    user_id_external: str | None = None

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list): raise ValueError("history debe ser una lista")
        return v[-200:]

# --- Endpoint de Token ---
@app.post("/token", summary="Obtener JWT para tenant")
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicio de base de datos no disponible.")
    try:
        resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active", True).execute()
        if not resp.data: raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Configuración inválida.")
        
        for row in resp.data:
            if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
                token = create_access_token({"tenant_id": str(row["id"]), "identifier": row["identifier"]})
                logger.info(f"Token generado para tenant: {row['identifier']}")
                return {"response": {"access_token": token, "token_type": "bearer"}}
        
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API Key inválida o inquilino inactivo")
    except Exception as e:
        logger.error(f"Error inesperado en autenticación: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno de autenticación.")

# --- Generador de Chat con Streaming ---
async def stream_chat_generator(
    input_messages: list, tools: list, openai_client: AsyncOpenAI, tenant_info: dict,
    rag_context: str, background_tasks: BackgroundTasks, request_data: ChatRequest
) -> AsyncGenerator[str, None]:
    
    tid, ident = tenant_info["id"], tenant_info["identifier"]
    full_assistant_response = ""

    async def yield_error(message: str):
        payload = {"type": "error", "data": message}
        yield f"data: {json.dumps(payload)}\n\n"

    try:
        logger.info(f"🤖 STREAM 1: Llamando a OpenAI para '{ident}'. RAG usado: {'SÍ' if rag_context else 'NO'}")
        stream1 = await openai_client.chat.completions.create(model="gpt-4.1", messages=input_messages, tools=tools, stream=True)
        
        accumulated_tool_calls = []
        async for chunk in stream1:
            delta = chunk.choices[0].delta
            if delta.content:
                full_assistant_response += delta.content
                payload = {"type": "content_delta", "data": delta.content}
                yield f"data: {json.dumps(payload)}\n\n"
            if delta.tool_calls:
                # Assuming one tool call for simplicity as per original logic
                if not accumulated_tool_calls:
                    accumulated_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                
                tc_chunk = delta.tool_calls[0]
                if tc_chunk.id:
                    accumulated_tool_calls[0]["id"] = tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        accumulated_tool_calls[0]["function"]["name"] = tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        accumulated_tool_calls[0]["function"]["arguments"] += tc_chunk.function.arguments

        if accumulated_tool_calls:
            tool_call = accumulated_tool_calls[0]
            logger.info(f"🔧 Función detectada en stream para '{ident}': {tool_call['function']['name']}")
            
            # Reconstruct the tool call object for handle_function_call
            class MockCall:
                def __init__(self, name, args):
                    self.name = name
                    self.arguments = args
            
            fc_item = MockCall(tool_call['function']['name'], tool_call['function']['arguments'])
            fc_result = await handle_function_call(fc_item, tid, ident)
            
            input_messages.append({"role": "assistant", "tool_calls": [tool_call]})
            input_messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(fc_result)})

            logger.info(f"📞 STREAM 2: Llamando a OpenAI con resultado de función para '{ident}'.")
            stream2 = await openai_client.chat.completions.create(model="gpt-4.1", messages=input_messages, stream=True)
            
            async for chunk in stream2:
                if chunk.choices[0].delta.content:
                    text_delta = chunk.choices[0].delta.content
                    full_assistant_response += text_delta
                    payload = {"type": "content_delta", "data": text_delta}
                    yield f"data: {json.dumps(payload)}\n\n"

    except OpenAIError as e:
        logger.error(f"Error en stream de OpenAI para '{ident}': {e}", exc_info=True)
        await yield_error("Error comunicándose con el asistente.")
        return

    last_user_message = request_data.history[-1].content if request_data.history else ""
    conv_id = request_data.conversation_id or f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{ident}"
    user_ext_id = request_data.user_id_external or "unknown_user"

    background_tasks.add_task(save_chat_history_to_db, tid, conv_id, user_ext_id, last_user_message, full_assistant_response.strip())
    
    payload = {"type": "end_of_stream", "conversation_id": conv_id}
    yield f"data: {json.dumps(payload)}\n\n"


# --- Endpoint /chat con Cache y Streaming ---
@app.post("/chat", summary="Chat multi-tenant con streaming, caché y RAG")
async def chat_endpoint(
    background_tasks: BackgroundTasks, data: ChatRequest = Body(...), tenant: TokenData = Depends(get_current_tenant)
):
    if not supabase or not pc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicios esenciales no disponibles.")

    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"🚀 Nuevo /chat (STREAMING) para tenant '{ident}' (ID: {tid}).")

    try:
        tenant_config = await get_or_load_tenant_config(tid, ident)
        openai_client = AsyncOpenAI(api_key=tenant_config["openai_api_key_decrypted"])
        
        tenant_default_lang = tenant_config.get("default_language_code", "es").lower()
        initial_lang_code = tenant_default_lang
        if data.history and (user_msg := data.history[-1].content.strip()):
            try:
                detected = detect_language(user_msg)
                if detected in SUPPORTED_LANG_CODES: initial_lang_code = detected
            except LangDetectException: pass
        system_md = tenant_config["system_md_template"] + f"\n\n[Instrucción: El idioma de la conversación es '{initial_lang_code}'. Responde en este idioma.]"
        
        rag_context_str = ""
        if (idx := tenant_config.get("pinecone_index_name")) and (ns := tenant_config.get("pinecone_namespace")) and data.history:
            try:
                user_query = data.history[-1].content
                emb_res = await openai_client.embeddings.create(input=[user_query], model="text-embedding-3-small")
                pinecone_index = pc.Index(idx)
                query_res = pinecone_index.query(namespace=ns, vector=emb_res.data[0].embedding, top_k=5, include_metadata=True)
                relevant = [m.metadata.get('text', '') for m in query_res.matches if m.score > 0.30 and m.metadata]
                if relevant:
                    rag_context_str = "\n\n--- CONTEXTO ---\n" + "\n\n".join(relevant) + "\n--- FIN CONTEXTO ---"
                    logger.info(f"🎯 Contexto RAG construido para '{ident}'.")
            except (PineconeException, OpenAIError) as e:
                logger.error(f"❌ Error durante RAG para '{ident}': {e}")

        effective_system_message = system_md + rag_context_str
        input_messages = [{"role": "system", "content": effective_system_message}] + [msg.model_dump() for msg in data.history]
        
        generator = stream_chat_generator(
            input_messages=input_messages, tools=tenant_config["custom_tools"], openai_client=openai_client,
            tenant_info={"id": tid, "identifier": ident}, rag_context=rag_context_str,
            background_tasks=background_tasks, request_data=data
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error de configuración del servidor: {e}")
    except Exception as e:
        logger.error(f"Error INESPERADO en /chat para '{ident}': {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Error interno inesperado."})

# --- Health Check y Uvicorn run ---
@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload_flag = os.environ.get("DEV_MODE") == "true"
    logger.info(f"Iniciando Uvicorn en host 0.0.0.0 puerto {port} (Reload: {reload_flag})")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_flag)