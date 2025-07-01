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

from openai import AsyncOpenAI, OpenAIError
from supabase import create_client, Client
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from pinecone import Pinecone, PineconeException
from langdetect import detect as detect_language, LangDetectException, DetectorFactory
DetectorFactory.seed = 0

load_dotenv()
REQUIRED_VARS = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "JWT_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "PINECONE_API_KEY"]
if missing_vars := [v for v in REQUIRED_VARS if not os.getenv(v)]:
    raise RuntimeError(f"Faltan variables de entorno obligatorias: {', '.join(missing_vars)}")

SUPABASE_URL, SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY, JWT_ALGORITHM = os.getenv("JWT_SECRET_KEY"), os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_tenant_agent_v3")
SUPPORTED_LANG_CODES = {"es", "it", "pt", "en", "fr", "de"}
TENANT_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Clientes de Supabase y Pinecone inicializados.")
except Exception as e:
    logger.error(f"Error inicializando clientes de servicio: {e}", exc_info=True)
    supabase, pc = None, None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cipher = Fernet(ENCRYPTION_MASTER_KEY)
app = FastAPI(title="Agente IA Multi-Tenant v3 (Chat Completions Fix)", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TokenData(BaseModel): tenant_id: str; identifier: str
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN))
    return jwt.encode({**data, "exp": expire}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_tenant(
    token_header: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="bp-access-token")
) -> TokenData:
    token = token_header or token_query
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No se proporcionó token de autenticación")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if not (tid := payload.get("tenant_id")) or not (ident := payload.get("identifier")):
            raise JWTError("Token inválido")
        return TokenData(tenant_id=tid, identifier=ident)
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Token inválido o expirado: {e}")

# ... (Las demás funciones helper como load_supabase_file, get_tenant_config_from_db, etc. se mantienen igual)
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

class EmailCheckModel(BaseModel): email: EmailStr

def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    name, args_str = call_obj.function.name, call_obj.function.arguments
    logger.info(f"Procesando webhook '{name}' para {tenant_identifier}")
    if not supabase: return {"success": False, "error": "Servicio de configuración no disponible."}
    
    try:
        resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
        url = resp_webhook.data.get("recoleccion_webhook_url") if resp_webhook.data else None
        if not url: return {"success": False, "error": "Falta URL de webhook."}

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
        logger.info(f"📚 Historial guardado para conv_id: {conversation_id}")
    except Exception as e:
        logger.error(f"Error guardando historial para conv_id {conversation_id}: {e}", exc_info=True)

class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel):
    history: List[ChatMessage]
    conversation_id: Optional[str] = None
    user_id_external: Optional[str] = None

@app.post("/token")
async def token_endpoint(api_key_botpress: str = Form(...)):
    # ... (código sin cambios)
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

async def stream_chat_generator(
    messages: list, tools: list, client: AsyncOpenAI, tenant_info: dict, bg_tasks: BackgroundTasks, req_data: ChatRequest
) -> AsyncGenerator[str, None]:
    tid, ident = tenant_info["id"], tenant_info["identifier"]
    full_assistant_response = ""
    
    try:
        stream = await client.chat.completions.create(model="gpt-4.1", messages=messages, tools=tools, stream=True)
        
        tool_calls = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_assistant_response += delta.content
                yield f"data: {json.dumps({'type': 'content_delta', 'data': delta.content})}\n\n"
            
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if tc_chunk.index >= len(tool_calls):
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id: tc["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name: tc["function"]["name"] = tc_chunk.function.name
                        if tc_chunk.function.arguments: tc["function"]["arguments"] += tc_chunk.function.arguments

        if tool_calls:
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            
            # For simplicity, assuming one tool call, but can be extended
            tool_call_obj = type('obj', (object,), tool_calls[0])() 
            result = handle_function_call(tool_call_obj, tid, ident)
            
            messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": json.dumps(result)})
            
            stream2 = await client.chat.completions.create(model="gpt-4.1", messages=messages, stream=True)
            async for chunk in stream2:
                if chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    full_assistant_response += delta_content
                    yield f"data: {json.dumps({'type': 'content_delta', 'data': delta_content})}\n\n"

    except OpenAIError as e:
        logger.error(f"Error en stream de OpenAI para '{ident}': {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': 'Error comunicándose con el asistente.'})}\n\n"
        return
    finally:
        user_msg = req_data.history[-1].content if req_data.history else ""
        conv_id = req_data.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
        user_ext_id = req_data.user_id_external or "unknown"
        if full_assistant_response:
            bg_tasks.add_task(save_chat_history_to_db, tid, conv_id, user_ext_id, user_msg, full_assistant_response.strip())
        
        yield f"data: {json.dumps({'type': 'end_of_stream', 'conversation_id': conv_id})}\n\n"


@app.post("/chat")
async def chat_endpoint(bg_tasks: BackgroundTasks, data: ChatRequest, tenant: TokenData = Depends(get_current_tenant)):
    if not all([supabase, pc]):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicios esenciales no disponibles.")
    
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"🚀 /chat (v3) para tenant '{ident}' (ID: {tid}).")
    
    try:
        config = await get_or_load_tenant_config(tid, ident)
        client = AsyncOpenAI(api_key=config["openai_api_key_decrypted"])
        
        lang = config.get("default_language_code", "es")
        if data.history and (user_msg := data.history[-1].content):
            try:
                detected = detect_language(user_msg)
                if detected in SUPPORTED_LANG_CODES: lang = detected
            except LangDetectException: pass
        
        system_msg = config["system_md_template"] + f"\n\n[Instrucción: Idioma de la conversación es '{lang}'. Responde en este idioma.]"
        
        rag_ctx = ""
        if (idx := config.get("pinecone_index_name")) and (ns := config.get("pinecone_namespace")) and data.history:
            try:
                emb = await client.embeddings.create(input=[data.history[-1].content], model="text-embedding-3-small")
                matches = pc.Index(idx).query(namespace=ns, vector=emb.data[0].embedding, top_k=5, include_metadata=True).matches
                relevant = [m.metadata['text'] for m in matches if m.score > 0.3 and 'text' in m.metadata]
                if relevant:
                    rag_ctx = "\n\n--- CONTEXTO ---\n" + "\n\n".join(relevant) + "\n--- FIN CONTEXTO ---"
                    logger.info(f"🎯 Contexto RAG construido para '{ident}'.")
            except (PineconeException, OpenAIError) as e:
                logger.error(f"❌ Error durante RAG para '{ident}': {e}")
        
        messages = [{"role": "system", "content": system_msg + rag_ctx}] + [msg.model_dump() for msg in data.history]
        
        generator = stream_chat_generator(messages, config["custom_tools"], client, {"id": tid, "identifier": ident}, bg_tasks, data)
        return StreamingResponse(generator, media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error INESPERADO en /chat para '{ident}': {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": f"Error interno: {e}"})

@app.get("/health")
async def health_check(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)