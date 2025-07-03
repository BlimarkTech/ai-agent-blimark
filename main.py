# -*- coding: utf-8 -*-

import os
import logging
import json
import asyncio
import requests
from typing import Dict, Any, AsyncGenerator, List, Optional
from datetime import datetime, timedelta, timezone
from json.decoder import JSONDecodeError

from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Depends, status, Form, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

from openai import AsyncOpenAI, OpenAIError
from supabase import create_client, Client
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from pinecone import Pinecone, PineconeException
from langdetect import detect as detect_language, LangDetectException, DetectorFactory

# --- Setup ---
DetectorFactory.seed = 0
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_tenant_agent_final")

# --- Constants & Config ---
REQUIRED_VARS = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "JWT_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "PINECONE_API_KEY"]
if missing_vars := [v for v in REQUIRED_VARS if not os.getenv(v)]:
    raise RuntimeError(f"Faltan variables de entorno obligatorias: {', '.join(missing_vars)}")

SUPABASE_URL, SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config")
JWT_SECRET_KEY, JWT_ALGORITHM = os.getenv("JWT_SECRET_KEY"), os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPPORTED_LANG_CODES = {"es", "it", "pt", "en", "fr", "de"}

# --- Global Cache & Clients ---
TENANT_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
PINECONE_INDEX_CACHE: Dict[str, Any] = {}
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Clientes de Supabase y Pinecone inicializados.")
except Exception as e:
    logger.error(f"Error inicializando clientes de servicio: {e}", exc_info=True)
    supabase, pc = None, None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
cipher = Fernet(ENCRYPTION_MASTER_KEY)
app = FastAPI(title="Agente IA Multi-Tenant (Final Version)", version="10.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
from pydantic import BaseModel, EmailStr

class TokenData(BaseModel): tenant_id: str; identifier: str
class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel):
    history: List[ChatMessage]
    conversation_id: Optional[str] = None
    user_id_external: Optional[str] = None

# --- Auth & Security ---
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

# --- Helper Functions ---
def load_supabase_file(path: str) -> str:
    return supabase.storage.from_(SUPABASE_BUCKET).download(path).decode("utf-8")

# FIX: Re-added the get_data_critico function from the old codebase
def get_data_critico(tenant_id: str, clave: str) -> str | None:
    if not supabase: return None
    try:
        resp = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        return resp.data.get("valor") if resp and resp.data else None
    except Exception as e:
        logger.error(f"Excepción al obtener dato crítico '{clave}': {e}")
        return None

async def get_or_load_tenant_config(tenant_id: str, tenant_identifier: str) -> Dict[str, Any]:
    if tenant_id in TENANT_CONFIG_CACHE:
        return TENANT_CONFIG_CACHE[tenant_id]
    
    logger.info(f"Cargando configuración para tenant '{tenant_identifier}' desde Supabase.")
    try:
        db_resp = supabase.table("tenants").select("openai_api_key_encrypted, pinecone_index_name, pinecone_namespace, default_language_code, recoleccion_webhook_url").eq("id", tenant_id).single().execute()
        if not db_resp.data: raise RuntimeError("Tenant no encontrado en DB.")
        
        config = db_resp.data
        config["openai_api_key_decrypted"] = cipher.decrypt(config["openai_api_key_encrypted"].encode()).decode()
        config["system_md_template"] = load_supabase_file(f"{tenant_identifier}/system_message_{tenant_identifier}.md")
        config["custom_tools"] = json.loads(load_supabase_file(f"{tenant_identifier}/functions_{tenant_identifier}.json"))
        
        TENANT_CONFIG_CACHE[tenant_id] = config
        logger.info(f"✅ Configuración para '{tenant_identifier}' cargada y cacheada.")
        return config
    except Exception as e:
        logger.error(f"❌ Error crítico cargando config para '{tenant_identifier}': {e}", exc_info=True)
        raise RuntimeError("Fallo en la carga de configuración del tenant.")

def handle_function_call(tool_call: dict, tenant_config: dict, tenant_info: dict) -> dict:
    function_def = tool_call.get("function", {})
    name, args_str = function_def.get("name"), function_def.get("arguments")
    
    if not name or not args_str:
        return {"success": False, "error": "Llamada a función malformada."}
        
    logger.info(f"Procesando webhook '{name}' para '{tenant_info['identifier']}'")
    try:
        webhook_url = tenant_config.get("recoleccion_webhook_url")
        if not webhook_url: return {"success": False, "error": "URL de webhook no configurada."}

        args_dict = json.loads(args_str)
        payload = {**args_dict, "_tenant_info": tenant_info}
        response = requests.post(webhook_url, json=payload, timeout=15)
        response.raise_for_status()
        
        try:
            return {"success": True, "data": response.json()}
        except JSONDecodeError:
            return {"success": True, "data": "Webhook ejecutado exitosamente sin contenido de respuesta."}
            
    except Exception as e:
        logger.error(f"Error en webhook para '{name}': {e}")
        return {"success": False, "error": str(e)}

def save_chat_history_to_db(tenant_id: str, conv_id: str, user_ext_id: str, user_msg: str, assistant_msg: str):
    try:
        supabase.table("chat_history").insert([
            {"tenant_id": tenant_id, "conversation_id": conv_id, "user_id_external": user_ext_id, "role": "user", "content": user_msg},
            {"tenant_id": tenant_id, "conversation_id": conv_id, "user_id_external": user_ext_id, "role": "assistant", "content": assistant_msg}
        ]).execute()
        logger.info(f"📚 Historial guardado para conv_id: {conv_id}")
    except Exception as e:
        logger.error(f"Error guardando historial para conv_id {conv_id}: {e}")

# --- Core Logic ---
async def stream_chat_generator(messages: list, tools: list, client: AsyncOpenAI, config: dict, tenant_info: dict, bg_tasks: BackgroundTasks, req_data: ChatRequest) -> AsyncGenerator[str, None]:
    full_assistant_response = ""
    tool_calls = []
    
    try:
        stream = await client.chat.completions.create(model="gpt-4.1", messages=messages, tools=tools, stream=True, tool_choice="auto")
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_assistant_response += delta.content
                yield f"data: {json.dumps({'type': 'content_delta', 'data': delta.content})}\n\n"
            
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if len(tool_calls) <= tc_chunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id: tc["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name: tc["function"]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments: tc["function"]["arguments"] += tc_chunk.function.arguments

        if tool_calls:
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            
            for tool_call in tool_calls:
                result = handle_function_call(tool_call, config, tenant_info)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)})

                # FIX: Re-added the logic to fetch and append critical data (like the planner link)
                if result.get("success"):
                    function_name = tool_call.get("function", {}).get("name")
                    if function_name:
                        clave_critica = f"post_text_{function_name}"
                        texto_adicional = get_data_critico(tenant_info["tenant_id"], clave_critica)
                        if texto_adicional:
                            full_assistant_response += f"\n\n{texto_adicional}"
                            yield f"data: {json.dumps({'type': 'content_delta', 'data': f'\\n\\n{texto_adicional}'})}\n\n"
                            logger.info(f"Texto crítico '{clave_critica}' añadido a la respuesta.")

            stream2 = await client.chat.completions.create(model="gpt-4.1", messages=messages, stream=True)
            async for chunk in stream2:
                if chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    full_assistant_response += delta_content
                    yield f"data: {json.dumps({'type': 'content_delta', 'data': delta_content})}\n\n"

    except OpenAIError as e:
        logger.error(f"Error de OpenAI para '{tenant_info['identifier']}': {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': 'Error comunicándose con el asistente.'})}\n\n"
        return
    finally:
        conv_id = req_data.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
        if full_assistant_response and req_data.history:
            bg_tasks.add_task(save_chat_history_to_db, tenant_info["tenant_id"], conv_id, req_data.user_id_external or "unknown", req_data.history[-1].content, full_assistant_response.strip())
        yield f"data: {json.dumps({'type': 'end_of_stream', 'conversation_id': conv_id})}\n\n"

# --- API Endpoints ---
@app.post("/token")
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE)
    resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active", True).execute()
    for row in resp.data:
        if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
            token = create_access_token({"tenant_id": str(row["id"]), "identifier": row["identifier"]})
            return {"response": {"access_token": token, "token_type": "bearer"}}
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API Key inválida o inquilino inactivo")

@app.post("/chat")
async def chat_endpoint(bg_tasks: BackgroundTasks, data: ChatRequest, tenant: TokenData = Depends(get_current_tenant)):
    if not all([supabase, pc]): raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        config = await get_or_load_tenant_config(tenant.tenant_id, tenant.identifier)
        client = AsyncOpenAI(api_key=config["openai_api_key_decrypted"])
        
        lang = config.get("default_language_code", "es")
        if data.history and (user_msg := data.history[-1].content):
            try:
                if (detected := detect_language(user_msg)) in SUPPORTED_LANG_CODES: lang = detected
            except LangDetectException: pass
        
        system_msg = config["system_md_template"] + f"\n\n[Instrucción: Idioma es '{lang}'. Responde en este idioma.]"
        
        rag_ctx = ""
        if (idx := config.get("pinecone_index_name")) and (ns := config.get("pinecone_namespace")) and data.history:
            try:
                if idx not in PINECONE_INDEX_CACHE:
                    PINECONE_INDEX_CACHE[idx] = pc.Index(idx)
                pinecone_index = PINECONE_INDEX_CACHE[idx]
                emb = await client.embeddings.create(input=[data.history[-1].content], model="text-embedding-3-small")
                matches = pinecone_index.query(namespace=ns, vector=emb.data[0].embedding, top_k=5, include_metadata=True).matches
                if relevant_texts := [m.metadata['text'] for m in matches if m.score > 0.3 and 'text' in m.metadata]:
                    rag_ctx = "\n\n--- CONTEXTO ---\n" + "\n\n".join(relevant_texts) + "\n--- FIN CONTEXTO ---"
            except Exception as e:
                logger.error(f"Error en RAG para '{tenant.identifier}': {e}")
        
        messages = [{"role": "system", "content": system_msg + rag_ctx}] + [msg.model_dump() for msg in data.history]
        
        generator = stream_chat_generator(messages, config["custom_tools"], client, config, tenant.model_dump(), bg_tasks, data)
        return StreamingResponse(generator, media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error INESPERADO en /chat para '{tenant.identifier}': {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": f"Error interno: {e}"})

@app.get("/health")
async def health_check(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
