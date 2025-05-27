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

# OpenAI y Supabase
from openai import AsyncOpenAI, OpenAIError
from supabase import create_client, Client

# JWT y Criptografía
from jose import JWTError, jwt
from passlib.context import CryptContext # Asegúrate de bcrypt==4.0.1 en requirements.txt
from cryptography.fernet import Fernet

# --- IMPORTACIÓN PARA PINECONE ---
from pinecone import Pinecone, PineconeException

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
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config") # Bucket para system_message, functions_json
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY").encode()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_tenant_agent_pinecone_responses_api") # Nombre del logger consistente

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

app = FastAPI(title="Agente IA Multi-Tenant (Pinecone + Responses API)", version="1.3.4") # Nueva versión

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
) 
# La indentación del decorador @app.exception_handler estaba corregida en tu archivo, la mantengo.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body(); logger.error(f"RequestValidationError: {exc.errors()} – Body: {body.decode('utf-8')}")
    return await request_validation_exception_handler(request, exc)

# --- Modelos Pydantic y JWT (Idénticos a tu archivo) ---
class TokenData(BaseModel): tenant_id: str; identifier: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

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

# --- Helpers (Idénticos a tu archivo, con el logging que ya tenías) ---
def load_supabase_file(bucket: str, path: str, as_text: bool = True) -> str | bytes:
    if not supabase: raise RuntimeError("Supabase no disponible.")
    if ".." in path or path.startswith("/"): raise ValueError(f"Ruta inválida: {path}")
    logger.info(f"Descargando archivo de Supabase Storage: {bucket}/{path}")
    try: 
        data_bytes = supabase.storage.from_(bucket).download(path)
        logger.info(f"Archivo {path} descargado exitosamente ({len(data_bytes)} bytes).")
        return data_bytes.decode("utf-8") if as_text else data_bytes
    except Exception as e: 
        logger.error(f"Error descargando {path} de Supabase: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo cargar archivo de configuración: {path}")

def get_tenant_openai_key(tenant_id: str, tenant_identifier: str) -> str:
    if not supabase: raise RuntimeError("Supabase no disponible para obtener API Key de OpenAI.")
    logger.info(f"Obteniendo API Key de OpenAI para tenant: {tenant_identifier} (ID: {tenant_id})")
    resp = supabase.table("tenants").select("openai_api_key_encrypted").eq("id", tenant_id).single().execute()
    if not resp.data: 
        logger.error(f"Tenant {tenant_identifier} no encontrado al buscar API Key de OpenAI.")
        raise RuntimeError(f"Inquilino {tenant_identifier} no encontrado (para API Key OpenAI).")
    encrypted = resp.data.get("openai_api_key_encrypted")
    if not encrypted or not encrypted.strip():
        logger.error(f"API Key de OpenAI encriptada faltante para {tenant_identifier}.")
        raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
    try: 
        decrypted_key = cipher.decrypt(encrypted.encode()).decode()
        logger.info(f"API Key de OpenAI obtenida y desencriptada para {tenant_identifier}.")
        return decrypted_key
    except Exception as e:
        logger.error(f"Error desencriptando API Key de OpenAI para {tenant_identifier}: {e}", exc_info=True)
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}.")

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None:
    if not supabase: logger.error(f"Supabase no disponible al buscar dato crítico '{clave}' para {tenant_identifier}"); return None
    try:
        logger.info(f"Buscando dato crítico '{clave}' para tenant {tenant_identifier} (ID: {tenant_id})")
        resp = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        if resp and hasattr(resp, 'data') and resp.data and isinstance(resp.data, dict):
            valor = resp.data.get("valor")
            if valor: logger.info(f"Dato crítico '{clave}' encontrado para {tenant_identifier}: '{valor[:50]}...'"); return valor
        logger.info(f"Dato crítico '{clave}' no encontrado o valor vacío para {tenant_identifier}.")
        return None
    except Exception as e: logger.error(f"Excepción al obtener dato crítico '{clave}' para {tenant_identifier}: {e}", exc_info=True); return None

class EmailCheckModel(BaseModel): email: EmailStr

async def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    name = call_obj.name
    arguments_str = call_obj.arguments
    logger.info(f"Procesando llamada a función (webhook): '{name}' para tenant {tenant_identifier} con argumentos: {arguments_str}")
    
    if not supabase: return {"success": False, "error": "Servicio de configuración (Supabase) no disponible para manejar función."}
    
    resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
    if not resp_webhook.data or not resp_webhook.data.get("recoleccion_webhook_url"):
        logger.warning(f"No hay URL de webhook configurada en tabla 'tenants' para {tenant_identifier} para la función '{name}'.")
        return {"success": False, "error": f"Función '{name}' no puede ser procesada: falta URL de webhook."}
    
    url = resp_webhook.data.get("recoleccion_webhook_url")
    try:
        args_dict = json.loads(arguments_str)
        if "email" in args_dict and args_dict["email"]:
            try: EmailCheckModel(email=args_dict["email"])
            except ValidationError as ve:
                logger.warning(f"Validación de email fallida en handle_function_call para '{args_dict['email']}' en {tenant_identifier}: {ve.errors()}")
                return {"success": False, "error": f"El email proporcionado no es válido: {args_dict['email']}"} 
        
        payload = {**args_dict, "_tenant_info": {"id": tenant_id, "identifier": tenant_identifier, "function_name": name}}
        logger.info(f"Enviando datos de función '{name}' a webhook '{url}' para {tenant_identifier}. Payload: {payload}")
        r = requests.post(url, json=payload, timeout=15) 
        r.raise_for_status() 
        logger.info(f"Webhook para {tenant_identifier} (función '{name}') respondió con status {r.status_code}.")
        
        try:
            response_data_webhook = r.json()
        except json.JSONDecodeError:
            logger.warning(f"Respuesta del webhook para {name} de {tenant_identifier} no es JSON. Raw: {r.text[:200]}")
            response_data_webhook = {"raw_response": r.text[:500]} 

        return {"success": True, "status_code": r.status_code, "function_name": name, "response_data": response_data_webhook}

    except requests.exceptions.Timeout:
        logger.error(f"Timeout llamando a webhook '{url}' para {tenant_identifier} (función '{name}').")
        return {"success": False, "error": "El servicio externo tardó demasiado en responder."}
    except requests.exceptions.HTTPError as http_err:
        status_code_webhook = getattr(http_err.response, 'status_code', 'N/A')
        logger.error(f"Error HTTP {status_code_webhook} en webhook '{url}' para {tenant_identifier} (función '{name}'). Respuesta: {getattr(http_err.response, 'text', 'N/A')[:200]}")
        return {"success": False, "error": f"Error en el servicio externo (código: {status_code_webhook})."}
    except Exception as e:
        logger.error(f"Error general en handle_function_call para '{name}' de {tenant_identifier}: {e}", exc_info=True)
        return {"success": False, "error": f"Error interno procesando la función: {str(e)}"}

# --- Modelos Pydantic para Chat (Idénticos a tu archivo) ---
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

# --- Endpoint de Token (Idéntico a tu archivo) ---
@app.post("/token", summary="Obtener JWT para tenant")
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicio de base de datos no disponible.")
    logger.info(f"Solicitud de token recibida. API Key (primeros 10 chars): {api_key_botpress[:10]}...")
    try:
        resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active", True).execute()
        if not resp.data: 
            logger.warning("No se encontraron tenants activos en la base de datos al solicitar token.")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Configuración de tenants inválida o no hay tenants activos.")
        for row in resp.data:
            if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
                token = create_access_token({"tenant_id": str(row["id"]), "identifier": row["identifier"]})
                logger.info(f"Token generado exitosamente para tenant: {row['identifier']}")
                return {"response": {"access_token": token, "token_type": "bearer"}}
        logger.warning(f"API Key de Botpress no válida o tenant inactivo. API Key (primeros 10): {api_key_botpress[:10]}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API Key inválida o inquilino inactivo")
    except Exception as e:
        logger.error(f"Error inesperado durante la autenticación de token: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno durante la autenticación.")

# --- Endpoint /chat con Pinecone y Responses API ---
@app.post("/chat", response_model=ChatApiResponse, summary="Chat multi-tenant con Pinecone + Responses API")
async def chat_endpoint(data: ChatRequest = Body(...), tenant: TokenData = Depends(get_current_tenant)):
    if not supabase or not pc:
        logger.critical("Servicios esenciales (Supabase/Pinecone) no están disponibles al inicio de /chat.")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicios esenciales no disponibles.")
    
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"🚀 Nuevo /chat para tenant '{ident}' (ID: {tid}). User ID Ext: {data.user_id_external}, Conv ID: {data.conversation_id}")
    
    try:
        # 1. Cargar configuración del tenant y API Key de OpenAI del tenant
        tenant_openai_key = get_tenant_openai_key(tid, ident)
        openai_client_for_tenant = AsyncOpenAI(api_key=tenant_openai_key) 
        
        logger.info(f"Cargando archivos de configuración para tenant '{ident}' desde bucket '{SUPABASE_BUCKET}'...")
        system_md = load_supabase_file(SUPABASE_BUCKET, f"{ident}/system_message_{ident}.md")
        tools_json_str = load_supabase_file(SUPABASE_BUCKET, f"{ident}/functions_{ident}.json")
        custom_tools = json.loads(tools_json_str)
        logger.info(f"Configuración para '{ident}' cargada: System Message ({len(system_md)} chars), Tools ({len(custom_tools)}).")
        
        # 2. Obtener configuración de Pinecone para RAG
        vs_conf_resp = supabase.table("tenants").select("pinecone_index_name, pinecone_namespace, vector_store_provider").eq("id", tid).single().execute()
        if not vs_conf_resp.data: 
            logger.error(f"Configuración de Vector Store (Pinecone) no encontrada para tenant '{ident}'.")
            raise RuntimeError(f"Configuración de Vector Store no encontrada para {ident}")
        
        pinecone_index_name = vs_conf_resp.data.get("pinecone_index_name")
        pinecone_namespace = vs_conf_resp.data.get("pinecone_namespace")
        vector_provider = vs_conf_resp.data.get("vector_store_provider", "pinecone") 
        
        logger.info(f"🌲 Configuración Pinecone para '{ident}': Index='{pinecone_index_name}', Namespace='{pinecone_namespace}', Provider='{vector_provider}'")
        
        rag_context_str = "" 
        
        # --- UMBRAL DE SIMILITUD PARA RAG ---
        # Ajustar este valor según las pruebas con los logs y la calidad de las respuestas.
        # Para chunks grandes (ej. 2000 caracteres), un umbral más alto puede ser apropiado.
        # Empezar con 0.65 e iterar.
        SIMILARITY_THRESHOLD = 0.30 
        logger.info(f"🔬 Usando umbral de similitud para RAG: {SIMILARITY_THRESHOLD}")
        # --- FIN DEL AJUSTE ---

        if vector_provider == "pinecone" and pinecone_index_name and pinecone_namespace and data.history:
            user_query = data.history[-1].content if data.history and data.history[-1].role == "user" else ""
            if user_query:
                try:
                    logger.info(f"🔍 INICIANDO RAG para '{ident}'. Consulta del Usuario: '{user_query[:150]}...'")
                    
                    query_embedding_response = await openai_client_for_tenant.embeddings.create(
                        input=[user_query], model="text-embedding-3-small" 
                    )
                    query_vector = query_embedding_response.data[0].embedding
                    logger.info(f"✅ Embedding generado para la consulta de '{ident}'. Dimensión: {len(query_vector)}")
                    
                    pinecone_index_client = pc.Index(pinecone_index_name)
                    logger.info(f"🔗 Consultando Pinecone: Index='{pinecone_index_name}', Namespace='{pinecone_namespace}', TopK=5")
                    
                    query_results = pinecone_index_client.query(
                        namespace=pinecone_namespace, 
                        vector=query_vector,
                        top_k=5, 
                        include_metadata=True 
                    )
                    
                    match_count = len(query_results.matches) if query_results and hasattr(query_results, 'matches') else 0
                    logger.info(f"📊 Pinecone devolvió {match_count} resultados para la consulta de '{ident}'.")
                    
                    relevant_texts = []
                    if match_count > 0:
                        for i, match in enumerate(query_results.matches):
                            metadata = match.metadata if hasattr(match, 'metadata') else {}
                            doc_name = metadata.get('document_name', 'N/A') if metadata else 'N/A'
                            text_preview = (metadata.get('text', '')[:100] + '...') if metadata and metadata.get('text') else 'N/A'
                            logger.info(f"📄 Resultado RAG {i+1} para '{ident}': ID={match.id}, Score={match.score:.4f}, Doc='{doc_name}', Texto='{text_preview}'")
                            
                            if match.score > SIMILARITY_THRESHOLD: 
                                text_content = metadata.get('text', '') if metadata else ''
                                if text_content:
                                    relevant_texts.append(text_content)
                                    logger.info(f"👍 Chunk INCLUIDO para '{ident}' (Score: {match.score:.4f} > {SIMILARITY_THRESHOLD}, Doc: '{doc_name}')")
                                else:
                                    logger.warning(f"🤔 Chunk con Score > {SIMILARITY_THRESHOLD} para '{ident}' pero sin contenido de texto en metadatos. ID={match.id}")
                            else:
                                logger.info(f"👎 Chunk EXCLUIDO para '{ident}' (Score: {match.score:.4f} <= {SIMILARITY_THRESHOLD}, Doc: '{doc_name}')")
                    
                    if relevant_texts:
                        rag_context_str = "\n\n" + "="*60 + "\n"
                        rag_context_str += "INFORMACIÓN RELEVANTE DE LA BASE DE CONOCIMIENTO (UTILIZA ESTO PARA RESPONDER):\n"
                        rag_context_str += "="*60 + "\n\n"
                        for i, text_chunk in enumerate(relevant_texts, 1):
                            rag_context_str += f"--- Fragmento de Documento {i} ---\n{text_chunk}\n\n"
                        rag_context_str += "="*60 + "\n"
                        rag_context_str += "INSTRUCCIÓN FINAL: Basa tu respuesta ÚNICAMENTE en la 'INFORMACIÓN RELEVANTE DE LA BASE DE CONOCIMIENTO' proporcionada arriba. Si la respuesta no se encuentra en estos documentos, indica explícitamente que no tienes esa información específica en tu base de conocimiento actual y no intentes adivinar.\n"
                        rag_context_str += "="*60
                        logger.info(f"🎯 Contexto RAG construido para '{ident}'. {len(relevant_texts)} chunks incluidos. Longitud total del contexto: {len(rag_context_str)} caracteres.")
                    else: 
                        logger.warning(f"⚠️ No se encontró contexto RAG suficientemente relevante en Pinecone para '{ident}' (todos los scores <= {SIMILARITY_THRESHOLD} o chunks sin texto). No se añadirá contexto RAG.")
                
                except PineconeException as pe: 
                    logger.error(f"❌ Error de API de Pinecone durante RAG para '{ident}': {pe}", exc_info=True)
                except OpenAIError as oe: 
                    logger.error(f"❌ Error de API de OpenAI (Embeddings) durante RAG para '{ident}': {oe}", exc_info=True)
                except Exception as e_rag: 
                    logger.error(f"❌ Error general inesperado durante el proceso RAG para '{ident}': {e_rag}", exc_info=True)
        else:
            if not (vector_provider == "pinecone" and pinecone_index_name and pinecone_namespace):
                logger.warning(f"⚠️ RAG con Pinecone no está completamente configurado para '{ident}'. Provider='{vector_provider}', Index='{pinecone_index_name}', Namespace='{pinecone_namespace}'.")
            if not data.history:
                 logger.info(f"ℹ️ No hay historial de chat para '{ident}', no se realizará RAG.")
            logger.info(f"↪️ Saltando RAG para '{ident}'.")
        
        effective_system_message = system_md + rag_context_str 
        
        logger.info(f"📝 System message final para '{ident}' - Longitud total: {len(effective_system_message)} chars. ¿Contexto RAG añadido?: {'SÍ' if rag_context_str else 'NO'}")
        logger.debug(f"SYSTEM_MESSAGE_FINAL (primeros 500 chars para '{ident}'):\n{effective_system_message[:500]}")
        if rag_context_str:
            logger.debug(f"CONTEXTO_RAG_AÑADIDO (primeros 500 chars para '{ident}'):\n{rag_context_str[:500]}")

        input_messages_for_api = [{"role": "system", "content": effective_system_message}]
        for msg in data.history: 
            input_messages_for_api.append({"role": msg.role, "content": msg.content})
        
        effective_tools_for_api = custom_tools if custom_tools else []
        
        logger.info(f"🤖 Llamando a OpenAI Responses API para '{ident}'. Modelo: gpt-4.1 (o similar). Tools: {len(effective_tools_for_api)}. RAG usado: {'SÍ' if rag_context_str else 'NO'}")
        
        response1 = await openai_client_for_tenant.responses.create(
            model="gpt-4.1", 
            input=input_messages_for_api,
            tools=effective_tools_for_api
        )
        
        detected_fc_item, initial_text = None, ""
        
        if response1.output: 
            for item in response1.output:
                if getattr(item, "type", "") == "function_call": 
                    detected_fc_item = item
                    logger.info(f"🔧 Función detectada por LLM: '{detected_fc_item.name}' para '{ident}'. Argumentos (raw): {detected_fc_item.arguments}")
                    break 
                if hasattr(item, "content") and item.content: 
                    for text_content_part in item.content:
                        if hasattr(text_content_part, "text") and text_content_part.text:
                            initial_text += text_content_part.text
        else:
            logger.warning(f"Respuesta de OpenAI (response1.output) vacía o None para '{ident}'.")

        final_text_to_return = initial_text.strip()
        
        if detected_fc_item:
            function_name_called = detected_fc_item.name
            logger.info(f"➡️ Procesando función '{function_name_called}' para '{ident}' mediante llamada a webhook.")
            
            fc_result_payload = await handle_function_call(detected_fc_item, tid, ident)
            
            current_input_for_api_after_fc = list(input_messages_for_api) 
            current_input_for_api_after_fc.append({ 
                "type": "function_call", "call_id": detected_fc_item.call_id,
                "name": function_name_called, "arguments": detected_fc_item.arguments
            })
            current_input_for_api_after_fc.append({ 
                "type": "function_call_output", "call_id": detected_fc_item.call_id,
                "output": json.dumps(fc_result_payload) 
            })
            
            logger.info(f"📞 Realizando SEGUNDA llamada a Responses API para '{ident}' con resultado de función '{function_name_called}'.")
            response2 = await openai_client_for_tenant.responses.create(
                model="gpt-4.1", 
                input=current_input_for_api_after_fc,
                tools=effective_tools_for_api 
            )
            
            text_from_second_call = ""
            if response2.output:
                for item2 in response2.output:
                    if hasattr(item2, "content") and item2.content:
                        for text_part2 in item2.content:
                            if hasattr(text_part2, "text") and text_part2.text:
                                text_from_second_call += text_part2.text
            else:
                logger.warning(f"Respuesta de OpenAI (response2.output) vacía o None para '{ident}' después de función.")
            
            final_text_to_return = text_from_second_call.strip() or "Información procesada."
            logger.info(f"💬 Texto de SEGUNDA llamada para '{ident}' (después de función): '{final_text_to_return[:150]}...'")
            
            if fc_result_payload and fc_result_payload.get("success", False):
                clave_texto_post_funcion = f"post_text_{function_name_called}"
                texto_personalizado = get_data_critico(tid, clave_texto_post_funcion, ident)
                if texto_personalizado:
                    if texto_personalizado not in final_text_to_return:
                        final_text_to_return += f"\n\n{texto_personalizado}"
                        logger.info(f"➕ Texto personalizado de datos_criticos para '{clave_texto_post_funcion}' AÑADIDO para '{ident}'.")
                    else:
                        logger.info(f"ℹ️ Texto personalizado para '{clave_texto_post_funcion}' ya presente en respuesta para '{ident}'.")
                else:
                    logger.info(f"ℹ️ Función '{function_name_called}' exitosa, pero no se encontró texto personalizado con clave '{clave_texto_post_funcion}' para '{ident}'.")
            else:
                logger.warning(f"⚠️ Función '{function_name_called}' no fue exitosa o fc_result_payload es None. No se buscará texto post-función. Payload: {fc_result_payload}")
        
        else: # No se detectó función en la primera llamada
            if not final_text_to_return:
                logger.warning(f"⚠️ Respuesta inicial de OpenAI vacía para '{ident}'. Devolviendo mensaje genérico.")
                final_text_to_return = "No he podido generar una respuesta en este momento. ¿Podrías reformular tu pregunta o intentarlo de nuevo?"
            logger.info(f"💬 Respuesta directa (sin función) para '{ident}': '{final_text_to_return[:150]}...' RAG usado: {'SÍ' if rag_context_str else 'NO'}")
        
        conv_id = data.conversation_id or f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{ident}" 
        user_ext_id = data.user_id_external or "unknown_user"
        
        try:
            if data.history: 
                last_user_msg = data.history[-1]
                supabase.table("chat_history").insert({
                    "tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id,
                    "role": last_user_msg.role, "content": last_user_msg.content
                }).execute()
            
            if final_text_to_return: 
                supabase.table("chat_history").insert({
                    "tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id,
                    "role": "assistant", "content": final_text_to_return
                }).execute()
            logger.info(f"📚 Historial de chat guardado para conv_id: {conv_id}")
        except Exception as e_hist:
            logger.error(f"Error guardando historial de chat para conv_id {conv_id}: {e_hist}", exc_info=True)
        
        return ChatApiResponse(response=ChatResponseData(text=final_text_to_return))
    
    except OpenAIError as oe:
        logger.error(f"Error CRÍTICO de API de OpenAI en /chat para '{ident}': {oe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error crítico comunicándose con OpenAI: {str(oe)}")
    except PineconeException as pe:
        logger.error(f"Error CRÍTICO de API de Pinecone en /chat para '{ident}': {pe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error crítico comunicándose con Pinecone: {str(pe)}")
    except RuntimeError as e_rt: 
        logger.error(f"Error de configuración (RuntimeError) en /chat para '{ident}': {str(e_rt)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error de configuración del servidor: {str(e_rt)}")
    except Exception as e_gen:
        logger.error(f"Error INESPERADO y CRÍTICO en /chat para '{ident}': {str(e_gen)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno inesperado procesando su solicitud.")

# --- Health Check y Uvicorn run (Idéntico a tu archivo) ---
@app.get("/health", summary="Health check")
async def health_check(): return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    logger.info(f"Iniciando Uvicorn en host 0.0.0.0 puerto {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True if os.environ.get("DEV_MODE") == "true" else False)
