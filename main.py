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
from openai import AsyncOpenAI, OpenAIError # OpenAIError para manejo de errores más genérico
from supabase import create_client, Client

# JWT y Criptografía
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet

# --- NUEVA IMPORTACIÓN PARA PINECONE ---
from pinecone import Pinecone, PineconeException

# ----------------------------
# Carga y validación de variables de entorno
# ----------------------------
load_dotenv()

REQUIRED_VARS = [
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "JWT_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY", "PINECONE_API_KEY" # OPENAI_API_KEY ya no es global, se obtiene por tenant
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
# --- NUEVA VARIABLE DE ENTORNO PARA PINECONE ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multi_tenant_agent_pinecone_responses_api")

# ----------------------------
# Clientes de servicio
# ----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Conexión a Supabase establecida.")
except Exception as e:
    logger.error(f"Error conectando a Supabase: {e}", exc_info=True)
    supabase = None

# --- INICIALIZAR CLIENTE DE PINECONE ---
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
app = FastAPI(title="Agente IA Multi-Tenant (Pinecone + Responses API)", version="1.3.0") # Actualizamos versión
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body(); logger.error(f"RequestValidationError: {exc.errors()} – Body: {body.decode('utf-8')}")
    return await request_validation_exception_handler(request, exc)

# --- Modelos Pydantic y JWT (Sin cambios respecto a tu versión 1.2.0) ---
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

# --- Helpers (Sin cambios load_supabase_file, get_data_critico, EmailCheckModel) ---
# La función get_tenant_openai_key sigue igual, ya que la usas para el cliente de OpenAI
def load_supabase_file(bucket: str, path: str, as_text: bool = True) -> str | bytes:
    if not supabase: raise RuntimeError("Supabase no disponible.")
    if ".." in path or path.startswith("/"): raise ValueError(f"Ruta inválida: {path}")
    logger.info(f"Descargando archivo {bucket}/{path}")
    try: data_bytes = supabase.storage.from_(bucket).download(path); return data_bytes.decode("utf-8") if as_text else data_bytes
    except Exception as e: logger.error(f"Error descargando {path}: {e}"); raise RuntimeError(f"No se pudo cargar archivo: {path}")

def get_tenant_openai_key(tenant_id: str, tenant_identifier: str) -> str:
    if not supabase: raise RuntimeError("Supabase no disponible para obtener API Key de OpenAI.")
    resp = supabase.table("tenants").select("openai_api_key_encrypted").eq("id", tenant_id).single().execute()
    if not resp.data: raise RuntimeError(f"Inquilino {tenant_identifier} no encontrado (para API Key OpenAI).")
    encrypted = resp.data.get("openai_api_key_encrypted")
    if not encrypted or not encrypted.strip():
        logger.error(f"API Key de OpenAI encriptada faltante para {tenant_identifier}")
        raise RuntimeError(f"API Key de OpenAI no configurada para {tenant_identifier}.")
    try: return cipher.decrypt(encrypted.encode()).decode()
    except Exception as e:
        logger.error(f"Error desencriptando API Key de OpenAI para {tenant_identifier}: {e}")
        raise RuntimeError(f"Error procesando API Key de OpenAI para {tenant_identifier}.")

def get_data_critico(tenant_id: str, clave: str, tenant_identifier: str) -> str | None:
    if not supabase: logger.error(f"Supabase no disponible al buscar '{clave}' para {tenant_identifier}"); return None
    try:
        logger.info(f"Buscando dato crítico '{clave}' para tenant {tenant_identifier} (ID: {tenant_id})")
        resp = supabase.table("datos_criticos").select("valor").eq("clave", clave).eq("tenant_id", tenant_id).maybe_single().execute()
        if resp and hasattr(resp, 'data') and resp.data and isinstance(resp.data, dict):
            valor = resp.data.get("valor")
            if valor: logger.info(f"Dato '{clave}' encontrado para {tenant_identifier}: '{valor[:50]}...'"); return valor
        logger.info(f"Dato crítico '{clave}' no encontrado o valor vacío para {tenant_identifier}.")
        return None
    except Exception as e: logger.error(f"Excepción al obtener dato crítico '{clave}' para {tenant_identifier}: {e}", exc_info=True); return None

class EmailCheckModel(BaseModel): email: EmailStr

# --- handle_function_call (Genérica como en tu v1.2.0, `call_obj` es el `detected_fc` de Responses API) ---
async def handle_function_call(call_obj, tenant_id: str, tenant_identifier: str) -> dict:
    # `call_obj` para Responses API es el objeto item con type="function_call"
    # que tiene `name`, `arguments` (string JSON), y `call_id`.
    name = call_obj.name
    arguments_str = call_obj.arguments

    logger.info(f"Ejecutando función '{name}' para tenant {tenant_identifier}")
    if not supabase: return {"success": False, "error": "Servicio de configuración no disponible."}
    
    resp_webhook = supabase.table("tenants").select("recoleccion_webhook_url").eq("id", tenant_id).single().execute()
    if not resp_webhook.data or not resp_webhook.data.get("recoleccion_webhook_url"):
        logger.warning(f"No hay URL de webhook configurada para {tenant_identifier} para la función '{name}'.")
        return {"success": False, "error": f"Función '{name}' no configurada (falta URL de webhook)."}
    
    url = resp_webhook.data.get("recoleccion_webhook_url")
    try:
        args_dict = json.loads(arguments_str)
        if "email" in args_dict and args_dict["email"]:
            try: EmailCheckModel(email=args_dict["email"])
            except ValidationError as ve:
                logger.warning(f"Validación de email fallida para '{args_dict['email']}' en {tenant_identifier}: {ve.errors()}")
                return {"success": False, "error": f"Email inválido: {ve.errors()}"}
        
        payload = {**args_dict, "_tenant_info": {"id": tenant_id, "identifier": tenant_identifier, "function_name": name}}
        logger.info(f"Enviando función '{name}' a webhook '{url}' para {tenant_identifier}")
        r = requests.post(url, json=payload, timeout=10); r.raise_for_status()
        logger.info(f"Webhook para {tenant_identifier} respondió con status {r.status_code}")
        return {"success": True, "status_code": r.status_code, "function_name": name,
                "response_data": r.json() if r.content and 'application/json' in r.headers.get('Content-Type','').lower() else {"raw_response": r.text[:200]}}
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"Error HTTP en webhook para {tenant_identifier} (Status: {getattr(http_err.response, 'status_code', 'N/A')})")
        return {"success": False, "error": f"Error webhook ({getattr(http_err.response, 'status_code', 'N/A')})."}
    except Exception as e:
        logger.error(f"Error general en función '{name}' para {tenant_identifier}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# --- Modelos Pydantic para Chat (Los que usabas con Responses API) ---
class ChatMessage(BaseModel): role: str; content: str # Responses API input no usa tool_calls aquí
class ChatRequest(BaseModel):
    history: list[ChatMessage]; conversation_id: str | None = None; user_id_external: str | None = None
    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if not isinstance(v, list): raise ValueError("history debe ser una lista")
        return v[-200:] if len(v) > 200 else v
class ChatResponseData(BaseModel): type: str = "text"; text: str | None
class ChatApiResponse(BaseModel): response: ChatResponseData

# --- Endpoint de Token (Sin cambios respecto a tu v1.2.0) ---
@app.post("/token", summary="Obtener JWT para tenant")
# ... (código de token_endpoint sin cambios)
async def token_endpoint(api_key_botpress: str = Form(...)):
    if not supabase: raise HTTPException(503, "Servicio BD no disponible")
    resp = supabase.table("tenants").select("id,identifier,api_key_hash").eq("is_active", True).execute()
    if not resp.data: raise HTTPException(401, "No hay tenants activos o configuración incorrecta")
    for row in resp.data:
        if pwd_context.verify(api_key_botpress, row["api_key_hash"]):
            token = create_access_token({"tenant_id": str(row["id"]), "identifier": row["identifier"]})
            return {"response": {"access_token": token, "token_type": "bearer"}}
    raise HTTPException(401, "API Key inválida o inquilino inactivo")

# --- Endpoint /chat con Pinecone y Responses API ---
@app.post("/chat", response_model=ChatApiResponse, summary="Chat multi-tenant con Pinecone + Responses API")
async def chat_endpoint(data: ChatRequest = Body(...), tenant: TokenData = Depends(get_current_tenant)):
    if not supabase or not pc: # OpenAI client se instancia por tenant
        raise HTTPException(503, "Servicios esenciales (Supabase/Pinecone) no disponibles.")
    
    tid, ident = tenant.tenant_id, tenant.identifier
    logger.info(f"Nuevo /chat para tenant {ident} (ID: {tid}) usando Responses API y Pinecone RAG.")

    try:
        # 1. Cargar configuración del tenant y API Key de OpenAI del tenant
        tenant_openai_key = get_tenant_openai_key(tid, ident)
        # Inicializar cliente de OpenAI con la clave del tenant para esta solicitud
        openai_client_for_tenant = AsyncOpenAI(api_key=tenant_openai_key)

        system_md = load_supabase_file(SUPABASE_BUCKET, f"{ident}/system_message_{ident}.md")
        tools_json_str = load_supabase_file(SUPABASE_BUCKET, f"{ident}/functions_{ident}.json")
        custom_tools = json.loads(tools_json_str) # Lista de definiciones de tools para Responses API

        # 2. Obtener configuración de Pinecone para RAG
        vs_conf_resp = supabase.table("tenants").select("pinecone_index_name, pinecone_namespace, vector_store_provider").eq("id", tid).single().execute()
        if not vs_conf_resp.data: raise RuntimeError(f"Configuración de Vector Store no encontrada para {ident}")
        
        pinecone_index_name = vs_conf_resp.data.get("pinecone_index_name")
        pinecone_namespace = vs_conf_resp.data.get("pinecone_namespace")
        vector_provider = vs_conf_resp.data.get("vector_store_provider", "pinecone")

        rag_context_str = ""
        if vector_provider == "pinecone" and pinecone_index_name and pinecone_namespace and data.history:
            user_query = data.history[-1].content if data.history and data.history[-1].role == "user" else ""
            if user_query:
                try:
                    logger.info(f"Generando embedding para consulta RAG: '{user_query[:50]}...'")
                    # Usar el cliente de OpenAI del tenant para generar embeddings
                    query_embedding_response = await openai_client_for_tenant.embeddings.create(
                        input=[user_query], model="text-embedding-3-small" # Modelo de embedding
                    )
                    query_vector = query_embedding_response.data[0].embedding
                    
                    pinecone_index_client = pc.Index(pinecone_index_name)
                    logger.info(f"Consultando Pinecone (Index: {pinecone_index_name}, Namespace: {pinecone_namespace})")
                    query_results = pinecone_index_client.query(
                        namespace=pinecone_namespace, vector=query_vector,
                        top_k=3, include_metadata=True # Obtener 3 chunks más relevantes
                    )
                    
                    # Extraer el texto de los metadatos de los chunks relevantes
                    relevant_texts = [match.metadata.get('text', '') for match in query_results.matches if match.score > 0.75] # Ajustar umbral de similitud
                    if relevant_texts:
                        rag_context_str = "\n\nContexto Relevante:\n" + "\n---\n".join(relevant_texts)
                        logger.info(f"Contexto RAG obtenido de Pinecone para {ident}.")
                    else: logger.info(f"No se encontró contexto RAG relevante en Pinecone para {ident}.")
                except PineconeException as pe: logger.error(f"Error de Pinecone durante RAG para {ident}: {pe}", exc_info=True)
                except OpenAIError as oe: logger.error(f"Error de OpenAI (Embeddings) durante RAG para {ident}: {oe}", exc_info=True)
                except Exception as e_rag: logger.error(f"Error general durante RAG para {ident}: {e_rag}", exc_info=True)
        
        # 3. Preparar 'input' para Responses API
        effective_system_message = system_md + rag_context_str # Inyectar contexto RAG
        
        input_messages_for_api = [{"role": "system", "content": effective_system_message}]
        for msg in data.history:
            input_messages_for_api.append({"role": msg.role, "content": msg.content})
            
        # 4. Llamada inicial a Responses API
        # Ya NO se usa la herramienta 'file_search' de OpenAI. Solo custom_tools.
        effective_tools_for_api = custom_tools if custom_tools else [] # Responses API espera lista

        logger.info(f"Llamando a OpenAI Responses API para {ident}. Modelo: gpt-4.1. Tools: {len(effective_tools_for_api) > 0}")
        response1 = await openai_client_for_tenant.responses.create(
            model="gpt-4.1", # O el modelo que uses con Responses API
            input=input_messages_for_api,
            tools=effective_tools_for_api
        )
        
        # 5. Procesar la salida de Responses API (Lógica de tu v1.2.0)
        # El objeto 'detected_fc' en tu v1.2.0 era el 'item' del bucle.
        detected_fc_item, initial_text = None, ""
        for item in response1.output:
            if getattr(item, "type", "") == "function_call": # item es FunctionCallContentPart
                detected_fc_item = item
                logger.info(f"Función detectada: {detected_fc_item.name} para {ident}"); break
            if hasattr(item, "content") and item.content: # item.content es lista de TextContentPart
                for text_content_part in item.content:
                    if hasattr(text_content_part, "text") and text_content_part.text:
                         initial_text += text_content_part.text
        final_text_to_return = initial_text

        if detected_fc_item: # Si se detectó una llamada a función
            function_name_called = detected_fc_item.name
            logger.info(f"Procesando función '{function_name_called}' para {ident}")
            
            # Adaptación de la lógica de validación de campos para Blimark Tech (si sigue siendo necesaria)
            fc_result_payload = None 
            if function_name_called == "recolectarInformacionContacto" and ident == "blimark_tech":
                args_dict = json.loads(detected_fc_item.arguments)
                required_fields = ["nombre", "apellidos", "email", "telefono", "pais", "mensaje"]
                missing_fields = [f for f in required_fields if not args_dict.get(f,"").strip()]
                if missing_fields:
                    final_text_to_return = f"Para procesar tu solicitud de contacto, necesito: {', '.join(missing_fields)}. ¿Podrías proporcionarlos?"
                    logger.info(f"Respondiendo con solicitud de campos faltantes para {ident}.")
                    # No se asigna fc_result_payload, por lo tanto no se hará la segunda llamada
                else:
                    fc_result_payload = await handle_function_call(detected_fc_item, tid, ident)
            else: # Para todas las demás funciones o tenants
                fc_result_payload = await handle_function_call(detected_fc_item, tid, ident)
            
            if fc_result_payload is not None: # Si la función se ejecutó (no se abortó por campos faltantes)
                # Añadir el resultado de la función al historial de 'input' para la segunda llamada
                current_input_for_api = list(input_messages_for_api) # Copiar el input actual
                current_input_for_api.append({ # Representar la llamada a función que hizo el modelo
                    "type": "function_call",
                    "call_id": detected_fc_item.call_id,
                    "name": function_name_called,
                    "arguments": detected_fc_item.arguments
                })
                current_input_for_api.append({ # Añadir el resultado de nuestra ejecución
                    "type": "function_call_output",
                    "call_id": detected_fc_item.call_id,
                    "output": json.dumps(fc_result_payload)
                })
                
                logger.info(f"Realizando segunda llamada a Responses API para {ident} con resultado de función.")
                response2 = await openai_client_for_tenant.responses.create(
                    model="gpt-4.1",
                    input=current_input_for_api,
                    tools=effective_tools_for_api
                )
                text_from_second_call = ""
                for item2 in response2.output:
                    if hasattr(item2, "content") and item2.content:
                        for text_part2 in item2.content:
                            if hasattr(text_part2, "text") and text_part2.text:
                                text_from_second_call += text_part2.text
                final_text_to_return = text_from_second_call or "Información procesada."
                logger.info(f"Texto de segunda llamada para {ident}: '{final_text_to_return}'")

                # Añadir texto personalizado de datos_criticos si la función fue exitosa
                if fc_result_payload.get("success", False):
                    clave_texto_post_funcion = f"post_text_{function_name_called}"
                    texto_personalizado = get_data_critico(tid, clave_texto_post_funcion, ident)
                    if texto_personalizado and texto_personalizado not in final_text_to_return:
                        final_text_to_return += f"\n\n{texto_personalizado}"
                        logger.info(f"Texto personalizado para '{clave_texto_post_funcion}' añadido para {ident}.")
                    elif texto_personalizado: logger.info(f"Texto para '{clave_texto_post_funcion}' ya presente o idéntico para {ident}.")
                    else: logger.info(f"Función '{function_name_called}' exitosa, pero no se encontró texto con clave '{clave_texto_post_funcion}' para {ident}.")
        
        else: # No se detectó función
            if not final_text_to_return : # Si la respuesta inicial estaba vacía
                 final_text_to_return = "No pude generar una respuesta en este momento."
            logger.info(f"Texto de primera llamada (sin función) para {ident}: '{final_text_to_return}'")

        # 6. Guardar historial y retornar (Sin cambios respecto a tu v1.2.0)
        conv_id = data.conversation_id or f"conv_{datetime.now(timezone.utc).isoformat()}"
        user_ext_id = data.user_id_external or "unknown_user"
        if data.history: # Guardar último mensaje del usuario
            last_user_msg = data.history[-1]
            supabase.table("chat_history").insert({
                "tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id,
                "role": last_user_msg.role, "content": last_user_msg.content
            }).execute()
        if final_text_to_return: # Guardar respuesta del asistente
            supabase.table("chat_history").insert({
                "tenant_id": tid, "conversation_id": conv_id, "user_id_external": user_ext_id,
                "role": "assistant", "content": final_text_to_return
            }).execute()
        
        return ChatApiResponse(response=ChatResponseData(text=final_text_to_return))

    except OpenAIError as oe:
        logger.error(f"Error de API de OpenAI en /chat para {ident}: {oe}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error comunicándose con OpenAI: {str(oe)}")
    except PineconeException as pe:
        logger.error(f"Error de API de Pinecone en /chat para {ident}: {pe}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error comunicándose con Pinecone: {str(pe)}")
    except RuntimeError as e_rt:
        logger.error(f"RuntimeError en /chat para {ident}: {str(e_rt)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error de configuración del servidor: {str(e_rt)}")
    except Exception as e_gen:
        logger.error(f"Error inesperado en /chat para {ident}: {str(e_gen)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno inesperado procesando su solicitud.")

# --- Health Check y Uvicorn run (Sin cambios respecto a tu v1.2.0) ---
@app.get("/health", summary="Health check")
async def health_check(): return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)