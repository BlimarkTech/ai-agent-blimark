# upload_to_pinecone.py

import os
import logging
import json
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI as SyncOpenAI, OpenAIError
from pinecone import Pinecone, PineconeException
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import time
from cryptography.fernet import Fernet

# Cargar variables de entorno del archivo .env
load_dotenv()

# --- Configuración del Logger ---
# Se configura el sistema de logging para mostrar información útil en la consola.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pinecone_sync_manager_strict_keys")

# --- Configuración de Constantes y Claves ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Limpia el nombre del bucket para evitar errores de formato (espacios, comillas).
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "vector-store").strip()

# Línea de depuración para verificar el nombre del bucket en la ejecución.
logger.info(f"DEBUG: Nombre del bucket limpio y listo para usar: '[{SUPABASE_BUCKET}]'")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ENCRYPTION_MASTER_KEY_ENV = os.getenv("ENCRYPTION_MASTER_KEY")

if not ENCRYPTION_MASTER_KEY_ENV:
    raise ValueError("La variable de entorno ENCRYPTION_MASTER_KEY no está configurada.")
ENCRYPTION_MASTER_KEY = ENCRYPTION_MASTER_KEY_ENV.encode()

# Constantes del modelo y del proceso
EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_UPSERT_BATCH_SIZE = 100

# Parámetros para la división de texto (chunking)
MARKDOWN_CHUNK_SIZE = 2000
MARKDOWN_CHUNK_OVERLAP = 200
RECURSIVE_CHUNK_SIZE = MARKDOWN_CHUNK_SIZE
RECURSIVE_CHUNK_OVERLAP = MARKDOWN_CHUNK_OVERLAP

# --- Inicialización de Clientes de Servicios ---
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    cipher = Fernet(ENCRYPTION_MASTER_KEY)
    logger.info("Clientes de Supabase, Pinecone y cifrador inicializados.")
except Exception as e:
    logger.fatal(f"Error inicializando clientes: {e}", exc_info=True)
    exit(1)

# --- Funciones Auxiliares ---

def calculate_content_hash(content_bytes: bytes) -> str:
    """Calcula un hash MD5 para el contenido de un archivo."""
    return hashlib.md5(content_bytes).hexdigest()

def get_tenant_openai_key_for_script(tenant_uuid_str: str, tenant_identifier: str) -> str | None:
    """Obtiene y desencripta la clave API de OpenAI para un tenant específico."""
    try:
        resp = supabase_client.table("tenants").select("openai_api_key_encrypted").eq("id", tenant_uuid_str).single().execute()
        encrypted_key = resp.data.get("openai_api_key_encrypted")
        if not encrypted_key or not encrypted_key.strip():
            logger.error(f"{tenant_identifier}: No se encontró 'openai_api_key_encrypted'.")
            return None
        return cipher.decrypt(encrypted_key.encode()).decode()
    except Exception as e:
        logger.error(f"{tenant_identifier}: Error desencriptando la clave de OpenAI: {e}")
        return None

def get_markdown_chunks(markdown_content: str) -> list[str]:
    """Divide un documento Markdown en chunks de texto manejables."""
    try:
        md_splitter = MarkdownTextSplitter(chunk_size=MARKDOWN_CHUNK_SIZE, chunk_overlap=MARKDOWN_CHUNK_OVERLAP)
        initial_chunks = md_splitter.split_text(markdown_content)
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE,
            chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        
        final_chunks = []
        for chunk in initial_chunks:
            if len(chunk) > RECURSIVE_CHUNK_SIZE:
                final_chunks.extend(recursive_splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Documento dividido en {len(final_chunks)} chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Error al dividir el texto Markdown: {e}")
        return []

def get_embedding_openai(client: SyncOpenAI, text_chunk: str) -> list[float] | None:
    """Genera el embedding para un chunk de texto usando OpenAI."""
    try:
        # Reemplaza el caracter nulo que causa errores en la API de OpenAI
        clean_text = text_chunk.replace("\x00", "\uFFFD")
        if not clean_text.strip():
            return None
        resp = client.embeddings.create(input=[clean_text], model=EMBEDDING_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        return None

def delete_vectors_from_pinecone(index, namespace: str, vector_ids: list[str]):
    """Elimina una lista de vectores de Pinecone."""
    if not vector_ids:
        return
    try:
        index.delete(ids=vector_ids, namespace=namespace)
        logger.info(f"Eliminados {len(vector_ids)} vectores obsoletos en el namespace '{namespace}'.")
    except Exception as e:
        logger.error(f"Error eliminando vectores en el namespace '{namespace}': {e}")

def sync_tenant_documents_to_pinecone(tenant: dict):
    """Función principal que sincroniza los documentos de un tenant con Pinecone."""
    tenant_id = str(tenant["id"])
    identifier = tenant["identifier"]
    index_name = tenant["pinecone_index_name"]
    namespace = tenant["pinecone_namespace"]
    
    logger.info(f"Iniciando sincronización para tenant '{identifier}' (ID: {tenant_id}) -> Índice '{index_name}', Namespace '{namespace}'")
    
    openai_api_key = get_tenant_openai_key_for_script(tenant_id, identifier)
    if not openai_api_key:
        logger.error(f"{identifier}: No se pudo obtener la clave de OpenAI. Omitiendo tenant.")
        return
    
    openai_client = SyncOpenAI(api_key=openai_api_key)
    pinecone_index = pinecone_client.Index(index_name)
    
    try:
        files_in_storage = supabase_client.storage.from_(SUPABASE_BUCKET).list(path=f"{identifier}/")
    except Exception as e:
        logger.error(f"{identifier}: Error al listar archivos en Supabase Storage: {e}")
        return
    
    md_files = [f for f in files_in_storage if f["name"].endswith(".md")]
    logger.info(f"{identifier}: Se encontraron {len(md_files)} archivos .md en el bucket.")
    
    try:
        manifest_response = supabase_client.table("vector_store_file_manifest").select("*").eq("tenant_id", tenant_id).execute()
        manifest = {m["file_path_in_storage"]: m for m in manifest_response.data}
    except Exception as e:
        logger.error(f"{identifier}: Error al obtener el manifiesto de archivos: {e}")
        manifest = {}

    for file_metadata in md_files:
        file_path = f"{identifier}/{file_metadata['name']}"
        document_name = os.path.splitext(file_metadata['name'])[0]
        
        try:
            file_content_bytes = supabase_client.storage.from_(SUPABASE_BUCKET).download(file_path)
        except Exception as e:
            logger.error(f"{identifier}: Error al descargar el archivo '{file_path}': {e}")
            continue
        
        content_hash = calculate_content_hash(file_content_bytes)
        old_manifest_entry = manifest.get(file_path)
        
        is_new = not old_manifest_entry
        is_modified = not is_new and old_manifest_entry["content_hash"] != content_hash
        
        if not is_new and not is_modified:
            logger.info(f"{identifier}: Archivo '{file_path}' sin cambios. Omitiendo.")
            continue
            
        logger.info(f"{identifier}: Procesando archivo {'nuevo' if is_new else 'modificado'}: '{file_path}'")
        
        if old_manifest_entry and old_manifest_entry.get("pinecone_vector_ids"):
            delete_vectors_from_pinecone(pinecone_index, namespace, old_manifest_entry["pinecone_vector_ids"])
            
        text_content = file_content_bytes.decode("utf-8")
        chunks = get_markdown_chunks(text_content)
        new_vector_ids, vectors_to_upsert = [], []
        
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding_openai(openai_client, chunk_text)
            if not embedding:
                continue
            
            vector_id = f"{identifier}-{document_name}-chunk{i}"
            new_vector_ids.append(vector_id)
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk_text,
                    "document_name": document_name,
                    "tenant": identifier,
                    "chunk_index": i,
                    "source": file_path
                }
            })
            
        if vectors_to_upsert:
            logger.info(f"Subiendo {len(vectors_to_upsert)} vectores a Pinecone para '{file_path}'.")
            for i in range(0, len(vectors_to_upsert), PINECONE_UPSERT_BATCH_SIZE):
                batch = vectors_to_upsert[i:i + PINECONE_UPSERT_BATCH_SIZE]
                try:
                    pinecone_index.upsert(vectors=batch, namespace=namespace)
                except Exception as e:
                    logger.error(f"{identifier}: Error en upsert de Pinecone (lote {(i//PINECONE_UPSERT_BATCH_SIZE)+1}): {e}")
                time.sleep(0.1)
                
        manifest_data = {
            "tenant_id": tenant_id,
            "file_path_in_storage": file_path,
            "supabase_last_modified": file_metadata["metadata"]["lastModified"],
            "content_hash": content_hash,
            "pinecone_vector_ids": new_vector_ids,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if is_new:
                supabase_client.table("vector_store_file_manifest").insert(manifest_data).execute()
            else:
                supabase_client.table("vector_store_file_manifest").update(manifest_data).eq("id", old_manifest_entry["id"]).execute()
            logger.info(f"{identifier}: Manifiesto {'creado' if is_new else 'actualizado'} para '{file_path}'.")
        except Exception as e:
            logger.error(f"{identifier}: Error actualizando el manifiesto para '{file_path}': {e}")
            
    logger.info(f"Sincronización completada para el tenant '{identifier}'.")

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    logger.info("=== INICIANDO SINCRONIZACIÓN: Supabase → Pinecone ===")
    try:
        active_tenants = supabase_client.table("tenants") \
            .select("id,identifier,pinecone_index_name,pinecone_namespace") \
            .eq("is_active", True) \
            .eq("vector_store_provider", "pinecone").execute().data
        
        for tenant_data in active_tenants:
            sync_tenant_documents_to_pinecone(tenant_data)
            
    except Exception as e:
        logger.fatal(f"Error fatal en el script principal: {e}", exc_info=True)
    
    logger.info("=== SCRIPT DE SINCRONIZACIÓN FINALIZADO ===")
