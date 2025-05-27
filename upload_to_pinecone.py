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

load_dotenv()

# --- Configuración ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "vector-store")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ENCRYPTION_MASTER_KEY_ENV = os.getenv("ENCRYPTION_MASTER_KEY")

if not ENCRYPTION_MASTER_KEY_ENV:
    raise ValueError("La variable de entorno ENCRYPTION_MASTER_KEY no está configurada y es obligatoria.")
ENCRYPTION_MASTER_KEY = ENCRYPTION_MASTER_KEY_ENV.encode()

EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_UPSERT_BATCH_SIZE = 100
MARKDOWN_CHUNK_SIZE = 1000
MARKDOWN_CHUNK_OVERLAP = 50
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pinecone_sync_manager_strict_keys")

# --- Inicialización de Clientes Globales ---
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    cipher = Fernet(ENCRYPTION_MASTER_KEY)
    logger.info("Clientes de Supabase y Pinecone, y Cifrador inicializados.")
except Exception as e:
    logger.fatal(f"Error fatal inicializando clientes globales: {e}", exc_info=True); exit(1)

# --- Funciones Helper ---
def calculate_content_hash(content_bytes: bytes) -> str:
    return hashlib.md5(content_bytes).hexdigest()

def get_tenant_openai_key_for_script(tenant_uuid_str: str, tenant_identifier: str) -> str | None:
    """Obtiene y desencripta la API Key de OpenAI para un tenant. Retorna None si no se encuentra o hay error."""
    if not supabase_client:
        logger.error("Cliente de Supabase no disponible en get_tenant_openai_key_for_script.")
        return None
    try:
        resp = supabase_client.table("tenants").select("id, openai_api_key_encrypted").eq("id", tenant_uuid_str).single().execute()
        if not resp.data:
            logger.error(f"No se encontró el tenant con ID {tenant_uuid_str} (identifier: {tenant_identifier}).")
            return None
        
        encrypted_key = resp.data.get("openai_api_key_encrypted")
        if not encrypted_key or not encrypted_key.strip():
            logger.error(f"API Key de OpenAI encriptada faltante para {tenant_identifier}. No se pueden generar embeddings.")
            return None # Estricto: si no hay clave, no hay fallback
        
        return cipher.decrypt(encrypted_key.encode()).decode()
    except Exception as e:
        logger.error(f"Error obteniendo/desencriptando API Key de OpenAI para {tenant_identifier}: {e}")
        return None # Estricto: si hay error, no hay fallback

def get_markdown_chunks(markdown_content: str) -> list[str]:
    # ... (sin cambios)
    final_chunks = []
    try:
        md_splitter = MarkdownTextSplitter(chunk_size=MARKDOWN_CHUNK_SIZE, chunk_overlap=MARKDOWN_CHUNK_OVERLAP)
        initial_chunks = md_splitter.split_text(markdown_content)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE, chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""], length_function=len
        )
        for chunk in initial_chunks:
            if len(chunk) > RECURSIVE_CHUNK_SIZE:
                final_chunks.extend(recursive_splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        logger.debug(f"Contenido dividido en {len(final_chunks)} chunks.")
        return final_chunks
    except Exception as e:
        logger.error(f"Error dividiendo MD: {e}"); return []


def get_embedding_openai(openai_s_client_for_tenant: SyncOpenAI, text_chunk: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    # ... (sin cambios)
    if not openai_s_client_for_tenant:
        logger.error("Cliente de OpenAI para tenant no proporcionado a get_embedding_openai.")
        return None
    try:
        text_chunk_cleaned = text_chunk.replace("\x00", "\uFFFD")
        if not text_chunk_cleaned.strip(): return None
        response = openai_s_client_for_tenant.embeddings.create(input=[text_chunk_cleaned], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error embedding con cliente de tenant: '{text_chunk_cleaned[:50]}...': {e}"); return None

def delete_vectors_from_pinecone(pinecone_index: Pinecone.Index, namespace: str, vector_ids: list[str]):
    # ... (sin cambios)
    if not vector_ids: logger.info(f"No hay IDs para eliminar en ns {namespace}."); return
    try:
        logger.info(f"Eliminando {len(vector_ids)} vectores de ns '{namespace}'...")
        delete_response = pinecone_index.delete(ids=vector_ids, namespace=namespace)
        logger.info(f"Respuesta de eliminación de Pinecone: {delete_response}")
    except Exception as e: logger.error(f"Error eliminando vectores de ns '{namespace}': {e}")


def sync_tenant_documents_to_pinecone(tenant_row_from_db: dict):
    # ... (código inicial de obtención de tenant_uuid_str, tenant_identifier, etc. sin cambios) ...
    tenant_uuid_str = str(tenant_row_from_db['id'])
    tenant_identifier = tenant_row_from_db['identifier']
    pinecone_index_name = tenant_row_from_db['pinecone_index_name']
    pinecone_namespace = tenant_row_from_db['pinecone_namespace']

    logger.info(f"--- Iniciando Sincronización para Tenant: {tenant_identifier} (ID: {tenant_uuid_str}) ---")

    tenant_openai_api_key = get_tenant_openai_key_for_script(tenant_uuid_str, tenant_identifier)
    if not tenant_openai_api_key: # Si no se pudo obtener la API Key, no se puede continuar para este tenant
        logger.error(f"IMPOSIBLE PROCESAR: No se pudo obtener la API Key de OpenAI para {tenant_identifier}. Se omitirá este tenant.")
        return # Termina la ejecución para este tenant
    
    try:
        openai_s_client_for_this_tenant = SyncOpenAI(api_key=tenant_openai_api_key)
        logger.info(f"Cliente de OpenAI inicializado para {tenant_identifier} con su API Key.")
    except Exception as e_client_init:
        logger.error(f"IMPOSIBLE PROCESAR: Error inicializando cliente de OpenAI para {tenant_identifier} con su API Key: {e_client_init}. Se omitirá este tenant.")
        return # Termina la ejecución para este tenant

    # ... (El resto de la lógica de sync_tenant_documents_to_pinecone sigue igual, ya que
    #      depende de que openai_s_client_for_this_tenant esté correctamente instanciado) ...
    try:
        pinecone_index = pinecone_client.Index(pinecone_index_name)
        storage_path_prefix = f"{tenant_identifier}/"
        
        try:
            supabase_files_list = supabase_client.storage.from_(SUPABASE_BUCKET).list(path=storage_path_prefix)
            current_supabase_files = { 
                f['name']: { 
                    'path': f"{storage_path_prefix}{f['name']}", 
                    'last_modified': datetime.fromisoformat(f['metadata']['lastModified'].replace("Z", "+00:00")) if f.get('metadata') and f['metadata'].get('lastModified') else datetime.now(timezone.utc), 
                    'size': f.get('metadata', {}).get('size', 0)
                } 
                for f in supabase_files_list if f['name'].endswith('.md')
            }
            logger.info(f"Encontrados {len(current_supabase_files)} archivos .md en Supabase para {tenant_identifier}.")
        except Exception as e_storage_list: 
            logger.error(f"Error listando de Supabase para {tenant_identifier}: {e_storage_list}"); return
        
        try:
            manifest_response = supabase_client.table("vector_store_file_manifest").select("*").eq("tenant_id", tenant_uuid_str).execute()
            manifest_files = {mf['file_path_in_storage']: mf for mf in manifest_response.data}
            logger.info(f"Encontrados {len(manifest_files)} registros en manifiesto para {tenant_identifier}.")
        except Exception as e_manifest_fetch: 
            logger.error(f"Error obteniendo manifiesto para {tenant_identifier}: {e_manifest_fetch}"); return
        
        manifest_paths = set(manifest_files.keys()); supabase_paths = set(details['path'] for details in current_supabase_files.values())
        deleted_in_supabase_paths = manifest_paths - supabase_paths
        for deleted_path in deleted_in_supabase_paths:
            logger.info(f"Archivo ELIMINADO de Supabase: {deleted_path}")
            manifest_entry = manifest_files[deleted_path]
            vector_ids_to_delete = manifest_entry.get('pinecone_vector_ids', [])
            if vector_ids_to_delete: delete_vectors_from_pinecone(pinecone_index, pinecone_namespace, vector_ids_to_delete)
            try: 
                supabase_client.table("vector_store_file_manifest").delete().eq("id", manifest_entry['id']).execute()
                logger.info(f"Manifiesto eliminado para {deleted_path}.")
            except Exception as e_m_del: logger.error(f"Error eliminando manifiesto para {deleted_path}: {e_m_del}")

        for file_name_md, supabase_file_details in current_supabase_files.items():
            file_path = supabase_file_details['path']
            doc_name_base = os.path.splitext(file_name_md)[0]
            try: 
                file_content_bytes = supabase_client.storage.from_(SUPABASE_BUCKET).download(file_path)
            except Exception as e_dl: 
                logger.error(f"Error descargando {file_path}: {e_dl}"); continue
            
            current_content_hash = calculate_content_hash(file_content_bytes)
            
            is_new_file = file_path not in manifest_files; is_modified_file = False
            if not is_new_file:
                manifest_entry = manifest_files[file_path]
                if manifest_entry.get('content_hash') != current_content_hash: is_modified_file = True
            
            if is_new_file: logger.info(f"Archivo NUEVO: {file_path}")
            elif is_modified_file:
                logger.info(f"Archivo MODIFICADO: {file_path}")
                old_vector_ids = manifest_entry.get('pinecone_vector_ids', [])
                if old_vector_ids: delete_vectors_from_pinecone(pinecone_index, pinecone_namespace, old_vector_ids)
            else: 
                logger.info(f"SIN CAMBIOS: {file_path}. Saltando."); continue
            
            content_md = file_content_bytes.decode('utf-8')
            text_chunks = get_markdown_chunks(content_md)
            if not text_chunks: 
                logger.warning(f"No chunks para {file_path}. Saltando."); continue
            
            new_pinecone_vector_ids = []
            vectors_for_upsert = []
            for i, chunk_text in enumerate(text_chunks):
                embedding_vector = get_embedding_openai(openai_s_client_for_this_tenant, chunk_text)
                if embedding_vector:
                    vector_id = f"{tenant_identifier}-{doc_name_base}-chunk{i}"
                    new_pinecone_vector_ids.append(vector_id)
                    metadata = { "text": chunk_text, "document_name": doc_name_base, "tenant": tenant_identifier, "chunk_index": i, "source_file_path": file_path }
                    vectors_for_upsert.append({"id": vector_id, "values": embedding_vector, "metadata": metadata})
            
            if vectors_for_upsert:
                for j in range(0, len(vectors_for_upsert), PINECONE_UPSERT_BATCH_SIZE):
                    batch = vectors_for_upsert[j : j + PINECONE_UPSERT_BATCH_SIZE]
                    try: 
                        pinecone_index.upsert(vectors=batch, namespace=pinecone_namespace)
                    except Exception as e_upsert: 
                        logger.error(f"Error upsert para {file_path}, batch {j // PINECONE_UPSERT_BATCH_SIZE + 1}: {e_upsert}")
                    time.sleep(0.1)
            
            manifest_data_to_upsert = { 
                "tenant_id": tenant_uuid_str, 
                "file_path_in_storage": file_path, 
                "supabase_last_modified": supabase_file_details['last_modified'].isoformat(), 
                "content_hash": current_content_hash, 
                "pinecone_vector_ids": new_pinecone_vector_ids, 
                "processed_at": datetime.now(timezone.utc).isoformat() 
            }
            try:
                if is_new_file: 
                    supabase_client.table("vector_store_file_manifest").insert(manifest_data_to_upsert).execute()
                else: 
                    supabase_client.table("vector_store_file_manifest").update(manifest_data_to_upsert).eq("id", manifest_entry['id']).execute()
                logger.info(f"Manifiesto actualizado/creado para {file_path}.")
            except Exception as e_m_upsert: 
                logger.error(f"Error upsert manifiesto {file_path}: {e_m_upsert}")
        
        logger.info(f"--- Sincronización para Tenant: {tenant_identifier} completada ---")

    except Exception as e:
        logger.error(f"Error inesperado en sync_tenant_documents_to_pinecone para {tenant_identifier}: {e}", exc_info=True)


# --- Script Principal ---
if __name__ == "__main__":
    # ... (logging y validación de clientes globales sin cambios) ...
    logger.info("===== INICIANDO SCRIPT DE SINCRONIZACIÓN SUPABASE <> PINECONE (CLAVES POR TENANT - ESTRICTO) =====")
    if not supabase_client or not pinecone_client:
        logger.fatal("Cliente Supabase o Pinecone no inicializado. Terminando script.")
        exit(1)
    try:
        tenants_resp = supabase_client.table("tenants").select("id, identifier, pinecone_index_name, pinecone_namespace, openai_api_key_encrypted").eq("is_active", True).eq("vector_store_provider", "pinecone").execute()
        if tenants_resp.data:
            for tenant_data_row in tenants_resp.data:
                sync_tenant_documents_to_pinecone(tenant_data_row)
        else:
            logger.info("No se encontraron tenants activos para 'pinecone'."); 
            if tenants_resp.error: logger.error(f"Error obteniendo tenants: {tenants_resp.error}")
    except Exception as e_main: 
        logger.fatal(f"Error fatal script: {e_main}", exc_info=True)
    logger.info("===== SCRIPT DE SINCRONIZACIÓN FINALIZADO =====")

