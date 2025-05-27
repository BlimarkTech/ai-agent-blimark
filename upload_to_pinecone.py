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

load_dotenv()

# --- Configuración ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "config") # Cambia a "vector_store" si ese es tu bucket
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_UPSERT_BATCH_SIZE = 100

MARKDOWN_CHUNK_SIZE = 1000
MARKDOWN_CHUNK_OVERLAP = 50
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pinecone_sync_manager")

# --- Inicialización de Clientes ---
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    openai_s_client = SyncOpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Clientes de Supabase, OpenAI (Sync) y Pinecone inicializados.")
except Exception as e:
    logger.fatal(f"Error fatal inicializando clientes: {e}", exc_info=True); exit(1)

# --- Funciones Helper ---
def calculate_content_hash(content_bytes: bytes) -> str:
    """Calcula un hash MD5 del contenido del archivo."""
    return hashlib.md5(content_bytes).hexdigest()

def get_markdown_chunks(markdown_content: str) -> list[str]:
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

def get_embedding_openai(text_chunk: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    try:
        text_chunk_cleaned = text_chunk.replace("\x00", "\uFFFD")
        if not text_chunk_cleaned.strip(): return None
        response = openai_s_client.embeddings.create(input=[text_chunk_cleaned], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error embedding: '{text_chunk_cleaned[:50]}...': {e}"); return None

def delete_vectors_from_pinecone(pinecone_index: Pinecone.Index, namespace: str, vector_ids: list[str]):
    """Elimina una lista de vectores de Pinecone por sus IDs."""
    if not vector_ids:
        logger.info(f"No hay vector IDs para eliminar en namespace {namespace}.")
        return
    try:
        logger.info(f"Eliminando {len(vector_ids)} vectores del namespace '{namespace}' en Pinecone...")
        delete_response = pinecone_index.delete(ids=vector_ids, namespace=namespace)
        logger.info(f"Respuesta de eliminación de Pinecone: {delete_response}")
    except PineconeException as pe:
        logger.error(f"Error de Pinecone eliminando vectores del namespace '{namespace}': {pe}")
    except Exception as e:
        logger.error(f"Error inesperado eliminando vectores del namespace '{namespace}': {e}")

def sync_tenant_documents_to_pinecone(tenant_row_from_db: dict):
    """
    Sincroniza los documentos .md de un tenant desde Supabase Storage a Pinecone.
    Detecta archivos nuevos, modificados y eliminados.
    """
    tenant_uuid = tenant_row_from_db['id'] # El UUID de la tabla tenants
    tenant_identifier = tenant_row_from_db['identifier']
    pinecone_index_name = tenant_row_from_db['pinecone_index_name']
    pinecone_namespace = tenant_row_from_db['pinecone_namespace']

    if not all([tenant_uuid, tenant_identifier, pinecone_index_name, pinecone_namespace]):
        logger.error(f"Configuración incompleta para tenant {tenant_row_from_db.get('identifier', 'ID DESCONOCIDO')}. Saltando.")
        return

    logger.info(f"--- Iniciando Sincronización para Tenant: {tenant_identifier} (ID: {tenant_uuid}) ---")
    logger.info(f"Pinecone Index: {pinecone_index_name}, Namespace: {pinecone_namespace}")

    try:
        pinecone_index = pinecone_client.Index(pinecone_index_name)
        
        # 1. Obtener estado actual de archivos en Supabase Storage para este tenant
        storage_path_prefix = f"{tenant_identifier}/"
        try:
            supabase_files_list = supabase_client.storage.from_(SUPABASE_BUCKET).list(path=storage_path_prefix)
            # Convertir a un dict para fácil acceso: {'file_name.md': {'last_modified': ..., 'id': ..., ...}}
            current_supabase_files = {
                f['name']: {
                    'path': f"{storage_path_prefix}{f['name']}",
                    'last_modified': datetime.fromisoformat(f['metadata']['lastModified'].replace("Z", "+00:00")) if f['metadata'] else datetime.now(timezone.utc),
                    'size': f['metadata']['size'] if f['metadata'] else 0
                }
                for f in supabase_files_list if f['name'].endswith('.md')
            }
            logger.info(f"Encontrados {len(current_supabase_files)} archivos .md en Supabase Storage para {tenant_identifier}.")
        except Exception as e_storage_list:
            logger.error(f"Error listando archivos de Supabase Storage para {tenant_identifier}: {e_storage_list}")
            return

        # 2. Obtener estado del manifiesto desde la tabla `vector_store_file_manifest` para este tenant
        try:
            manifest_response = supabase_client.table("vector_store_file_manifest").select("*").eq("tenant_id", str(tenant_uuid)).execute()
            # Convertir a un dict para fácil acceso: {'file_path_in_storage': {manifest_data}}
            manifest_files = {mf['file_path_in_storage']: mf for mf in manifest_response.data}
            logger.info(f"Encontrados {len(manifest_files)} registros en el manifiesto para {tenant_identifier}.")
        except Exception as e_manifest_fetch:
            logger.error(f"Error obteniendo manifiesto de Supabase para {tenant_identifier}: {e_manifest_fetch}")
            return

        # 3. Identificar archivos eliminados de Supabase Storage (en manifiesto pero no en Supabase)
        manifest_paths = set(manifest_files.keys())
        supabase_paths = set(details['path'] for details in current_supabase_files.values())
        
        deleted_in_supabase_paths = manifest_paths - supabase_paths
        for deleted_path in deleted_in_supabase_paths:
            logger.info(f"Archivo ELIMINADO de Supabase detectado: {deleted_path}")
            manifest_entry = manifest_files[deleted_path]
            vector_ids_to_delete = manifest_entry.get('pinecone_vector_ids', [])
            if vector_ids_to_delete:
                delete_vectors_from_pinecone(pinecone_index, pinecone_namespace, vector_ids_to_delete)
            
            # Eliminar del manifiesto
            try:
                supabase_client.table("vector_store_file_manifest").delete().eq("id", manifest_entry['id']).execute()
                logger.info(f"Entrada del manifiesto eliminada para {deleted_path}.")
            except Exception as e_manifest_delete:
                 logger.error(f"Error eliminando entrada del manifiesto para {deleted_path}: {e_manifest_delete}")

        # 4. Identificar archivos nuevos o modificados en Supabase Storage
        for file_name_md, supabase_file_details in current_supabase_files.items():
            file_path = supabase_file_details['path']
            doc_name_base = os.path.splitext(file_name_md)[0] # ej. hoteles_maldivas
            
            # Descargar contenido para hashing y procesamiento
            try:
                file_content_bytes = supabase_client.storage.from_(SUPABASE_BUCKET).download(file_path)
            except Exception as e_download:
                logger.error(f"Error descargando {file_path} de Supabase: {e_download}")
                continue # Saltar este archivo

            current_content_hash = calculate_content_hash(file_content_bytes)
            
            is_new_file = file_path not in manifest_files
            is_modified_file = False
            if not is_new_file:
                manifest_entry = manifest_files[file_path]
                # Comparamos hash para detectar modificación de contenido.
                # Podrías también comparar supabase_file_details['last_modified'] con manifest_entry['supabase_last_modified']
                # pero el hash es más robusto para cambios de contenido.
                if manifest_entry.get('content_hash') != current_content_hash:
                    is_modified_file = True
            
            if is_new_file:
                logger.info(f"Archivo NUEVO detectado: {file_path}")
            elif is_modified_file:
                logger.info(f"Archivo MODIFICADO detectado: {file_path} (Hash diferente)")
                # Eliminar vectores antiguos del archivo modificado
                old_vector_ids = manifest_entry.get('pinecone_vector_ids', [])
                if old_vector_ids:
                    delete_vectors_from_pinecone(pinecone_index, pinecone_namespace, old_vector_ids)
            else:
                logger.info(f"Archivo SIN CAMBIOS: {file_path}. Saltando procesamiento.")
                continue # Saltar a la siguiente iteración del bucle de archivos

            # Procesar (o re-procesar) el archivo nuevo o modificado
            logger.info(f"Procesando para carga/actualización: {file_path}")
            content_md = file_content_bytes.decode('utf-8')
            text_chunks = get_markdown_chunks(content_md)
            if not text_chunks:
                logger.warning(f"No se generaron chunks para {file_path}. Saltando archivo."); continue

            new_pinecone_vector_ids = []
            vectors_for_upsert = []
            for i, chunk_text in enumerate(text_chunks):
                embedding_vector = get_embedding_openai(chunk_text)
                if embedding_vector:
                    vector_id = f"{tenant_identifier}-{doc_name_base}-chunk{i}" # ID consistente
                    new_pinecone_vector_ids.append(vector_id)
                    metadata = {
                        "text": chunk_text, "document_name": doc_name_base,
                        "tenant": tenant_identifier, "chunk_index": i,
                        "source_file_path": file_path
                    }
                    vectors_for_upsert.append({"id": vector_id, "values": embedding_vector, "metadata": metadata})
                else:
                    logger.warning(f"No se pudo generar embedding para chunk {i} de {file_path}. Saltando chunk.")
            
            if vectors_for_upsert:
                for j in range(0, len(vectors_for_upsert), PINECONE_UPSERT_BATCH_SIZE):
                    batch = vectors_for_upsert[j : j + PINECONE_UPSERT_BATCH_SIZE]
                    try:
                        logger.info(f"Upsert de {len(batch)} vectores para {file_path} a ns '{pinecone_namespace}' (Batch {j//PINECONE_UPSERT_BATCH_SIZE + 1})")
                        pinecone_index.upsert(vectors=batch, namespace=pinecone_namespace)
                    except Exception as e_upsert:
                        logger.error(f"Error en upsert para {file_path}, batch {j//PINECONE_UPSERT_BATCH_SIZE + 1}: {e_upsert}")
                    time.sleep(0.1)
            
            # Actualizar/Insertar entrada en el manifiesto
            manifest_data_to_upsert = {
                "tenant_id": str(tenant_uuid),
                "file_path_in_storage": file_path,
                "supabase_last_modified": supabase_file_details['last_modified'].isoformat(),
                "content_hash": current_content_hash,
                "pinecone_vector_ids": new_pinecone_vector_ids, # Lista de los nuevos IDs de vectores
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            try:
                if is_new_file:
                    supabase_client.table("vector_store_file_manifest").insert(manifest_data_to_upsert).execute()
                    logger.info(f"Nueva entrada de manifiesto creada para {file_path}.")
                else: # Archivo modificado
                    supabase_client.table("vector_store_file_manifest").update(manifest_data_to_upsert).eq("id", manifest_entry['id']).execute()
                    logger.info(f"Entrada de manifiesto actualizada para {file_path}.")
            except Exception as e_manifest_upsert:
                logger.error(f"Error actualizando/insertando manifiesto para {file_path}: {e_manifest_upsert}")

        logger.info(f"--- Sincronización para Tenant: {tenant_identifier} completada ---")

    except PineconeException as pe:
        logger.error(f"Error de Pinecone para {tenant_identifier}: {pe}", exc_info=True)
    except Exception as e:
        logger.error(f"Error inesperado en sync_tenant_documents_to_pinecone para {tenant_identifier}: {e}", exc_info=True)

# --- Script Principal ---
if __name__ == "__main__":
    logger.info("===== INICIANDO SCRIPT DE SINCRONIZACIÓN SUPABASE <> PINECONE =====")
    if not supabase_client or not pinecone_client or not openai_s_client:
        logger.fatal("Uno o más clientes globales no están inicializados. Terminando script.")
        exit(1)

    try:
        # Obtener todos los tenants activos configurados para Pinecone
        tenants_resp = supabase_client.table("tenants").select("id, identifier, pinecone_index_name, pinecone_namespace").eq("is_active", True).eq("vector_store_provider", "pinecone").execute()
        
        if tenants_resp.data:
            for tenant_data_row in tenants_resp.data:
                sync_tenant_documents_to_pinecone(tenant_data_row)
        else:
            logger.info("No se encontraron tenants activos configurados para 'pinecone'.")
            if tenants_resp.error:
                 logger.error(f"Error obteniendo tenants de Supabase: {tenants_resp.error}")
    except Exception as e_main:
        logger.fatal(f"Error fatal en el script principal de sincronización: {e_main}", exc_info=True)
    
    logger.info("===== SCRIPT DE SINCRONIZACIÓN SUPABASE <> PINECONE FINALIZADO =====")
