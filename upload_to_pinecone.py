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
    raise ValueError("La variable de entorno ENCRYPTION_MASTER_KEY no está configurada.")
ENCRYPTION_MASTER_KEY = ENCRYPTION_MASTER_KEY_ENV.encode()

EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_UPSERT_BATCH_SIZE = 100

# --- Valores de chunking ajustados para RAG con OpenAI ---
# ~2000 caracteres (~400-500 tokens), solapamiento 200 caracteres
MARKDOWN_CHUNK_SIZE = 2000
MARKDOWN_CHUNK_OVERLAP = 200
# Desactivar recursión adicional usando mismos valores
RECURSIVE_CHUNK_SIZE = MARKDOWN_CHUNK_SIZE
RECURSIVE_CHUNK_OVERLAP = MARKDOWN_CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pinecone_sync_manager_strict_keys")

# --- Inicialización de clientes ---
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    cipher = Fernet(ENCRYPTION_MASTER_KEY)
    logger.info("Clientes de Supabase, Pinecone y cifrador inicializados.")
except Exception as e:
    logger.fatal(f"Error inicializando clientes: {e}", exc_info=True)
    exit(1)

# --- Helpers ---
def calculate_content_hash(content_bytes: bytes) -> str:
    return hashlib.md5(content_bytes).hexdigest()

def get_tenant_openai_key_for_script(tenant_uuid_str: str, tenant_identifier: str) -> str | None:
    try:
        resp = supabase_client.table("tenants") \
            .select("openai_api_key_encrypted") \
            .eq("id", tenant_uuid_str).single().execute()
        enc = resp.data.get("openai_api_key_encrypted")
        if not enc or not enc.strip():
            logger.error(f"{tenant_identifier}: falta openai_api_key_encrypted.")
            return None
        return cipher.decrypt(enc.encode()).decode()
    except Exception as e:
        logger.error(f"{tenant_identifier}: error desencriptando OpenAI key: {e}")
        return None

def get_markdown_chunks(markdown_content: str) -> list[str]:
    try:
        md_splitter = MarkdownTextSplitter(
            chunk_size=MARKDOWN_CHUNK_SIZE,
            chunk_overlap=MARKDOWN_CHUNK_OVERLAP
        )
        initial = md_splitter.split_text(markdown_content)
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE,
            chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = []
        for c in initial:
            if len(c) > RECURSIVE_CHUNK_SIZE:
                chunks.extend(recursive.split_text(c))
            else:
                chunks.append(c)
        logger.info(f"Split en {len(chunks)} chunks con tamaño {MARKDOWN_CHUNK_SIZE}±{MARKDOWN_CHUNK_OVERLAP}.")
        return chunks
    except Exception as e:
        logger.error(f"Error dividiendo MD: {e}")
        return []

def get_embedding_openai(client: SyncOpenAI, text_chunk: str) -> list[float] | None:
    try:
        txt = text_chunk.replace("\x00", "\uFFFD")
        if not txt.strip():
            return None
        resp = client.embeddings.create(input=[txt], model=EMBEDDING_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Error embedding: {e}")
        return None

def delete_vectors_from_pinecone(index, namespace: str, vector_ids: list[str]):
    if not vector_ids:
        return
    try:
        index.delete(ids=vector_ids, namespace=namespace)
        logger.info(f"Eliminados {len(vector_ids)} vectores en ns '{namespace}'.")
    except Exception as e:
        logger.error(f"Error eliminando vectores en ns '{namespace}': {e}")

def sync_tenant_documents_to_pinecone(tenant: dict):
    tid = str(tenant["id"])
    ident = tenant["identifier"]
    idx_name = tenant["pinecone_index_name"]
    namespace = tenant["pinecone_namespace"]
    logger.info(f"Sync tenant '{ident}' (ID: {tid}) -> index '{idx_name}', ns '{namespace}'")
    api_key = get_tenant_openai_key_for_script(tid, ident)
    if not api_key:
        logger.error(f"{ident}: no OpenAI key, omitiendo.")
        return
    client = SyncOpenAI(api_key=api_key)
    index = pinecone_client.Index(idx_name)

    # Listar archivos .md
    try:
        files = supabase_client.storage.from_(SUPABASE_BUCKET).list(path=f"{ident}/")
    except Exception as e:
        logger.error(f"{ident}: error listando storage: {e}")
        return
    md_files = [f for f in files if f["name"].endswith(".md")]
    logger.info(f"{ident}: {len(md_files)} archivos .md encontrados.")

    # Obtener manifiesto
    try:
        mf_resp = supabase_client.table("vector_store_file_manifest") \
            .select("*").eq("tenant_id", tid).execute()
        manifest = {m["file_path_in_storage"]: m for m in mf_resp.data}
    except Exception as e:
        logger.error(f"{ident}: error leyendo manifiesto: {e}")
        manifest = {}

    # Procesar cada archivo
    for md in md_files:
        path = f"{ident}/{md['name']}"
        old = manifest.get(path)
        try:
            b = supabase_client.storage.from_(SUPABASE_BUCKET).download(path)
        except Exception as e:
            logger.error(f"{ident}: error descargando '{path}': {e}")
            continue
        h = calculate_content_hash(b)
        is_new = (path not in manifest)
        is_mod = (not is_new and manifest[path]["content_hash"] != h)
        if not is_new and not is_mod:
            logger.info(f"{ident}: '{path}' sin cambios.")
            continue

        if old and old.get("pinecone_vector_ids"):
            delete_vectors_from_pinecone(index, namespace, old["pinecone_vector_ids"])

        text = b.decode("utf-8")
        chunks = get_markdown_chunks(text)
        new_ids, vectors = [], []
        for i, chunk in enumerate(chunks):
            emb = get_embedding_openai(client, chunk)
            if not emb:
                continue
            vid = f"{ident}-{os.path.splitext(md['name'])[0]}-chunk{i}"
            new_ids.append(vid)
            vectors.append({"id": vid, "values": emb,
                            "metadata": {"text": chunk, "document_name": os.path.splitext(md['name'])[0],
                                         "tenant": ident, "chunk_index": i, "source": path}})
        if vectors:
            for i in range(0, len(vectors), PINECONE_UPSERT_BATCH_SIZE):
                batch = vectors[i:i+PINECONE_UPSERT_BATCH_SIZE]
                try:
                    index.upsert(vectors=batch, namespace=namespace)
                except Exception as e:
                    logger.error(f"{ident} upsert error batch {(i//PINECONE_UPSERT_BATCH_SIZE)+1}: {e}")
                time.sleep(0.1)

        manifest_data = {
            "tenant_id": tid, "file_path_in_storage": path,
            "supabase_last_modified": md["metadata"]["lastModified"],
            "content_hash": h, "pinecone_vector_ids": new_ids,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            if is_new:
                supabase_client.table("vector_store_file_manifest").insert(manifest_data).execute()
            else:
                supabase_client.table("vector_store_file_manifest") \
                    .update(manifest_data).eq("id", old["id"]).execute()
            logger.info(f"{ident}: manifiesto {'creado' if is_new else 'actualizado'} para '{path}'.")
        except Exception as e:
            logger.error(f"{ident}: error actualizando manifiesto '{path}': {e}")

    logger.info(f"Sync completado para tenant '{ident}'.")

if __name__ == "__main__":
    logger.info("=== INICIANDO SYNC Supabase → Pinecone ===")
    try:
        tenants = supabase_client.table("tenants") \
            .select("id,identifier,pinecone_index_name,pinecone_namespace,openai_api_key_encrypted") \
            .eq("is_active", True).eq("vector_store_provider", "pinecone").execute().data
        for t in tenants:
            sync_tenant_documents_to_pinecone(t)
    except Exception as e:
        logger.fatal(f"Error fatal en script principal: {e}", exc_info=True)
    logger.info("=== SCRIPT DE SINCRONIZACIÓN FINALIZADO ===")
