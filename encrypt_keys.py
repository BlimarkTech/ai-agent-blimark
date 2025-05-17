import os
from cryptography.fernet import Fernet
from supabase import create_client

print("ENV ENCRYPTION_MASTER_KEY:", os.getenv("ENCRYPTION_MASTER_KEY"))
print("ENV SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("ENV SUPABASE_SERVICE_KEY:", os.getenv("SUPABASE_SERVICE_KEY"))

master_key = os.environ["ENCRYPTION_MASTER_KEY"].encode()
cipher = Fernet(master_key)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
print("Inicializando Supabase client…")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client inicializado.")

def encrypt_and_store(tenant_identifier: str, plain_api_key: str):
    print(f"\n– Procesando tenant '{tenant_identifier}'")
    token = cipher.encrypt(plain_api_key.encode()).decode()
    print(f"  Token cifrado (inicio y fin): {token[:8]}…{token[-8:]}")

    resp = supabase.table("tenants")\
        .update({"openai_api_key_encrypted": token})\
        .eq("identifier", tenant_identifier)\
        .execute()

    # Éxito si resp.data no es []
    if not resp.data:
        raise RuntimeError(f"Error actualizando {tenant_identifier}: no se actualizó ninguna fila.")
    print(f"  ✅ {tenant_identifier} actualizado. Filas afectadas: {len(resp.data)}")

if __name__ == "__main__":
    encrypt_and_store(
        "cliente-1", # Reemplaza "cliente-1" por el "identifier" del cliente de Supabase Tabla tenants
        "API_KEY_OPENAI_CLIENTE_1" # Agrega aquí la API Key real de OpenAI de este cliente
    )
    encrypt_and_store(
        "cliente-2", # Reemplaza "cliente-2" por el "identifier" del cliente de Supabase Tabla tenants
        "API_KEY_OPENAI_CLIENTE_2" # Agrega aquí la API Key real de OpenAI de este cliente
    )
    print("\n¡Script terminado!")
