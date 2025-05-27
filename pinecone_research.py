from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('multi-tenant-agent-docs')

# Usa un vector dummy de la dimensión correcta
dummy_vector = [0.0] * 1536
top_k = 100  # más que tus 18 vectores

response = index.query(
    vector=dummy_vector,
    top_k=top_k,
    namespace='arenatours',
    include_metadata=True
)
for match in response['matches']:
    print(f"ID: {match['id']}")
    print(f"Metadata: {match['metadata']}")
    print('-' * 40)
