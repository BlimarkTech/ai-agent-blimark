import openai
import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Iniciar la aplicación FastAPI
app = FastAPI()

@app.get("/chat")
def chat(query: str):
    response = openai_client.responses.create(
        model="gpt-4o",
        input=query
    )
    return {"response": response.output[0].content[0].text}
