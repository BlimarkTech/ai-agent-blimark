# main.py
import os
import logging
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Asegúrate de que está importado
from pydantic import BaseModel
from openai import AsyncOpenAI
import json # Necesario si los argumentos de la función vienen como string JSON

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente OpenAI
# Asegúrate de que la variable de entorno OPENAI_API_KEY esté configurada en Render
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")
    # Considera lanzar un error o manejar esto de forma adecuada si la key es esencial
openai_client = AsyncOpenAI(api_key=openai_api_key)

app = FastAPI()

# --- CORS Middleware (Correcto) ---
origins = ["*"] # Permite todos los orígenes para probar
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- FIN CORS Middleware ---

# Modelo para la solicitud POST
class ChatRequest(BaseModel):
    query: str

# Instrucciones del sistema (Completo y Correcto)
SYSTEM_MESSAGE = """
1. **Rol y objetivo:**
- Actúa como agente de servicio al cliente de la Agencia de Marketing *Blimark Tech*.
- *Blimark Tech* es una Agencia que presta servicios de Inteligencia Artificial aplicada al Marketing.
- Tu objetivo principal es captar leads y programar reuniones con potenciales clientes.
2. **Saludo, presentación y flujo de conversación:**
- Saluda al usuario con amabilidad, dale una cordial bienvenida y agradécele por ponerse en contacto con nosotros.
- Preséntate como Agente de *Blimark Tech* y pregúntale en qué podemos ayudarle hoy.
- Responde a todas las consultas del usuario en frases claras y concretas.
- Nunca respondas preguntas o consultas que no tengan que ver con los servicios que ofrece *Blimark Tech*. Responde que no puedes ayudar con eso.
3. **Consulta de datos de contacto:**
- Siempre que el usuario desee saber cómo ponerse en contacto con la empresa o hablar con un agente humano, consulta los datos de contacto en el vector store.
- Proporciónalos directamente al usuario.
4. **Consulta de servicios y precios:**
- Obtén la información sobre servicios, productos, precios y ofertas únicamente del vector store.
- Si el usuario solicita información sobre servicios o precios, consulta el vector store antes de responder.
- Si no encuentras información sobre un servicio o precio específico, informa al usuario que no dispones de esa información en este momento y sugiérele agendar una reunión con uno de nuestros expertos para recibir asistencia personalizada.
5. **Programación de reuniones:**
- Si el usuario expresa interés en programar una reunión, utiliza la función `recolectarInformacionContacto` para obtener sus datos.
- Confirma los datos con el usuario antes de finalizar.
- Si falta información obligatoria (nombre, email, mensaje), pídesela amablemente.
6. **Resolución de dudas:**
- Responde preguntas sobre *Blimark Tech* usando la información del vector store.
- Si no encuentras la respuesta, indica que no tienes esa información y ofrece agendar una reunión.
**Restricciones:**
- Usa exclusivamente el vector store para URLs, servicios, precios y datos de contacto.
- No inventes información ni muestres metadatos internos.
- No respondas preguntas ajenas a *Blimark Tech*.
- Mantén respuestas en máximo 500 caracteres.
- Sé transparente sobre tus límites.
"""

# --- Definición de tools (ESTRUCTURA CORREGIDA Y AJUSTADA PARA RESPONSES API) ---
tools = [
    { # Herramienta 0: File Search
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"]
    }, # Coma separadora
    { # Herramienta 1: Function (Estructura ajustada según error anterior)
        "type": "function",
        "name": "recolectarInformacionContacto", # <-- Nombre directamente aquí
        "description": "Recolecta información de contacto de un lead y un breve mensaje sobre sus necesidades.", # <-- Descripción directamente aquí
        "parameters": { # <-- Parámetros directamente aquí
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del lead."},
                "apellidos": {"type": "string", "description": "Apellidos del lead."},
                "email": {"type": "string", "description": "Correo electrónico del lead."},
                "telefono": {"type": "string", "description": "Número de teléfono del lead."},
                "pais": {"type": "string", "description": "País de residencia del lead."},
                "mensaje": {"type": "string", "description": "Breve descripción de los servicios."}
            },
            "required": ["nombre", "email", "mensaje"]
        }
    } # Cierre del diccionario de la herramienta 1
] # Cierre de la lista tools
# --- FIN Definición de tools ---


@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    logger.info(f"Query recibida: {request.query}")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": request.query}
        ] # Cierre de la lista messages

        # <<< Llamada a Responses API (Se mantiene) >>>
        response = await openai_client.responses.create(
            model="gpt-4.1", # Revisa compatibilidad o usa gpt-4o / gpt-4-turbo
            input=messages,
            tools=tools,
        )

        results = []
        # <<< Procesamiento de response.output (Se mantiene) >>>
        for output in response.output:
            if hasattr(output, "content") and output.content:
                if hasattr(output.content[0], 'text'):
                    results.append({"response": output.content[0].text})
            elif hasattr(output, "function_call"):
                results.append({
                    "function_call": {
                        "name": output.function_call.name,
                        "arguments": output.function_call.arguments
                    }
                })
            elif hasattr(output, "file_search_call"):
                file_results = getattr(output.file_search_call, "results", [])
                results.append({
                    "file_search_results": [
                        { # Apertura del diccionario del resultado
                            "text": getattr(res, 'text', ''),
                            "file_id": getattr(res, 'file_id', '')
                        } # Cierre del diccionario del resultado
                        for res in file_results
                    ] # Cierre de la lista de resultados
                })

        if not results:
            logger.warning("No se generaron resultados válidos desde la API.")
            return JSONResponse(
                status_code=500,
                content={"error": "No se pudo procesar la respuesta adecuadamente."},
                media_type="application/json; charset=utf-8"
            )

        # Devuelve el primer resultado (o ajusta si necesitas manejar múltiples)
        return JSONResponse(content=results[0], media_type="application/json; charset=utf-8")

    except Exception as e:
        # Loguea el error detallado en Render
        logger.error(f"Error en /chat endpoint: {e}", exc_info=True)
        # Devuelve un error 500 genérico y consistente al cliente (Botpress)
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno procesando la solicitud."},
            media_type="application/json; charset=utf-8"
        )

# Ejecución local (opcional)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

