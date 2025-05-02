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
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --- MODIFICACIÓN CORS ---
# Habilitar CORS para permitir solicitudes desde cualquier origen durante las pruebas
origins = ["*"] # Permite todos los orígenes para probar

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False, # Debe ser False si allow_origins es ["*"]
    allow_methods=["*"],    # Permite POST y también OPTIONS (preflight)
    allow_headers=["*"],    # Permite todas las cabeceras
)
# --- FIN MODIFICACIÓN CORS ---

# Modelo para la solicitud POST
class ChatRequest(BaseModel):
    query: str

# Instrucciones del sistema (Se mantiene tu versión)
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
*(…misma estructura que antes…)*
6. **Resolución de dudas:**
*(…misma estructura que antes…)*
**Restricciones:**
- Usa exclusivamente el vector store para URLs, servicios, precios y datos de contacto.
- No inventes información ni muestres metadatos internos.
- No respondas preguntas ajenas a *Blimark Tech*.
- Mantén respuestas en máximo 500 caracteres.
- Sé transparente sobre tus límites.
"""

# Definición de tools (CORREGIDA la sintaxis faltante de {} y ,)
tools = [
    { # Falta el { de apertura del diccionario
        "type": "file_search",
        # "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"] # Descomenta si lo necesitas
    }, # Falta la , para separar elementos de la lista
    { # Falta el { de apertura del diccionario
        "type": "function",
        # <<< LA DEFINICIÓN DE LA FUNCIÓN DEBE ESTAR DENTRO DE UN OBJETO "function" >>>
        "function": {
            "name": "recolectarInformacionContacto",
            "description": "Recolecta información de contacto de un lead y un breve mensaje sobre sus necesidades.",
            "parameters": {
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
        } # Cierre del objeto "function"
    } # Cierre del diccionario de la herramienta función
]

@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    logger.info(f"Query recibida: {request.query}")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": request.query}
        ]

        # <<< SE MANTIENE TU LLAMADA A LA RESPONSES API >>>
        response = await openai_client.responses.create(
            model="gpt-4.1", # Asegúrate que este modelo sea compatible con Responses API o usa uno recomendado como gpt-4o / gpt-4-turbo
            input=messages,
            tools=tools,
            # include=["file_search_call.results"] # Verifica si este parámetro es válido en Responses API
        )

        results = []
        # <<< SE MANTIENE TU LÓGICA PARA PROCESAR response.output >>>
        for output in response.output:
            # Si es texto generado
            if hasattr(output, "content") and output.content:
                # Asumiendo que content es una lista y quieres el primer elemento de texto
                if hasattr(output.content[0], 'text'):
                     results.append({"response": output.content[0].text})
            # Si es llamada a función
            elif hasattr(output, "function_call"):
                results.append({
                    "function_call": {
                        "name": output.function_call.name,
                        "arguments": output.function_call.arguments # Puede ser un string JSON que necesites parsear
                    }
                })
            # Si es file_search
            elif hasattr(output, "file_search_call"):
                # Asegúrate de que la estructura de file_search_call sea la que esperas
                file_results = getattr(output.file_search_call, "results", [])
                results.append({
                    "file_search_results": [
                        {
                            "text": getattr(res, 'text', ''), # Acceso más seguro a atributos
                            "file_id": getattr(res, 'file_id', '')
                        } for res in file_results
                    ]
                })

        if not results:
             logger.warning("No se generaron resultados válidos desde la API.")
             return JSONResponse(
                 content={"response": "No se pudo procesar la respuesta adecuadamente."},
                 media_type="application/json; charset=utf-8"
             )

        # Si solo hay una respuesta, devuélvela directo
        if len(results) == 1:
            return JSONResponse(content=results[0], media_type="application/json; charset=utf-8")
        # Si hay varias (texto + file_search), devuélvelas todas
        else:
            return JSONResponse(content={"results": results}, media_type="application/json; charset=utf-8")

    except Exception as e:
        logger.error(f"Error en /chat: {e}", exc_info=True) # Loguea el traceback
        # Devuelve un error 500 genérico
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno procesando la solicitud."},
            media_type="application/json; charset=utf-8"
        )

# Para ejecutar localmente (opcional)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
