import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI   # Cliente asíncrono

# Configuración de logging y cliente
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# Instrucciones de sistema
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

# Definición de tools
tools = [
    {
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"]
    },
    {
        "type": "function",
        "name": "recolectarInformacionContacto",
        "description": "Recolecta información de contacto de un lead y un breve mensaje sobre sus necesidades.",
        "parameters": {
            "type": "object",
            "properties": {
                "nombre":    {"type": "string", "description": "Nombre del lead."},
                "apellidos": {"type": "string", "description": "Apellidos del lead."},
                "email":     {"type": "string", "description": "Correo electrónico del lead."},
                "telefono":  {"type": "string", "description": "Número de teléfono del lead."},
                "pais":      {"type": "string", "description": "País de residencia del lead."},
                "mensaje":   {"type": "string", "description": "Breve descripción de los servicios."}
            },
            "required": ["nombre", "email", "mensaje"]
        }
    }
]

@app.get("/chat")
async def chat(query: str):
    logger.info(f"Query recibida: {query}")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": query}
        ]
        response = await openai_client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )
        output = response.output[0]

        if output.content:
            text = output.content[0].text
            return JSONResponse(
                content={"response": text},
                media_type="application/json; charset=utf-8"
            )

        if output.function_call:
            return JSONResponse(
                content={
                    "function_call": {
                        "name":      output.function_call.name,
                        "arguments": output.function_call.arguments
                    }
                },
                media_type="application/json; charset=utf-8"
            )

        return JSONResponse(
            content={"response": "No se pudo procesar la respuesta"},
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error("Error en /chat:", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            media_type="application/json; charset=utf-8"
        )
