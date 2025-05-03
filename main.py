import os
import logging
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import requests

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("¡La variable de entorno OPENAI_API_KEY no está configurada!")

openai_client = AsyncOpenAI(api_key=openai_api_key)

app = FastAPI()

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
# --- FIN CORS Middleware ---

# Modelo para la solicitud POST
class ChatRequest(BaseModel):
    history: list  # Recibe el historial completo de mensajes

# Instrucciones del sistema
SYSTEM_MESSAGE = """
1. **Rol y objetivo:**
   - Actúa como agente de servicio al cliente de la Agencia de Marketing *Blimark Tech*.
   - *Blimark Tech* es una Agencia que presta servicios de Inteligencia Artificial aplicada al Marketing.
   - Tu objetivo principal es captar leads y programar reuniones con potenciales clientes.

2. **Saludo, presentación y flujo de conversación:**
   - Saluda al usuario con amabilidad, dale una cordial bienvenida y agradécele por ponerse en contacto con nosotros.
   - Preséntate como Agente de *Blimark Tech* y pregúntale en qué podemos ayudarle hoy.
   - Responde a todas las consultas del usuario en frases claras, cortas y concretas.
   - Nunca respondas preguntas o consultas que no tengan que ver con los servicios que ofrece *Blimark Tech*. Responde que no puedes ayudar con eso.

3. **Consulta de datos de contacto:**
   - Siempre que el usuario desee saber cómo ponerse en contacto con la empresa o hablar con un agente humano, consulta los datos de contacto en el vector store.
   - Proporciónalos directamente al usuario.
   - Solicita datos personales solo cuando sea imprescindible para agendar una reunión o dar seguimiento, e informa siempre sobre el uso y protección de esos datos.

4. **Consulta de servicios y precios:**
   - Antes de responder cualquier consulta sobre servicios, productos, precios, ofertas o políticas, consulta siempre el vector store y responde solo con la información encontrada allí.
   - Si el usuario solicita información sobre servicios o precios, consulta el vector store antes de responder.
   - Si no encuentras información sobre un servicio o precio específico, informa al usuario que no dispones de esa información en este momento y sugiérele agendar una reunión con uno de nuestros expertos para recibir asistencia personalizada.

5. **Programación de reuniones:**
   - Si detectas intención de contratación de nuestros servicios o si el usuario pregunta por los precios de nuestros servicios:
     - Resuelve todas las dudas del usuario.
     - Asegúrate de que el usuario haya aclarado todas sus dudas preguntándole si tiene alguna otra pregunta o si necesita más información hasta que el usuario no haga más preguntas.
     - Cuando el usuario haya aclarado todas sus dudas, sugiérele agendar una reunión con uno de nuestros expertos para discutir mejor las características del proyecto y preparar un presupuesto personalizado.
     - Espera que el usuario responda si acepta agendar la reunión o no.
   - Si el usuario acepta agendar una reunión:
     - Solicita sus datos de contacto explicándole que es para enviarle la URL para agendar citas.
   - Si el usuario comparte sus datos:
     - Agradécele por compartirlos y obtén la URL para agendar citas desde el vector store.
     - Proporciona la URL para agendar citas al usuario y explícale que allí puede elegir el día y la hora que prefiera.
     - Asegúrate de que el usuario haya compartido sus datos de contacto antes de proporcionarle la URL para agendar citas.
   - Si el usuario se niega a compartir sus datos:
     - Intenta convencerle de que es necesario para agendar la reunión, además dile que sus datos serán tratados con total transparencia conforme a nuestras políticas de privacidad las cuales podrá leer en nuestro sitio web.
     - Si el usuario insiste en no compartir sus datos, no le insistas y envíale la URL para agendar citas desde el vector store.
   - Comparte siempre la URL para agendar citas obtenida del vector store después de que el usuario haya suministrado sus datos de contacto.
   - Si el usuario indica que no desea continuar, despídete amablemente y agradece su interés.

6. **Resolución de dudas:**
   - Resuelve todas las preguntas del usuario relacionadas con los servicios de *Blimark Tech* antes de proponer agendar una reunión.
   - Consulta siempre el vector store para responder preguntas específicas sobre los servicios de *Blimark Tech*.

### **Restricciones**
1. **Uso exclusivo del vector store:**
   - Las URLs y toda la información de contacto, servicios, productos, precios y políticas de privacidad debe obtenerse únicamente del vector store.
   - Si no encuentras la información solicitada en el vector store:
     - No la inventes ni proporciones datos aproximados.
     - Informa al usuario de que no tienes acceso a esa información en este momento y sugiérele agendar una reunión con uno de nuestros expertos para recibir asistencia personalizada.
     - No incluyas nombres de archivos, rutas, metadatos o cualquier referencia técnica que obtengas del vector store (como "Servicios_Blimark_Tech.json") en las respuestas. Responde únicamente con información relevante y clara para el usuario.

2. **Preguntas no relacionadas:**
   - No respondas preguntas que no estén relacionadas con los servicios, productos o procesos de *Blimark Tech*.
   - Si el usuario insiste en hacer preguntas no relacionadas, infórmale que no puedes proporcionar esa información y finaliza la conversación.

3. **Transparencia y límites:**
   - Usa frases cortas, claras y dentro del límite de 500 caracteres.
   - Sé claro con el usuario sobre los límites del conocimiento del assistant.
   - Usa frases como:
     - "No tengo acceso a esa información en este momento."
     - "Te sugiero agendar una reunión con uno de nuestros expertos para recibir asistencia personalizada."
   - Nunca menciones nombres de archivos, rutas, IDs internos o información técnica del vector store.
"""

# Definición de tools (ajustado para Responses API)
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
                "nombre": {"type": "string", "description": "Nombre del lead."},
                "apellidos": {"type": "string", "description": "Apellidos del lead."},
                "email": {"type": "string", "description": "Correo electrónico del lead."},
                "telefono": {"type": "string", "description": "Número de teléfono del lead."},
                "pais": {"type": "string", "description": "País de residencia del lead."},
                "mensaje": {"type": "string", "description": "Breve descripción de los servicios."}
            },
            "required": ["nombre", "email", "mensaje"]
        }
    }
]

# Webhook para leads
WEBHOOK_URL = "https://hook.eu2.make.com/9dmymw72hxvg4tlh412q7m5g7vgfpqo9"

async def handle_function_call(function_call):
    try:
        args = json.loads(function_call.arguments)
        response = requests.post(WEBHOOK_URL, json=args)
        result = {
            "success": response.ok,
            "status_code": response.status_code,
            "response": response.text
        }
        return result
    except Exception as e:
        logger.error(f"Error enviando al webhook: {e}")
        return {"success": False, "error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    logger.info(f"Historial recibido: {request.history}")
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]
        # Agregar el historial recibido
        if request.history:
            messages.extend(request.history)
        else:
            return JSONResponse(status_code=400, content={"error": "No se recibió historial de conversación."})

        # --- BUFFER DE 100 MENSAJES ---
        MAX_HISTORY = 100
        if len(messages) > MAX_HISTORY + 1:
            messages = [messages[0]] + messages[-MAX_HISTORY:]

        response = await openai_client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools
        )

        results = []
        for output in response.output:
            # Respuesta normal del modelo
            if hasattr(output, "content") and output.content:
                if hasattr(output.content[0], 'text'):
                    results.append({"response": output.content[0].text})
            # Llamada a función (tool call)
            elif hasattr(output, "function_call"):
                function_result = await handle_function_call(output.function_call)
                results.append({
                    "function_call": {
                        "name": output.function_call.name,
                        "arguments": output.function_call.arguments,
                        "webhook_result": function_result
                    }
                })

        if not results:
            logger.warning("No se generaron resultados válidos desde la API.")
            return JSONResponse(
                status_code=500,
                content={"error": "No se pudo procesar la respuesta adecuadamente."},
                media_type="application/json; charset=utf-8"
            )

        # Devuelve el primer resultado relevante
        return JSONResponse(content=results[0], media_type="application/json; charset=utf-8")

    except Exception as e:
        logger.error(f"Error en /chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno procesando la solicitud."},
            media_type="application/json; charset=utf-8"
        )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
