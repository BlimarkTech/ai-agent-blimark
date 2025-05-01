import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Iniciar la aplicación FastAPI
app = FastAPI()

# Instrucciones de sistema (tal como las tienes en el Playground)
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
   - Si detectas intención de contratación de nuestros servicios o si el usuario pregunta por los precios de nuestros servicios:
     - Resuelve todas las dudas del usuario.
     - Asegúrate de que el usuario haya aclarado todas sus dudas preguntándole si tiene alguna otra pregunta o si necesita más información hasta que el usuario no haga más preguntas.
     - Cuando el usuario haya aclarado todas sus dudas sugiérele agendar una reunión con uno de nuestros expertos para discutir mejor las características del proyecto y preparar un presupuesto personalizado.
     - Espera que el usuario responda si acepta agendar la reunión o no.
   - Si el usuario acepta agendar una reunión:
     - Solicita sus datos de contacto explicándole que es para enviarle la URL para agendar citas.
   - Si el usuario comparte sus datos:
     - Agradécele por compartirlos y obtén la URL para agendar citas desde el vector store.
     - Proporciona la URL para agendar citas al usuario y explícale que allí puede elegir el día y la hora que prefiera.
     - Asegúrate que el usuario haya compartido sus datos de contacto antes de proporcionarle la URL para agendar citas.
   - Si el usuario se niega a compartir sus datos:
     - Intenta convencerle de que es necesario para agendar la reunión, además dile que sus datos serán tratados con total transparencia conforme a nuestras políticas de privacidad las cuales podrá leer en nuestro sitio web.
     - Si el usuario insiste en no compartir sus datos, no le insistas y envíale la URL para agendar citas desde el vector store.
   - Comparte siempre la URL para agendar citas obtenida del vector store después de que el usuario haya suministrado sus datos de contacto.

6. **Resolución de dudas:**
   - Resuelve todas las preguntas del usuario relacionadas con los servicios de *Blimark Tech* antes de proponer agendar una reunión.
   - Consulta siempre el vector store para responder preguntas específicas sobre los servicios de *Blimark Tech*.

**Restricciones:**
- Usa exclusivamente el vector store para URLs, servicios, precios y datos de contacto.
- No inventes información ni muestres metadatos internos.
- No respondas preguntas ajenas a *Blimark Tech*.
- Mantén respuestas en máximo 500 caracteres.
- Sé transparente sobre tus límites.
"""

# Configuración del File Search Tool apuntando al vector store
tools = [
    {
        "type": "file_search",
        "vector_store_ids": ["vs_UJO3EkBk4HnIk1M0Ivv7Wmnz"]
    }
]

# Definición de la función recolectarInformacionContacto
functions = [
    {
        "name": "recolectarInformacionContacto",
        "description": "Recolecta información de contacto de un lead y un breve mensaje sobre sus necesidades.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "nombre": {
                    "type": "string",
                    "description": "Nombre del lead."
                },
                "apellidos": {
                    "type": "string",
                    "description": "Apellidos del lead."
                },
                "email": {
                    "type": "string",
                    "description": "Correo electrónico del lead."
                },
                "telefono": {
                    "type": "string",
                    "description": "Número de teléfono del lead, incluyendo código de país si es posible."
                },
                "pais": {
                    "type": "string",
                    "description": "País de residencia del lead."
                },
                "mensaje": {
                    "type": "string",
                    "description": "Breve descripción de los servicios que el lead necesita (ejemplo: diseño web, servicios SEO, campañas SEM, etc.)."
                }
            },
            "required": ["nombre", "email", "mensaje"]
        }
    }
]

@app.get("/chat")
async def chat(query: str):
    # Construir el array de mensajes con sistema y usuario
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": query}
    ]

    # Llamada a la Responses API con herramientas y función
    response = await openai_client.responses.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        functions=functions
    )

    # Procesar salida o llamada a función
    output = response.output[0]
    if output.content:
        text = output.content[0].text
        return {"response": text}
    elif output.function_call:
        return {
            "function_call": {
                "name": output.function_call.name,
                "arguments": output.function_call.arguments
            }
        }
    else:
        return {"response": None}

