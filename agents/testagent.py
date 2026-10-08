from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL

client = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
)

response = client.invoke([{"role": "user", "content": "Tell me a joke"}])

print(response.content)