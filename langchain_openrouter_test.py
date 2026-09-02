import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

model_name = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it:free")

llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Kei AI Local Test",
    },
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep the response brief."),
    ("human", "Say hello and tell me which model you are using."),
])

if __name__ == "__main__":
    response = llm.invoke(prompt.format_messages())
    print(response.content)
