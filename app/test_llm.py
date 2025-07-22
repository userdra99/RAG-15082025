from llama_index.llms.openai import OpenAI
import os

api_base = os.environ.get("OPENAI_API_BASE", "http://nginx:80/v1")
api_key = os.environ.get("OPENAI_API_KEY", "sk-12345")

print("DEBUG: Using OpenAI API base:", api_base)
print("DEBUG: Using OpenAI API key:", api_key)

llm = OpenAI(
    model="gpt-3.5-turbo",
    api_base=api_base,
    api_key=api_key
)

response = llm.complete("Hello, how are you?")
print("LLM response:", response) 