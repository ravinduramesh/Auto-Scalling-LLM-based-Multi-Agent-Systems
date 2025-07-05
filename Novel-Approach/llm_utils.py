import os

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

if os.getenv("LLM_SELECTION") == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif os.getenv("LLM_SELECTION") == "gemini":
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def callLLM(messages):
    if os.getenv("LLM_SELECTION") == "openai":
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            response_format={ "type": "text"}
        )
    if os.getenv("LLM_SELECTION") == "gemini":
        response = client.chat.completions.create(
            messages=messages,
            model="gemini-2.5-flash",
            response_format={ "type": "text"}
        )

    content = response.choices[0].message.content.strip("'")
    return content