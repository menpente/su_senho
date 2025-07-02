from langchain.llms.base import LLM
from typing import Optional, List
import os
import requests

class GroqLLM(LLM):
    model: str = "llama3"
    temperature: float = 0.2
    api_key: str = os.getenv("GROQ_API_KEY")
    endpoint: str = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = requests.post(self.endpoint, json=data, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
