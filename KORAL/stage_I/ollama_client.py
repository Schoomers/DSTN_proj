"""Drop-in replacement for OpenAI client that calls Ollama instead"""
import requests

class OllamaClient:
    
    def __init__(self, model = "llama3", api_key=None, base_url=None, **kwargs):
    # Handle base_url with /v1 suffix (OpenAI style)
        if base_url and base_url.endswith('/v1'):
            base_url = base_url[:-3]
        
        self.base_url = base_url or "http://localhost:11434"
        self.api_key = api_key  # Ignored, just for compatibility
        self.model = model

    class ChatCompletion:
        def __init__(self, client):
            self.client = client
        
        def create(self, model=None, messages=None, **kwargs):
            # Convert messages to prompt
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            # Call Ollama
            response = requests.post(
                f"{self.client.base_url}/api/generate",
                json={
                    "model": self.client.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",   # 🔥 THIS IS KEY FOR OLLAMA
                    "options": {
                        "temperature": 0.1
                    }
                }
            ).json()
            
            # Return in OpenAI format
            class Response:
                class Choice:
                    class Message:
                        content = response['response']
                    message = Message()
                choices = [Choice()]
            return Response()
    
    @property
    def chat(self):
        class Chat:
            completions = self.ChatCompletion(self)
        return Chat()
