"""Drop-in replacement for the OpenAI client that calls a local Ollama server.

Used by stage_I.ssd_kg_pipeline_ollama as `OpenAI`. The public surface mirrors
openai>=1.0 just enough for the pipeline to work:

    client = OllamaClient(base_url="http://localhost:11434", api_key="ollama")
    resp = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[...],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    text = resp.choices[0].message.content
"""

from __future__ import annotations
import json as _json
import requests


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str, raw: dict):
        self.choices = [_Choice(content)]
        self.raw = raw


class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", api_key=None, base_url=None, **kwargs):
        # Strip OpenAI-style /v1 suffix if present so the native /api/... paths work.
        if base_url and base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/")[:-3]
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.api_key = api_key  # unused, kept for API parity
        self.model = model
        self.chat = _Chat(self)


class _Chat:
    def __init__(self, client: OllamaClient):
        self._client = client
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: OllamaClient):
        self._client = client

    def create(
        self,
        model: str | None = None,
        messages=None,
        temperature: float = 0.1,
        response_format=None,
        timeout: int = 300,
        **kwargs,
    ) -> _Response:
        model = model or self._client.model
        messages = messages or []

        # Ollama /api/chat expects the native message list. Preserve roles.
        payload = {
            "model": model,
            "messages": [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in messages
            ],
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        url = f"{self._client.base_url}/api/chat"
        r = requests.post(url, json=payload, timeout=timeout)

        try:
            data = r.json()
        except Exception as ex:
            raise RuntimeError(
                f"Ollama returned non-JSON body (status={r.status_code}): {r.text[:500]}"
            ) from ex

        # Ollama may return an error with no "message" key. Surface it clearly.
        if "error" in data:
            raise RuntimeError(f"Ollama error ({r.status_code}): {data['error']}")

        msg = data.get("message") or {}
        content = msg.get("content")
        if content is None:
            # Fallback for older /api/generate-style servers that return "response".
            content = data.get("response")
        if content is None:
            raise RuntimeError(
                f"Ollama response had no message.content or response field: {list(data.keys())[:6]}"
            )

        return _Response(content=str(content), raw=data)
