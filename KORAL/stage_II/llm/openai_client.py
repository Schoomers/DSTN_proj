#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI Chat Completions client wrapper (GPT-4o)."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from stage_II.utils.io import safe_get_env

@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]

class OpenAIChatClient:
    """Minimal client to avoid hard dependency on openai SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: int = 120,
    ):
        self.model = model
        self.api_key = api_key or safe_get_env("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Export it before running Stage II.")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            payload["seed"] = int(seed)

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
        raw = r.json()
        text = raw["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=raw)
