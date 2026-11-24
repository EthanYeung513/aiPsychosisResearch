"""Minimal OpenAI-compatible chat client.

Written against the OpenAI chat-completions response shape. Compatibility with any specific server
(vLLM, llama.cpp, Ollama, OpenRouter) is unverified until a smoke run is done against it.

Failures raise ChatError: transport errors, HTTP errors, malformed bodies and empty content. Callers must
never treat an error as a model response. A reply cut off at the token limit is returned with
truncated=True so callers can decide (the runner records it; the judge treats it as invalid).
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class ChatError(RuntimeError):
    pass


Message = dict[str, str]  # {"role": ..., "content": ...}


@dataclass(frozen=True)
class Reply:
    content: str
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class ChatClient(Protocol):
    def chat(self, messages: list[Message]) -> Reply: ...

    def describe(self) -> dict: ...


def parse_completion(body: object) -> Reply:
    """Extract the first choice from a chat-completions body; any shape problem raises ChatError."""
    try:
        choice = body["choices"][0]  # type: ignore[index]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise ChatError(f"malformed completion body: {type(e).__name__}: {e}") from e
    if not isinstance(content, str) or not content.strip():
        raise ChatError(f"empty or non-text content: {content!r}")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ChatError(f"non-string finish_reason: {finish_reason!r}")
    return Reply(content.strip(), finish_reason)


@dataclass
class OpenAICompatibleClient:
    base_url: str  # e.g. http://localhost:8000/v1
    model: str
    api_key_env: str | None = None  # name of env var holding the key; the key itself is never logged
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_delay_s: float = 2.0
    extra_body: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "extra_body": self.extra_body,
        }

    def _payload(self, messages: list[Message]) -> dict:
        body: dict = {"model": self.model, "messages": messages, **self.extra_body}
        for key in ("temperature", "max_tokens", "seed"):
            value = getattr(self, key)
            if value is not None:
                body[key] = value
        return body

    def _post(self, url: str, data: bytes, headers: dict) -> object:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, messages: list[Message]) -> Reply:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ChatError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(self._payload(messages)).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return parse_completion(self._post(url, data, headers))
            # OSError covers URLError/HTTPError, timeouts and connection resets; ValueError covers bad JSON/UTF-8.
            except (OSError, http.client.HTTPException, ValueError, ChatError) as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_s * (attempt + 1))
        raise ChatError(f"{self.model} @ {self.base_url} failed after {self.max_retries} attempts: {last}")
