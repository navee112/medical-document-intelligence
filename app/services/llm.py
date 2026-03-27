import json
from typing import Any

import httpx

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_PROVIDER,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT_S,
)


def _completion_urls() -> list[str]:
    base = LLM_BASE_URL.strip()
    if not base:
        return []
    if base.endswith("/chat/completions"):
        return [base]

    normalized = base.rstrip("/")
    urls = [f"{normalized}/chat/completions"]
    if not normalized.endswith("/v1"):
        urls.append(f"{normalized}/v1/chat/completions")
    return urls


def _request_openai_compatible(
    prompt: str,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    urls = _completion_urls()
    if not (LLM_API_KEY and urls):
        return None

    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        for url in urls:
            try:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except Exception:
                continue
    return None


def _request_ollama(prompt: str, json_mode: bool = False) -> str | None:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                return content.strip()
        except Exception:
            return None
    return None


def _request_completion(prompt: str, response_format: dict[str, Any] | None = None) -> str | None:
    provider = LLM_PROVIDER or ""
    if provider == "openai_compatible":
        return _request_openai_compatible(prompt, response_format=response_format)
    if provider == "ollama":
        return _request_ollama(prompt, json_mode=bool(response_format))

    # Auto fallback when provider is misconfigured.
    content = _request_ollama(prompt, json_mode=bool(response_format))
    if content is not None:
        return content
    return _request_openai_compatible(prompt, response_format=response_format)


def get_llm_status() -> dict[str, Any]:
    provider = LLM_PROVIDER or "unknown"

    if provider == "openai_compatible":
        configured = bool(LLM_API_KEY and _completion_urls())
        return {
            "provider": provider,
            "model": MODEL_NAME,
            "configured": configured,
            "endpoint": LLM_BASE_URL,
        }

    if provider == "ollama":
        return {
            "provider": provider,
            "model": OLLAMA_MODEL,
            "configured": bool(OLLAMA_BASE_URL),
            "endpoint": OLLAMA_BASE_URL,
        }

    return {
        "provider": provider,
        "model": OLLAMA_MODEL,
        "configured": bool(OLLAMA_BASE_URL or LLM_BASE_URL),
        "endpoint": OLLAMA_BASE_URL or LLM_BASE_URL,
    }


def call_llm(prompt: str) -> str:
    content = _request_completion(prompt)
    if content is not None:
        return content

    status = get_llm_status()
    provider = status.get("provider", "unknown")
    model = status.get("model", "unknown")
    return (
        "LLM backend unavailable. "
        f"Current provider={provider}, model={model}. "
        "Start Ollama and pull the configured model, or configure an OpenAI-compatible endpoint."
    )


def call_llm_json(prompt: str) -> dict[str, Any]:
    content = _request_completion(prompt, response_format={"type": "json_object"})
    if content is None:
        content = _request_completion(prompt)
        if content is None:
            return {}

    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
        return {}
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {}
    return {}
