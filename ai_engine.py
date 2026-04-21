"""AI response engine: rule-based templates + optional Ollama fallback."""


#Design notes:
#if short_message and casual:
    #use_gemma_4b()
#elif complex_question:
    #use_gemma_12b()

#So llm responses are reserved for when they add the most value, and we can use fast templates for common cases.
#Small messages or casual conversation can use a smaller, faster model or template system. 
#Complex questions or when we want more personality can use a larger model.

#Test 3 models:

#Gemma 3 4B
#LLaMA 3.2 3B
#Gemma 3 12B

#Measure:

#time to first token
#total response time

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from typing import Dict, Optional

import dialogue

import requests
import json


def _ollama_generate(prompt: str, model: str = "llama3", timeout: float = 3.0) -> Optional[str]:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    try:
        import requests

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.3")),
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"]
    except Exception:
        logging.debug("HTTP Ollama call failed, falling back to CLI", exc_info=True)

    try:
        proc = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            output = proc.stdout.strip()
            if output:
                return output
    except Exception:
        logging.debug("Ollama CLI call failed", exc_info=True)

    return None


def _persona_block() -> str:
    return (
        "You are Kei Tendou, acting as the user's focus assistant. "
        "Stay in character. Reply briefly, clearly, and with calm but firm accountability. "
        "Do not mention prompts, instructions, roleplay, or being an AI model. "
        "Use the desktop activity context to keep the user on task."
    )


def _build_prompt(trigger: str, context: Dict, template: str) -> str:
    try:
        context_json = json.dumps(context, ensure_ascii=False)
    except Exception:
        context_json = str(context)
    return (
        f"{_persona_block()}\n\n"
        f"System trigger: {trigger}\n"
        f"Context: {context_json}\n"
        f"Baseline reply: {template}\n\n"
        "Write a short spoken reply."
    )


def _build_conversation_prompt(user_text: str, context: Dict) -> str:
    try:
        context_json = json.dumps(context, ensure_ascii=False)
    except Exception:
        context_json = str(context)
    return (
        f"{_persona_block()}\n\n"
        f"Desktop context: {context_json}\n"
        f"User said: {user_text}\n\n"
        "Reply as Kei in 1-3 sentences. If the user seems distracted, steer them back to work."
    )


def _openrouter_generate(prompt: str, model: Optional[str] = None, timeout: float = 6.0) -> Optional[str]:
    """Send a chat completion request to OpenRouter and return the assistant text.

    Reads API key from `GEMMA3_4B_API_KEY` or `OPENROUTER_API_KEY`.
    Endpoint can be overridden via `OPENROUTER_ENDPOINT` (defaults to api.openrouter.ai path).
    """
    api_key = (
        os.environ.get("GEMMA3_4B_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("YUUKA_OPENROUTER_API_KEY")
    )
    if not api_key:
        logging.debug("OpenRouter API key not configured")
        return None

    configured = os.environ.get("OPENROUTER_ENDPOINT")
    # Try configured endpoint first, otherwise try the common endpoints
    endpoints = [configured] if configured else [
        "https://api.openrouter.ai/v1/chat/completions",
        "https://openrouter.ai/api/v1/chat/completions",
    ]
    model_name = model or os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "gemma3-4b"))

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _persona_block()},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", os.environ.get("YUUKA_OPENROUTER_TEMPERATURE", 0.3))),
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", 256)),
    }

    logging.info(
        "OpenRouter request: model=%s key_present=%s timeout=%s",
        model_name,
        bool(api_key),
        timeout,
    )
    logging.debug(
        "OpenRouter payload meta: temperature=%s max_tokens=%s messages=%d",
        payload["temperature"],
        payload["max_tokens"],
        len(payload["messages"]),
    )

    for endpoint in endpoints:
        if not endpoint:
            continue
        logging.info("Attempting OpenRouter endpoint: %s", endpoint)
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            logging.info("OpenRouter HTTP status: %s", getattr(resp, "status_code", None))
            resp.raise_for_status()
            data = resp.json()

            result_text: Optional[str] = None

            # Common chat response shape: choices[0].message.content
            if isinstance(data, dict):
                choices = data.get("choices")
                if choices and isinstance(choices, list):
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("delta")
                        if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                            result_text = msg["content"]
                        else:
                            # fallback to text/content keys
                            for k in ("text", "content"):
                                if k in first and isinstance(first[k], str):
                                    result_text = first[k]

                # other possible keys
                if not result_text:
                    for key in ("response", "output", "output_text", "text"):
                        if key in data and isinstance(data[key], str):
                            result_text = data[key]

                if not result_text:
                    dlist = data.get("data")
                    if isinstance(dlist, list) and dlist:
                        first = dlist[0]
                        if isinstance(first, dict):
                            for k in ("text", "content", "response"):
                                if k in first and isinstance(first[k], str):
                                    result_text = first[k]

            if result_text is not None:
                result_text = result_text.strip()
                logging.info("OpenRouter response extracted (len=%d)", len(result_text))
                logging.debug("OpenRouter response preview: %s", result_text[:1000])
                return result_text
        except Exception:
            logging.exception("OpenRouter request failed for endpoint %s", endpoint)
            # try the next endpoint if available
            continue

    logging.warning("All OpenRouter endpoints failed")
    return None


def generate_with_openrouter(trigger: str, context: Optional[Dict] = None, *, model: Optional[str] = None, timeout: float = 6.0) -> Optional[str]:
    context = context or {}
    template = get_template_response(trigger, context)
    prompt = _build_prompt(trigger, context, template)
    model = model or os.environ.get("OPENROUTER_MODEL", "gemma3-4b")
    candidate = _openrouter_generate(prompt, model=model, timeout=timeout)
    if not candidate:
        return None
    candidate = candidate.strip()
    if candidate and candidate.lower() != template.lower():
        return candidate
    return None


def get_template_response(trigger: str, context: Optional[Dict] = None) -> str:
    context = context or {}
    try:
        return dialogue.get_response(trigger, context)
    except Exception:
        logging.exception("dialogue.get_response() failed")
        return "Okay."


def generate_with_ollama(trigger: str, context: Optional[Dict] = None, *, model: Optional[str] = None, timeout: float = 6.0) -> Optional[str]:
    context = context or {}
    template = get_template_response(trigger, context)
    prompt = _build_prompt(trigger, context, template)
    model = model or os.environ.get("YUUKA_OLLAMA_MODEL", "llama3")
    candidate = _ollama_generate(prompt, model=model, timeout=timeout)
    if not candidate:
        return None
    candidate = candidate.strip()
    if candidate and candidate.lower() != template.lower():
        return candidate
    return None


def generate_conversation_reply(
    user_text: str,
    context: Optional[Dict] = None,
    *,
    timeout: float = 8.0,
) -> str:
    context = context or {}
    fallback = get_template_response("conversation", context)
    enabled_or = os.environ.get("YUUKA_ENABLE_OPENROUTER", "0").lower() in ("1", "true", "yes")
    enabled_ollama = os.environ.get("YUUKA_ENABLE_OLLAMA", "0").lower() in ("1", "true", "yes")

    if not (enabled_or or enabled_ollama):
        return fallback

    prompt = _build_conversation_prompt(user_text, context)

    # Prefer OpenRouter when enabled
    if enabled_or:
        model = os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "gemma3-4b"))
        candidate = _openrouter_generate(prompt, model=model, timeout=timeout)
        if candidate:
            return candidate.strip()

    # Fallback to Ollama if configured
    if enabled_ollama:
        model = os.environ.get("YUUKA_OLLAMA_MODEL", "llama3")
        candidate = _ollama_generate(prompt, model=model, timeout=timeout)
        if candidate:
            return candidate.strip()

    return fallback


def generate_response(trigger: str, context: Optional[Dict] = None, *, use_ollama: bool = True) -> str:
    context = context or {}
    template = get_template_response(trigger, context)
    enabled_or = os.environ.get("YUUKA_ENABLE_OPENROUTER", "0").lower() in ("1", "true", "yes")
    enabled_ollama = os.environ.get("YUUKA_ENABLE_OLLAMA", "0").lower() in ("1", "true", "yes")
    enabled = enabled_or or enabled_ollama

    always_or = os.environ.get("YUUKA_ALWAYS_USE_OPENROUTER", "0").lower() in ("1", "true", "yes")
    always_ollama = os.environ.get("YUUKA_ALWAYS_USE_OLLAMA", "0").lower() in ("1", "true", "yes")
    always = always_or or always_ollama

    if not use_ollama or not enabled:
        return template

    try:
        fallback_prob = float(os.environ.get("YUUKA_OLLAMA_FALLBACK_PROB", os.environ.get("YUUKA_LLM_FALLBACK_PROB", "0.18")))
    except Exception:
        fallback_prob = 0.18

    if not always and random.random() > fallback_prob:
        return template

    # Prefer OpenRouter when enabled
    if enabled_or:
        candidate = generate_with_openrouter(trigger, context, timeout=4.0)
        if candidate:
            return candidate

    # Fallback to Ollama if configured
    if enabled_ollama:
        candidate = generate_with_ollama(trigger, context, timeout=4.0)
        if candidate:
            return candidate

    return template
