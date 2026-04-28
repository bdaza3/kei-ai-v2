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
from typing import Any, Dict, Optional

import dialogue
from memory import get_memory

import requests
import json
import re


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text_value = content.get("text") or content.get("content")
        if isinstance(text_value, str):
            return text_value
    return ""


def _strip_json_fences(text: str) -> str:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    return candidate.strip()


def _normalize_openrouter_model(model_name: Optional[str]) -> str:
    raw = (model_name or "").strip()
    if not raw:
        return "google/gemma-3-4b-it:free"

    aliases = {
        "gemma3-4b": "google/gemma-3-4b-it:free",
        "gemma-3-4b": "google/gemma-3-4b-it:free",
        "gemma3": "google/gemma-3-4b-it:free",
        "gemma3-12b": "google/gemma-3-12b-it:free",
        "gemma-3-12b": "google/gemma-3-12b-it:free",
    }
    lowered = raw.lower()
    return aliases.get(lowered, raw)


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
        "You are Kei Tendou, acting as the user's assistant. "
        "Stay in character. Reply briefly, clearly, and with calm but firm accountability. "
        "Do not mention prompts, instructions, roleplay, or being an AI model. "
    )


def _memory_block(user_text: str) -> str:
    memory = get_memory()
    bundle = memory.build_prompt_memory(user_text)
    try:
        return json.dumps(bundle, ensure_ascii=False)
    except Exception:
        return str(bundle)


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
        f"Character memory (identity, relationship, style examples, short-term, long-term): {_memory_block(user_text)}\n\n"
        f"Desktop context: {context_json}\n"
        f"User said: {user_text}\n\n"
        "Reply as Kei in 1-3 sentences. Use the style examples as tone references. "
        "Keep a tsundere voice: sharp honesty with underlying care. "
        "If the user seems distracted, steer them back to work. "
        "Use plain, simple words. Avoid formal words like 'inquire', 'dawdle', 'presume', 'regarding'. "
        "The user in this chat is Sensei; never treat Sensei as another person. "
        "If asked who you are or what you are doing, answer only from character memory and current context. "
        "Do not invent project names, tasks, or events. If unknown, say you do not have that detail yet."
    )


def _build_bilingual_conversation_prompt(user_text: str, context: Dict) -> str:
    try:
        context_json = json.dumps(context, ensure_ascii=False)
    except Exception:
        context_json = str(context)
    return (
        f"{_persona_block()}\n\n"
        f"Character memory (identity, relationship, style examples, short-term, long-term): {_memory_block(user_text)}\n\n"
        f"Desktop context: {context_json}\n"
        f"User said in English: {user_text}\n\n"
        "You must answer in valid JSON only. No markdown. No extra text.\n"
        "Return exactly these fields:\n"
        "{\n"
        '  "japanese": "natural conversational Japanese for speech",\n'
        '  "english": "natural English for display"\n'
        "}\n\n"
        "Rules:\n"
        "- Japanese must sound natural, warm, and like a real person talking.\n"
        "- English must be simple, clear, and feel like a VN line.\n"
        "- Keep both fields concise, but not flat or robotic.\n"
        "- Use a little tsundere attitude when it fits.\n"
        "- Kei can sound a little annoyed, but she should still sound caring.\n"
        "- Do not use stiff or formal wording.\n"
        "- Use 2 short sentences when possible. One tiny sentence is too short.\n"
        "- Give a full answer: identity, current action, or advice, then a small tsundere follow-up.\n"
        "- Aim for roughly 35 to 70 Japanese characters in the Japanese field when natural.\n"
        "- The English field should match the Japanese meaning closely and be slightly longer if needed.\n"
        "- Use simple words only. Avoid formal words like inquire, dawdle, presume, regarding.\n"
        "- The user is Sensei. Do not confuse Sensei with anyone else.\n"
        "- Do not invent tasks, projects, or events. If unknown, say so plainly.\n"
        "- If the user asks who you are or what you are doing, answer clearly and directly, then add one small personality line.\n"
        "- For identity or current-action questions, give a slightly fuller answer: what you are, what you are doing, and one tsundere remark.\n"
        "Example:\n"
        '{"japanese":"今日は何をしますか？","english":"What will you do today?"}'
    )


def _simplify_wording(text: str) -> str:
    out = text or ""
    substitutions = {
        r"\binquire\b": "ask",
        r"\binquired\b": "asked",
        r"\bdawdle\b": "waste time",
        r"\bpresume\b": "assume",
        r"\bregarding\b": "about",
        r"\bclarify\b": "explain",
        r"\befficiently\b": "well",
        r"\bimmediately\b": "now",
    }
    for pattern, replacement in substitutions.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidate = _strip_json_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(candidate[start:])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _normalize_bilingual_payload(payload: Dict[str, Any], raw_text: str = "") -> Dict[str, str]:
    japanese = str(payload.get("japanese") or payload.get("ja") or payload.get("japanese_text") or "").strip()
    english = str(payload.get("english") or payload.get("en") or payload.get("english_text") or "").strip()
    if not english and raw_text:
        english = raw_text.strip()
    if not japanese:
        japanese = str(payload.get("spoken") or payload.get("voice") or "").strip()
    japanese = _simplify_wording(japanese)
    english = _simplify_wording(english)
    return {
        "japanese": japanese,
        "english": english,
    }


def _openrouter_generate_json(prompt: str, model: Optional[str] = None, timeout: float = 6.0) -> Optional[Dict[str, str]]:
    raw = _openrouter_generate(prompt, model=model, timeout=timeout)
    if not raw:
        return None

    parsed = _extract_json_object(raw)
    if parsed:
        return _normalize_bilingual_payload(parsed, raw_text=raw)

    return {
        "japanese": "",
        "english": _simplify_wording(raw.strip()),
    }


def _openrouter_generate(prompt: str, model: Optional[str] = None, timeout: float = 6.0) -> Optional[str]:
    """Send a chat completion request to OpenRouter and return the assistant text.

    Reads API key from `GEMMA3_4B_API_KEY` or `OPENROUTER_API_KEY`.
    Endpoint can be overridden via `OPENROUTER_ENDPOINT` (defaults to api.openrouter.ai path).
    """
    api_key = (
        os.environ.get("GEMMA3_4B_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
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
    model_name = _normalize_openrouter_model(
        model or os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "google/gemma-3-4b-it:free"))
    )

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
                        if isinstance(msg, dict) and "content" in msg:
                            result_text = _extract_message_text(msg.get("content"))
                        else:
                            # fallback to text/content keys
                            for k in ("text", "content"):
                                if k in first:
                                    result_text = _extract_message_text(first[k])
                                    if result_text:
                                        break

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
    model = _normalize_openrouter_model(model or os.environ.get("OPENROUTER_MODEL", "google/gemma-3-4b-it:free"))
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
    memory = get_memory()
    fallback = get_template_response("conversation", context)
    enabled_or = os.environ.get("YUUKA_ENABLE_OPENROUTER", "0").lower() in ("1", "true", "yes")
    enabled_ollama = os.environ.get("YUUKA_ENABLE_OLLAMA", "0").lower() in ("1", "true", "yes")

    if not (enabled_or or enabled_ollama):
        memory.remember_turn(user_text, fallback, context)
        return fallback

    prompt = _build_conversation_prompt(user_text, context)

    # Prefer OpenRouter when enabled
    if enabled_or:
        model = _normalize_openrouter_model(
            os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "google/gemma-3-4b-it:free"))
        )
        candidate = _openrouter_generate(prompt, model=model, timeout=timeout)
        if candidate:
            out = _simplify_wording(candidate.strip())
            memory.remember_turn(user_text, out, context)
            return out

    # Fallback to Ollama if configured
    if enabled_ollama:
        model = os.environ.get("YUUKA_OLLAMA_MODEL", "llama3")
        candidate = _ollama_generate(prompt, model=model, timeout=timeout)
        if candidate:
            out = _simplify_wording(candidate.strip())
            memory.remember_turn(user_text, out, context)
            return out

    fallback_simple = _simplify_wording(fallback)
    memory.remember_turn(user_text, fallback_simple, context)
    return fallback_simple


def generate_openrouter_conversation_reply(
    user_text: str,
    context: Optional[Dict] = None,
    *,
    model: Optional[str] = None,
    timeout: float = 8.0,
) -> Optional[str]:
    """Generate a conversation reply using OpenRouter only.

    Returns None when OpenRouter is not configured or the request fails.
    """
    context = context or {}
    memory = get_memory()
    prompt = _build_conversation_prompt(user_text, context)
    model_name = _normalize_openrouter_model(
        model or os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "google/gemma-3-4b-it:free"))
    )
    candidate = _openrouter_generate(prompt, model=model_name, timeout=timeout)
    if candidate:
        out = _simplify_wording(candidate.strip())
        memory.remember_turn(user_text, out, context)
        return out
    return None


def generate_openrouter_bilingual_reply(
    user_text: str,
    context: Optional[Dict] = None,
    *,
    model: Optional[str] = None,
    timeout: float = 8.0,
) -> Optional[Dict[str, str]]:
    """Generate one OpenRouter response with Japanese speech text and English display text."""
    context = context or {}
    memory = get_memory()
    prompt = _build_bilingual_conversation_prompt(user_text, context)
    model_name = _normalize_openrouter_model(
        model or os.environ.get("OPENROUTER_MODEL", os.environ.get("YUUKA_OPENROUTER_MODEL", "google/gemma-3-4b-it:free"))
    )

    # Prefer strict JSON mode if the endpoint/model supports it.
    api_key = os.environ.get("GEMMA3_4B_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logging.debug("OpenRouter API key not configured")
        return None

    configured = os.environ.get("OPENROUTER_ENDPOINT")
    endpoints = [configured] if configured else [
        "https://api.openrouter.ai/v1/chat/completions",
        "https://openrouter.ai/api/v1/chat/completions",
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return only valid JSON with keys japanese and english."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", os.environ.get("YUUKA_OPENROUTER_TEMPERATURE", 0.3))),
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", 256)),
        "response_format": {"type": "json_object"},
    }

    last_raw: Optional[str] = None
    for endpoint in endpoints:
        if not endpoint:
            continue
        logging.info("Attempting OpenRouter bilingual endpoint: %s", endpoint)
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                logging.warning("OpenRouter rate limited on %s; trying next endpoint if available", endpoint)
                continue
            resp.raise_for_status()
            data = resp.json()

            raw_text: Optional[str] = None
            if isinstance(data, dict):
                choices = data.get("choices")
                if choices and isinstance(choices, list):
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("delta")
                        if isinstance(msg, dict):
                            raw_text = _extract_message_text(msg.get("content"))

                if not raw_text:
                    for key in ("response", "output", "output_text", "text"):
                        if key in data:
                            raw_text = _extract_message_text(data[key])
                            if raw_text:
                                break

            if raw_text:
                last_raw = raw_text
                parsed = _extract_json_object(raw_text)
                if parsed:
                    normalized = _normalize_bilingual_payload(parsed, raw_text=raw_text)
                    if normalized["english"] or normalized["japanese"]:
                        # If the model answered too briefly, keep a little more of the same answer in the English line.
                        if len(normalized["english"]) < 18 and len(normalized["japanese"]) < 12:
                            normalized["english"] = normalized["english"] or "I am here, Sensei."
                            normalized["japanese"] = normalized["japanese"] or "はい、Sensei。"
                        memory.remember_turn(user_text, normalized.get("english") or normalized.get("japanese") or "", context)
                        return normalized
                else:
                    # Some models ignore JSON mode. Keep the text as English fallback.
                    fallback = {"japanese": "", "english": _simplify_wording(raw_text.strip())}
                    memory.remember_turn(user_text, fallback["english"], context)
                    return fallback
        except Exception:
            logging.exception("OpenRouter bilingual request failed for endpoint %s", endpoint)
            continue

    if last_raw:
        fallback = {"japanese": "", "english": _simplify_wording(last_raw.strip())}
        memory.remember_turn(user_text, fallback["english"], context)
        return fallback

    return None


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
