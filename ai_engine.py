"""Simplified AI response engine for Kei.

This file keeps a small, readable surface:
- llm_chat(): the one low-level request helper
- build_system_prompt(): the core system instructions
- generate_plain_text_reply(): the plain text LLM path
- compatibility wrappers for the rest of the project
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import dialogue
import requests
from memory import get_memory
from tools import TOOLS, get_active_window

DEFAULT_MODEL = "google/gemma-3-4b-it:free"
OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"


def _get_openrouter_url() -> str:
    configured = (os.environ.get("OPENROUTER_ENDPOINT") or "").strip()
    return configured or OPENROUTER_URL


def _normalize_openrouter_model(model_name: Optional[str]) -> str:
    raw = (model_name or "").strip()
    if not raw:
        return DEFAULT_MODEL

    aliases = {
        "gemma3-4b": DEFAULT_MODEL,
        "gemma-3-4b": DEFAULT_MODEL,
        "gemma3": DEFAULT_MODEL,
        "gemma3-12b": "google/gemma-3-12b-it:free",
        "gemma-3-12b": "google/gemma-3-12b-it:free",
    }
    lowered = raw.lower()
    return aliases.get(lowered, raw)


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
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


def build_system_prompt() -> str:
    return (
        f"{_persona_block()}\n\n"
        "Reply as Kei in 1-3 sentences. Use the style examples as tone references. "
        "Always answer the question directly. Do not echo or mirror the user's wording. "
        "Keep a tsundere voice: sharp honesty with underlying care. "
        "If the user seems distracted, steer them back to work. "
        "Use plain, simple words. Avoid formal words like 'inquire', 'dawdle', 'presume', 'regarding'. "
        "The user in this chat is Sensei; never treat or refer to Sensei as another person such as him, her, or they. Use 'Sensei' or 'you' only."
        "If asked who you are or what you are doing, answer only from character memory and current context. "
        "Do not invent project names, tasks, or events. If unknown, say you do not have that detail yet."
    )


def build_user_prompt(user_text: str, context: Optional[Dict[str, Any]] = None) -> str:
    context = context or {}
    try:
        context_json = json.dumps(context, ensure_ascii=False)
    except Exception:
        context_json = str(context)
    return (
        f"Character memory: {_memory_block(user_text)}\n\n"
        f"Desktop context: {context_json}\n\n"
        f"User said: {user_text}\n\n"
        "Rules:\n"
        "- Never repeat the user's question. Never start with a paraphrase of the prompt.\n"
        "- Answer the question directly. Start with the actual answer first.\n"
        "- English must be simple, clear, and feel like a VN line.\n"
        "- Keep both fields concise, but not flat or robotic.\n"
        "- Use a little tsundere attitude when it fits.\n"
        "- Kei can sound a little annoyed, but she should still sound caring.\n"
        "- Do not use stiff or formal wording.\n"
        "- Use 2 short sentences when possible. One tiny sentence is too short.\n"
        "- Give a full answer: identity, current action, or advice, then a small tsundere follow-up.\n"
        "- Use simple words only. Avoid formal words like inquire, dawdle, presume, regarding.\n"
        "- The user is Sensei. Do not confuse Sensei with anyone else.\n"
        "- Do not invent tasks, projects, or events. If unknown, say so plainly.\n"
        "- If the user asks who you are or what you are doing, answer clearly and directly, then add one small personality line.\n"
        "- For identity or current-action questions, give a slightly fuller answer: what you are, what you are doing, and one tsundere remark.\n"
        "Reply in 1-3 short sentences. Use simple words. "
        "If it fits, keep the tone a little tsundere but warm."
    )


def llm_chat(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    timeout: float = 20.0,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Low-level OpenRouter chat wrapper used by all higher-level response functions."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMMA3_4B_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    resolved_model = _normalize_openrouter_model(model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": min(temperature, 0.2),
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = _get_openrouter_url()
    logging.info("OpenRouter request to %s using model %s", endpoint, payload["model"])
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 429:
        detail = response.text[:300]
        raise RuntimeError(f"OpenRouter rate limited for model {payload['model']}: {detail}")
    response.raise_for_status()
    return response.json()


def _parse_text_reply(data: Dict[str, Any]) -> Optional[str]:
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or choice.get("delta") or {}
        text = _extract_message_text(msg.get("content", ""))
        if text:
            return _simplify_wording(text.strip())
    except Exception:
        pass
    return None


def dispatch_tool_call(tool_call: Dict[str, Any]) -> Any:
    """Execute a single tool call and return the JSON-serializable result."""
    func_block = tool_call.get("function") or {}
    tool_name = str(func_block.get("name") or "").strip()
    raw_args = func_block.get("arguments") or "{}"

    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except Exception:
            arguments = {}
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        arguments = {}

    logging.info("Tool dispatch: name=%s args=%s", tool_name, arguments)

    if tool_name == "get_active_window":
        result = get_active_window()
        logging.info("Tool result: %s", result)
        return result

    raise ValueError(f"Unsupported tool: {tool_name}")


def _extract_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            return tool_calls
    except Exception:
        pass
    return []


def generate_tool_call_reply(
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """Generic agent loop: the model may call one or more tools, and we continue until it replies naturally."""
    context = context or {}
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(user_text, context)},
    ]

    try:
        for _ in range(5):
            data = llm_chat(messages, tools=TOOLS, model=model, timeout=timeout)
            tool_calls = _extract_tool_calls(data)
            if not tool_calls:
                reply = _parse_text_reply(data)
                if reply:
                    return reply
                return get_template_response("conversation", context)

            assistant_tool_calls: List[Dict[str, Any]] = []
            tool_messages: List[Dict[str, Any]] = []

            for tool_call in tool_calls:
                tool_name = (tool_call.get("function") or {}).get("name") or "unknown"
                logging.info("Model requested tool call: %s", tool_name)
                tool_result = dispatch_tool_call(tool_call)
                logging.info("Tool call executed: %s -> %s", tool_name, tool_result)
                assistant_tool_calls.append(tool_call)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "tool_call_1"),
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            messages.append({
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            })
            messages.extend(tool_messages)

        return get_template_response("conversation", context)
    except RuntimeError as exc:
        logging.warning("Tool-calling reply failed due to rate limit: %s", exc)
        return get_template_response("conversation", context)
    except Exception as exc:
        logging.warning("Tool-calling reply failed: %s", exc)
        return generate_plain_text_reply(user_text, context, model=model, timeout=timeout)


def generate_plain_text_reply(
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> str:
    """Plain text LLM request/response path. This is the primary path for typed chat."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(user_text, context)},
    ]

    try:
        data = llm_chat(messages, model=model, timeout=timeout)
        reply = _parse_text_reply(data)
        if reply:
            return reply
    except RuntimeError as exc:
        logging.warning("OpenRouter rate limit reached: %s", exc)
        return get_template_response("conversation", context or {})
    except Exception as exc:
        logging.warning("OpenRouter plain-text generation failed: %s", exc)

    return get_template_response("conversation", context or {})


def get_template_response(trigger: str, context: Optional[Dict[str, Any]] = None) -> str:
    context = context or {}
    try:
        return dialogue.get_response(trigger, context)
    except Exception:
        logging.exception("dialogue.get_response() failed")
        return "Okay."


def generate_with_openrouter(trigger: str, context: Optional[Dict[str, Any]] = None, *, model: Optional[str] = None, timeout: float = 20.0) -> Optional[str]:
    prompt = build_user_prompt(str(trigger), context)
    messages = [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": prompt}]
    try:
        data = llm_chat(messages, model=model, timeout=timeout)
        return _parse_text_reply(data)
    except Exception:
        logging.exception("generate_with_openrouter failed")
        return None


def generate_with_ollama(prompt: str, model: str = "llama3", timeout: float = 3.0) -> Optional[str]:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.3")),
        }
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"].strip()
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


def generate_conversation_reply(
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 20.0,
) -> str:
    """Compatibility wrapper used by the rest of the project."""
    context = context or {}
    memory = get_memory()
    fallback = get_template_response("conversation", context)
    try:
        reply = generate_tool_call_reply(user_text, context, timeout=timeout)
        memory.remember_turn(user_text, reply, context)
        return reply
    except Exception:
        logging.exception("generate_conversation_reply failed")
        memory.remember_turn(user_text, fallback, context)
        return fallback


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


def generate_openrouter_bilingual_reply(
    user_text: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[Dict[str, str]]:
    """Compatibility JSON reply for older UI code. Returns japanese + english strings."""
    context = context or {}
    prompt = (
        "Return only valid JSON with keys 'japanese' and 'english'.\n"
        "Rules:\n"
        "- Never repeat the user's question or echo the prompt.\n"
        "- Answer the question directly. Start with the fact or answer first.\n"
        "- Do not output the user's exact wording back to them.\n"
        "- Keep the English output brief and natural.\n"
        f"User said: {user_text}\n"
        "Respond with the answer, not a reworded question."
    )
    messages = [
        {"role": "system", "content": "Return only valid JSON with keys japanese and english."},
        {"role": "user", "content": prompt},
    ]

    try:
        data = llm_chat(messages, model=model, timeout=timeout)
        raw_text = _parse_text_reply(data)
        if not raw_text:
            return None

        parsed = _extract_json_object(raw_text)
        if parsed:
            japanese = str(parsed.get("japanese") or parsed.get("ja") or "").strip()
            english = str(parsed.get("english") or parsed.get("en") or "").strip()
            if not english:
                english = raw_text
            return {"japanese": japanese, "english": english}

        return {"japanese": "", "english": raw_text}
    except RuntimeError as exc:
        logging.warning("OpenRouter rate limited in bilingual reply: %s", exc)
        local_reply = get_template_response("conversation", context or {})
        return {"japanese": "", "english": local_reply}
    except Exception:
        logging.exception("generate_openrouter_bilingual_reply failed")
        return None


__all__ = [
    "DEFAULT_MODEL",
    "build_system_prompt",
    "build_user_prompt",
    "generate_plain_text_reply",
    "generate_conversation_reply",
    "generate_with_openrouter",
    "generate_with_ollama",
    "generate_openrouter_bilingual_reply",
    "get_template_response",
    "llm_chat",
]
