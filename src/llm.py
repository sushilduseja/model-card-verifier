"""LLM client with graceful fallback and provider routing."""
import json
import os
import httpx
import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

litellm.suppress_debug_info = True


def _to_litellm_model(model: str) -> tuple[str, dict]:
    """Returns (litellm_model_string, extra_kwargs)"""
    if model.startswith("nvidia/"):
        model_name = model.replace("nvidia/", "", 1)
        return (
            f"nvidia_nim/{model_name}",
            {"api_base": os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")},
        )
    return model, {}


def _make_timeout(seconds: int) -> httpx.Timeout:
    """Hard socket-level timeout — prevents SSL read stalls."""
    return httpx.Timeout(timeout=float(seconds), connect=10.0)


def completion_with_fallback(primary_model: str, primary_key: str, messages: list, **kwargs):
    fallback_models = _fallback_models()
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    expect_json = bool(kwargs.pop("expect_json", False))
    kwargs["timeout"] = _make_timeout(timeout_seconds)
    
    transient_errors = (
        RateLimitError,
        Timeout,
        APIConnectionError,
        ServiceUnavailableError,
        InternalServerError,
    )
    fallback_trigger_errors = transient_errors + (AuthenticationError,)
    
    last_error = None

    # Try primary model
    from src import log
    try:
        primary_response = _call_llm(primary_model, primary_key, messages, **kwargs)
        if _response_is_acceptable(primary_response, expect_json):
            return primary_response
        content = ""
        try:
            content = primary_response.choices[0].message.content or ""
        except Exception:
            pass
        log.fallback(primary_model, fallback_models[0] if fallback_models else "none", f"unusable ({len(content)} chars)")
        last_error = RuntimeError(f"{primary_model} returned unusable content")
    except fallback_trigger_errors as e:
        log.fallback(primary_model, fallback_models[0] if fallback_models else "none", type(e).__name__)
        last_error = e

    # Try fallbacks in order
    if not fallback_models:
        raise last_error or RuntimeError("primary failed, no fallbacks configured")

    for fallback_model in fallback_models:
        fallback_key = _fallback_key_for_model(fallback_model)
        if not fallback_key:
            log.warn(f"skipping fallback {fallback_model} (no key)")
            continue

        log.info(f"trying fallback {fallback_model}...")
        try:
            fallback_response = _call_llm(fallback_model, fallback_key, messages, **kwargs)
            if _response_is_acceptable(fallback_response, expect_json):
                return fallback_response
            log.warn(f"{fallback_model} returned unusable content")
            last_error = RuntimeError(f"{fallback_model} returned unusable content")
            continue
        except fallback_trigger_errors as e:
            log.warn(f"{fallback_model} failed: {type(e).__name__}")
            last_error = e
            continue
        except Exception:
            raise

    raise last_error or RuntimeError("all models failed")


def _fallback_models() -> list[str]:
    models_csv = os.getenv("FALLBACK_MODELS", "").strip()
    return [m.strip() for m in models_csv.split(",") if m.strip()]


def _fallback_key_for_model(model: str) -> str:
    if model.startswith("nvidia/"):
        return os.getenv("NVIDIA_API_KEY", "")
    if model.startswith("groq/"):
        return os.getenv("GROQ_API_KEY", "")
    return os.getenv("NVIDIA_API_KEY") or os.getenv("GROQ_API_KEY", "")


def _response_has_content(response: object) -> bool:
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        return bool(content and str(content).strip())
    except Exception:
        return False


def _response_is_acceptable(response: object, expect_json: bool) -> bool:
    if not _response_has_content(response):
        return False
    if not expect_json:
        return True
    try:
        content = str(response.choices[0].message.content).strip()  # type: ignore[attr-defined]
        candidate = _extract_json_candidate(content)
        if not candidate:
            return False
        json.loads(candidate)
        return True
    except Exception:
        return False


def _extract_json_candidate(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if "```" in text:
            text = text.split("```", 1)[0]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    start = None
    for idx, ch in enumerate(text):
        if ch in "[{":
            start = idx
            break
    if start is None:
        return ""

    opening = text[start]
    closing = "]" if opening == "[" else "}"
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return ""


def _call_llm(model: str, api_key: str, messages: list, **kwargs) -> dict:
    import time as _time
    from src import log
    
    litellm_model, extra_kwargs = _to_litellm_model(model)
    kwargs["model"] = litellm_model
    kwargs["messages"] = messages
    
    if "nvidia" in litellm_model:
        os.environ["NVIDIA_API_KEY"] = api_key
        os.environ["NVIDIA_NIM_API_KEY"] = api_key
        kwargs.pop("api_key", None)
    else:
        kwargs["api_key"] = api_key
    
    for k, v in extra_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    
    t0 = _time.monotonic()
    resp = completion(**kwargs)
    latency_ms = int((_time.monotonic() - t0) * 1000)
    try:
        tokens = resp.usage.total_tokens if resp.usage else None
    except Exception:
        tokens = None
    log.llm_call(litellm_model, latency_ms, tokens)
    return resp
