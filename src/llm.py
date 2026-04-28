"""LLM client with graceful fallback when primary provider hits rate limits."""
import json
import os
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


OPENCODE_ZEN_MODELS = set(
    m.strip()
    for m in os.getenv("OPENCODE_ZEN_MODELS", "").split(",")
    if m.strip()
)


def _to_litellm_model(model: str) -> tuple[str, bool, dict]:
    """Returns (litellm_model_string, is_zen, extra_kwargs)"""
    parts = model.split("/", 1)
    if len(parts) == 2:
        provider, model_name = parts
        if provider in {"zen", "opencode"}:
            return "openai/" + model_name, True, {}
        if provider == "nvidia":
            return (
                f"nvidia/{model_name}",
                False,
                {"api_base": os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")},
            )
    # bare model name — check explicit Zen list
    if model in OPENCODE_ZEN_MODELS:
        return "openai/" + model, True, {}
    return model, False, {}


def completion_with_fallback(primary_model: str, primary_key: str, messages: list, **kwargs):
    fallback_models = _fallback_models()
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    expect_json = bool(kwargs.pop("expect_json", False))
    kwargs.setdefault("timeout", timeout_seconds)
    transient_errors = (
        RateLimitError,
        Timeout,
        APIConnectionError,
        ServiceUnavailableError,
        InternalServerError,
    )
    fallback_trigger_errors = transient_errors + (AuthenticationError,)
    
    if not fallback_models:
        # No fallback configured, just retry
        max_retries = 3
        retry_delay = int(os.getenv("FALLBACK_RETRY_DELAY", "5"))
        
        for attempt in range(max_retries):
            try:
                return _call_llm(primary_model, primary_key, messages, **kwargs)
            except transient_errors:
                if attempt < max_retries - 1:
                    print(f"  [retry] {primary_model} transient failure, retrying in {retry_delay}s... (attempt {attempt + 2}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                else:
                    print(f"  [error] {primary_model} failed after {max_retries} attempts")
                    raise
        return None
    
    last_error = None
    try:
        primary_response = _call_llm(primary_model, primary_key, messages, **kwargs)
        if _response_is_acceptable(primary_response, expect_json):
            return primary_response
        last_error = RuntimeError(f"{primary_model} returned empty content")
        print(f"  [fallback] {primary_model} returned unusable content, trying fallbacks...")
    except fallback_trigger_errors as primary_err:
        last_error = primary_err
        for fallback_model in fallback_models:
            fallback_key = _fallback_key_for_model(fallback_model, primary_key)
            if not fallback_key or "your_" in fallback_key:
                print(f"  [warn] skipping fallback {fallback_model} (no valid key)")
                continue

            print(f"  [fallback] {primary_model} failed, falling back to {fallback_model}...")
            try:
                fallback_response = _call_llm(fallback_model, fallback_key, messages, **kwargs)
                if _response_is_acceptable(fallback_response, expect_json):
                    return fallback_response
                print(f"  [fallback] {fallback_model} returned unusable content")
                last_error = RuntimeError(f"{fallback_model} returned unusable content")
                continue
            except fallback_trigger_errors as fallback_err:
                print(f"  [fallback] {fallback_model} also failed: {fallback_err}")
                last_error = fallback_err
                continue
            except Exception:
                raise
    else:
        for fallback_model in fallback_models:
            fallback_key = _fallback_key_for_model(fallback_model, primary_key)
            if not fallback_key or "your_" in fallback_key:
                print(f"  [warn] skipping fallback {fallback_model} (no valid key)")
                continue
            print(f"  [fallback] trying {fallback_model}...")
            try:
                fallback_response = _call_llm(fallback_model, fallback_key, messages, **kwargs)
                if _response_is_acceptable(fallback_response, expect_json):
                    return fallback_response
                print(f"  [fallback] {fallback_model} returned unusable content")
                last_error = RuntimeError(f"{fallback_model} returned unusable content")
            except fallback_trigger_errors as fallback_err:
                print(f"  [fallback] {fallback_model} also failed: {fallback_err}")
                last_error = fallback_err
                continue
            except Exception:
                raise
    raise last_error or RuntimeError("all fallback models failed")


def _fallback_models() -> list[str]:
    models_csv = os.getenv("FALLBACK_MODELS", "").strip()
    if models_csv:
        return [m.strip() for m in models_csv.split(",") if m.strip()]

    single = os.getenv("FALLBACK_MODEL", "").strip()
    return [single] if single else []


def _fallback_key_for_model(model: str, primary_key: str) -> str:
    if model.startswith("nvidia/"):
        return os.getenv("NVIDIA_API_KEY", "")
    if model.startswith("gemini/"):
        return (
            os.getenv("FALLBACK_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
    return os.getenv("FALLBACK_API_KEY") or primary_key


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
    litellm_model, is_zen, extra_kwargs = _to_litellm_model(model)
    kwargs.setdefault("timeout", int(os.getenv("LLM_TIMEOUT_SECONDS", "45")))
    
    kwargs["model"] = litellm_model
    kwargs["messages"] = messages
    
    if litellm_model.startswith("nvidia/"):
        os.environ["NVIDIA_API_KEY"] = api_key
        os.environ["NVIDIA_NIM_API_KEY"] = api_key
        kwargs.pop("api_key", None)
    else:
        kwargs["api_key"] = api_key
    
    for k, v in extra_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v
    
    if is_zen:
        kwargs["api_base"] = os.getenv("ZEN_API_BASE") or os.getenv("OPENCODE_ZEN_API_BASE")

    return completion(**kwargs)
