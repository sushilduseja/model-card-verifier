"""Structured, timed logging for the audit pipeline."""
import time
import os

VERBOSE = os.getenv("AUDIT_VERBOSE", "true").lower() != "false"
_pipeline_start = time.monotonic()
_step_start = time.monotonic()


def _elapsed() -> str:
    total = time.monotonic() - _pipeline_start
    return f"+{total:6.1f}s"


def step_start(label: str) -> None:
    global _step_start
    _step_start = time.monotonic()
    print(f"\n[{_elapsed()}] >>> {label}")


def step_done(label: str, detail: str = "") -> None:
    elapsed = time.monotonic() - _step_start
    suffix = f"  ({detail})" if detail else ""
    print(f"[{_elapsed()}] <<< {label} done in {elapsed:.1f}s{suffix}")


def info(msg: str) -> None:
    print(f"[{_elapsed()}]     {msg}")


def warn(msg: str) -> None:
    print(f"[{_elapsed()}] WARN {msg}")


def fallback(primary: str, fallback_model: str, reason: str) -> None:
    print(f"[{_elapsed()}] FALL {primary} -> {fallback_model}  ({reason})")


def skip(item: str, reason: str) -> None:
    print(f"[{_elapsed()}] SKIP {item}  reason={reason}")


def llm_call(model: str, latency_ms: int, tokens: int | None = None, cached: bool = False) -> None:
    if not VERBOSE:
        return
    tok = f"  tokens={tokens}" if tokens else ""
    cac = "  [cached]" if cached else ""
    print(f"[{_elapsed()}]  LLM {model}  {latency_ms}ms{tok}{cac}")


def fail_safe(test_id: str, reason: str) -> None:
    print(f"[{_elapsed()}] SAFE {test_id}  {reason}")