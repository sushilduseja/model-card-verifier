"""Run test prompts against the model under audit."""
import time
import os
import httpx
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout
from src.models import TestCase, TestResult
from src.llm import _call_llm, _make_timeout
from src import log

litellm.suppress_debug_info = True


def run_test(
    test: TestCase,
    model_id: str,
    api_key: str,
    api_base: str | None = None,
    tester_model: str = "",
    judge_model: str = "",
    retries: int = 2,
) -> TestResult:
    from src import log
    
    t0 = time.monotonic()
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

    last_error = None
    actual = ""
    for attempt in range(retries + 1):
        try:
            # Route through _call_llm so nvidia/ → nvidia_nim/ translation applies
            response = _call_llm(
                model_id,
                api_key,
                [{"role": "user", "content": test.prompt}],
                max_tokens=800,
                timeout=_make_timeout(timeout_seconds),
            )
            actual = (response.choices[0].message.content or "").strip()
            break
        except (RateLimitError, Timeout, APIConnectionError) as e:
            last_error = e
            if attempt < retries:
                log.warn(f"{model_id} attempt {attempt + 2}/{retries + 1}: {type(e).__name__}")
                time.sleep(1)
            else:
                actual = f"[ERROR: {str(e)}]"
        except Exception as e:
            actual = f"[ERROR: {str(e)}]"
            break

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    status = "ok" if not actual.startswith("[ERROR") else "err"
    log.info(f"run  {test.test_id}  {status}  {elapsed_ms}ms")

    return TestResult(
        test_id=test.test_id,
        claim_id=test.claim_id,
        test_type=test.test_type,
        prompt=test.prompt,
        expected_behavior=test.expected_behavior,
        actual_response=actual,
        model_under_audit=model_id,
        tester_model=tester_model,
        judge_model=judge_model,
    )


def run_all_tests(
    tests: list[TestCase],
    model_id: str,
    api_key: str,
    api_base: str | None = None,
    tester_model: str = "",
    judge_model: str = "",
) -> list[TestResult]:
    max_workers = int(os.getenv("RUN_WORKERS", "5"))
    results = [None] * len(tests)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_test, t, model_id, api_key, api_base, tester_model, judge_model): i
            for i, t in enumerate(tests)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results
