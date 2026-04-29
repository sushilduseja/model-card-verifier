"""Run test prompts against the model under audit."""
import time
import os
import httpx
import litellm
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout
from src.models import TestCase, TestResult
from src.llm import _call_llm, _make_timeout

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
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    time.sleep(0.5)

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
            actual = response.choices[0].message.content.strip()
            break
        except (RateLimitError, Timeout, APIConnectionError) as e:
            last_error = e
            if attempt < retries:
                print(f"  [retry] {model_id} attempt {attempt + 2}/{retries + 1}: {type(e).__name__}")
                time.sleep(3)
            else:
                actual = f"[ERROR: {str(e)}]"
        except Exception as e:
            actual = f"[ERROR: {str(e)}]"
            break

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
    return [run_test(t, model_id, api_key, api_base, tester_model, judge_model) for t in tests]
