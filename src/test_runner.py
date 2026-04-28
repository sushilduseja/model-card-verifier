"""Run test prompts against the model under audit."""
import time
import os
import litellm
from litellm import completion
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout
from src.models import TestCase, TestResult

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
    provider_prefixes = {
        "openai",
        "groq",
        "gemini",
        "nvidia",
        "zen",
        "opencode",
        "anthropic",
        "mistral",
        "vertex_ai",
        "openrouter",
        "deepinfra",
    }
    normalized_model = model_id
    if api_base:
        prefix = model_id.split("/", 1)[0]
        if prefix not in provider_prefixes:
            normalized_model = f"openai/{model_id}"

    timeout_seconds = int(os.getenv("TEST_TIMEOUT_SECONDS", "30"))
    kwargs = {
        "model": normalized_model,
        "messages": [{"role": "user", "content": test.prompt}],
        "api_key": api_key,
        "max_tokens": 800,
    }
    if api_base:
        kwargs["api_base"] = api_base

    time.sleep(0.5)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = completion(**kwargs, timeout=timeout_seconds)
            actual = response.choices[0].message.content.strip()
            break
        except (RateLimitError, Timeout, APIConnectionError) as e:
            last_error = e
            if attempt < retries:
                print(f"  [retry] {model_id} failed ({e}), attempt {attempt + 2}/{retries + 1}")
                time.sleep(2)
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
