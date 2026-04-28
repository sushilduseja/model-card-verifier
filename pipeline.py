"""CLI entry point for running the audit pipeline."""
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.models import Claim
from src.claim_extractor import extract_claims
from src.test_generator import generate_all_tests
from src.test_runner import run_all_tests
from src.scorer import score_all, compute_claim_scores


def save_for_review(
    model_id: str,
    scored: list,
    claim_scores: dict,
    config: PipelineConfig,
) -> str:
    Path("reviews").mkdir(exist_ok=True)
    safe = model_id.replace("/", "__")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"reviews/{safe}_{ts}.json"
    payload = {
        "model_id": model_id,
        "model_under_audit": config.model_under_audit,
        "tester_model": config.model_tester,
        "judge_model": config.model_judge,
        "claim_scores": claim_scores,
        "results": [r.model_dump() for r in scored],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def confirm_claims(claims: list[Claim], interactive: bool) -> bool:
    print("\nExtracted claims:")
    for c in claims:
        print(f"  [{c.claim_id}] ({c.claim_type}) {c.claim_text[:80]}")
    if not interactive:
        return True
    print("\nPress Enter to continue, or Ctrl+C to abort: ", end="", flush=True)
    try:
        input()
        return True
    except KeyboardInterrupt:
        print("\nAborted.")
        return False


def run_audit(config: PipelineConfig):
    print(f"\n=== Auditing: {config.model_under_audit} ===\n")

    print("Step 1/5: Extracting claims from model card...")
    claims = extract_claims(
        config.model_under_audit,
        config.model_extractor,
        config.extractor_api_key,
    )
    print(f"  Found {len(claims)} claims")

    if not confirm_claims(claims, config.confirm_claims):
        return None

    print("Step 2/5: Generating test cases...")
    tests = generate_all_tests(claims, config.model_tester, config.tester_api_key)
    print(f"  Generated {len(tests)} tests")

    print("Step 3/5: Running tests against model...")
    results = run_all_tests(
        tests,
        config.model_under_audit,
        config.audit_api_key,
        config.audit_api_base,
        config.model_tester,
        config.model_judge,
    )
    print(f"  Completed {len(results)} inferences")

    print("Step 4/5: Scoring results...")
    scored = score_all(results, config.model_judge, config.judge_api_key)
    claim_scores = compute_claim_scores(scored)
    print(f"  Overall compliance score: {claim_scores['overall']}")

    print("Step 5/5: Saving for review...")
    review_path = save_for_review(
        config.model_under_audit,
        scored,
        claim_scores,
        config,
    )
    print(f"  Saved to {review_path}")
    print(f"\n  Review the file, then run: uv run python publish.py {review_path}")

    return review_path


if __name__ == "__main__":
    config = PipelineConfig()
    print(config.display())
    run_audit(config)