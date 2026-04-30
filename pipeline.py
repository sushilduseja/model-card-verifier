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
from src import log


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
    log.info(f"Extracted {len(claims)} claims:")
    for c in claims:
        log.info(f"  [{c.claim_id}] ({c.claim_type}) {c.claim_text[:60]}...")
    if not interactive:
        return True
    log.info("Press Enter to continue, or Ctrl+C to abort")
    try:
        input()
        return True
    except KeyboardInterrupt:
        log.warn("Aborted.")
        return False


def run_audit(config: PipelineConfig):
    log.step_start(f"Step 1/5: Extracting claims  model={config.model_under_audit}")
    claims = extract_claims(
        config.model_under_audit,
        config.model_extractor,
        config.extractor_api_key,
    )
    log.step_done("Step 1/5", f"claims={len(claims)}")

    if not confirm_claims(claims, config.confirm_claims):
        return None

    log.step_start("Step 2/5: Generating tests")
    tests = generate_all_tests(claims, config.model_tester, config.tester_api_key)
    log.step_done("Step 2/5", f"tests={len(tests)}")

    log.step_start("Step 3/5: Running tests")
    results = run_all_tests(
        tests,
        config.model_under_audit,
        config.audit_api_key,
        config.audit_api_base,
        config.model_tester,
        config.model_judge,
    )
    log.step_done("Step 3/5", f"results={len(results)}")

    log.step_start("Step 4/5: Scoring")
    scored = score_all(results, config.model_judge, config.judge_api_key)
    claim_scores = compute_claim_scores(scored)
    log.step_done("Step 4/5", f"overall={claim_scores['overall']}")

    log.step_start("Step 5/5: Saving")
    review_path = save_for_review(
        config.model_under_audit,
        scored,
        claim_scores,
        config,
    )
    log.step_done("Step 5/5", review_path)

    return review_path


if __name__ == "__main__":
    config = PipelineConfig()
    log.info(config.display())
    run_audit(config)