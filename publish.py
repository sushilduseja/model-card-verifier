"""CLI entry point for publishing a reviewed audit to HuggingFace."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ScoredResult
from src.publisher import publish_results


def publish_from_review(review_path: str):
    with open(review_path, encoding="utf-8") as f:
        data = json.load(f)

    scored = [ScoredResult(**r) for r in data["results"]]
    claim_scores = data["claim_scores"]
    model_id = data["model_id"]

    print(f"Publishing audit for: {model_id}")
    print(f"Overall compliance score: {claim_scores['overall']}")
    print(f"Tests: {len(scored)}, Claims: {len(claim_scores['per_claim'])}")

    confirm = input("\nPublish to HuggingFace? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    publish_results(model_id, scored, claim_scores)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python publish.py reviews/<file>.json")
        sys.exit(1)
    publish_from_review(sys.argv[1])