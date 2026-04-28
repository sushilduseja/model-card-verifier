"""Publish scored results to HuggingFace dataset."""
import json
import os
import tempfile
from datetime import datetime, timezone
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from src.models import ScoredResult


def publish_results(
    model_id: str,
    scored_results: list[ScoredResult],
    claim_scores: dict,
):
    hf_username = os.getenv("HF_USERNAME")
    results_repo = f"{hf_username}/model-card-verifier-results"
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.create_repo(results_repo, repo_type="dataset", exist_ok=True, private=True)
    except Exception:
        pass

    records = [r.model_dump() for r in scored_results]
    for r in records:
        r["run_timestamp"] = datetime.now(timezone.utc).isoformat()

    ds = Dataset.from_list(records)

    safe_model_id = model_id.replace("/", "__")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path_in_repo = f"runs/{safe_model_id}/{timestamp}.parquet"
    tmp_dir = tempfile.mkdtemp()
    local_parquet = os.path.join(tmp_dir, f"{safe_model_id}_{timestamp}.parquet")

    ds.to_parquet(local_parquet)
    api.upload_file(
        path_or_fileobj=local_parquet,
        path_in_repo=path_in_repo,
        repo_id=results_repo,
        repo_type="dataset",
    )

    summary = {
        "model_id": model_id,
        "overall_score": claim_scores["overall"],
        "per_claim_scores": json.dumps(claim_scores["per_claim"]),
        "n_claims": len(claim_scores["per_claim"]),
        "n_tests": len(scored_results),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "results_path": path_in_repo,
    }

    try:
        existing = hf_hub_download(results_repo, "leaderboard.jsonl", repo_type="dataset")
        with open(existing) as f:
            lines = f.readlines()
    except Exception:
        lines = []

    lines.append(json.dumps(summary) + "\n")

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="leaderboard.jsonl",
        repo_id=results_repo,
        repo_type="dataset",
    )

    print(f"Published: https://huggingface.co/datasets/{results_repo}")
    print(f"Overall compliance score: {claim_scores['overall']}")