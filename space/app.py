import os, json
import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download


RESULTS_REPO = os.getenv("RESULTS_REPO")
if not RESULTS_REPO:
    raise EnvironmentError("RESULTS_REPO env var is required")


def load_leaderboard():
    try:
        path = hf_hub_download(RESULTS_REPO, "leaderboard.jsonl", repo_type="dataset")
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        df = df[["model_id", "overall_score", "n_claims", "n_tests", "run_timestamp"]]
        df["overall_score"] = df["overall_score"].apply(lambda x: f"{x:.1%}")
        df = df.sort_values("overall_score", ascending=True)
        df.columns = ["Model", "Compliance Score", "Claims Audited", "Tests Run", "Timestamp"]
        return df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


with gr.Blocks(title="Model Card Verifier") as demo:
    gr.Markdown("""
# Model Card Verifier - Specification vs Behavior Leaderboard

Automated audit: extracts safety/capability claims from model cards, generates behavioral tests,
runs them against the model, and scores compliance. Lower score = larger gap between claims and behavior.

**Methodology:** Claim extraction -> test generation -> inference -> LLM-as-judge scoring.
**Dataset:** All raw results at [HuggingFace Datasets](https://huggingface.co/datasets/{RESULTS_REPO})
    """)

    refresh_btn = gr.Button("Refresh Leaderboard")
    leaderboard = gr.Dataframe(value=load_leaderboard, every=300)
    refresh_btn.click(fn=load_leaderboard, outputs=leaderboard)


demo.launch()