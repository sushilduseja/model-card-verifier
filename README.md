# Model Card Verifier

Automated audit pipeline that extracts safety/capability claims from HuggingFace model cards, generates behavioral tests, runs them against a target model, and scores compliance.

## Pipeline

1. **Claim Extraction** - Fetches model card from HF Hub, extracts claims via LLM
2. **Test Generation** - Generates confirmatory and adversarial test cases per claim
3. **Test Execution** - Runs tests against the target model via API
4. **Scoring** - LLM-as-judge scores compliance with published claims
5. **Review** - Results saved locally for human review before publishing

## Workflow

```bash
# 1. Setup
cp .env.example .env  # add your API keys
uv sync

# 2. Run audit (saves to reviews/, does NOT publish)
uv run python pipeline.py mistralai/Mistral-7B-Instruct-v0.2 llama-3.3-70b-versatile

# 3. Review the output in reviews/

# 4. Publish after review
uv run python publish.py reviews/<file>.json
```

## Models

| Role | Model | Provider |
|---|---|---|
| Claim extraction | `llama-3.3-70b-versatile` | Groq |
| Test generation | `llama-3.3-70b-versatile` | Groq |
| Model under audit | configurable | Groq or OpenCode Zen |
| LLM-as-judge | `llama-3.3-70b-versatile` | Groq |

## Directory Structure

```
model-card-verifier/
├── src/
│   ├── claim_extractor.py
│   ├── test_generator.py
│   ├── test_runner.py
│   ├── scorer.py
│   └── publisher.py
├── space/
│   └── app.py              # Gradio leaderboard Space
├── pipeline.py              # Audit runner (saves to reviews/)
├── publish.py               # Human-gated publish command
├── reviews/                 # Audit output (not committed)
└── pyproject.toml           # Dependencies (uv)
```

## Environment

```env
GROQ_API_KEY=your_groq_key
OPENCODE_ZEN_API_KEY=your_opencode_zen_key
OPENCODE_ZEN_API_BASE=https://api.opencode.ai/v1
OPENCODE_ZEN_MODELS=MiniMax-Text-01,big-pickle-v1
HF_TOKEN=your_hf_write_token
HF_USERNAME=your_hf_username
```

`OPENCODE_ZEN_MODELS` is a comma-separated list of model IDs that should route to OpenCode Zen. All other models route to Groq.

## Deploy Leaderboard

Create a HF Space with Gradio, then push `space/` contents. Add `RESULTS_REPO` as a Space secret.