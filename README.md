# Model Card Verifier

Automated audit pipeline that extracts safety and capability claims from HuggingFace model cards, generates behavioral tests, runs them against a target model, and produces a compliance score.

## The Problem

Model cards on HuggingFace describe what an AI model can and cannot do—its capabilities, limitations, and safety boundaries. But how do we know if those claims are actually true? A model might claim to "refuse harmful requests" or "follow system instructions accurately," yet behave differently in practice.

This project automates the verification of those claims through behavioral testing.

## Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────┐    ┌──────────┐
│  Model Card     │───▶│  Claim Extraction │───▶│ Test Generation │───▶│ Test Run   │───▶│  Scoring │
│  (HF Hub)       │    │  (LLM extracts   │    │  (Generates     │    │  (Inference│    │  (Judge   │
│                 │    │   claims from    │    │   confirmatory  │    │   against  │    │   scores  │
│                 │    │   card text)     │    │   & adversarial │    │   target)  │    │   claims) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └────────────┘    └──────────┘
```

1. **Claim Extraction** - Fetches the model's card from HuggingFace Hub, uses an LLM to identify measurable claims about capabilities, safety boundaries, and limitations
2. **Test Generation** - For each claim, generates both confirmatory tests (designed to pass if the claim is true) and adversarial tests (designed to find cases where the claim might fail)
3. **Test Execution** - Runs each test prompt against the model being audited via API
4. **Scoring** - A separate LLM acts as judge, evaluating whether the model's responses comply with the original claims
5. **Review** - Results are saved locally for human review before any publication

## Workflow

```bash
# 1. Setup
cp .env.example .env  # Add your API keys
uv sync

# 2. Run audit (saves to reviews/, does NOT publish)
uv run python pipeline.py mistralai/Mistral-7B-Instruct-v0.2 llama-3.3-70b-versatile

# 3. Review the output in reviews/

# 4. Publish after review (optional)
uv run python publish.py reviews/<file>.json
```

## Models

| Role | Model | Provider |
|------|-------|----------|
| Claim extraction | `groq/llama-3.1-8b-instant` | Groq |
| Test generation | `groq/llama-3.1-8b-instant` | Groq |
| Model under audit | `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM |
| LLM-as-judge | `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM |

The configuration uses Groq for extraction and test generation (fast, reliable for single LLM calls) and NVIDIA NIM for inference and scoring (cost-effective, good throughput).

## Directory Structure

```
model-card-verifier/
├── src/
│   ├── claim_extractor.py    # Extracts claims from model card text
│   ├── test_generator.py     # Generates behavioral test cases
│   ├── test_runner.py       # Runs tests against target model
│   ├── scorer.py             # LLM-as-judge scores compliance
│   ├── publisher.py          # Publishes results to leaderboard
│   ├── llm.py                # Provider routing and fallback
│   ├── config.py             # Configuration management
│   └── log.py                # Structured logging
├── space/
│   └── app.py                # Gradio leaderboard Space
├── pipeline.py               # Main audit runner
├── publish.py                # Publish to leaderboard
├── reviews/                  # Audit output (not committed)
└── pyproject.toml           # Dependencies
```

## Environment

```env
# Provider keys
GROQ_API_KEY=your_groq_key
NVIDIA_API_KEY=your_nvidia_nim_key

# Model configuration
MODEL_EXTRACTOR=groq/llama-3.1-8b-instant
MODEL_TESTER=groq/llama-3.1-8b-instant
MODEL_UNDER_AUDIT=nvidia/nemotron-3-nano-30b-a3b
MODEL_JUDGE=nvidia/nemotron-3-nano-30b-a3b

# Performance tuning
RUN_WORKERS=2
SCORE_WORKERS=2
LLM_TIMEOUT_SECONDS=90
```

## Deploy Leaderboard

Create a HuggingFace Space with Gradio, then push the contents of `space/`. Add `RESULTS_REPO` as a Space secret to enable result publishing.

## Output

The pipeline produces a JSON file in `reviews/` containing:
- Extracted claims from the model card
- Generated test cases (confirmatory and adversarial)
- Test results (pass/fail with model responses)
- Per-claim compliance scores (0-1)
- Overall compliance score
- Timing information for each pipeline stage

This allows maintainers to review results before publishing to the leaderboard.