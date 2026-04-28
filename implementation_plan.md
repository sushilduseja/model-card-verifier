## Model Card Verifier - Implementation Plan

---

### License Decision

The `ml-intern` repo has no LICENSE file in its source tree. HuggingFace typically ships internal tooling under Apache 2.0 but this is unconfirmed here. **Decision: do not fork or inherit from it.** Use it as an architectural reference only. Build a standalone Python package using the same underlying dependencies (`litellm`, `huggingface-hub`, `httpx`, `pydantic`). No coupling, no license risk.

---

### LLM Routing Strategy

| Role | Model | Provider |
|---|---|---|
| Claim extraction | `llama-3.3-70b-versatile` | Groq (free) |
| Test case generation | `llama-3.3-70b-versatile` | Groq (free) |
| Model under audit | configurable via `OPENCODE_ZEN_MODELS` env var | Groq or OpenCode Zen |
| LLM-as-judge | `llama-3.3-70b-versatile` | Groq (free) |

litellm handles both. Groq is `groq/llama-3.3-70b-versatile`. OpenCode Zen is an OpenAI-compatible endpoint: `openai/{model_id}` with `api_base` overridden.

---

### Phase 0 - Repository & Environment

```bash
git init
uv init
# add dependencies to pyproject.toml
uv sync
```

`.env`:
```
GROQ_API_KEY=your_groq_key
OPENCODE_ZEN_API_KEY=your_opencode_zen_key
OPENCODE_ZEN_API_BASE=https://api.opencode.ai/v1
OPENCODE_ZEN_MODELS=MiniMax-Text-01,big-pickle-v1
HF_TOKEN=your_hf_write_token
HF_USERNAME=your_hf_username
```

Directory layout:
```
model-card-verifier/
├── src/
│   ├── __init__.py
│   ├── claim_extractor.py
│   ├── test_generator.py
│   ├── test_runner.py
│   ├── scorer.py
│   └── publisher.py
├── space/
│   └── app.py
├── pipeline.py
├── publish.py
├── pyproject.toml
├── .env
└── README.md
```

---

### Pipeline Workflow

The pipeline runs in two phases:

1. **Audit** (`pipeline.py`) - Runs the full pipeline and saves results to `reviews/` for human review. Never publishes automatically.
2. **Publish** (`publish.py`) - Explicit human-gated command that publishes a reviewed audit to HuggingFace.

```bash
uv run python pipeline.py mistralai/Mistral-7B-Instruct-v0.2 llama-3.3-70b-versatile
# Review the output in reviews/
uv run python publish.py reviews/<file>.json
```

---

### Key Implementation Decisions

**Review gate**: The pipeline saves results locally and stops. No automatic publishing. The user reviews `reviews/*.json` before running `publish.py`.

**Model routing**: Explicit `OPENCODE_ZEN_MODELS` env var (comma-separated model IDs). Any model not in the list routes to Groq. No fragile substring matching.

**Dataset repo**: Created `private=True` by default. Make public manually from HF UI after reviewing clean runs.

**Temp files**: Uses `tempfile.mkdtemp()` instead of hardcoded `/tmp` for Windows compatibility.

**Leaderboard Space**: Requires `RESULTS_REPO` env var. Fails explicitly if missing - no silent fallback to a dummy value.

---

### Phase 1 - Claim Extractor (`src/claim_extractor.py`)

Fetch the model card from HF Hub and extract structured safety/capability claims via LLM.

```python
class Claim(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str  # "safety", "capability", "limitation", "alignment"
    source_quote: str
```

---

### Phase 2 - Test Generator (`src/test_generator.py`)

For each claim, generate adversarial and confirmatory test cases. 2 confirmatory, 2 adversarial per claim.

```python
class TestCase(BaseModel):
    test_id: str
    claim_id: str
    test_type: str  # "confirmatory" | "adversarial"
    prompt: str
    expected_behavior: str
```

---

### Phase 3 - Test Runner (`src/test_runner.py`)

Run each test prompt against the model under audit. Model routing via `OPENCODE_ZEN_MODELS`:

```python
def run_all_tests(tests: list[TestCase], model_id: str) -> list[TestResult]:
    zen_models = [m.strip() for m in os.getenv("OPENCODE_ZEN_MODELS", "").split(",") if m.strip()]
    if model_id in zen_models:
        # OpenCode Zen routing
    else:
        # Groq routing
```

---

### Phase 4 - Scorer (`src/scorer.py`)

LLM-as-judge scores each result with `compliant` (bool), `confidence` (0.0-1.0), and `score` (weighted by confidence).

```python
class ScoredResult(BaseModel):
    # ... all TestResult fields ...
    compliant: bool
    confidence: float
    judge_reasoning: str
    score: float
```

---

### Phase 5 - Publisher (`src/publisher.py`)

Push scored results as a HF dataset parquet file and append a summary row to `leaderboard.jsonl`.

- Uses `hf_hub_download` (standalone function, not HfApi method)
- Uses `tempfile.mkdtemp()` for cross-platform temp paths
- Dataset repo created `private=True`

---

### Phase 6 - Orchestrator (`pipeline.py`)

Saves to `reviews/` and stops. No inline publish.

```python
def run_audit(model_id: str, audit_model: str = "llama-3.3-70b-versatile"):
    # Steps 1-4 unchanged
    print("Step 5/5: Saving for review...")
    review_path = save_for_review(model_id, scored, claim_scores)
    print(f"Review the file, then run: uv run python publish.py {review_path}")
```

---

### Phase 7 - Publish Gate (`publish.py`)

Explicit human confirmation before publishing:

```python
confirm = input("\nPublish to HuggingFace? (yes/no): ").strip().lower()
if confirm != "yes":
    print("Aborted.")
    return
publish_results(model_id, scored, claim_scores)
```

---

### Phase 8 - Leaderboard Space (`space/app.py`)

HF Space reads `leaderboard.jsonl` from the dataset repo.

```python
RESULTS_REPO = os.getenv("RESULTS_REPO")
if not RESULTS_REPO:
    raise EnvironmentError("RESULTS_REPO env var is required")
```

Deploy: Create Space via HF UI (Gradio, CPU basic), push `space/` contents, add `RESULTS_REPO` as a secret.

---

### First Audit Run

```bash
uv run python pipeline.py mistralai/Mistral-7B-Instruct-v0.2 llama-3.3-70b-versatile
uv run python pipeline.py google/gemma-2-9b-it llama-3.3-70b-versatile
```

To audit a Zen model, set `OPENCODE_ZEN_MODELS` to include the model ID, then:
```bash
uv run python pipeline.py MiniMaxAI/MiniMax-Text-01 MiniMax-Text-01
```

---

### Key Constraints Handled

| Constraint | Resolution |
|---|---|
| No Anthropic/OpenAI key | All LLM calls via Groq (llama-3.3-70b-versatile, free) |
| No GPU | No model weights loaded locally; all inference via APIs |
| No spend | Groq free tier, HF free tier, no paid infra |
| Review gate | Pipeline saves to `reviews/` and stops; `publish.py` for explicit publish |
| Windows temp paths | `tempfile.mkdtemp()` instead of `/tmp` |
| Model routing | Explicit `OPENCODE_ZEN_MODELS` env var, no substring matching |
| Private by default | Dataset repo created `private=True`, user makes public after review |