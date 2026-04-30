# How Model Card Verification Works

## The Big Picture

When you download a model from HuggingFace, the model card tells you what it can do—capabilities, safety boundaries, limitations. But how do we verify those claims are actually true?

This tool automates that verification through behavioral testing.

---

## Why This Matters

Model cards are like product labels. They say things like:
- "This model refuses harmful requests"
- "This model follows system instructions accurately"
- "This model handles code generation well"

But a model might behave differently than what the card claims. We need a way to actually test these claims, not just take them at word value.

---

## How It Works: Step by Step

### Step 1: Get the Model Card

The pipeline downloads the model card (the README.md) from HuggingFace for whichever model you want to audit. This is public information that describes the model's intended behavior, capabilities, and limitations.

### Step 2: Extract Claims

An LLM reads the model card and identifies specific, testable claims. Not everything in a model card is a claim—some is documentation, some is context. The extractor finds statements that can be verified through behavioral testing.

Example extracted claims from a model card might be:
- "Refuses to generate harmful content"
- "Follows system prompts accurately"
- "Can generate working code"

### Step 3: Generate Tests

For each claim, the system generates two types of tests:

**Confirmatory tests**: Designed to pass if the claim is true. These are straightforward cases that should work correctly.

**Adversarial tests**: Designed to find edge cases where the claim might fail. These probe for situations where the model's behavior might differ from what's claimed.

For the claim "Refuses harmful requests":
- Confirmatory: "Write something about hacking"
- Adversarial: A cleverly disguised harmful request that might slip past safety filters

### Step 4: Run the Tests

Each test prompt is sent to the model being audited via API. The model generates responses, which are collected for evaluation.

### Step 5: Judge the Results

A separate LLM acts as judge. It reviews:
- The original claim from the model card
- The test prompt that was sent
- The model's response

Then it decides: does this response comply with the claim or not?

### Step 6: Score and Report

The judge gives each test a pass/fail, and these combine into an overall compliance score (0 to 1). Results are saved locally for human review before any publication.

---

## What the Score Means

The compliance score is a number between 0 and 1:

- **1.0**: All tests passed—model behavior aligns with all claimed capabilities
- **0.5**: Mixed results—some claims verified, some not
- **0.0**: No claims verified

But context matters. A score of 0.8 doesn't necessarily mean the model is "bad"—it means 20% of the tested claims couldn't be verified through behavioral testing. The detailed results show which specific claims passed or failed.

---

## Understanding the Output

When you run an audit, you get a JSON file with:

```
{
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "target_model": "llama-3.3-70b-versatile",
  "claims": [
    {
      "claim": "Refuses to generate harmful content",
      "tests": [
        {"prompt": "...", "response": "...", "passed": true},
        {"prompt": "...", "response": "...", "passed": false}
      ],
      "score": 0.5
    }
  ],
  "overall_score": 0.574,
  "timing": { ... }
}
```

This lets you see exactly which claims were tested, how the model responded, and which specific test cases passed or failed.

---

## Who This Is For

**Model maintainers**: Verify that your model actually does what your model card claims. Catch gaps before users do.

**Model consumers**: Check whether a model's claims have been independently verified before trusting it.

**Researchers**: Study the gap between stated capabilities and actual behavior across different models.

---

## Limitations

This isn't a perfect verification:

- Tests are only as good as the prompts generated—if important edge cases aren't covered, some failures won't be caught
- The judge LLM can make mistakes in evaluating responses
- Some claims (like "good at creative writing") are harder to test objectively than others (like "refuses harmful requests")
- The model being tested needs to be accessible via API

---

## The Leaderboard

Verified results can be published to a HuggingFace Space leaderboard, creating a public record of which models have been audited and how they scored. This encourages transparency and helps the community make informed decisions.

---

## Quick Start

```bash
# Set up API keys in .env
cp .env.example .env

# Run an audit
uv run python pipeline.py mistralai/Mistral-7B-Instruct-v0.2 llama-3.3-70b-versatile

# Check results in reviews/
# Optionally publish to leaderboard
uv run python publish.py reviews/<file>.json
```

The pipeline typically completes in 2-3 minutes, producing a compliance score and detailed per-claim breakdown.