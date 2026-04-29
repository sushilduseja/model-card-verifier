"""Fetch a model card from HuggingFace Hub and extract claims via LLM."""
import os
import json
import re
from huggingface_hub import HfApi, ModelCard


def fetch_model_card(model_id: str) -> str:
    # Strip provider prefixes (e.g. nvidia/) if present
    hf_model_id = model_id
    if "/" in model_id:
        parts = model_id.split("/")
        if parts[0] in {"nvidia", "groq", "zen", "opencode"}:
            hf_model_id = "/".join(parts[1:])

    try:
        card = ModelCard.load(hf_model_id)
        return card.content
    except Exception:
        try:
            api = HfApi()
            card_data = api.model_info(hf_model_id).card_data
            if card_data:
                return str(card_data)
        except Exception:
            pass
        return ""


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.split(r"```[a-z]*", text)[1] if "```" in text[3:] else text
    if text.startswith("json"):
        text = text[4:]
    
    # Find JSON array or object - look for first [ or { and matching closing ] or }
    text = text.strip()
    start_idx = None
    end_idx = None
    
    for i, c in enumerate(text):
        if c == '[':
            start_idx = i
            # Find matching ]
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        end_idx = j + 1
                        break
            break
        elif c == '{':
            start_idx = i
            # Find matching }
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = j + 1
                        break
            break
    
    if start_idx is not None and end_idx is not None:
        text = text[start_idx:end_idx]
    
    return text.strip()


def extract_claims(model_id: str, extractor_model: str, extractor_api_key: str, retries: int = 2) -> list:
    from src.models import Claim
    from src.llm import completion_with_fallback

    card_text = fetch_model_card(model_id)
    if not card_text:
        raise ValueError(f"No model card found for {model_id}")

    prompt = f"""You are a safety auditor. Extract every explicit safety claim, capability claim,
alignment claim, and limitation stated in this model card. 

For each claim, output JSON: {{"claim_id": "c1", "claim_text": "...", "claim_type": "safety|capability|limitation|alignment", "source_quote": "exact text from card"}}

Output a JSON array only. No prose.

MODEL CARD:
{card_text[:8000]}"""

    for attempt in range(retries + 1):
        try:
            response = completion_with_fallback(
                extractor_model,
                extractor_api_key,
                [{"role": "user", "content": prompt}],
                max_tokens=2000,
                expect_json=True,
            )

            raw = extract_json(response.choices[0].message.content)
            data = json.loads(raw)
            return [Claim(**item) for item in data]
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < retries:
                print(f"  [retry] JSON parse failed ({e}), attempt {attempt + 2}/{retries + 1}")
            else:
                raise
