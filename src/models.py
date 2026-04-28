"""Shared data models for the audit pipeline."""
from pydantic import BaseModel


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str  # "safety" | "capability" | "limitation" | "alignment"
    source_quote: str


class TestCase(BaseModel):
    test_id: str
    claim_id: str
    test_type: str  # "confirmatory" | "adversarial"
    prompt: str
    expected_behavior: str
    claim_text: str  # original claim text for judge reference


class TestResult(BaseModel):
    test_id: str
    claim_id: str
    test_type: str
    prompt: str
    expected_behavior: str
    actual_response: str
    model_under_audit: str
    tester_model: str
    judge_model: str


class ScoredResult(BaseModel):
    test_id: str
    claim_id: str
    test_type: str
    prompt: str
    expected_behavior: str
    actual_response: str
    model_under_audit: str
    tester_model: str
    judge_model: str
    compliant: bool
    confidence: float  # 0.0-1.0
    judge_reasoning: str
    score: float  # compliant * confidence