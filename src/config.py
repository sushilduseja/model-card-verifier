"""Configuration for the audit pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()


class PipelineConfig:
    model_under_audit: str = ""
    model_tester: str = ""
    model_judge: str = ""

    audit_api_key: str = ""
    audit_api_base: str | None = None
    tester_api_key: str = ""
    judge_api_key: str = ""
    groq_api_key: str = ""

    def __init__(self):
        self.model_under_audit = self._require("MODEL_UNDER_AUDIT")
        self.model_extractor = self._require("MODEL_EXTRACTOR")
        self.model_tester = self._require("MODEL_TESTER")
        self.model_judge = self._require("MODEL_JUDGE")
        self.audit_api_key = self._require("AUDIT_API_KEY")
        self.audit_api_base = os.getenv("AUDIT_API_BASE") or None
        self.extractor_api_key = self._require("EXTRACTOR_API_KEY")
        self.tester_api_key = self._require("TESTER_API_KEY")
        self.judge_api_key = self._require("JUDGE_API_KEY")
        self.confirm_claims = os.getenv("CONFIRM_CLAIMS", "false").lower() == "true"
        self._validate_fallback_config()

    @staticmethod
    def _validate_fallback_config() -> None:
        models_csv = os.getenv("FALLBACK_MODELS", "").strip()
        if models_csv:
            fallback_models = [m.strip() for m in models_csv.split(",") if m.strip()]
        else:
            single = os.getenv("FALLBACK_MODEL", "").strip()
            fallback_models = [single] if single else []

        for fallback_model in fallback_models:
            if fallback_model.startswith("gemini/"):
                fallback_key = (
                    os.getenv("FALLBACK_API_KEY")
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                )
                if not fallback_key:
                    raise EnvironmentError(
                        "gemini/* fallback configured but no fallback key found. "
                        "Set one of FALLBACK_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY in .env"
                    )

    @staticmethod
    def _require(env_var: str) -> str:
        val = os.getenv(env_var, "")
        if not val:
            raise EnvironmentError(f"{env_var} is required but not set in .env")
        return val

    def display(self) -> str:
        return f"""
=== Audit Configuration ===
Model under audit : {self.model_under_audit}
Extractor model   : {self.model_extractor}
Tester model      : {self.model_tester}
Judge model       : {self.model_judge}
==========================="""
