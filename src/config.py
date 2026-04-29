"""Configuration for the audit pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()


class PipelineConfig:
    model_under_audit: str = ""
    model_extractor: str = ""
    model_tester: str = ""
    model_judge: str = ""

    # Role keys mapped from provider keys
    audit_api_key: str = ""
    extractor_api_key: str = ""
    tester_api_key: str = ""
    judge_api_key: str = ""
    
    audit_api_base: str | None = None

    def __init__(self):
        # 1. Load Model Roles
        self.model_under_audit = self._require("MODEL_UNDER_AUDIT")
        self.model_extractor = self._require("MODEL_EXTRACTOR")
        self.model_tester = self._require("MODEL_TESTER")
        self.model_judge = self._require("MODEL_JUDGE")
        
        # 2. Load Provider Keys
        groq_key = self._require("GROQ_API_KEY")
        nvidia_key = self._require("NVIDIA_API_KEY")
        
        # 3. Map Roles to Keys based on model prefix
        self.extractor_api_key = groq_key if self.model_extractor.startswith("groq/") else nvidia_key
        self.tester_api_key = groq_key if self.model_tester.startswith("groq/") else nvidia_key
        self.judge_api_key = nvidia_key if self.model_judge.startswith("nvidia/") else groq_key
        self.audit_api_key = nvidia_key if self.model_under_audit.startswith("nvidia/") else groq_key
        
        self.confirm_claims = os.getenv("CONFIRM_CLAIMS", "false").lower() == "true"

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
