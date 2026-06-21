from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Clinical Auditor AI Agent"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./clinical_auditor.db"
    fake_doctor_id: str = "dev-doctor-001"

    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_enabled: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_medical_evidence_collection: str = "medical_evidence"
    qdrant_doctor_memory_collection: str = "doctor_memory"

    embedding_model: str = "intfloat/multilingual-e5-base"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
