from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Clinical Auditor AI Agent"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./clinical_auditor.db"
    fake_doctor_id: str = "dev-doctor-001"
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440
    use_langgraph_audit: bool = True

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_enabled: bool = False
    gemini_temperature: float = 0.0
    gemini_timeout_seconds: int = 60
    gemini_report_enable: bool = False
    gemini_report_model: str = ""
    gemini_report_timeout_seconds: int | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZTc2MGI0YmMtODcwMi00ZWM5LWEyYzgtZTUyZmY3MmEyZWRhIn0.yUyjw4EbS6Z1qhL3TmgFMRR_BCGWeuOTRcXXvs4Kp_8"
    qdrant_medical_evidence_collection: str = "clinical_evidence"
    qdrant_doctor_memory_collection: str = "doctor_memory"

    embedding_model: str = "intfloat/multilingual-e5-base"
    disable_local_embeddings: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
