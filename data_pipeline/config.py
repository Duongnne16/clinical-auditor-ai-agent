from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = Path("data")
    embedding_model: str = "intfloat/multilingual-e5-base"
    medical_evidence_collection: str = "medical_evidence"
