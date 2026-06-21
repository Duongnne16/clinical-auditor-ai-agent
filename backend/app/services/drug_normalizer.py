class DrugNormalizer:
    """Placeholder for drug-name normalization and mapping."""

    def normalize(self, name: str) -> str:
        return " ".join(name.strip().lower().split())
