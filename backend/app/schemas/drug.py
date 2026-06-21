from pydantic import BaseModel, Field


class Drug(BaseModel):
    name: str = Field(min_length=1)
    active_ingredient: str | None = None
    dosage: str | None = None
