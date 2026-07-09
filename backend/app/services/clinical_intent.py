from __future__ import annotations

from enum import Enum

from backend.app.services.intent_router import (
    DRUG_INTERACTION_QUERY,
    OUT_OF_SCOPE,
    SINGLE_DRUG_QUERY,
)


class ClinicalIntent(str, Enum):
    PRESCRIPTION_CHECK = "prescription_check"
    DRUG_INFORMATION_QUERY = "drug_information_query"
    OUT_OF_SCOPE = "out_of_scope"


CHAT_INTENT_TO_CLINICAL_INTENT = {
    DRUG_INTERACTION_QUERY: ClinicalIntent.DRUG_INFORMATION_QUERY,
    SINGLE_DRUG_QUERY: ClinicalIntent.DRUG_INFORMATION_QUERY,
    OUT_OF_SCOPE: ClinicalIntent.OUT_OF_SCOPE,
}


def clinical_intent_from_chat_intent(intent: str) -> ClinicalIntent:
    return CHAT_INTENT_TO_CLINICAL_INTENT.get(intent, ClinicalIntent.OUT_OF_SCOPE)
