from backend.app.services.clinical_intent import (
    ClinicalIntent,
    clinical_intent_from_chat_intent,
)


def test_clinical_intent_has_exactly_three_intents() -> None:
    assert set(ClinicalIntent) == {
        ClinicalIntent.PRESCRIPTION_CHECK,
        ClinicalIntent.DRUG_INFORMATION_QUERY,
        ClinicalIntent.OUT_OF_SCOPE,
    }
    assert len(ClinicalIntent) == 3


def test_clinical_intent_exact_enum_values() -> None:
    assert ClinicalIntent.PRESCRIPTION_CHECK.value == "prescription_check"
    assert ClinicalIntent.DRUG_INFORMATION_QUERY.value == "drug_information_query"
    assert ClinicalIntent.OUT_OF_SCOPE.value == "out_of_scope"


def test_drug_interaction_query_maps_to_drug_information_query() -> None:
    assert (
        clinical_intent_from_chat_intent("drug_interaction_query")
        == ClinicalIntent.DRUG_INFORMATION_QUERY
    )


def test_single_drug_query_maps_to_drug_information_query() -> None:
    assert (
        clinical_intent_from_chat_intent("single_drug_query")
        == ClinicalIntent.DRUG_INFORMATION_QUERY
    )


def test_out_of_scope_maps_to_out_of_scope() -> None:
    assert clinical_intent_from_chat_intent("out_of_scope") == ClinicalIntent.OUT_OF_SCOPE


def test_unknown_chat_intent_maps_to_out_of_scope() -> None:
    assert (
        clinical_intent_from_chat_intent("unknown_intent")
        == ClinicalIntent.OUT_OF_SCOPE
    )
