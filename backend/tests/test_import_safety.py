import importlib


PIPELINE_MODULES = [
    "data_pipeline.config",
    "data_pipeline.crawlers.crawl_drugs_com",
    "data_pipeline.crawlers.trungtamthuoc_ingredients_crawler",
    "data_pipeline.cleaning.clean_longchau",
    "data_pipeline.cleaning.clean_duocthu",
    "data_pipeline.cleaning.clean_drugs_com",
    "data_pipeline.cleaning.inspect_trungtamthuoc_raw",
    "data_pipeline.cleaning.clean_trungtamthuoc_raw",
    "data_pipeline.inspection.inspect_longchau_drugs",
    "data_pipeline.processing.build_drug_mapping",
    "data_pipeline.processing.build_longchau_drug_mapping",
    "data_pipeline.processing.build_evidence_ingredient_catalog",
    "data_pipeline.processing.chunk_evidence",
    "data_pipeline.processing.chunk_trungtamthuoc_ingredients",
    "data_pipeline.processing.embed_and_ingest",
    "backend.app.services.drug_mapping_service",
    "backend.app.services.ingredient_evidence_resolver",
    "backend.app.services.medication_line_parser",
    "backend.app.services.normalize_drugs_service",
    "backend.app.services.qdrant_retriever_service",
    "backend.app.services.prescription_check_service",
    "backend.app.services.risk_analyzer_service",
    "backend.app.services.gemini_risk_analyzer_client",
    "backend.app.services.report_generator_service",
    "backend.app.services.safety_layer_service",
    "backend.app.services.prescription_audit_service",
    "backend.app.api.routes.prescriptions",
]


def test_pipeline_modules_are_safe_to_import() -> None:
    for module_name in PIPELINE_MODULES:
        importlib.import_module(module_name)
