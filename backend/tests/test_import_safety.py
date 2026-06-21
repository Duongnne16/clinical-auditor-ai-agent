import importlib


PIPELINE_MODULES = [
    "data_pipeline.config",
    "data_pipeline.crawlers.crawl_longchau",
    "data_pipeline.crawlers.crawl_duocthu",
    "data_pipeline.crawlers.crawl_drugs_com",
    "data_pipeline.cleaning.clean_longchau",
    "data_pipeline.cleaning.clean_duocthu",
    "data_pipeline.cleaning.clean_drugs_com",
    "data_pipeline.processing.build_drug_mapping",
    "data_pipeline.processing.chunk_evidence",
    "data_pipeline.processing.embed_and_ingest",
]


def test_pipeline_modules_are_safe_to_import() -> None:
    for module_name in PIPELINE_MODULES:
        importlib.import_module(module_name)
