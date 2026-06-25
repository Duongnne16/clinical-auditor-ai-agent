import json
from pathlib import Path

import pytest

from data_pipeline.processing.build_evidence_ingredient_catalog import (
    build_catalog,
    normalize_text,
)


INPUT_PATH = Path(
    "data/processed/trungtamthuoc_v2/ingredients_chunks.jsonl"
)


@pytest.mark.skipif(
    not INPUT_PATH.exists(), reason="Full evidence chunks are unavailable"
)
def test_full_evidence_catalog_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "catalog"
    report = build_catalog(INPUT_PATH, output_dir)
    records = [
        json.loads(line)
        for line in (output_dir / "evidence_ingredient_catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_slug = {record["slug"]: record for record in records}

    assert report["unique_ingredients"] > 1000
    assert {"paracetamol", "metformin"} <= set(by_slug)
    assert "aciclovir" in by_slug or "acyclovir" in by_slug

    amlodipine_matches = [
        record
        for record in records
        if any(
            token in normalize_text(
                " ".join(
                    [record["slug"], record["entity_name"], *record["aliases"]]
                )
            )
            for token in ("amlodipine", "amlodipin")
        )
    ]
    assert amlodipine_matches, "Amlodipine/Amlodipin not found in catalog"
    assert amlodipine_matches[0]["slug"]

    povidone_matches = [
        record
        for record in records
        if "povidone iodine"
        in {
            normalize_text(record["entity_name"]),
            *map(normalize_text, record["aliases"]),
        }
    ]
    assert povidone_matches, "Povidone Iodine not found in catalog"
    assert povidone_matches[0]["slug"]
    assert report["invalid_lines"] == 0
    assert report["invalid_chunks"] == 0
