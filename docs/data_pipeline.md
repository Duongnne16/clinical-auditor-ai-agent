# Data pipeline

The offline pipeline is organized as:

1. Source-specific crawlers for Long Châu, Dược Thư, and Drugs.com.
2. Source-specific cleaning functions.
3. Drug-name mapping.
4. Evidence chunking with source metadata.
5. Embedding with `intfloat/multilingual-e5-base` and ingestion into the
   Qdrant `medical_evidence` collection.

Every operation must be called explicitly. Importing pipeline modules never
starts a crawler, downloads an embedding model, connects to Qdrant, or writes
files.

## Long Châu product mapping

Build the normalized product file from chunked Long Châu JSON:

```powershell
python -m data_pipeline.processing.build_drug_mapping
```

The command searches `data/processed/longchau_chunked` recursively, falls back
to `data/raw/longchau_chunked`, and atomically writes
`data/processed/longchau_drug_products.json`. Products that use description
fallbacks or uncertain brand extraction are marked for review.

## Trung Tam Thuoc ingredient crawler

This crawler stores raw ingredient monographs only. It does not crawl products,
chunk content, load embedding models, or connect to Qdrant.

Run a small test crawl:

```powershell
python -m data_pipeline.crawlers.trungtamthuoc_ingredients_crawler --limit 3 --overwrite
```

Inspect the completed raw ingredient JSONL without modifying it:

```powershell
python -m data_pipeline.cleaning.inspect_trungtamthuoc_raw
```

The quality report is written to
`data/cleaned/trungtamthuoc/inspection_report.json`.

Safely clean the raw JSONL without truncating or chunking long sections:

```powershell
python -m data_pipeline.cleaning.clean_trungtamthuoc_raw --overwrite
```

Chunk the cleaned ingredient monographs for later retrieval:

```powershell
python -m data_pipeline.processing.chunk_trungtamthuoc_ingredients --overwrite
```
