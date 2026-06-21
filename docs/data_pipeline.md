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
