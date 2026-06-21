# Architecture

The MVP is split into a FastAPI backend, an offline data pipeline, SQLite, and
Qdrant integration boundaries.

## Storage

- SQLite stores MVP metadata and placeholder doctor notes only.
- Qdrant collection `medical_evidence` stores embedded medical evidence.
- Qdrant collection `doctor_memory` will store semantic doctor memory later.

The API does not connect to Qdrant, Gemini, or an embedding model during import
or startup. SQLite tables are initialized only in the FastAPI lifespan.

## Deferred components

LangGraph, JWT authentication, frontend code, live Gemini generation, vector
retrieval, and clinical interaction rules are deferred to later phases.
