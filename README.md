# Clinical Auditor AI Agent

Backend skeleton for an AI-assisted prescription auditing system. This stage
contains API contracts and extension points only; it does not perform clinical
analysis or call Gemini, Qdrant, embedding models, or crawlers automatically.

## Storage boundaries

- SQLite is used only for MVP metadata and placeholder doctor notes.
- Medical evidence belongs in the Qdrant collection `medical_evidence`.
- Semantic doctor memory will belong in the Qdrant collection `doctor_memory`
  in a later phase.

## Local setup

Python 3.11 or newer is recommended.

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, then run FastAPI:

   ```powershell
   Copy-Item .env.example .env
   uvicorn backend.app.main:app --reload
   ```

   API documentation is available at `http://127.0.0.1:8000/docs`.

4. Run tests:

   ```powershell
   pytest
   ```

## Current limitations

LangGraph, real JWT authentication, the frontend, live Gemini calls, vector
retrieval, crawling, and drug-interaction logic are intentionally out of scope
for this skeleton. Placeholder responses are not clinical advice.
