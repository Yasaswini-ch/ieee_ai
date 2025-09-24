# Legal/Policy RAG System (OpenAI + FAISS)

This app ingests legal/policy documents (PDF, DOCX, JSONL), chunks them by sections, indexes them with OpenAI embeddings into a FAISS vector store, and answers user queries with citations. It also supports contradiction detection (NLI), explain mode (plain English), and time-machine compliance (compare historical versions).

Live Demo at:https://legalanalyzer.streamlit.app/

## Features
- Ingestion: PDF, DOCX, JSONL with metadata (doc_id, section_id, version, title, text)
- Embeddings: OpenAI Embeddings (text-embedding-3-small by default)
- Vector DB: FAISS (local)
- RAG: Retrieve top-K relevant chunks and generate answers via OpenAI GPT
- Citations: Exact references `doc_id/section_id/version`
- Contradiction Detection: Optional NLI model to flag contradictions among retrieved chunks
- Explain Mode: "Explain like I’m 5" toggle to simplify language
- Time-Machine Compliance: Query historical versions and diff changes
- Frontend: Streamlit web app
https://github.com/Yasaswini-ch/ieee_ai/blob/main/legal-rag/data/index/WhatsApp%20Image%202025-09-24%20at%2015.54.42_0bb5b459.jpg
## Setup
1) Python 3.10+
2) Create a virtual environment (recommended)
3) Install dependencies:
```
pip install -r requirements.txt
```
4) Set your OpenAI API key as an environment variable (do NOT hardcode keys):
- Linux/macOS:
```
export OPENAI_API_KEY="YOUR_KEY_HERE"
```
- Windows PowerShell:
```
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
```

Optionally, copy `.env.example` to `.env` and fill in values, then use a loader like `python-dotenv` (already included) or set environment variables directly.

## Run
```
streamlit run streamlit_app.py
```

## Project Structure
```
legal-rag/
  README.md
  requirements.txt
  .env.example
  streamlit_app.py
  src/
    ingestion.py
    chunking.py
    embedding_store.py
    retrieval.py
    generation.py
    contradictions.py
    versioning.py
    utils.py
  data/
    index/
```

- `data/index/` stores FAISS index files and metadata JSON.
- Upload documents via the Streamlit UI or place files under a directory and point the app to ingest them.

## Notes
- Contradiction detection uses a small NLI model via `transformers`. It is optional and will degrade gracefully if the model cannot be loaded.
- JSONL format expected: one JSON object per line with at least `doc_id`, `version`, `title`, `text`, and optionally `section_id`.
- Use `version` consistently to enable time-machine comparisons.

## Evaluation
- Basic evaluation hooks are included to measure accuracy, citation traceability, and ambiguous query coverage — extend `utils.py` as needed.
