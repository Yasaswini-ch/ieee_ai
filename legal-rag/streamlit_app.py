import streamlit as st
from pathlib import Path
from typing import List, Dict
import os

from src.utils import get_env, ensure_dir, load_json, save_json, redact_key
from src.ingestion import ingest_pdf, ingest_docx, ingest_jsonl, to_records
from src.chunking import simple_chunk
from src.embedding_store import EmbeddingStore
from src.retrieval import retrieve_top_k
from src.generation import answer, general_answer
from src.contradictions import detect_contradictions
from src.versioning import compare_versions

INDEX_DIR = get_env("INDEX_DIR", "./data/index")

st.set_page_config(page_title="Legal/Policy RAG", layout="wide")

st.sidebar.title("RAG Index")
openai_key = os.getenv("OPENAI_API_KEY")
st.sidebar.write(f"OpenAI key: {redact_key(openai_key)}")

index_dir = st.sidebar.text_input("Index directory", value=str(INDEX_DIR))
ensure_dir(index_dir)

# Embedding provider controls
st.sidebar.subheader("Embedding Provider")
emb_provider = st.sidebar.selectbox("Provider", options=["OpenAI", "Gemini", "Local"], index=0)
emb_provider_key = emb_provider.lower()
local_model = st.sidebar.text_input("Local embedding model", value="sentence-transformers/all-MiniLM-L6-v2") if emb_provider_key == "local" else None
gem_embed_model = st.sidebar.text_input("Gemini embedding model", value="models/text-embedding-004") if emb_provider_key == "gemini" else None
if emb_provider_key == "local":
    st.sidebar.info("Local embeddings enabled. No OpenAI/Gemini calls for indexing.")
if emb_provider_key == "gemini":
    st.sidebar.caption("Requires GEMINI_API_KEY in environment or .env")

store = EmbeddingStore(
    index_dir,
    use_local=(emb_provider_key == "local"),
    local_model=local_model,
    provider=emb_provider_key,
    gemini_model=gem_embed_model,
)
loaded = store.load()
st.sidebar.success("Index loaded" if loaded else "No existing index. Build one.")

st.sidebar.subheader("Upload Documents")
files = st.sidebar.file_uploader("Upload PDF/DOCX/JSONL", accept_multiple_files=True, type=["pdf", "docx", "jsonl"]) 
version_default = st.sidebar.text_input("Default version for uploads", value="v1")

if st.sidebar.button("Ingest & Index"):
    all_records: List[Dict] = []
    for f in files or []:
        name = f.name
        stem = Path(name).stem
        ext = Path(name).suffix.lower()
        if ext == ".pdf":
            sections = ingest_pdf(f.read(), doc_id=stem, version=version_default)
        elif ext == ".docx":
            sections = ingest_docx(f.read(), doc_id=stem, version=version_default)
        elif ext == ".jsonl":
            sections = ingest_jsonl(f.read())
        else:
            st.sidebar.warning(f"Unsupported type: {ext}")
            continue
        recs = to_records(sections)
        recs = simple_chunk(recs, max_chars=2000, overlap=200)
        all_records.extend(recs)
    if all_records:
        try:
            added = store.add(all_records)
            store.save()
            st.sidebar.success(f"Indexed {added} chunks from {len(files)} files.")
        except RuntimeError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Unexpected error during indexing: {e}")
    else:
        st.sidebar.warning("No records to index.")

st.title("Legal/Policy RAG System")

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_area("Enter your query", height=100, placeholder="e.g., What are the data retention requirements?")
with col2:
    k = st.number_input("Top K", min_value=1, max_value=20, value=5)
    explain = st.checkbox("Explain Mode (plain English)", value=False)
    show_contradictions = st.checkbox("Contradiction Detection", value=False)
    # LLM provider controls
    llm_provider = st.selectbox("LLM Provider", options=["OpenAI", "Gemini", "Ollama"], index=0)
    local_llm_model = st.text_input("Local LLM model (Ollama)", value="llama3.1:8b") if llm_provider.lower() == "ollama" else None
    gemini_chat_model = st.text_input("Gemini chat model", value="gemini-1.5-flash") if llm_provider.lower() == "gemini" else None
    bypass_rag = st.checkbox("Bypass RAG (use LLM directly)", value=False, help="Answers directly with the selected LLM (e.g., Gemini) without using retrieved context.")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    elif store.index is None or store.index.ntotal == 0:
        st.warning("Please build the index first by uploading and indexing documents.")
    else:
        results = retrieve_top_k(store, query, k=k)
        chunks = [m for _, m in results]
        # Decide whether to use RAG or general fallback
        use_general = False
        if bypass_rag:
            use_general = True
        else:
            if not chunks:
                use_general = True
            else:
                # If the top score is too low, prefer a general model answer to avoid poor context
                top_score = results[0][0] if results else 0.0
                if top_score < 0.2:
                    use_general = True

        try:
            with st.spinner("Generating answer..."):
                if use_general:
                    ans = general_answer(query, provider=llm_provider, gemini_model=gemini_chat_model)
                else:
                    ans = answer(
                        query,
                        chunks,
                        explain=explain,
                        llm_provider=llm_provider,
                        local_model=local_llm_model,
                        gemini_model=gemini_chat_model,
                    )
            st.subheader("Answer")
            st.write(ans)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error while generating answer: {e}")

        # Only show citations/contradictions when RAG was used successfully
        if not use_general and chunks:
            st.subheader("Citations")
            for score, meta in results:
                st.markdown(f"- **{meta['doc_id']} / {meta['section_id']} / {meta['version']}** — {meta.get('title','')} (score: {score:.3f})")

            if show_contradictions and len(chunks) >= 2:
                st.subheader("Contradiction Warnings")
                contras = detect_contradictions(chunks)
                if not contras:
                    st.write("No contradictions detected among top chunks.")
                else:
                    for (i, j), s in contras:
                        a = chunks[i]; b = chunks[j]
                        st.markdown(f"- Possible contradiction between `" +
                                    f"{a['doc_id']}/{a['section_id']}/{a['version']}" +
                                    "` and `" +
                                    f"{b['doc_id']}/{b['section_id']}/{b['version']}" + f"` (score {s:.2f})")

st.divider()

st.subheader("Time-Machine Compliance")
with st.form("diff_form"):
    doc_id = st.text_input("doc_id", value="")
    section_id = st.text_input("section_id", value="")
    v_old = st.text_input("old version", value="")
    v_new = st.text_input("new version", value="")
    submitted = st.form_submit_button("Compare Versions")
    if submitted:
        if store.metadata:
            diff = compare_versions(store.metadata, doc_id, section_id, v_old, v_new)
            st.code(diff or "No diff.")
        else:
            st.info("No metadata available. Build index first.")
