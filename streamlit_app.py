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
from src.tfidf_retriever import tfidf_retrieve
from src.extractive_qa import best_span
from src.reranker import rerank
from src.summarize import summarize_records
from src.contradictions import detect_contradictions

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
    prefer_non_llm = st.checkbox("Prefer Non-LLM Extractive QA", value=True, help="Use TF-IDF + extractive QA first; LLM used only as fallback.")
    low_conf_thresh = st.slider("RAG low-confidence threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    extractive_top_n = st.slider("Extractive QA top_n contexts", min_value=1, max_value=5, value=3)
    use_reranker = st.checkbox("Use cross-encoder re-ranking", value=True)
    rerank_model = st.text_input("Re-ranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None
    rerank_top_m = st.slider("Re-rank top M", min_value=3, max_value=20, value=5) if use_reranker else 0
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
        # 1) If user wants to bypass RAG entirely, use general LLM immediately
        if bypass_rag:
            try:
                with st.spinner("Generating answer..."):
                    ans = general_answer(query, provider=llm_provider, gemini_model=gemini_chat_model)
                st.subheader("Answer")
                st.write(ans)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
            st.stop()

        # 2) Prefer Non-LLM path: TF-IDF + optional Re-rank + Extractive QA over indexed metadata
        used_extractive = False
        extractive_answer = ""
        tfidf_results: List[tuple[float, Dict]] = []
        if prefer_non_llm:
            corpus = store.metadata or []
            if not corpus:
                st.info("No indexed metadata available. Please ingest documents first.")
            else:
                tfidf_results = tfidf_retrieve(corpus, query, k=max(k, rerank_top_m if use_reranker else k))
                top_contexts = [m for _, m in tfidf_results]
                if use_reranker and top_contexts:
                    rr = rerank(query, top_contexts, model_name=rerank_model, top_m=rerank_top_m)
                    # rr is list of (score, record); use the records in order
                    top_contexts = [m for _, m in rr]
                with st.spinner("Running extractive QA..."):
                    extractive_answer, scored = best_span(query, top_contexts, top_n=min(extractive_top_n, len(top_contexts)))
                if extractive_answer:
                    used_extractive = True
                    st.subheader("Answer")
                    st.write(extractive_answer)
                    # Citations from the contexts actually used
                    st.subheader("Citations")
                    for meta in top_contexts[:extractive_top_n]:
                        st.markdown(f"- **{meta['doc_id']} / {meta['section_id']} / {meta['version']}** — {meta.get('title','')}")
        if used_extractive:
            st.stop()

        # 3) If extractive not used or failed, try vector RAG (with optional re-ranking)
        results = retrieve_top_k(store, query, k=k)
        chunks = [m for _, m in results]
        if use_reranker and chunks:
            rr = rerank(query, chunks, model_name=rerank_model, top_m=min(rerank_top_m, len(chunks)))
            chunks = [m for _, m in rr]
            # Reconstruct scores for display (normalize to descending rank if needed)
            results = [(1.0 - (i / max(1, len(chunks) - 1)), chunks[i]) for i in range(len(chunks))]
        use_general = False
        if not chunks:
            use_general = True
        else:
            top_score = results[0][0] if results else 0.0
            if top_score < low_conf_thresh:
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

# ---------------- Policy Summary Cards ----------------
st.divider()
st.subheader("Policy Summary Cards")
if not store.metadata:
    st.info("No indexed metadata available. Please ingest documents first.")
else:
    from collections import defaultdict
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for rec in store.metadata:
        by_doc[rec.get("doc_id", "")].append(rec)

    doc_ids = sorted(list(by_doc.keys()))
    selected_docs = st.multiselect("Select documents to summarize", options=doc_ids, default=doc_ids[:1])
    sentences_max = st.slider("Bullets per summary", min_value=3, max_value=10, value=6)

    if st.button("Generate Summary Cards"):
        import json
        for did in selected_docs:
            records = by_doc.get(did, [])
            with st.spinner(f"Summarizing {did}..."):
                card = summarize_records(records, sentences_max=sentences_max)
            st.markdown(f"### {card['doc_id']} — {card['title']}")
            if card.get("version_set"):
                st.caption("Versions: " + ", ".join(card["version_set"]))
            for b in card.get("bullets", []):
                st.write(b)

            # Downloads
            j = json.dumps(card, ensure_ascii=False, indent=2)
            st.download_button(
                label=f"Download JSON ({did})",
                file_name=f"summary_{did}.json",
                mime="application/json",
                data=j,
            )
            md = f"# Summary: {card['doc_id']} — {card['title']}\n\n" + "\n".join(card.get("bullets", []))
            st.download_button(
                label=f"Download Markdown ({did})",
                file_name=f"summary_{did}.md",
                mime="text/markdown",
                data=md,
            )
