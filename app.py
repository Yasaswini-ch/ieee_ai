import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Import your custom chunker & classifier
from winning_technical_implementation import DocumentChunker, QueryClassifier

# --- Citation dataclass ---
@dataclass
class CitationOut:
    id: int
    section_path: str
    paragraph_range: str
    legal_weight: str
    text_snippet: str
    score: float

# --- RAG Service ---
class RAGService:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.chunker = DocumentChunker()
        self.classifier = QueryClassifier()
        self.document_chunks: List[Dict[str,Any]] = []
        self.embeddings = None
        self.index = None

    def ingest_document(self, text: str):
        self.document_chunks = self.chunker.chunk_document(text)
        texts = [c['text'] for c in self.document_chunks]
        embs = self.encoder.encode(texts, show_progress_bar=True)
        embs = np.array(embs).astype('float32')
        faiss.normalize_L2(embs)
        self.embeddings = embs
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        st.success(f"Ingested {len(self.document_chunks)} chunks.")

    def search_chunks(self, query: str, top_k=5) -> List[CitationOut]:
        qtype, _ = self.classifier.classify_query(query)
        q_emb = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k*3)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.document_chunks):
                continue
            chunk = self.document_chunks[idx]
            base_weight = {'mandatory':1.0,'guideline':0.8,'recommendation':0.6}.get(
                chunk.get('legal_weight','recommendation'),0.6)
            adjusted = float(score) * base_weight
            if qtype == 'compliance' and chunk.get('legal_weight') == 'mandatory':
                adjusted *= 1.3
            results.append((adjusted, idx))
        results.sort(key=lambda x: x[0], reverse=True)
        seen, out = set(), []
        for adjusted, idx in results:
            if idx in seen: continue
            seen.add(idx)
            ch = self.document_chunks[idx]
            out.append(CitationOut(
                id=idx,
                section_path=ch.get('section_path','Unknown'),
                paragraph_range=ch.get('paragraph_range',''),
                legal_weight=ch.get('legal_weight','recommendation'),
                text_snippet=(ch['text'][:500] + '...') if len(ch['text'])>500 else ch['text'],
                score=adjusted
            ))
            if len(out) >= top_k:
                break
        return out

    def _build_prompt(self, query: str, citations: List[CitationOut]) -> str:
        ctx_parts = []
        for c in citations:
            tag = f"[id={c.id} section={c.section_path} para={c.paragraph_range} weight={c.legal_weight}]"
            ctx_parts.append(f"{tag}\n{c.text_snippet}\n")
        context_block = "\n-----\n".join(ctx_parts)
        return (
            "You are an assistant that answers **only** from the provided legal document excerpts. "
            "Always attach the citation IDs used. If answer is not in the context, say: "
            "'I don't know based on the provided document.'\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUERY: {query}\n\n"
            "TASK: Answer in 3-6 sentences. After the answer, list citations in the form "
            "(id=XX, section=YYY, para=ZZ)."
        )

    def generate_answer_with_citations(self, query: str, top_k=5, model="gpt-4o-mini"):
        citations = self.search_chunks(query, top_k)
        prompt = self._build_prompt(query, citations)
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=500,
            temperature=0.0
        )
        answer = resp["choices"][0]["message"]["content"]
        return {
            "query": query,
            "answer": answer,
            "citations": [c.__dict__ for c in citations],
            "prompt": prompt
        }

# --- Streamlit UI ---
st.title("📄 Dynamic Legal Document QA (RAG)")

uploaded_file = st.file_uploader("Upload a legal document (.txt)", type=["txt"])
rag = RAGService()

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    rag.ingest_document(text)

query = st.text_input("Enter your question:")
if st.button("Ask") and query:
    if not rag.document_chunks:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Generating answer..."):
            result = rag.generate_answer_with_citations(query)
        st.subheader("Answer:")
        st.write(result['answer'])
        st.subheader("Citations:")
        for c in result['citations']:
            st.write(c)
