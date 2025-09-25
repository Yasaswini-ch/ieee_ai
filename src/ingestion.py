from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable
import io
import json
import re

from pypdf import PdfReader
from docx import Document


@dataclass
class Section:
    doc_id: str
    section_id: str
    version: str
    title: str
    text: str


_heading_re = re.compile(r"^(\d+\.|[A-Z]\.|[IVX]+\.)?\s*[A-Za-z].{0,80}$")


def _split_by_headings(text: str) -> List[tuple[str, str]]:
    lines = [l.strip() for l in text.splitlines()]
    sections: List[tuple[str, str]] = []
    title = ""
    buf: List[str] = []
    for line in lines:
        if line and _heading_re.match(line) and len(line) <= 120:
            if buf or title:
                sections.append((title or "Section", "\n".join(buf).strip()))
                buf = []
            title = line
        else:
            buf.append(line)
    if buf or title:
        sections.append((title or "Section", "\n".join(buf).strip()))
    return [(t if t else "Section", b) for t, b in sections if b.strip()]


def ingest_pdf(file_bytes: bytes, doc_id: str, version: str) -> List[Section]:
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = []
    for page in reader.pages:
        try:
            full_text.append(page.extract_text() or "")
        except Exception:
            full_text.append("")
    text = "\n".join(full_text)
    chunks = _split_by_headings(text)
    sections: List[Section] = []
    for idx, (title, body) in enumerate(chunks, start=1):
        sections.append(Section(doc_id=doc_id, section_id=f"s{idx}", version=version, title=title, text=body))
    if not sections:  # fallback: page-based
        for i, page_text in enumerate(full_text, start=1):
            sections.append(Section(doc_id=doc_id, section_id=f"p{i}", version=version, title=f"Page {i}", text=page_text))
    return sections


def ingest_docx(file_bytes: bytes, doc_id: str, version: str) -> List[Section]:
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    paragraphs = [p.text.strip() for p in doc.paragraphs]
    text = "\n".join(paragraphs)
    chunks = _split_by_headings(text)
    sections: List[Section] = []
    for idx, (title, body) in enumerate(chunks, start=1):
        sections.append(Section(doc_id=doc_id, section_id=f"s{idx}", version=version, title=title, text=body))
    if not sections:
        # fallback: chunk by ~800 words
        words = text.split()
        step = 800
        for i in range(0, len(words), step):
            body = " ".join(words[i:i+step])
            sections.append(Section(doc_id=doc_id, section_id=f"w{i//step+1}", version=version, title=f"Chunk {i//step+1}", text=body))
    return sections


def ingest_jsonl(file_bytes: bytes) -> List[Section]:
    sections: List[Section] = []
    for line in io.BytesIO(file_bytes).read().decode("utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        doc_id = obj.get("doc_id", "doc")
        version = obj.get("version", "v1")
        title = obj.get("title", "Section")
        text = obj.get("text", "")
        section_id = obj.get("section_id") or f"s{len(sections)+1}"
        if text:
            sections.append(Section(doc_id=doc_id, section_id=section_id, version=version, title=title, text=text))
    return sections


def to_records(sections: Iterable[Section]) -> List[Dict]:
    return [
        {
            "doc_id": s.doc_id,
            "section_id": s.section_id,
            "version": s.version,
            "title": s.title,
            "text": s.text,
        }
        for s in sections
    ]
