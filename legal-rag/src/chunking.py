from __future__ import annotations
from typing import List, Dict


def simple_chunk(records: List[Dict], max_chars: int = 2000, overlap: int = 200) -> List[Dict]:
    out: List[Dict] = []
    for rec in records:
        text = rec["text"].strip()
        if len(text) <= max_chars:
            out.append(rec)
            continue
        start = 0
        idx = 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end]
            new = dict(rec)
            new["text"] = chunk_text
            new["section_id"] = f"{rec['section_id']}_c{idx}"
            new["title"] = f"{rec['title']} (part {idx})"
            out.append(new)
            if end == len(text):
                break
            start = end - overlap
            idx += 1
    return out
