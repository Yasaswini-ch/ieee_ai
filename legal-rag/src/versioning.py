from __future__ import annotations
from typing import List, Dict, Tuple
import difflib


def diff_text(old: str, new: str, context: int = 2) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="", n=context, fromfile="old", tofile="new")
    return "\n".join(diff)


def build_version_map(records: List[Dict]) -> dict[tuple[str, str], dict[str, Dict]]:
    mp: dict[tuple[str, str], dict[str, Dict]] = {}
    for r in records:
        key = (r["doc_id"], r["section_id"])  # version-agnostic key
        mp.setdefault(key, {})[r["version"]] = r
    return mp


def compare_versions(records: List[Dict], doc_id: str, section_id: str, v_old: str, v_new: str) -> str:
    mp = build_version_map(records)
    key = (doc_id, section_id)
    versions = mp.get(key, {})
    a = versions.get(v_old)
    b = versions.get(v_new)
    if not a or not b:
        return "No matching sections/versions found for comparison."
    return diff_text(a.get("text", ""), b.get("text", ""))
