"""
Shared JSON Schema Extraction, Heuristic Repair, and Single-Retry Utilities.
Used across NLP SLU (Stage 3) and Role Interviewer (Stage 4) to recover malformed
outputs from small quantized local models without code duplication.
"""

import ast
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _extract_balanced_braces(text: str) -> Optional[str]:
    """Extracts the first balanced {...} substring, tolerating trailing truncation."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    # Close truncated object cleanly when small models hit token budget
    if depth > 0:
        return text[start:] + ("}" * depth)
    return None


def extract_and_repair_json(
    raw_text: str,
    expected_keys: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool, List[str]]:
    """
    Attempts strict JSON parsing first; if invalid, applies deterministic syntax repairs:
    1) Markdown fence removal
    2) Prose wrapper / truncated brace extraction
    3) Trailing comma removal & Python bool/None/unquoted-key normalization
    Returns (parsed_dict_or_none, was_repaired, applied_fixes).
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, False, []

    stripped = raw_text.strip()
    try:
        direct = json.loads(stripped)
        if isinstance(direct, dict):
            return direct, False, []
    except Exception:
        pass

    fixes: List[str] = []
    candidate = stripped

    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", candidate, flags=re.IGNORECASE).strip()
        fixes.append("strip_markdown_fences")

    balanced = _extract_balanced_braces(candidate)
    if balanced and balanced != candidate:
        candidate = balanced
        fixes.append("extract_brace_object")
    elif balanced and balanced.endswith("}") and not stripped.rstrip().endswith("}"):
        candidate = balanced
        fixes.append("close_truncated_braces")

    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("remove_trailing_commas")

    # Convert unquoted top/nested keys after { or ,
    quoted_keys = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate)
    if quoted_keys != candidate:
        candidate = quoted_keys
        fixes.append("quote_unquoted_keys")

    py_lit = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", candidate)))
    if py_lit != candidate:
        candidate = py_lit
        fixes.append("normalize_python_literals")

    for cand in (candidate, py_lit):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, True, fixes
        except Exception:
            continue

    # Final safe literal fallback for single-quoted Python-style dicts
    try:
        obj = ast.literal_eval(balanced or stripped)
        if isinstance(obj, dict):
            fixes.append("python_dict_literal")
            return obj, True, fixes
    except Exception:
        pass

    return None, False, fixes


def _has_expected_keys(obj: Optional[Dict[str, Any]], expected_keys: Optional[Sequence[str]]) -> bool:
    if not isinstance(obj, dict):
        return False
    if not expected_keys:
        return bool(obj)
    return all(k in obj for k in expected_keys)


def repair_and_parse_with_retry(
    raw_text: str,
    expected_keys: Optional[Sequence[str]] = None,
    enable_repair: bool = True,
    retry_fn: Optional[Callable[[str, Sequence[str]], str]] = None,
) -> Dict[str, Any]:
    """
    Parses LLM JSON output, optionally applies syntax repair and at most 1 constrained retry.
    Returns structured status dict with `parsed`, `raw_valid`, `schema_valid`, `repaired`, `retried`, `fixes`.
    """
    stripped = (raw_text or "").strip()
    raw_valid = False
    try:
        raw_obj = json.loads(stripped)
        raw_valid = _has_expected_keys(raw_obj, expected_keys)
        if isinstance(raw_obj, dict) and not enable_repair:
            return {
                "parsed": raw_obj if _has_expected_keys(raw_obj, expected_keys) or not expected_keys else raw_obj,
                "raw_valid": raw_valid,
                "schema_valid": _has_expected_keys(raw_obj, expected_keys),
                "repaired": False,
                "retried": False,
                "fixes": [],
            }
    except Exception:
        raw_obj = None

    if not enable_repair:
        return {
            "parsed": None,
            "raw_valid": False,
            "schema_valid": False,
            "repaired": False,
            "retried": False,
            "fixes": [],
        }

    parsed, repaired, fixes = extract_and_repair_json(raw_text, expected_keys=expected_keys)
    retried = False
    if (parsed is None or (expected_keys and not _has_expected_keys(parsed, expected_keys))) and retry_fn is not None:
        retried = True
        retry_raw = retry_fn(raw_text, expected_keys or [])
        parsed2, rep2, fixes2 = extract_and_repair_json(retry_raw, expected_keys=expected_keys)
        if parsed2 is not None and (_has_expected_keys(parsed2, expected_keys) or parsed is None):
            parsed = parsed2
            repaired = repaired or rep2 or True
            fixes.extend(["schema_repair_retry"] + fixes2)

    return {
        "parsed": parsed,
        "raw_valid": raw_valid,
        "schema_valid": _has_expected_keys(parsed, expected_keys),
        "repaired": repaired,
        "retried": retried,
        "fixes": fixes,
    }
