"""
Pure post-processing helper utilities (no CUDA / vLLM side effects).

This module extracts the lightweight parsing / overlap / time utilities from the
former in-process postprocess implementation so tests and other pure-Python
callers can use them directly without importing the compatibility shim
(omoai.pipeline.postprocess) which is deprecated.

Provided functions:
  _parse_time_to_seconds
  _dedup_overlap
  _parse_vietnamese_labeled_text
  join_punctuated_segments  (re-export wrapper to scripts/post.py)

Design:
- Absolutely no torch / vLLM imports.
- Safe to import at process start (no CUDA initialization).
- join_punctuated_segments is delegated to postprocess_utils to keep single
  dynamic loader path for scripts/post.py.

Deprecation Path:
- Tests should migrate to import from this module instead of
  omoai.pipeline.postprocess.
- The compatibility shim will be removed after deprecation window.

Enhancements (parsing robustness):
- Unicode normalization (NFC) + diacritic stripping for header detection.
- Diacritic-insensitive detection of "Điểm chính" (covers all tone-mark variants).
- Relaxed points header: accepts forms like "Điểm chính", "Điểm chính:", "Điểm chính -".
- Title-only case no longer incorrectly infers abstract from the single labeled line.

"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Sequence
from typing import Any


def join_punctuated_segments(
    segments: Sequence[dict],
    join_separator: str = " ",
    paragraph_gap_seconds: float = 3.0,
) -> str:
    """
    Join punctuated segment texts with optional paragraph breaks based on time gaps.

    - Uses segment["text_punct"] when present; falls back to segment["text"] or empty string.
    - Inserts two newlines when the gap between previous end and current start >= paragraph_gap_seconds.
    - Otherwise joins with join_separator.
    """
    parts: list[str] = []
    prev_end: float | None = None
    for seg in segments:
        # parse times robustly from str/float/int
        try:
            start = float(seg.get("start", 0.0))
        except Exception:
            start = 0.0
        try:
            end = float(seg.get("end", start))
        except Exception:
            end = start

        text = str(seg.get("text_punct") or seg.get("text") or "").strip()
        if not text:
            prev_end = end
            continue

        if prev_end is not None and (start - prev_end) >= float(paragraph_gap_seconds):
            # paragraph break
            if parts:
                parts.append("\n\n")
            parts.append(text)
        else:
            if parts and parts[-1] not in ("\n\n",):
                parts.append(join_separator)
            parts.append(text)

        prev_end = end

    # collapse any accidental separator before paragraph breaks
    out = "".join(parts)
    out = out.replace(f"{join_separator}\n\n", "\n\n")
    return out


# ---------------------------------------------------------------------------
# Unicode / diacritic helpers (used only for header detection)
# ---------------------------------------------------------------------------


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip_diacritics(s: str) -> str:
    """
    Remove combining diacritics; used only for header detection (NOT for captured content).
    """
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch))


def _norm_header(s: str) -> str:
    """
    Diacritic-insensitive, lowercase representation for header detection.
    Also normalizes Vietnamese 'đ' / 'Đ' to plain 'd' so headers like 'Tiêu đề'
    become 'tieu de' enabling consistent pattern checks.
    """
    base = _strip_diacritics(_nfc(s)).lower()
    base = base.replace("đ", "d").replace("Đ", "d")
    return base.strip()


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------


def _parse_time_to_seconds(ts: Any) -> float | None:
    """
    Parse flexible timestamp formats into seconds (float).

    Supported examples:
      "01:02:03:045" -> 3723.045  (HH:MM:SS:MS)
      "00:00:01.500" -> 1.5       (HH:MM:SS.mmm)
      "1:02"         -> 62.0      (M:SS)
      2.25           -> 2.25

    Returns None on invalid input.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return float(ts)
        except Exception:
            return None
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None

    # HH:MM:SS:MS (colon separated 4 parts, last is ms)
    if s.count(":") == 3 and "." not in s:
        parts = s.split(":")
        if len(parts) != 4:
            return None
        try:
            hh, mm, ss, ms = parts
            total = (int(hh) * 3600) + (int(mm) * 60) + int(ss) + (int(ms) / 1000.0)
            return float(total)
        except Exception:
            return None

    # HH:MM:SS.mmm or MM:SS.mmm or M:SS
    colon_parts = s.split(":")
    try:
        if len(colon_parts) == 3:
            # HH:MM:SS(.mmm)?
            hh, mm, rest = colon_parts
            if "." in rest:
                ss_str, ms_str = rest.split(".", 1)
                seconds = int(ss_str) + float(f"0.{ms_str}")
            else:
                seconds = int(rest)
            return (int(hh) * 3600) + (int(mm) * 60) + float(seconds)
        if len(colon_parts) == 2:
            # M:SS(.mmm)?
            mm, rest = colon_parts
            if "." in rest:
                ss_str, ms_str = rest.split(".", 1)
                seconds = int(ss_str) + float(f"0.{ms_str}")
            else:
                seconds = int(rest)
            return (int(mm) * 60) + float(seconds)
    except Exception:
        return None

    # Fallback: plain float
    try:
        return float(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Overlap de-duplication
# ---------------------------------------------------------------------------


def _dedup_overlap(prev: str, nxt: str, max_tokens: int = 4) -> str:
    """
    Remove duplicated token overlap at boundary:
      prev: "... xin chao tat ca moi nguoi"
      nxt:  "moi nguoi hom nay the nao"
      -> "hom nay the nao"  (overlap = 'moi nguoi')

    Scans for the longest (<= max_tokens) suffix of prev matching prefix of nxt.
    """
    if not isinstance(prev, str) or not isinstance(nxt, str):
        return nxt
    prev_tokens = prev.strip().split()
    next_tokens = nxt.strip().split()
    max_scan = min(max_tokens, len(prev_tokens), len(next_tokens))
    for k in range(max_scan, 0, -1):
        if prev_tokens[-k:] == next_tokens[:k]:
            return " ".join(next_tokens[k:])
    return nxt


# ---------------------------------------------------------------------------
# Vietnamese / English labeled summary parsing
# ---------------------------------------------------------------------------

# Original regex patterns (retain for content capture)
_LABEL_TITLE_PATTERNS = [
    r"(?i)^\s*tiêu đề:\s*(.+)",
    r"(?i)^\s*title:\s*(.+)",
]
_LABEL_ABSTRACT_PATTERNS = [
    r"(?i)^\s*tóm tắt:\s*(.+)",
    r"(?i)^\s*summary:\s*(.+)",
]
_LABEL_POINTS_PATTERNS = [
    r"(?i)^\s*đi[eê]m\s*ch[ií]nh\s*:?\s*$",
    r"(?i)^\s*main\s*points\s*:?\s*$",
]

# Canonical (diacritic-stripped) headers for points
_CANON_POINTS_HEADERS = {
    "diem chinh",
    "main points",
}


def _is_points_header(line: str) -> bool:
    norm = _norm_header(line)
    norm = re.sub(r"[:\-]\s*$", "", norm).strip()
    return norm in _CANON_POINTS_HEADERS


def _parse_vietnamese_labeled_text(text: str) -> dict[str, Any] | None:
    """
    Parse structured Vietnamese/English labeled summary blocks.

    Returns dict (bullets-only schema):
      {
        "title": str,
        "abstract": str,
        "bullets": List[str]
      }
    or None if no recognized labels.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    original_lines = [ln.rstrip("\r") for ln in raw.splitlines()]
    if not original_lines:
        return None

    title: str = ""
    abstract: str = ""
    points: list[str] = []
    points_mode = False

    debug = bool(os.environ.get("OMOAI_PARSE_DEBUG"))
    debug_log = []

    def dbg(msg: str):
        if debug:
            debug_log.append(msg)

    def match_any(patterns: Sequence[str], line: str):
        for p in patterns:
            m = re.match(p, line)
            if m:
                return m
        return None

    # Pre-compute a diacritic-stripped lower version for header heuristic fallback
    norm_lines = [_norm_header(line) for line in original_lines]

    for idx, line in enumerate(original_lines):
        line_nfc = _nfc(line)
        norm = norm_lines[idx]

        dbg(
            f"[LINE {idx}] raw='{line}' norm='{norm}' mode={'POINTS' if points_mode else 'SCAN'}"
        )

        if not title:
            m = match_any(_LABEL_TITLE_PATTERNS, line_nfc)
            if m:
                title = m.group(1).strip()
                dbg(f"  -> Title matched: {title}")
                continue

        if not abstract:
            m = match_any(_LABEL_ABSTRACT_PATTERNS, line_nfc)
            if m:
                abstract = m.group(1).strip()
                dbg(f"  -> Abstract matched: {abstract[:40]}...")
                continue

        # Enter points mode if header OR heuristic 'diem' + 'chinh' both present (fallback)
        if not points_mode:
            header_match = match_any(
                _LABEL_POINTS_PATTERNS, line_nfc
            ) or _is_points_header(line_nfc)
            heuristic = ("diem" in norm and "chinh" in norm) or (
                "main" in norm and "points" in norm
            )
            if header_match or heuristic:
                points_mode = True
                dbg(
                    f"  -> Points header detected (header_match={bool(header_match)} heuristic={heuristic})"
                )
                continue
        else:
            # Accept common bullet markers: -, *, •, ‣, \u2013, \u2014, or enumerated 1. / 1)
            bullet_m = re.match(r"^\s*(?:[-*•‣\u2013\u2014]|\d+[\.)])\s+(.+)", line_nfc)
            if bullet_m:
                bullet = bullet_m.group(1).strip()
                if bullet:
                    points.append(bullet)
                    dbg(f"  -> Bullet: {bullet}")
                continue
            if line_nfc.strip() == "":
                dbg("  -> Blank inside points block (ignored)")
                continue
            # If a new header appears, terminate points mode
            if match_any(
                _LABEL_TITLE_PATTERNS
                + _LABEL_ABSTRACT_PATTERNS
                + _LABEL_POINTS_PATTERNS,
                line_nfc,
            ) or _is_points_header(line_nfc):
                points_mode = False
                dbg("  -> Points block terminated by new header")
                # do not 'continue' so that new header could be processed in next loop iteration

    if not (title or abstract or points):
        dbg("No labels detected; returning None")
        if debug:
            print("\n".join(debug_log))
        return None

    # Abstract fallback logic
    if not abstract:
        lines_no_blank = [line for line in original_lines if line.strip()]
        single_line_only_title = (
            title and len(lines_no_blank) == 1 and norm_lines[0].startswith("tieu de:")
        )
        if not single_line_only_title:
            para_candidates = [p.strip() for p in raw.split("\n\n") if p.strip()]
            dbg(f"Fallback abstract candidates: {len(para_candidates)}")
            if para_candidates:
                if title and para_candidates and title in para_candidates[0]:
                    if len(para_candidates) > 1:
                        abstract = para_candidates[1]
                        dbg("  -> Abstract inferred from second paragraph")
                if not abstract and para_candidates:
                    abstract = para_candidates[0]
                    dbg("  -> Abstract inferred from first paragraph")
        else:
            dbg("Single-line title-only text; abstract left empty")

    if debug:
        dbg(f"FINAL title='{title}' abstract_len={len(abstract)} points={len(points)}")
        print("\n".join(debug_log))

    return {
        "title": title,
        "abstract": abstract,
        "bullets": points,
    }


__all__ = [
    "_dedup_overlap",
    "_parse_time_to_seconds",
    "_parse_vietnamese_labeled_text",
    "join_punctuated_segments",
]
