import argparse
import gc
import json
import os
import re
import textwrap
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from omoai.logging_system.logger import get_logger, setup_logging
from omoai.pipeline.postprocess_core_utils import (
    _dedup_overlap as _dedup_overlap,
)
from omoai.pipeline.postprocess_core_utils import (
    _parse_time_to_seconds as _parse_time_to_seconds,
)

# Import Policy:
# This module is now import-safe for vLLM child processes spawned under the 'spawn'
# method. We only wanted to block ad-hoc in-process library usage that would
# initialize vLLM/CUDA in the API parent before the dedicated subprocess.
#
# Instead of raising ImportError (which breaks vLLM internal spawned workers),
# we set a flag that can be inspected if needed. Dynamic imports from pipeline
# utilities should be removed; those callers should rely on the subprocess path.
SAFE_SCRIPT_IMPORT = (__name__ != "__main__")

try:  # optional torch for CUDA cache clearing; not required in dry-run
    import torch  # type: ignore
except ImportError:  # pragma: no cover - keep runnable without torch
    torch = None  # type: ignore

# Initialize unified logging for the script
setup_logging()
logger = get_logger(__name__)

# Environment flag for debug GPU memory clearing
DEBUG_EMPTY_CACHE = os.environ.get("OMOAI_DEBUG_EMPTY_CACHE", "false").lower() == "true"

try:  # optional streaming JSON parser for very large ASR files
    import ijson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ijson = None  # type: ignore

try:  # optional progress bar
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - keep runnable without tqdm
    tqdm = None  # type: ignore

# Defer vLLM import until inside build_llm() after multiprocessing start method is safely set.
VLLM_AVAILABLE: bool | None = None  # resolved lazily in main()

# Placeholder SamplingParams (never used before real vLLM import; kept to satisfy type references)
class SamplingParams:  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def load_asr_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_asr_top_level(path: Path) -> dict[str, Any]:
    """Read only top-level keys of an ASR JSON using ijson if available.

    Skips the heavy `segments` array to reduce memory footprint.
    Fallback to full load if ijson is not available or on error.
    """
    if ijson is None:
        return load_asr_json(path)
    top: dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            for key, value in ijson.kvitems(f, ""):
                if key == "segments":
                    continue
                top[key] = value
    except (ValueError, KeyError, TypeError):
        return load_asr_json(path)
    return top


def iter_asr_segments(path: Path):
    """Yield segments one-by-one using ijson if available; otherwise load all."""
    if ijson is None:
        asr = load_asr_json(path)
        for seg in asr.get("segments", []) or []:
            yield seg
        return
    try:
        with open(path, "rb") as f:
            for seg in ijson.items(f, "segments.item"):
                yield seg
    except (ValueError, KeyError, TypeError):
        asr = load_asr_json(path)
        for seg in asr.get("segments", []) or []:
            yield seg


def get_tokenizer(llm: Any):
    return llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else llm.tokenizer


def apply_chat_template(llm: Any, messages: list[dict[str, str]]) -> str:
    """Build a prompt for chat-style models with a safe fallback.

    Tries tokenizer.apply_chat_template first. If unavailable or failing,
    constructs a simple role-tagged prompt that works for base models too.
    """
    tokenizer = get_tokenizer(llm)
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        if tmpl:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    except (ValueError, KeyError, TypeError):
        pass
    # Fallback: simple role-tagged format
    parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append("[SYSTEM]\n" + content)
        elif role == "user":
            parts.append("[USER]\n" + content)
        elif role == "assistant":
            parts.append("[ASSISTANT]\n" + content)
        else:
            parts.append(content)
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def generate_chat(llm: Any, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
    prompt = apply_chat_template(llm, messages)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


def generate_chat_batch(llm: Any, list_of_messages: list[list[dict[str, str]]], temperature: float, max_tokens: int) -> list[str]:
    """Batched chat generation for multiple prompts to improve throughput."""
    if not list_of_messages:
        return []
    prompts = [apply_chat_template(llm, m) for m in list_of_messages]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


def _compute_input_token_limit(max_model_len: int, prompt_overhead_tokens: int = 128, output_margin_tokens: int = 128) -> int:
    """Conservative upper bound for input tokens so total prompt + output fits model context."""
    return max(256, int(max_model_len) - int(prompt_overhead_tokens) - int(output_margin_tokens))


def _split_text_by_token_budget(llm: Any, text: str, max_input_tokens: int) -> list[str]:
    """Split text into chunks whose tokenized length is <= max_input_tokens.

    Uses the model tokenizer to split exactly on token boundaries to guarantee
    reversibility when concatenated. Falls back to sentence/char-based splitting
    if tokenization fails for any reason.
    """
    if not text:
        return []
    try:
        tokenizer = get_tokenizer(llm)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_input_tokens:
            return [text]
        chunks: list[str] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + max_input_tokens, n)
            piece = tokenizer.decode(tokens[start:end])
            chunks.append(piece)
            start = end
        return chunks
    except (ValueError, KeyError, TypeError):
        # Fallback on sentence/char splitter with an approximate size
        approx_chars = max(500, int(max_input_tokens * 4))
        return split_text_into_chunks(text, max_chars=approx_chars, overlap_sentences=1)


def _split_text_by_token_budget_with_counts(llm: Any, text: str, max_input_tokens: int) -> list[tuple[str, int]]:
    """Return list of (chunk_text, token_count) within token budget to avoid re-encoding."""
    if not text:
        return []
    try:
        tokenizer = get_tokenizer(llm)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_input_tokens:
            return [(text, len(tokens))]
        out: list[tuple[str, int]] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + max_input_tokens, n)
            piece_tokens = tokens[start:end]
            piece = tokenizer.decode(piece_tokens)
            out.append((piece, len(piece_tokens)))
            start = end
        return out
    except (ValueError, KeyError, TypeError):
        approx_chars = max(500, int(max_input_tokens * 4))
        chunks = split_text_into_chunks(text, max_chars=approx_chars, overlap_sentences=1)
        return [(c, max(1, len(c) // 3)) for c in chunks]


def punctuate_text_with_splitting(
    llm: Any,
    text: str,
    system_prompt: str,
    max_model_len: int,
    temperature: float = 0.0,
    batch_prompts: int = 1,
    show_progress: bool = False,
    user_prompt: str | None = None,
) -> str:
    """Punctuate text safely by splitting into token-budgeted chunks if needed.

    Ensures each generation request keeps input length within model limits.
    """
    if not text:
        return ""
    try:
        # tokenizer = get_tokenizer(llm)  # Unused variable removed
        pass
    except (ValueError, KeyError, TypeError):
        pass

    input_limit = _compute_input_token_limit(max_model_len)
    parts_counts = _split_text_by_token_budget_with_counts(llm, text, input_limit)
    if not parts_counts:
        return ""
    per_piece_max: list[int] = []
    for _, tokens_in in parts_counts:
        per_piece_max.append(max(64, min(int(max_model_len) - 64, int(tokens_in) + 128)))

    pieces_only = [p for p, _ in parts_counts]
    if batch_prompts <= 1:
        iterable = pieces_only
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, total=len(pieces_only), desc="Punctuating")
        out_pieces: list[str] = []
        for idx, piece in enumerate(iterable):
            if not piece.strip():
                continue
            out_pieces.append(
                punctuate_text(
                    llm,
                    piece,
                    system_prompt,
                    max_tokens=per_piece_max[idx],
                    temperature=temperature,
                    user_prompt=user_prompt,
                ),
            )
        return " ".join(p.strip() for p in out_pieces if p is not None).strip()

    # Batched path
    out_texts: list[str] = [""] * len(pieces_only)
    batched_indices: list[int] = list(range(len(pieces_only)))
    start = 0
    while start < len(batched_indices):
        end = min(start + int(max(1, batch_prompts)), len(batched_indices))
        idxs = batched_indices[start:end]
        group_max_tokens = max(per_piece_max[i] for i in idxs)
        list_of_messages: list[list[dict[str, str]]] = []
        for i in idxs:
            list_of_messages.append([
                {"role": "system", "content": system_prompt},
                ({"role": "user", "content": f"{user_prompt}\n\n{pieces_only[i]}" } if user_prompt else {"role": "user", "content": pieces_only[i] }),
            ])
        group_outs = generate_chat_batch(
            llm,
            list_of_messages,
            temperature=temperature,
            max_tokens=int(group_max_tokens),
        )
        for local_i, gi in enumerate(idxs):
            out_texts[gi] = group_outs[local_i] or ""
        start = end
    return " ".join(s.strip() for s in out_texts if s is not None).strip()

def punctuate_text(llm: Any, text: str, system_prompt: str, max_tokens: int | None = None, temperature: float = 0.0, user_prompt: str | None = None) -> str:
    if not text:
        return ""
    system = system_prompt
    # If a user_prompt is provided, prepend it to the user's content
    if user_prompt:
        user_content = f"{user_prompt}\n\n{text}"
    else:
        user_content = text
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    # Heuristic: allow output tokens ~= input tokens + margin
    if max_tokens is None:
        try:
            tokenizer = get_tokenizer(llm)
            max_tokens = min(4096, len(tokenizer.encode(user_content)) + 64)
        except (ValueError, KeyError, TypeError):
            max_tokens = 1024
    return generate_chat(llm, messages, temperature=temperature, max_tokens=int(max_tokens))


def _parse_structured_summary(text: str) -> dict:
    """
    Parse a multi-line structured summary string into a dict with keys:
    - title: str
    - abstract: str
    - bullets: List[str]
    Robustly handles different label names and bullet styles.
    """
    if not text:
        return {"title": "", "abstract": "", "bullets": []}
    lines = [line.rstrip() for line in text.splitlines()]
    title = ""
    abstract_lines: list[str] = []
    points: list[str] = []
    mode = "auto"  # auto, title, summary, points

    # Normalize and detect sections by common headings
    for _i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            # preserve paragraph breaks in abstract
            if mode in ("auto", "summary"):
                abstract_lines.append("")
            continue
        lower = line.lower()
        # Common heading markers (English and Vietnamese)
        if lower.startswith("title:") or lower.startswith("# ") or lower.startswith("title -") or lower.startswith("tiêu đề:"):
            title = line.split(":", 1)[-1].strip() if ":" in line else line.lstrip("# ").strip()
            mode = "summary"
            continue
        if lower.startswith("summary:") or lower.startswith("abstract:") or lower.startswith("summary -") or lower.startswith("tóm tắt:"):
            mode = "summary"
            # take text after colon if present
            after = line.split(":", 1)
            if len(after) > 1 and after[1].strip():
                abstract_lines.append(after[1].strip())
            continue
        if lower.startswith("points:") or lower.startswith("bullets:") or lower.startswith("key points:") or lower.startswith("items:") or lower.startswith("điểm chính:"):
            mode = "points"
            # take trailing content on same line after colon
            after = line.split(":", 1)
            if len(after) > 1 and after[1].strip():
                # may be a single bullet on same line
                pts = after[1].strip()
                # split by ';' if present
                if ";" in pts:
                    points.extend([p.strip() for p in pts.split(";") if p.strip()])
                else:
                    points.append(pts)
            continue
        # Bullet detection
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• ") or (line[0:2].isdigit() and line[2:3] == "."):
            bullet = line.lstrip("-*• ").strip()
            points.append(bullet)
            mode = "points"
            continue
        # If in points mode but line doesn't start with bullet, treat as continuation of last bullet
        if mode == "points" and points:
            points[-1] = f"{points[-1]} {line}"
            continue
        # Otherwise treat as summary/abstract content
        abstract_lines.append(line)
        mode = "summary"

    # If no explicit title but first line looks like a short heading, use it
    if not title and len(abstract_lines) > 0 and len(abstract_lines[0].split()) <= 10:
        # pop as title if next line starts a new paragraph
        candidate = abstract_lines.pop(0)
        title = candidate

    abstract = "\n".join([line for line in abstract_lines]).strip()
    return {"title": title, "abstract": abstract, "bullets": points}


def summarize_text(llm: Any, text: str, system_prompt: str, temperature: float = 0.2, user_prompt: str | None = None) -> dict:
    """
    Call the LLM to summarize and parse the multi-line structured string into a dict.
    Returns a dict with keys: title, abstract, bullets.
    """
    if not text:
        return {"title": "", "abstract": "", "bullets": []}
    system = system_prompt
    # Prepend user_prompt to the user content if provided
    if user_prompt:
        user_content = f"{user_prompt}\n\n{text}"
    else:
        user_content = text
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    content = generate_chat(llm, messages, temperature=temperature, max_tokens=800)
    raw = (content or "").strip()
    parsed = _parse_structured_summary(raw)
    # Include the raw model content for downstream consumers
    parsed_with_raw = dict(parsed)
    parsed_with_raw["raw"] = raw
    return parsed_with_raw
def prepare_timestamp_context(segments: list[dict]) -> str:
    """
    Create a numbered and timestamped transcript string from ASR segments.
    """
    def _format_timestamp(seconds: float) -> str:
        """Format seconds into [MM:SS] string for LLM prompt."""
        if seconds is None:
            seconds = 0.0
        try:
            seconds = max(0.0, float(seconds))
        except (TypeError, ValueError):
            seconds = 0.0
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return f"[{minutes:02d}:{seconds:02d}]"

    timestamped_transcript = []
    word_counter = 1
    for segment in segments:
        # Look for word-level timestamps in 'word_segments' (from alignment) or 'words'
        words = segment.get("word_segments") or segment.get("words")
        if not words:
            continue
        for word in words:
            if "start" in word and "end" in word:
                start_time = _format_timestamp(word['start'])
                timestamped_transcript.append(
                    f"{word_counter}. {start_time} {word['word']}"
                )
                word_counter += 1
    return "\n".join(timestamped_transcript)


def prepare_sentence_timestamp_lines(segments: list[dict]) -> str:
    """Create sentence-level timestamped lines from ASR segments.

    Format: [HH:MM:SS] sentence_text
    Uses first word start if available; otherwise segment start.
    Dramatically reduces tokens vs. per-word format.
    """
    lines: list[str] = []
    for seg in segments:
        words = seg.get("word_segments") or seg.get("words") or []
        if words:
            first = next((w for w in words if "start" in w), None)
            start_s = float(first["start"]) if first else float(seg.get("start", 0.0) or 0.0)
        else:
            start_s = float(seg.get("start", 0.0) or 0.0)

        # Prefer punctuated if available, else raw
        text = (seg.get("text_punct") or seg.get("text_raw") or seg.get("text") or "").strip()
        if not text:
            continue

        hh = int(start_s // 3600)
        mm = int((start_s % 3600) // 60)
        ss = int(start_s % 60)
        ts = f"[{hh:02d}:{mm:02d}:{ss:02d}]"
        lines.append(f"{ts} {text}")
    return "\n".join(lines)


def pack_sentence_lines_to_budget(llm: Any, lines_text: str, max_model_len: int, out_margin: int = 128, overhead: int = 64) -> list[str]:
    """Pack sentence lines into chunks that fit token budget.

    Returns list of chunk strings, each fitting within the input limit.
    """
    if not lines_text.strip():
        return []

    try:
        tokenizer = get_tokenizer(llm)
        input_limit = max(256, int(max_model_len) - overhead - out_margin)

        # Split by lines to preserve sentence boundaries
        lines = [ln for ln in lines_text.splitlines() if ln.strip()]
        if not lines:
            return []

        chunks: list[str] = []
        cur: list[str] = []
        cur_tokens = 0

        for ln in lines:
            try:
                t = len(tokenizer.encode(ln))
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Tokenization failed for line: {e}")
                # Fallback estimation if tokenization fails
                t = max(1, len(ln) // 3)

            if cur_tokens and cur_tokens + t > input_limit:
                if cur:
                    chunks.append("\n".join(cur))
                cur, cur_tokens = [], 0
            cur.append(ln)
            cur_tokens += t

        if cur:
            chunks.append("\n".join(cur))
        return chunks
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Token packing failed: {e}")
        # Fallback: split by character count
        approx_chars = max(500, int(max_model_len * 3))  # rough char-to-token ratio
        return split_text_into_chunks(lines_text, max_chars=approx_chars, overlap_sentences=0)


def generate_timestamped_summary(llm: Any, timestamped_transcript: str, system_prompt: str) -> str:
    """
    Generate a summary with timestamps from the LLM.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please create a summary of the following transcript:\n\n{timestamped_transcript}"},
    ]
    return generate_chat(llm, messages, temperature=0.2, max_tokens=800)
def parse_llm_response(response_text: str, include_raw: bool = False) -> dict:
    """
    Parse the LLM response to extract the summary and timestamps.

    Supports common formats where each line is:
        [HH:MM:SS] Topic text
    or
        [MM:SS] Topic text

    Falls back to a more permissive parser if lines are not per-timestamp.
    """
    import re

    def _parse_time_to_seconds(time_str: str) -> float:
        """Convert [MM:SS] or [HH:MM:SS] string to seconds."""
        try:
            parts = time_str.strip('[]').split(':')
            if len(parts) == 2:  # [MM:SS]
                m, s = map(int, parts)
                return m * 60 + s
            if len(parts) == 3:  # [HH:MM:SS]
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
        except ValueError:
            pass
        return 0.0

    def _format_timestamp(seconds: float) -> str:
        """Format seconds into [HH:MM:SS] string for final output."""
        if seconds is None:
            seconds = 0.0
        try:
            seconds = max(0.0, float(seconds))
        except (TypeError, ValueError):
            seconds = 0.0
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

    # First pass: line-based parser "[time] text" (most common)
    timestamps: list[dict] = []
    lines = (response_text or "").splitlines()
    line_pattern = re.compile(r"^\s*\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.+?)\s*$")
    matched_any = False
    for ln in lines:
        m = line_pattern.match(ln)
        if not m:
            continue
        matched_any = True
        # Reconstruct original time string for robust parsing
        hh = m.group(1)
        mm = m.group(2)
        ss = m.group(3)
        time_str = f"[{hh}:{mm}:{ss}]" if ss is not None else f"[{hh}:{mm}]"
        start_time = _parse_time_to_seconds(time_str)
        text = (m.group(4) or "").strip()
        if text:
            timestamps.append({"text": text, "start": start_time})

    if matched_any and timestamps:
        # Standardize summary_text to HH:MM:SS + text, one per line
        summary_text = "\n".join(
            f"{_format_timestamp(item['start'])} {item['text']}" for item in timestamps
        ).strip()
        result = {"summary_text": summary_text, "timestamps": timestamps}
        if include_raw and (response_text or "").strip() != (summary_text or "").strip():
            result["raw"] = response_text
        return result

    # Fallback: permissive parser — find bracketed times anywhere and take following text on the same line
    timestamps = []
    summary_text = response_text or ""
    pat = re.compile(r"\[(\d{2}):(\d{2})(?::(\d{2}))?\]")
    for m in pat.finditer(summary_text):
        start_time = _parse_time_to_seconds(m.group(0))
        # Take text to end-of-line
        line_end = summary_text.find("\n", m.end())
        if line_end == -1:
            line_end = len(summary_text)
        segment = summary_text[m.end():line_end]
        text = segment.strip()
        timestamps.append({"text": text, "start": start_time})
        # Normalize time format inside summary_text
        summary_text = summary_text.replace(m.group(0), _format_timestamp(start_time))

    result = {"summary_text": summary_text.strip(), "timestamps": timestamps}
    if include_raw and (response_text or "").strip() != (result["summary_text"] or "").strip():
        result["raw"] = response_text
    return result


"""Use shared helpers from core utils to avoid duplication."""


def join_punctuated_segments(
    segments: list[dict[str, Any]],
    join_separator: str = " ",
    use_vi_sentence_segmenter: bool = False,
) -> str:
    """Join per-segment punctuated texts into a coherent transcript.

    Strategy:
    - Append segment by segment into a buffer, joining with a separator
    - Emit complete sentences when sentence-ending punctuation is detected
    - Keep incomplete tail to be continued by the next segment
    - Deduplicate small overlaps at joins
    """
    extractor_fn: Callable[[str], tuple[list[str], str]] | None = None
    if use_vi_sentence_segmenter:
        with suppress(Exception):
            from underthesea import sent_tokenize  # type: ignore

            def _vi_extractor(text: str) -> tuple[list[str], str]:
                sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
                if not sents:
                    return [], text
                ends_with_term = bool(re.search(r"[\.\!\?…][”\)\"»\]]*$", text.strip()))
                if ends_with_term:
                    return sents, ""
                return sents[:-1], sents[-1] if sents else ""
            extractor_fn = _vi_extractor
    if extractor_fn is None:
        # Use a forward-search matcher to avoid variable-length lookbehind
        sentence_end_pattern = re.compile(r"[\.\!\?…][”\)\"»\]]*\s+")

        def _default_extractor(text: str) -> tuple[list[str], str]:
            sentences: list[str] = []
            last_cut = 0
            for m in sentence_end_pattern.finditer(text):
                end_idx = m.end()
                sent = text[last_cut:end_idx].strip()
                if sent:
                    sentences.append(sent)
                last_cut = end_idx
            tail = text[last_cut:]
            return sentences, tail
        extractor_fn = _default_extractor

    out_sentences: list[str] = []
    buffer = ""
    last_end: float | None = None

    for seg in segments:
        part = (seg.get("text_punct") or seg.get("text_raw") or "").strip()
        if not part:
            end_sec = _parse_time_to_seconds(seg.get("end"))
            last_end = end_sec if end_sec is not None else last_end
            continue

        # Deduplicate small overlaps
        part = _dedup_overlap(buffer, part, max_tokens=8)

        # Append to rolling buffer
        buffer = (buffer + (join_separator if buffer else "") + part).strip()
        end_sec2 = _parse_time_to_seconds(seg.get("end"))
        last_end = end_sec2 if end_sec2 is not None else last_end

        # Emit complete sentences from buffer; keep last tail incomplete
        if extractor_fn is None:
            # Should not happen; guard for linter and robustness
            sentences, tail = [], buffer
        else:
            sentences, tail = extractor_fn(buffer)
        if sentences:
            out_sentences.extend(s for s in sentences if s.strip())
            buffer = tail

    # Flush remaining tail
    if buffer.strip():
        out_sentences.append(buffer.strip())

    return " ".join(out_sentences)


def _count_tokens(llm: Any, text: str) -> int:
    try:
        tokenizer = get_tokenizer(llm)
        return len(tokenizer.encode(text))
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Token counting failed: {e}")
        # Fallback heuristic to avoid zeros in decision logic
        return max(1, len(text) // 3)


def _split_words(text: str) -> list[str]:
    return text.strip().split()


# --- Token-preserving alignment helpers ---
_PUNCT_CHARS = set(list(".,!?…:;\"'()[]{}«»\"\"''—-"))


def _tokenize_words_and_punct(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        if ch in _PUNCT_CHARS:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _is_word(tok: str) -> bool:
    if not tok:
        return False
    if tok in _PUNCT_CHARS:
        return False
    return True


def _join_tokens_with_spacing(tokens: list[str]) -> str:
    out: list[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue
        if tok in _PUNCT_CHARS:
            # Attach punctuation to previous token without extra space
            out[-1] = (out[-1] + tok)
        else:
            out.append(" " + tok)
    return "".join(out)


def _force_preserve_with_alignment(original_text: str, llm_text: str, adopt_case: bool = True) -> str:
    """Keep original words order/content; adopt punctuation and optionally case from LLM.

    Corrected behavior:
    - When LLM inserts or replaces words, prefer the LLM words (respecting adopt_case).
    - When LLM deletes words, remove those original words.
    - Always preserve punctuation coming from the LLM, mapped to the LLM word it follows.
    """
    if not original_text:
        return llm_text or ""

    orig_words = [t for t in _tokenize_words_and_punct(original_text) if _is_word(t)]
    llm_tokens = _tokenize_words_and_punct(llm_text or "")
    llm_words = [t for t in llm_tokens if _is_word(t)]

    if not orig_words:
        return llm_text or ""
    if not llm_words:
        return original_text

    # Map punctuation to the word it FOLLOWS.
    # Index -1: leading punctuation. Index i: punctuation after word i.
    punct_after: dict[int, list[str]] = {-1: []}
    for i in range(len(llm_words)):
        punct_after[i] = []

    llw_idx = -1
    for tok in llm_tokens:
        if _is_word(tok):
            llw_idx += 1
        else:
            punct_after[llw_idx].append(tok)

    a = [w.lower() for w in orig_words]
    b = [w.lower() for w in llm_words]
    sm = SequenceMatcher(None, a, b, autojunk=False)

    result_tokens: list[str] = []
    # leading punctuation from LLM
    result_tokens.extend(punct_after.get(-1, []))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # words matched: preserve original content, optionally adopt LLM case
            for oi, lj in zip(range(i1, i2), range(j1, j2), strict=False):
                orig_word = orig_words[oi]
                llm_word = llm_words[lj]
                if adopt_case:
                    # use LLM casing for matched words
                    out_word = llm_word
                else:
                    out_word = orig_word
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))
        elif tag == "replace":
            # LLM replaced original words: prefer LLM words (respect adopt_case)
            for lj in range(j1, j2):
                llm_word = llm_words[lj]
                out_word = llm_word if adopt_case else llm_word.lower()
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))
        elif tag == "delete":
            # LLM deleted some original words: skip the original words (do not emit them)
            # No addition to result_tokens here.
            continue
        elif tag == "insert":
            # LLM inserted new words: emit them (respect adopt_case)
            for lj in range(j1, j2):
                llm_word = llm_words[lj]
                out_word = llm_word if adopt_case else llm_word.lower()
                result_tokens.append(out_word)
                result_tokens.extend(punct_after.get(lj, []))

    return _join_tokens_with_spacing(result_tokens).strip()


def _distribute_punct_to_segments(
    punctuated_text: str,
    segments: list[dict[str, Any]],
    keep_nonempty_segments: bool = False,
) -> list[dict[str, Any]]:
    """Distribute a single punctuated text back to segments using fuzzy word alignment.

    Handles word count mismatches by using proportional distribution and fuzzy matching.
    """
    if not segments:
        return []
    if not punctuated_text.strip():
        # Log and optionally fallback
        print("[punct-align] punctuated_text empty; mapping empty outputs to segments")
        out_segments: list[dict[str, Any]] = []
        for s in segments:
            raw = (s.get("text_raw") or s.get("text") or "").strip()
            if keep_nonempty_segments and raw:
                piece = _add_basic_punctuation(raw)
                out_segments.append({**s, "text_punct": piece})
            else:
                out_segments.append({**s, "text_punct": ""})
        return out_segments

    # Get original segments with text
    text_segments = [(i, s) for i, s in enumerate(segments) if (s.get("text_raw") or s.get("text") or "").strip()]
    if not text_segments:
        return [dict(s) for s in segments]

    # Create concatenated original text
    original_concat = " ".join((s.get("text_raw") or s.get("text") or "").strip() for _, s in text_segments)
    original_words = _split_words(original_concat)
    punct_words = _split_words(punctuated_text)

    # If word counts match exactly, use precise distribution
    if len(original_words) == len(punct_words):
        return _distribute_exact_match(punctuated_text, segments, keep_nonempty_segments=keep_nonempty_segments)

    # Handle word count mismatch with fuzzy alignment
    return _distribute_fuzzy_match(punctuated_text, segments, original_concat, keep_nonempty_segments=keep_nonempty_segments)


def _distribute_exact_match(
    punctuated_text: str,
    segments: list[dict[str, Any]],
    keep_nonempty_segments: bool = False,
) -> list[dict[str, Any]]:
    """Distribute when word counts match exactly."""
    seg_word_counts = [len(_split_words((s.get("text_raw") or s.get("text") or "").strip())) for s in segments]
    words_punct = _split_words(punctuated_text)

    out_segments: list[dict[str, Any]] = []
    cursor = 0
    for s, cnt in zip(segments, seg_word_counts, strict=False):
        if cnt == 0:
            out_segments.append({**s, "text_punct": ""})
        else:
            piece_words = words_punct[cursor : cursor + cnt]
            cursor += cnt
            piece = " ".join(piece_words).strip()
            if not piece and keep_nonempty_segments:
                raw = (s.get("text_raw") or s.get("text") or "").strip()
                piece = _add_basic_punctuation(raw) if raw else ""
            out_segments.append({**s, "text_punct": piece})
    return out_segments


def _distribute_fuzzy_match(
    punctuated_text: str,
    segments: list[dict[str, Any]],
    original_concat: str,
    keep_nonempty_segments: bool = False,
) -> list[dict[str, Any]]:
    """Distribute using a word-level alignment between original_concat and punctuated_text.

    Strategy:
    - Tokenize both original_concat and punctuated_text into words and punctuation.
    - Use SequenceMatcher on lowercased word sequences to get opcodes.
    - Map LLM words (and their trailing punctuation) back to original word indices.
    - For insertions (LLM-only words), attach them to the nearest original word (prefer previous).
    - For replacements of unequal sizes, distribute LLM words across the original span proportionally.
    - Finally, assemble per-segment punctuated text by collecting mapped outputs for the original word indices that belong to each segment.
    """
    # Defensive checks
    if not original_concat:
        return [dict(s) for s in segments]
    if not punctuated_text:
        # Nothing to distribute, return empty or fallback
        out_segments: list[dict[str, Any]] = []
        for s in segments:
            raw = (s.get("text_raw") or s.get("text") or "").strip()
            if keep_nonempty_segments and raw:
                out_segments.append({**s, "text_punct": _add_basic_punctuation(raw)})
            else:
                out_segments.append({**s, "text_punct": ""})
        print("[punct-align] fuzzy: empty punctuated_text; emitted fallback =", keep_nonempty_segments)
        return out_segments

    # Build original word list and per-segment word counts
    orig_words = _split_words(original_concat)
    seg_word_counts: list[int] = [len(_split_words((s.get("text_raw") or s.get("text") or "").strip())) for s in segments]

    if not orig_words:
        return [dict(s) for s in segments]

    # Tokenize LLM text into words and punctuation and map punctuation to following word index
    llm_tokens = _tokenize_words_and_punct(punctuated_text)
    llm_words = [t for t in llm_tokens if _is_word(t)]

    # Map punctuation that follows LLM words (index -1 stores leading punctuation)
    punct_after: dict[int, list[str]] = {-1: []}
    for i in range(len(llm_words)):
        punct_after[i] = []
    llw_idx = -1
    for tok in llm_tokens:
        if _is_word(tok):
            llw_idx += 1
        else:
            # punctuation
            punct_after.setdefault(llw_idx, []).append(tok)

    # Sequence match between original words and llm words (lowercased)
    a = [w.lower() for w in orig_words]
    b = [w.lower() for w in llm_words]
    sm = SequenceMatcher(None, a, b, autojunk=False)

    # Prepare mapping: for each original word index, a list of emitted LLM word+punct strings
    mapped_per_orig: list[list[str]] = [[] for _ in range(len(orig_words))]

    # Helper to render an llm word with its punctuation
    def render_llm_word(idx: int) -> str:
        w = llm_words[idx]
        toks = [w, *punct_after.get(idx, [])]
        # Join tokens with spacing rules: attach punctuation to word
        return _join_tokens_with_spacing(_tokenize_words_and_punct(" ".join(toks)))

    # Iterate opcodes and distribute LLM words
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # one-to-one mapping
            for oi, lj in zip(range(i1, i2), range(j1, j2), strict=False):
                mapped_per_orig[oi].append(render_llm_word(lj))
        elif tag == "replace":
            # distribute LLM words across original span proportionally
            orig_span = i2 - i1
            llm_span = j2 - j1
            if orig_span <= 0:
                # treat as pure insert
                attach_idx = i1 - 1 if i1 > 0 else 0
                for lj in range(j1, j2):
                    tgt = attach_idx
                    if 0 <= tgt < len(mapped_per_orig):
                        mapped_per_orig[tgt].append(render_llm_word(lj))
            elif llm_span == 0:
                # deletion: nothing to emit for those original words
                continue
            elif orig_span == llm_span:
                for oi, lj in zip(range(i1, i2), range(j1, j2), strict=False):
                    mapped_per_orig[oi].append(render_llm_word(lj))
            else:
                # proportional distribution of ljs to oi indices
                for offset_l, lj in enumerate(range(j1, j2)):
                    # compute fractional position within llm span and map to orig index
                    frac = offset_l / max(1, llm_span)
                    oi_rel = int(frac * orig_span)
                    oi = i1 + min(orig_span - 1, oi_rel)
                    mapped_per_orig[oi].append(render_llm_word(lj))
        elif tag == "delete":
            # original words removed by LLM: leave their mapped list empty (deleted)
            continue
        elif tag == "insert":
            # LLM inserted words with no original counterpart: attach to previous original word if exists, else next
            attach_idx = i1 - 1 if i1 > 0 else i1
            # clamp attach_idx
            if attach_idx < 0:
                attach_idx = 0
            if attach_idx >= len(mapped_per_orig):
                attach_idx = len(mapped_per_orig) - 1
            for lj in range(j1, j2):
                mapped_per_orig[attach_idx].append(render_llm_word(lj))

    # Now assemble per-segment punctuated text based on original word indices
    out_segments: list[dict[str, Any]] = []
    word_cursor = 0
    for seg_idx, s in enumerate(segments):
        raw_text = (s.get("text_raw") or s.get("text") or "").strip()
        cnt = seg_word_counts[seg_idx]
        if cnt == 0:
            out_segments.append({**s, "text_punct": ""})
            continue
        seg_tokens: list[str] = []
        # collect mapped outputs for each original word in this segment
        for wi in range(word_cursor, min(word_cursor + cnt, len(mapped_per_orig))):
            seg_tokens.extend(mapped_per_orig[wi])
        word_cursor += cnt
        if not seg_tokens:
            # Log and optionally fallback
            print(f"[punct-align] empty_map seg={seg_idx} cnt={cnt} keep_nonempty={keep_nonempty_segments}")
            if keep_nonempty_segments and raw_text:
                out_segments.append({**s, "text_punct": _add_basic_punctuation(raw_text)})
            else:
                out_segments.append({**s, "text_punct": ""})
        else:
            # Join tokens ensuring proper spacing/punctuation
            # seg_tokens may already contain word+punct strings; simply join with space and normalize whitespace
            piece = " ".join(t.strip() for t in seg_tokens if t.strip()).strip()
            # Cleanup double spaces before punctuation introduced by joins
            piece = re.sub(r"\s+([,\.!\?:;…])", r"\1", piece)
            out_segments.append({**s, "text_punct": piece})
    return out_segments


def _add_basic_punctuation(text: str) -> str:
    """Add basic punctuation to text as fallback."""
    if not text:
        return ""

    # Capitalize first letter
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # Add period at end if no punctuation
    if text[-1] not in '.!?':
        text += '.'

    return text

# --- Character-level alignment and quality metrics ---
def _align_chars(orig_word: str, llm_word: str) -> tuple[list[str], list[str], list[str]]:
    """Align characters between two words using SequenceMatcher.
    Returns: (orig_chars, llm_chars, tags) where tags are 'equal', 'replace', 'delete', 'insert'
    """
    sm = SequenceMatcher(None, orig_word, llm_word)
    orig_chars: list[str] = []
    llm_chars: list[str] = []
    tags: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_chars.extend(list(orig_word[i1:i2]))
        llm_chars.extend(list(llm_word[j1:j2]))
        tags.extend([tag] * max(i2 - i1, j2 - j1))
    return orig_chars, llm_chars, tags


def _compute_wer(orig_words: list[str], llm_words: list[str]) -> float:
    """Compute Word Error Rate (WER) between two lists of words."""
    if not orig_words:
        return 0.0 if not llm_words else 1.0
    sm = SequenceMatcher(None, orig_words, llm_words)
    edits = 0
    insertions = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            edits += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            edits += i2 - i1
        elif tag == 'insert':
            edits += j2 - j1
            insertions += j2 - j1
    # Use (number of edits) / (number of words in reference + number of insertions)
    # This is a common alternative WER calculation method
    denominator = len(orig_words) + insertions
    return edits / denominator if denominator > 0 else 0.0


def _compute_cer(orig_text: str, llm_text: str) -> float:
    """Compute Character Error Rate (CER) between two strings."""
    if not orig_text:
        return 0.0 if not llm_text else 1.0
    sm = SequenceMatcher(None, orig_text, llm_text)
    s = d = i = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            s += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            d += i2 - i1
        elif tag == 'insert':
            i += j2 - j1
    n = len(orig_text)
    return (s + d + i) / n


def _compute_per(orig_text: str, llm_text: str) -> float:
    """Compute Punctuation Error Rate (PER) by comparing only punctuation marks."""
    orig_punct = [c for c in orig_text if c in _PUNCT_CHARS]
    llm_punct = [c for c in llm_text if c in _PUNCT_CHARS]
    if not orig_punct:
        return 0.0 if not llm_punct else 1.0
    sm = SequenceMatcher(None, orig_punct, llm_punct)
    s = d = i = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            s += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            d += i2 - i1
        elif tag == 'insert':
            i += j2 - j1
    n = len(orig_punct)
    return (s + d + i) / n


def _compute_uwer_fwer(orig_text: str, llm_text: str) -> tuple[float, float]:
    """Compute Unpunctuated WER (U-WER) and Formatted WER (F-WER)."""
    # Remove punctuation for U-WER
    orig_no_punct = ''.join(c for c in orig_text if c not in _PUNCT_CHARS)
    llm_no_punct = ''.join(c for c in llm_text if c not in _PUNCT_CHARS)
    orig_words_u = _split_words(orig_no_punct)
    llm_words_u = _split_words(llm_no_punct)
    uwer = _compute_wer(orig_words_u, llm_words_u)
    # F-WER is WER on the original (punctuated) text
    orig_words_f = _split_words(orig_text)
    llm_words_f = _split_words(llm_text)
    fwer = _compute_wer(orig_words_f, llm_words_f)
    return uwer, fwer


def _generate_human_readable_diff(orig_text: str, llm_text: str) -> str:
    """Generate a side-by-side human-readable diff with color codes."""
    sm = SequenceMatcher(None, orig_text, llm_text)
    diff_lines: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        orig_seg = orig_text[i1:i2]
        llm_seg = llm_text[j1:j2]
        if tag == 'equal':
            diff_lines.append(f"  {orig_seg}")
        elif tag == 'replace':
            diff_lines.append(f"- {orig_seg}")
            diff_lines.append(f"+ {llm_seg}")
        elif tag == 'delete':
            diff_lines.append(f"- {orig_seg}")
        elif tag == 'insert':
            diff_lines.append(f"+ {llm_seg}")
    return "\n".join(diff_lines)


def _segmentwise_punctuate_segments(
    llm: Any,
    system_prompt: str,
    max_model_len: int,
    segments: list[dict[str, Any]],
    temperature: float = 0.0,
    user_prompt: str | None = None,
    batch_prompts: int = 1,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Punctuate each segment independently using the LLM (batched for throughput)."""
    non_empty_indices: list[int] = [i for i, s in enumerate(segments) if (s.get("text_raw") or "").strip()]
    if not non_empty_indices:
        return [dict(s) for s in segments]
    messages: list[list[dict[str, str]]] = []
    per_piece_max: list[int] = []
    try:
        tokenizer = get_tokenizer(llm)
    except (ValueError, KeyError, TypeError):
        tokenizer = None
    for i in non_empty_indices:
        raw = (segments[i].get("text_raw") or "").strip()
        if user_prompt:
            user_content = f"{user_prompt}\n\n{raw}"
        else:
            user_content = raw
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])
        if tokenizer is not None:
            with suppress(Exception):
                t_in = len(tokenizer.encode(raw))
                per_piece_max.append(max(64, min(int(max_model_len) - 64, int(t_in) + 128)))
                continue
        per_piece_max.append(512)
    # Run in groups
    outputs: list[str] = [""] * len(non_empty_indices)
    start = 0
    if show_progress and tqdm is not None:
        pbar = tqdm(total=len(non_empty_indices), desc="Punct segwise")
    else:
        pbar = None
    while start < len(non_empty_indices):
        end = min(start + int(max(1, batch_prompts)), len(non_empty_indices))
        group_msgs = messages[start:end]
        group_max = max(per_piece_max[start:end]) if per_piece_max else 512
        group_out = generate_chat_batch(llm, group_msgs, temperature=temperature, max_tokens=int(group_max))
        for j, text in enumerate(group_out):
            outputs[start + j] = text or ""
        if pbar is not None:
            pbar.update(len(group_msgs))
        start = end
    if pbar is not None:
        pbar.close()
    # Write back
    out_segments: list[dict[str, Any]] = [dict(s) for s in segments]
    for k, idx in enumerate(non_empty_indices):
        out_segments[idx]["text_punct"] = (outputs[k] or "").strip()
    return out_segments


def _safe_distribute_punct_to_segments(
    punctuated_text: str,
    segments: list[dict[str, Any]],
    *,
    llm: Any | None = None,
    system_prompt: str | None = None,
    max_model_len: int | None = None,
    temperature: float = 0.0,
    batch_prompts: int = 1,
    show_progress: bool = False,
    keep_nonempty_segments: bool = False,
) -> list[dict[str, Any]]:
    """Distribute punctuated text to segments using improved fuzzy matching.

    Behavior:
    - Default: strict alignment (may leave text_punct empty when LLM deletes words).
    - When keep_nonempty_segments is True: apply gentle fallback to avoid empty text_punct
      by minimally punctuating the original segment text.
    """
    return _distribute_punct_to_segments(punctuated_text, segments, keep_nonempty_segments=keep_nonempty_segments)


def _build_segment_batches_by_token_budget(
    llm: Any,
    segments: list[dict[str, Any]],
    token_budget: int,
    safety_margin: int = 64,
) -> list[dict[str, Any]]:
    """Group adjacent segments into batches that fit within a token budget.

    Returns a list of dicts with keys: start_idx, end_idx (exclusive), text,
    word_counts (per original segment), start (time), end (time).
    """
    tokenizer = get_tokenizer(llm)
    batches: list[dict[str, Any]] = []
    i = 0
    budget = max(1, int(token_budget) - int(safety_margin))
    while i < len(segments):
        j = i
        texts: list[str] = []
        word_counts: list[int] = []
        cur_tokens = 0
        while j < len(segments):
            raw = (segments[j].get("text_raw") or "").strip()
            if not raw:
                j += 1
                continue
            t = len(tokenizer.encode(raw))
            if cur_tokens and cur_tokens + t > budget:
                break
            texts.append(raw)
            word_counts.append(len(_split_words(raw)))
            cur_tokens += t
            j += 1
        if not texts:
            # Force progress to avoid infinite loop on empty texts
            j = max(j, i + 1)
        start_time = segments[i].get("start")
        end_time = segments[j - 1].get("end") if j - 1 < len(segments) else segments[-1].get("end")
        batches.append(
            {
                "start_idx": i,
                "end_idx": j,
                "text": " ".join(texts).strip(),
                "word_counts": word_counts,
                "start": start_time,
                "end": end_time,
            },
        )
        i = j
    return batches


def split_text_into_chunks(text: str, max_chars: int = 4000, overlap_sentences: int = 1) -> list[str]:
    import re
    if not text:
        return []
    # Simple sentence split on Vietnamese punctuation and newlines
    sentences = re.split(r"(?<=[\.\!\?\n])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if current_len + len(s) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            # overlap last N sentences
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(x) + 1 for x in current)
        current.append(s)
        current_len += len(s) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def build_chunks_from_segments_by_token_budget(
    llm: Any,
    segments: list[dict[str, Any]],
    token_budget: int,
    safety_margin: int = 64,
) -> list[str]:
    tokenizer = get_tokenizer(llm)
    chunks: list[str] = []
    current_texts: list[str] = []
    current_tokens = 0
    budget = max(1, token_budget - safety_margin)
    for seg in segments:
        raw = (seg.get("text_raw") or "").strip()
        if not raw:
            continue
        seg_tokens = len(tokenizer.encode(raw))
        if current_tokens + seg_tokens > budget and current_texts:
            chunks.append(" ".join(current_texts).strip())
            current_texts = []
            current_tokens = 0
        current_texts.append(raw)
        current_tokens += seg_tokens
    if current_texts:
        chunks.append(" ".join(current_texts).strip())
    return chunks


def summarize_long_text_map_reduce(
    llm: Any,
    text: str,
    system_prompt: str,
    temperature: float = 0.2,
    max_model_len: int = 2048,
    batch_prompts: int = 1,
    show_progress: bool = False,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    # Split into manageable chunks by tokenizer budget and summarize each, then reduce
    input_limit = _compute_input_token_limit(max_model_len, prompt_overhead_tokens=128, output_margin_tokens=256)
    chunks = _split_text_by_token_budget(llm, text, max_input_tokens=input_limit)
    if not chunks:
        return {"bullets": [], "abstract": ""}
    partial_bullets: list[str] = []
    partial_abstracts: list[str] = []
    if batch_prompts <= 1:
        iterable = chunks
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable, total=len(chunks), desc="Summarize map")
        for chunk in iterable:
            part = summarize_text(llm, chunk, system_prompt, temperature=temperature, user_prompt=user_prompt)
            partial_bullets.extend(part.get("bullets", [])[:5])
            if part.get("abstract"):
                partial_abstracts.append(part["abstract"])
    else:
        list_of_messages: list[list[dict[str, str]]] = []
        for c in chunks:
            if user_prompt:
                user_content = f"{user_prompt}\n\n{c}"
            else:
                user_content = f"{c}"
            list_of_messages.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        params_max = 800
        outs: list[str] = []
        start = 0
        while start < len(list_of_messages):
            end = min(start + int(max(1, batch_prompts)), len(list_of_messages))
            msgs = list_of_messages[start:end]
            outs.extend(generate_chat_batch(llm, msgs, temperature=temperature, max_tokens=params_max))
            start = end
        for content in outs:
            try:
                parsed = json.loads(content)
                pb = parsed.get("bullets", [])
                pa = parsed.get("abstract", "")
            except (ValueError, KeyError, TypeError):
                pb = [line.strip("- ") for line in content.splitlines() if line.strip()]
                pa = ""
            partial_bullets.extend(pb[:5])
            if pa:
                partial_abstracts.append(pa)
    # Reduce
    reduce_text = "\n".join([f"- {b}" for b in partial_bullets] + partial_abstracts)
    # Ensure the reduce step also respects token budget
    reduce_chunks = _split_text_by_token_budget(llm, reduce_text, max_input_tokens=input_limit)
    # Initialize containers so later references (raw assembly) are always bound
    reduce_raw_parts: list[str] = []
    outs2: list[str] = []
    if len(reduce_chunks) == 1:
        reduced = summarize_text(llm, reduce_chunks[0], system_prompt, temperature=temperature, user_prompt=user_prompt)
    else:
        # Summarize each reduce chunk then merge (batched if enabled)
        merged_bullets: list[str] = []
        merged_abstracts: list[str] = []
        if batch_prompts <= 1:
            iterable = reduce_chunks
            if show_progress and tqdm is not None:
                iterable = tqdm(iterable, total=len(reduce_chunks), desc="Summarize reduce")
            # Collect raw outputs to expose a concatenated raw signal
            reduce_raw_parts = []
            for rc in iterable:
                rpart = summarize_text(llm, rc, system_prompt, temperature=temperature, user_prompt=user_prompt)
                merged_bullets.extend(rpart.get("bullets", [])[:5])
                if rpart.get("abstract"):
                    merged_abstracts.append(rpart["abstract"])
                if rpart.get("raw"):
                    reduce_raw_parts.append(str(rpart.get("raw")))
        else:
            list_of_messages = []
            for rc in reduce_chunks:
                if user_prompt:
                    user_content = f"{user_prompt}\n\n{rc}"
                else:
                    user_content = f"{rc}"
                list_of_messages.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ])
            params_max = 800
            outs2 = []
            start = 0
            while start < len(list_of_messages):
                end = min(start + int(max(1, batch_prompts)), len(list_of_messages))
                outs2.extend(generate_chat_batch(llm, list_of_messages[start:end], temperature=temperature, max_tokens=params_max))
                start = end
            for content in outs2:
                try:
                    parsed = json.loads(content)
                    mb = parsed.get("bullets", [])
                    ma = parsed.get("abstract", "")
                except (ValueError, KeyError, TypeError):
                    mb = [line.strip("- ") for line in content.splitlines() if line.strip()]
                    ma = ""
                merged_bullets.extend(mb[:5])
                if ma:
                    merged_abstracts.append(ma)
        reduced = {
            "bullets": merged_bullets[:7],
            "abstract": " ".join(merged_abstracts)[:1000],
        }
    # Expose a single raw string where possible
    raw_out: str | None = None
    if isinstance(reduced, dict) and "raw" in reduced and reduced.get("raw"):
        raw_out = str(reduced.get("raw"))
    else:
        # If we had batched reduce or chunk-level reduce, fallback to joining available raw parts or batch outputs
        try:
            raw_candidates: list[str] = []
            # from single-chunk reduce loop
            if 'reduce_raw_parts' in locals() and reduce_raw_parts:
                raw_candidates.extend([str(x) for x in reduce_raw_parts if x])
            # from multi-chunk reduce batching
            if 'outs2' in locals() and outs2:
                raw_candidates.extend([str(x).strip() for x in outs2 if str(x).strip()])
            raw_out = "\n\n".join(raw_candidates) if raw_candidates else None
        except (ValueError, KeyError, TypeError):
            raw_out = None
    return {
        "bullets": reduced.get("bullets", partial_bullets[:7]),
        "abstract": reduced.get("abstract", " ".join(partial_abstracts)[:1000]),
        "raw": raw_out or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process ASR with vLLM (offline): punctuation + summary")
    parser.add_argument("--config", type=str, default="/home/cetech/omoai/config.yaml", help="Path to config.yaml")
    parser.add_argument("--asr-json", type=str, required=True, help="Path to ASR JSON")
    parser.add_argument("--model", type=str, default=None, help="HF model id for vLLM")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization type (e.g., awq, gptq)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model len (context)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory util ratio")
    parser.add_argument("--out", type=str, required=True, help="Path to output final JSON")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Max concurrent sequences (memory)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Max batched tokens (memory)")
    parser.add_argument("--auto-outdir", action="store_true", help="Create per-input folder under paths.out_dir/{stem-YYYYMMDD-HHMMSS}; if ASR JSON already resides under paths.out_dir/<name>, reuse that folder")
    # Punctuation controls (segmented batching only)
    parser.add_argument("--punct-auto-ratio", type=float, default=None, help="Ratio of context length used to form per-batch token budget")
    parser.add_argument("--punct-auto-margin", type=int, default=None, help="Margin tokens reserved for punctuation output")


    # Safety controls
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for vLLM")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Usability/runtime controls
    parser.add_argument("--dry-run", action="store_true", help="Simulate planning; no LLM calls; still write output with metadata/decisions")
    parser.add_argument("--stream-asr", action="store_true", help="Stream ASR JSON (requires ijson) to reduce memory")
    parser.add_argument("--progress", action="store_true", help="Enable progress bars (tqdm)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    # Removed: --use-vi-sentence-seg (no longer used in segmented batching-only flow)
    parser.add_argument("--batch-prompts", type=int, default=None, help="Max prompts per batched generation call")
    parser.add_argument(
        "--timestamped_summary",
        action="store_true",
        help="Generate a summary with timestamps.",
    )
    args = parser.parse_args()

    # Load centralized configuration using the project's Pydantic schemas
    try:
        from omoai.config.schemas import get_config
    except ImportError:
        # If running as a script with imports failing, ensure src/ is on sys.path then retry
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
        from omoai.config.schemas import get_config

    cfg = get_config()

    def cfg_get(path: list[str], default: Any | None = None) -> Any:
        """
        Access nested fields from the centralized Pydantic config using attribute access.
        Example: cfg_get(["punctuation", "llm", "model_id"]) -> cfg.punctuation.llm.model_id
        """
        cur: Any = cfg
        for key in path:
            if cur is None:
                return default
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
        return cur

    # Global defaults
    model_id_from_cfg = cfg.llm.model_id
    if model_id_from_cfg is None:
        raise ValueError("model_id not found in configuration. Please ensure it is set in the config file under the 'llm' section.")
    model_id_default = args.model or model_id_from_cfg
    quant_default = args.quantization or cfg_get(["llm", "quantization"], None)
    mml_default = int(args.max_model_len or cfg_get(["llm", "max_model_len"], 2048))
    gmu_default = float(args.gpu_memory_utilization or cfg_get(["llm", "gpu_memory_utilization"], 0.90))
    mns_default = int(args.max_num_seqs or cfg_get(["llm", "max_num_seqs"], 1))
    mbt_default = int(args.max_num_batched_tokens or cfg_get(["llm", "max_num_batched_tokens"], 512))

    # Punctuation overrides (segmented batching only)
    p_model_id = cfg_get(["punctuation", "llm", "model_id"], model_id_default)
    p_quant = cfg_get(["punctuation", "llm", "quantization"], quant_default)
    p_mml = int(cfg_get(["punctuation", "llm", "max_model_len"], mml_default))
    p_gmu = float(cfg_get(["punctuation", "llm", "gpu_memory_utilization"], gmu_default))
    p_mns = int(cfg_get(["punctuation", "llm", "max_num_seqs"], mns_default))
    p_mbt = int(cfg_get(["punctuation", "llm", "max_num_batched_tokens"], mbt_default))
    p_temp = float(cfg_get(["punctuation", "sampling", "temperature"], 0.0))
    # Optional user-provided prompt to guide punctuation (prepended to segment/chunk text)
    # p_user_prompt = cfg_get(["punctuation", "user_prompt"], None)  # Unused variable removed
    punct_auto_ratio = float(args.punct_auto_ratio or cfg_get(["punctuation", "auto_switch_ratio"], 0.98))
    punct_auto_margin = int(args.punct_auto_margin or cfg_get(["punctuation", "auto_margin_tokens"], 128))

    # Output quality controls (hardcoded)
    adopt_case = True
    preserve_original_words = bool(cfg_get(["punctuation", "preserve_original_words"], True))
    # keep_nonempty_segments is enforced and not user-configurable
    # keep_nonempty_segments = True  # Unused variable removed

    join_sep = str(cfg_get(["punctuation", "join_separator"], " "))
    use_vi_sentence_seg = bool(cfg_get(["punctuation", "use_vi_sentence_segmenter"], False))
    enable_paragraphs = bool(cfg_get(["punctuation", "enable_paragraphs"], True))

    # Safety controls
    # Optional user prompt for summarization (prepended to user content)
    s_user_prompt = cfg_get(["summarization", "user_prompt"], None)
    trust_remote_code = cfg_get(["llm", "trust_remote_code"], True)
    if args.trust_remote_code:
        trust_remote_code = True
    elif args.no_trust_remote_code:
        trust_remote_code = False

    # Summarization overrides (optional)
    s_model_id = cfg_get(["summarization", "llm", "model_id"], model_id_default)
    s_quant = cfg_get(["summarization", "llm", "quantization"], quant_default)
    s_mml = int(cfg_get(["summarization", "llm", "max_model_len"], mml_default))
    s_gmu = float(cfg_get(["summarization", "llm", "gpu_memory_utilization"], gmu_default))
    s_mns = int(cfg_get(["summarization", "llm", "max_num_seqs"], mns_default))
    s_mbt = int(cfg_get(["summarization", "llm", "max_num_batched_tokens"], mbt_default))
    s_temp = float(cfg_get(["summarization", "sampling", "temperature"], 0.2))

    def build_llm(model_id: str, quant: str | None, max_model_len: int, gmu: float, mns: int, mbt: int) -> Any:
        # Ensure spawn method & environment JUST before first real vLLM import.
        import multiprocessing as _mp
        try:
            if _mp.get_start_method(allow_none=True) != "spawn":
                _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("MULTIPROCESSING_START_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")

        # Import vLLM late to avoid CUDA context initialization in parent prematurely.
        from vllm import LLM as _real_llm  # type: ignore
        from vllm import SamplingParams as _RealSamplingParams

        # Bind real classes globally for subsequent calls.
        global SamplingParams
        SamplingParams = _RealSamplingParams  # type: ignore

        q = None if quant in (None, "auto", "infer", "compressed-tensors", "model") else str(quant)

        kwargs = dict(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gmu,
            trust_remote_code=trust_remote_code,
            max_num_seqs=mns,
            max_num_batched_tokens=mbt,
            enforce_eager=True,
            distributed_executor_backend="mp",
        )
        if q is not None:
            kwargs["quantization"] = q
        try:
            return _real_llm(**kwargs)  # type: ignore[arg-type, call-arg, misc]
        except Exception as e:
            if "Quantization method specified in the model config" in str(e):
                kwargs.pop("quantization", None)
                return _real_llm(**kwargs)  # type: ignore[arg-type, call-arg, misc]
            raise

    asr_json_path = Path(args.asr_json)
    # Optional streaming path for large ASR JSON
    if args.stream_asr and ijson is not None:
        asr_top = load_asr_top_level(asr_json_path)
        seg_iter = iter_asr_segments(asr_json_path)
        asr = asr_top
        segments = list(seg_iter)
    else:
        asr = load_asr_json(asr_json_path)
        segments = asr.get("segments", [])
    transcript_raw: str = asr.get("transcript_raw", "") if isinstance(asr, dict) else ""

    # English prompts that request Vietnamese outputs
    punct_system_default = (
        """<instruction>
    You are an expert in Vietnamese grammar and punctuation. Your task is to meticulously correct the input text by adding proper punctuation and capitalization.
    - Add all necessary punctuation, including commas, periods, question marks, etc.
    - Correct capitalization for the start of sentences and proper nouns (e.g., 'hà nội' -> 'Hà Nội').
    - Ensure the original wording, word order, and meaning are perfectly preserved.
    - Your output must be a single, coherent block of punctuated text and nothing else.
    </instruction>
    <example>
    <input>xin chào thế giới đây là một ví dụ về khôi phục dấu câu</input>
    <output>Xin chào thế giới, đây là một ví dụ về khôi phục dấu câu.</output>
    </example>
    <example>
    <input>bạn tên là gì tôi tên là nam</input>
    <output>Bạn tên là gì? Tôi tên là Nam.</output>
    </example>
    <example>
    <input>tôi đang xem một buổi lai trim trên phây búc về trí tuệ nhân tạo ai</input>
    <output>Tôi đang xem một buổi livestream trên Facebook về trí tuệ nhân tạo AI.</output>
    </example>
    <example>
    <input>hôm qua tại hà nội thủ tướng đã nói chúng ta cần phải nỗ lực hơn nữa để phát triển kinh tế</input>
    <output>Hôm qua tại Hà Nội, Thủ tướng đã nói: "Chúng ta cần phải nỗ lực hơn nữa để phát triển kinh tế."</output>
    </example>
    <policy>
    ABSOLUTE RULE: Do not delete, replace, or rephrase any words from the original input. Your only task is to add punctuation and capitalization. The original words must be kept exactly as they are.
    </policy>"""
    )
    sum_system_default = (
        """<instruction>
    You are a highly skilled Vietnamese text analysis engine. Your task is to generate a concise summary of the input text and format it as a single, valid JSON object.
    - The JSON object must contain exactly two keys: "bullets" and "abstract".
    - The "bullets" value must be an array of 3 to 7 short Vietnamese sentences (max 20 words each), highlighting the main points.
    - The "abstract" value must be a string containing a 2-3 sentence summary in Vietnamese.
    - The summary must be based exclusively on the provided text.
    - Your output MUST be only the JSON object. Do not include any other text, explanations, or markdown formatting before or after the JSON.
    </instruction>
    <example>
    <input>Hệ thống nhận dạng giọng nói đã trở thành một công nghệ phổ biến. Nó được sử dụng trong nhiều ứng dụng từ trợ lý ảo đến điều khiển bằng giọng nói trong xe hơi. Công nghệ này giúp tăng cường sự tiện lợi và hiệu quả.</input>
    <output>
    {
      "bullets": [
        "Hệ thống nhận dạng giọng nói là công nghệ phổ biến.",
        "Nó có nhiều ứng dụng như trợ lý ảo và điều khiển xe hơi.",
        "Công nghệ này giúp tăng sự tiện lợi và hiệu quả."
      ],
      "abstract": "Hệ thống nhận dạng giọng nói là một công nghệ phổ biến được sử dụng trong nhiều ứng dụng, từ trợ lý ảo đến điều khiển bằng giọng nói trong xe hơi. Công nghệ này giúp tăng cường sự tiện lợi và hiệu quả cho người dùng."
    }
    </output>
    </example>
    <example>
    <input>Trí tuệ nhân tạo đang thay đổi thế giới việc làm. Nhiều công việc lặp đi lặp lại có thể được tự động hóa, giúp con người tập trung vào các nhiệm vụ sáng tạo và chiến lược hơn. Tuy nhiên, điều này cũng đặt ra thách thức về đào tạo lại lực lượng lao động để họ có thể thích ứng với các vai trò mới. Các chính phủ và doanh nghiệp cần hợp tác để giải quyết vấn đề này.</input>
    <output>
    {
      "bullets": [
        "Trí tuệ nhân tạo đang làm thay đổi thị trường lao động.",
        "Các công việc lặp đi lặp lại đang được tự động hóa.",
        "Con người có thể tập trung vào công việc sáng tạo và chiến lược.",
        "Thách thức đặt ra là phải đào tạo lại lực lượng lao động.",
        "Chính phủ và doanh nghiệp cần hợp tác để giải quyết vấn đề."
      ],
      "abstract": "Trí tuệ nhân tạo đang thay đổi thế giới việc làm bằng cách tự động hóa các công việc lặp đi lặp lại, cho phép con người tập trung vào các nhiệm vụ sáng tạo hơn. Tuy nhiên, điều này tạo ra nhu cầu cấp thiết về việc đào tạo lại lực lượng lao động, đòi hỏi sự hợp tác giữa chính phủ và doanh nghiệp."
    }
    </output>
    </example>"""
    )
    punct_system = str(cfg_get(["punctuation", "system_prompt"], punct_system_default))
    sum_system = str(cfg_get(["summarization", "system_prompt"], sum_system_default))
    # Progress and batching controls
    progress_enabled = bool(args.progress)
    if args.no_progress:
        progress_enabled = False
    verbose = bool(args.verbose)
    prompt_batch_prompts = int(args.batch_prompts) if args.batch_prompts is not None else int(p_mns)

    if verbose:
        logger.info(
            f"[post] Config: adopt_case={adopt_case}, enable_paragraphs={enable_paragraphs}",
        )
        logger.info(
            f"[post] Token budget: ratio={punct_auto_ratio}, margin={punct_auto_margin}",
        )
        logger.info(f"[post] Trust remote code: {trust_remote_code}")

    # Resolve vLLM availability lazily now (after argument parsing & under __main__).
    global VLLM_AVAILABLE
    if VLLM_AVAILABLE is None:
        try:
            import vllm  # type: ignore  # noqa: F401
            VLLM_AVAILABLE = True
        except ImportError:
            VLLM_AVAILABLE = False

    # Early dry-run path: compute decisions without instantiating LLM
    # Also activate this path automatically when vLLM is unavailable.
    if bool(args.dry_run) or not VLLM_AVAILABLE:
        if not VLLM_AVAILABLE:
            logger.warning("[post] vLLM not available; running in offline dry-run mode (no LLM calls)")
        # Segmented batching only: estimate batches by approximate per-segment tokens
        approx_tokens = max(1, len((transcript_raw or "").strip()) // 3)
        ratio = max(0.5, min(1.0, float(punct_auto_ratio)))
        token_limit = int(ratio * int(p_mml)) - int(punct_auto_margin)
        token_limit = max(256, token_limit)

        seg_tokens: list[int] = []
        for seg in segments:
            raw = (seg.get("text_raw") or "").strip()
            if not raw:
                continue
            seg_tokens.append(max(1, len(raw) // 3))
        est_batches = 0
        cur = 0
        for t in seg_tokens:
            if cur and cur + t > token_limit:
                est_batches += 1
                cur = 0
            cur += t
        if seg_tokens:
            est_batches += 1

        # Summarization decision
        auto_ratio = float(cfg_get(["summarization", "auto_switch_ratio"], 0.98))
        margin_tokens = int(cfg_get(["summarization", "auto_margin_tokens"], 64))
        ratio_limit = max(0.5, min(1.0, auto_ratio))
        s_token_limit = int(ratio_limit * int(s_mml)) - margin_tokens
        approx_sum_tokens = approx_tokens
        sum_mode = "map_reduce" if ((approx_sum_tokens and approx_sum_tokens > s_token_limit) or bool(cfg_get(["summarization", "map_reduce"], False))) else "single"

        decisions: dict[str, Any] = {
            "punctuation_mode": "segments",
            "segments_count": len(segments),
            "estimated_batches": est_batches,
            "summarization_mode": sum_mode,
        }

        final = dict(asr)
        final.update(
            {
                "segments": segments,
                # Without an LLM, pass through raw transcript; real punctuation requires vLLM
                "transcript_punct": (asr.get("transcript_raw") or "") if isinstance(asr, dict) else "",
                "summary": {"bullets": [], "abstract": ""},
                "metadata": {
                    **(asr.get("metadata", {}) if isinstance(asr, dict) else {}),
                    "llm_model_punctuation": p_model_id,
                    "llm_model_summarization": s_model_id,
                    "llm_offline": True,
                    "dry_run": True,
                    "decisions": decisions,
                },
            },
        )
        out_path = Path(args.out)
        save_json(out_path, final)
        return

    # Stage 1: punctuation LLM instance and segmented-batching processing
    llm_punc: Any | None = None
    llm_punc = build_llm(p_model_id, p_quant, p_mml, p_gmu, p_mns, p_mbt)
    punct_segments: list[dict[str, Any]]
    transcript_punct: str

    # Segmented punctuation with batching by token budget
    ratio = max(0.5, min(1.0, float(punct_auto_ratio)))
    token_limit = int(ratio * p_mml) - int(punct_auto_margin)
    token_limit = max(256, token_limit)
    if verbose:
        logger.info(
            f"[post] Punctuation: token_limit={token_limit}, segments={len(segments)}",
        )
    batches = _build_segment_batches_by_token_budget(llm_punc, segments, token_limit, safety_margin=64)
    if verbose:
        logger.info(f"[post] Created {len(batches)} batches for punctuation")
    # Initialize output copy
    punct_segments = [dict(s) for s in segments]
    chunk_puncts: list[str] = []
    # Batched generation across batch texts for throughput
    non_empty_batches = [b for b in batches if b["text"]]
    list_of_messages: list[list[dict[str, str]]] = [
        [
            {"role": "system", "content": punct_system},
            {"role": "user", "content": b["text"]},
        ]
        for b in non_empty_batches
    ]
    try:
        tokenizer = get_tokenizer(llm_punc)
        est_in = [len(tokenizer.encode(b["text"])) for b in non_empty_batches]
    except (ValueError, KeyError, TypeError):
        est_in = [max(64, len(b["text"]) // 3) for b in non_empty_batches]
    est_out = [max(64, min(int(p_mml) - 64, i + 128)) for i in est_in]
    generated_texts: list[str] = []
    start_idx = 0
    if tqdm is not None and progress_enabled:
        pbar = tqdm(total=len(non_empty_batches), desc="Punct batches")
    else:
        pbar = None
    while start_idx < len(list_of_messages):
        end_idx = min(start_idx + int(max(1, prompt_batch_prompts)), len(list_of_messages))
        group_msgs = list_of_messages[start_idx:end_idx]
        group_max = max(est_out[start_idx:end_idx]) if est_out else 512
        generated = generate_chat_batch(llm_punc, group_msgs, temperature=p_temp, max_tokens=int(group_max))
        generated_texts.extend(generated)
        if pbar is not None:
            pbar.update(len(group_msgs))
        start_idx = end_idx
    if pbar is not None:
        pbar.close()
    gi = 0
    for b in batches:
        if not b["text"]:
            continue
        chunk_punct = generated_texts[gi]
        gi += 1
        sub_segments = segments[b["start_idx"] : b["end_idx"]]
        # Optionally preserve original words or allow LLM to rewrite (for spelling/clean-up)
        if preserve_original_words:
            original_concat = " ".join((s.get("text_raw") or "").strip() for s in sub_segments).strip()
            aligned_chunk = _force_preserve_with_alignment(original_concat, chunk_punct or "", adopt_case=adopt_case)
        else:
            aligned_chunk = (chunk_punct or "").strip()
        if aligned_chunk:
            chunk_puncts.append(aligned_chunk)
        # Distribute with LLM-aware fallback to fill per-segment punctuation if strict slicing fails
        distributed = _safe_distribute_punct_to_segments(
            aligned_chunk,
            sub_segments,
            llm=llm_punc,
            system_prompt=punct_system,
            max_model_len=p_mml,
            temperature=p_temp,
            batch_prompts=prompt_batch_prompts,
            show_progress=False,
        )
        # Write back
        for k, seg_out in enumerate(distributed):
            punct_segments[b["start_idx"] + k]["text_punct"] = seg_out.get("text_punct", "")
    # Build transcript from aggregated chunk outputs with deduplication
    if not chunk_puncts:
        transcript_punct = ""
    else:
        transcript_punct = chunk_puncts[0].strip()
        for cp in chunk_puncts[1:]:
            if cp:
                deduped = _dedup_overlap(transcript_punct, cp.strip(), max_tokens=8)
                transcript_punct = (transcript_punct + " " + deduped).strip()

    # Re-process segments through paragraph formatter for better flow
    transcript_punct = join_punctuated_segments(
        punct_segments,
        join_separator=join_sep,
        use_vi_sentence_segmenter=use_vi_sentence_seg,
    )

    # Keep llm_punc for reuse if settings match summarization

    # 2) Summarize
    long_text = transcript_punct
    # Auto-switch logic: single-pass unless use_map_reduce is forced, or total tokens exceed ratio*context
    use_map_reduce = bool(cfg_get(["summarization", "map_reduce"], False))
    # Stage 2: summarization LLM instance (reuse if identical settings)
    reuse = (
        s_model_id == p_model_id
        and (str(s_quant) == str(p_quant) or (s_quant in (None, "auto") and p_quant in (None, "auto")))
        and s_mml == p_mml
        and abs(float(s_gmu) - float(p_gmu)) < 1e-6
        and s_mns == p_mns
        and s_mbt == p_mbt
    )
    if reuse:
        llm_sum = llm_punc
    else:
        # Free punctuation engine VRAM before building a new model to avoid OOM
        with suppress(Exception):
            llm_punc = None
        # Only clear cache if debug flag is set or switching models (higher chance of OOM)
        if DEBUG_EMPTY_CACHE or not reuse:
            with suppress(Exception):
                torch.cuda.empty_cache()  # type: ignore[attr-defined]
        gc.collect()
        llm_sum = build_llm(s_model_id, s_quant, s_mml, s_gmu, s_mns, s_mbt)
    try:
        tokenizer_s = get_tokenizer(llm_sum)
        total_tokens_s = len(tokenizer_s.encode(long_text))
    except (ValueError, KeyError, TypeError):
        tokenizer_s = None
        total_tokens_s = 0

    # Allow configurable ratio threshold for auto-switch
    auto_ratio = float(cfg_get(["summarization", "auto_switch_ratio"], 0.98))
    margin_tokens = int(cfg_get(["summarization", "auto_margin_tokens"], 64))
    ratio_limit = max(0.5, min(1.0, auto_ratio))  # clamp for safety
    token_limit = int(ratio_limit * s_mml) - margin_tokens

    if not use_map_reduce and tokenizer_s is not None and total_tokens_s and total_tokens_s <= token_limit:
        if verbose:
            logger.info(f"[post] Summarization: single-pass, tokens={total_tokens_s}")
        summary = summarize_text(llm_sum, long_text, sum_system, temperature=s_temp, user_prompt=s_user_prompt)
    else:
        if verbose:
            logger.info(f"[post] Summarization: map-reduce, tokens={total_tokens_s}")
        summary = summarize_long_text_map_reduce(
            llm_sum,
            long_text,
            sum_system,
            temperature=s_temp,
            max_model_len=s_mml,
            batch_prompts=prompt_batch_prompts,
            show_progress=(tqdm is not None and progress_enabled),
        )
    # Defer freeing engines until after optional timestamped summary generation

    # Consistency check: validate all non-empty text_raw have non-empty text_punct
    non_empty_raw = sum(1 for s in punct_segments if (s.get("text_raw") or "").strip())
    non_empty_punct = sum(1 for s in punct_segments if (s.get("text_punct") or "").strip())
    punct_coverage = non_empty_punct / max(1, non_empty_raw)

    # Count punctuation marks for quality metrics
    punct_marks = len([c for c in transcript_punct if c in ".,!?…"])
    punct_density = punct_marks / max(1, len(transcript_punct))

    if verbose:
        logger.info(
            f"[post] Quality metrics: coverage={punct_coverage:.4f}, density={punct_density:.4f}, marks={punct_marks}",
        )
        logger.info(
            f"[post] Summary: bullets={len(summary.get('bullets', []))}, has_abstract={bool(summary.get('abstract'))}",
        )

    # Compose final output with enhanced metadata
    decisions = {
        "punctuation_mode": "segments",
        "segments_count": len(segments),
        "batches_used": len(batches),
        "punct_coverage": round(punct_coverage, 4),
        "punct_marks": punct_marks,
        "punct_density": round(punct_density, 4),
        "adopt_case": adopt_case,
        "summarization_mode": "map_reduce" if use_map_reduce else "single",
    }

    final = dict(asr)
    # --- Compute quality metrics & diffs ---
    try:
        original_text = (
            (asr.get("transcript_raw") or "")
            if isinstance(asr, dict)
            else ""
        )
        if not original_text and isinstance(asr, dict):
            original_text = asr.get("text") or ""
        if not original_text:
            # Fallback: join ASR segments' raw text
            try:
                original_text = " ".join(
                    (s.get("text") or s.get("text_raw") or "").strip()
                    for s in (segments or [])
                ).strip()
            except (TypeError, AttributeError) as e:
                logger.debug(f"Failed to extract original text: {e}")
                original_text = ""

        # Compute metrics only if we have something to compare
        if original_text and transcript_punct:
            wer_val = _compute_wer(_split_words(original_text), _split_words(transcript_punct))
            cer_val = _compute_cer(original_text, transcript_punct)
            per_val = _compute_per(original_text, transcript_punct)
            uwer_val, fwer_val = _compute_uwer_fwer(original_text, transcript_punct)
            alignment_conf = punct_coverage  # use coverage as a proxy confidence

            quality_metrics = {
                "wer": round(float(wer_val), 4),
                "cer": round(float(cer_val), 4),
                "per": round(float(per_val), 4),
                "uwer": round(float(uwer_val), 4),
                "fwer": round(float(fwer_val), 4),
                "alignment_confidence": round(float(alignment_conf), 4),
            }

            # Build a simple human-readable diff
            diff_output = _generate_human_readable_diff(original_text, transcript_punct)
            human_diff = {
                "original_text": original_text,
                "punctuated_text": transcript_punct,
                "diff_output": diff_output,
                "alignment_summary": (
                    f"coverage={punct_coverage:.4f}, density={punct_density:.4f}, marks={punct_marks}"
                ),
            }
        else:
            quality_metrics = None
            human_diff = None
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Quality metrics computation failed: {e}")
        # Keep pipeline robust; if metrics computation fails, omit them
        quality_metrics = None
        human_diff = None
    # Generate timestamped summary if requested
    if args.timestamped_summary:
        if not VLLM_AVAILABLE or args.dry_run:
            logger.warning("[post] Timestamped summary requested but vLLM not available or dry-run mode; skipping")
        else:
            try:
                logger.info("Starting timestamped summary generation...")
                system_prompt = cfg_get(["timestamped_summary", "system_prompt"], "You are a helpful assistant that processes transcripts to create summaries...")

                # Build a dedicated LLM instance for timestamped_summary using its own config
                ts_model_id = cfg_get(["timestamped_summary", "llm", "model_id"], model_id_default) or model_id_default
                ts_quant = cfg_get(["timestamped_summary", "llm", "quantization"], quant_default) or quant_default
                ts_mml = int(cfg_get(["timestamped_summary", "llm", "max_model_len"], None) or mml_default)
                ts_gmu = float(cfg_get(["timestamped_summary", "llm", "gpu_memory_utilization"], None) or gmu_default)
                ts_mns = int(cfg_get(["timestamped_summary", "llm", "max_num_seqs"], None) or mns_default)
                ts_mbt = int(cfg_get(["timestamped_summary", "llm", "max_num_batched_tokens"], None) or mbt_default)

                # Free previous engines to avoid OOM before starting a fresh instance
                with suppress(Exception):
                    llm_punc = None
                with suppress(Exception):
                    del llm_sum
                with suppress(Exception):
                    torch.cuda.empty_cache()  # type: ignore[attr-defined]
                gc.collect()

                llm_ts = build_llm(ts_model_id, ts_quant, ts_mml, ts_gmu, ts_mns, ts_mbt)

                # 1) Build sentence-level prompt text to cut tokens
                sentence_lines = prepare_sentence_timestamp_lines(final["segments"])
                if verbose:
                    logger.info(f"[post] Built sentence-level prompt with {len(sentence_lines.splitlines())} lines")

                # 2) Compute token counts and budget
                try:
                    tokenizer_ts = get_tokenizer(llm_ts)
                    t_in = len(tokenizer_ts.encode(sentence_lines)) if sentence_lines else 0
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Token counting for timestamped summary failed: {e}")
                    t_in = max(1, len(sentence_lines) // 3) if sentence_lines else 0

                ts_auto_ratio = float(cfg_get(["timestamped_summary", "auto_switch_ratio"], 0.98))
                ts_margin = int(cfg_get(["timestamped_summary", "auto_margin_tokens"], 128))
                token_limit = int(max(0.5, min(1.0, ts_auto_ratio)) * ts_mml) - ts_margin

                use_map_reduce = bool(cfg_get(["timestamped_summary", "map_reduce"], False)) or (t_in and token_limit > 0 and t_in > token_limit)

                if verbose:
                    logger.info(f"[post] Timestamped summary: tokens={t_in}, limit={token_limit}, map_reduce={use_map_reduce}")

                include_raw = cfg_get(["timestamped_summary", "return_raw"], False)

                if not use_map_reduce:
                    # 3) Single-pass on sentence lines
                    if verbose:
                        logger.info("[post] Using single-pass timestamped summary")
                    timestamped_summary_raw = generate_timestamped_summary(llm_ts, sentence_lines, system_prompt)
                    timestamped_summary = parse_llm_response(timestamped_summary_raw, include_raw=include_raw)
                    final["timestamped_summary"] = timestamped_summary
                else:
                    # 4) Chunked map-reduce: pack sentence lines into token-budgeted chunks
                    if verbose:
                        logger.info("[post] Using chunked map-reduce for timestamped summary")

                    chunks = pack_sentence_lines_to_budget(llm_ts, sentence_lines, ts_mml, out_margin=ts_margin, overhead=64)
                    if verbose:
                        logger.info(f"[post] Created {len(chunks)} chunks for timestamped summary")

                    chunk_msgs: list[list[dict[str, str]]] = []
                    for chunk_text in chunks:
                        if not chunk_text.strip():
                            continue
                        chunk_msgs.append([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": chunk_text},
                        ])

                    # Keep chunk outputs tight to maximize input space
                    params_max = min(192, ts_margin)
                    if chunk_msgs:
                        outs = generate_chat_batch(llm_ts, chunk_msgs, temperature=0.2, max_tokens=params_max)
                    else:
                        outs = []

                    # 5) Parse each chunk, merge timestamps
                    all_ts: list[tuple[str, float]] = []  # (text, start)
                    raw_concat: list[str] = []
                    for o in outs:
                        if not (o or "").strip():
                            continue
                        parsed = parse_llm_response(o, include_raw=False)
                        if include_raw:
                            raw_concat.append(o)
                        for item in parsed.get("timestamps", []) or []:
                            text_item = str(item.get("text", "")).strip()
                            start_item = float(item.get("start", 0.0))
                            if text_item:
                                all_ts.append((text_item, start_item))

                    # 6) Deduplicate and sort by time
                    from difflib import SequenceMatcher as _SequenceMatcher
                    all_ts.sort(key=lambda x: x[1])
                    merged: list[tuple[str, float]] = []
                    for text, s in all_ts:
                        if not text:
                            continue
                        if merged:
                            prev_text, ps = merged[-1]
                            # near-duplicate textual similarity or very close in time window
                            sim = _SequenceMatcher(None, prev_text.lower(), text.lower()).ratio()
                            if sim >= 0.85 or abs(s - ps) <= 2.0:
                                continue
                        merged.append((text, s))

                    # 7) Assemble final object
                    summary_text = " ".join([f"[{int(t//3600):02d}:{int((t%3600)//60):02d}:{int(t%60):02d}] {txt}" for (txt, t) in merged])
                    final_obj = {
                        "summary_text": summary_text,
                        "timestamps": [{"text": txt, "start": s} for (txt, s) in merged]
                    }
                    if include_raw and raw_concat:
                        final_obj["raw"] = "\n\n".join([r for r in raw_concat if r.strip()])
                    final["timestamped_summary"] = final_obj

                    if verbose:
                        logger.info(f"[post] Merged {len(merged)} unique timestamped topics")

                # Free timestamp model after use
                with suppress(Exception):
                    del llm_ts
                with suppress(Exception):
                    torch.cuda.empty_cache()  # type: ignore[attr-defined]
                gc.collect()
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"[post] Failed to generate timestamped summary: {e}")


    # Now it's safe to free any remaining engines (the process exits anyway)
    if not reuse:
        with suppress(Exception):
            del llm_sum
    with suppress(Exception):
        llm_punc = None
    # Only clear cache at end if debug flag is set
    if DEBUG_EMPTY_CACHE:
        with suppress(Exception):
            torch.cuda.empty_cache()  # type: ignore[attr-defined]
    gc.collect()

    final.update(
        {
            "segments": punct_segments,
            "transcript_punct": transcript_punct,
            "summary": summary,
            # New fields produced by postprocess for downstream consumers
            **({"quality_metrics": quality_metrics} if quality_metrics is not None else {}),
            **({"diffs": human_diff} if human_diff is not None else {}),
            "metadata": {
                **asr.get("metadata", {}),
                "llm_model_punctuation": p_model_id,
                "llm_model_summarization": s_model_id,
                "llm_offline": True,
                "decisions": decisions,
            },
        },
    )
    # Auto output dir per input file, if requested
    out_path = Path(args.out)
    if args.auto_outdir:
        from datetime import datetime
        # If ASR JSON already lives under paths.out_dir/<name>/asr.json, reuse that folder
        base_root = Path(str(cfg_get(["paths", "out_dir"], "data/output")))
        try:
            asr_parent = asr_json_path.resolve().parent
        except (OSError, ValueError):
            asr_parent = asr_json_path.parent
        if base_root in asr_parent.parents or asr_parent == base_root:
            base_dir = asr_parent
        else:
            # Otherwise, create a new folder based on input audio stem + timestamp
            audio_path_in = (asr.get("audio", {}) or {}).get("path")
            stem = Path(audio_path_in).stem if audio_path_in else asr_json_path.stem
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            candidate = base_root / f"{stem}-{timestamp}"
            if candidate.exists():
                suffix = 1
                while (base_root / f"{stem}-{timestamp}-{suffix:02d}").exists():
                    suffix += 1
                candidate = base_root / f"{stem}-{timestamp}-{suffix:02d}"
            candidate.mkdir(parents=True, exist_ok=True)
            base_dir = candidate
        out_path = base_dir / "final.json"

        # Optional separate files per config (supports legacy and structured OutputConfig)
        out_formats = []
        try:
            fmts = cfg_get(["output", "formats"], [])
            if isinstance(fmts, (list, tuple)):
                out_formats = [str(x).lower() for x in fmts]
        except (ValueError, KeyError, TypeError):
            out_formats = []
        write_text_files = bool(cfg_get(["output", "write_separate_files"], False)) or ("text" in out_formats or "md" in out_formats)

        if write_text_files:
            # Legacy defaults
            tf_name = str(cfg_get(["output", "transcript_file"], "transcript.txt"))
            sf_name = str(cfg_get(["output", "summary_file"], "summary.txt"))

            # Prefer structured OutputConfig filenames if available
            try:
                out_cfg = cfg.output  # type: ignore[attr-defined]
                if out_cfg and getattr(out_cfg, "transcript", None):
                    tcfg = out_cfg.transcript
                    # Use punctuated transcript file if provided
                    if getattr(tcfg, "file_punct", None):
                        tf_name = str(tcfg.file_punct)
                if out_cfg and getattr(out_cfg, "summary", None):
                    scfg = out_cfg.summary
                    if getattr(scfg, "file", None):
                        sf_name = str(scfg.file)
            except (ValueError, KeyError, TypeError):
                pass
            # transcript_punct plain text with optional wrapping
            with open(base_dir / tf_name, "w", encoding="utf-8") as tf:
                tf_text = (final.get("transcript_punct") or "").strip()
                # Prefer structured transcript.wrap_width if present (model validator maps legacy)
                wrap_width_cfg = cfg_get(["output", "transcript", "wrap_width"], None)
                if wrap_width_cfg is None:
                    wrap_width_cfg = cfg_get(["output", "wrap_width"], 100)
                try:
                    wrap_width = int(wrap_width_cfg) if wrap_width_cfg is not None else 0
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse wrap_width: {e}")
                    wrap_width = 0
                if wrap_width and wrap_width > 0:
                    paragraphs = tf_text.split("\n\n") if tf_text else []
                    wrapped_paragraphs: list[str] = []
                    for para in paragraphs:
                        p = para.strip()
                        if not p:
                            wrapped_paragraphs.append("")
                            continue
                        wrapped_paragraphs.append(
                            textwrap.fill(
                                p,
                                width=wrap_width,
                                replace_whitespace=False,
                                break_long_words=False,
                                break_on_hyphens=False,
                            ),
                        )
                    tf_text = "\n\n".join(wrapped_paragraphs).strip()
                tf.write(tf_text + "\n")
            # summary plain text (render bullets + abstract)
            summary = final.get("summary") or {}
            points = summary.get("bullets") or []
            abstract = summary.get("abstract") or ""
            with open(base_dir / sf_name, "w", encoding="utf-8") as sf:
                for p in points:
                    sf.write(f"- {p}\n")
                if points and abstract:
                    sf.write("\n")
                if abstract:
                    sf.write(abstract.strip() + "\n")

        # Write timed text formats if requested via formats
        try:
            fmts = set(out_formats)
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to parse output formats: {e}")
            fmts = set()

        def _fmt_time_srt(t: float) -> str:
            # SRT uses comma milliseconds
            import math as _m
            t = max(0.0, float(t))
            hh = int(t // 3600)
            mm = int((t % 3600) // 60)
            ss = int(t % 60)
            ms = _m.floor((t - _m.floor(t)) * 1000.0)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        def _fmt_time_vtt(t: float) -> str:
            # VTT uses dot milliseconds
            import math as _m
            t = max(0.0, float(t))
            hh = int(t // 3600)
            mm = int((t % 3600) // 60)
            ss = int(t % 60)
            ms = _m.floor((t - _m.floor(t)) * 1000.0)
            return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

        # Prepare filenames from structured config when available
        srt_name = str(cfg_get(["output", "transcript", "file_srt"], "transcript.srt"))
        vtt_name = str(cfg_get(["output", "transcript", "file_vtt"], "transcript.vtt"))

        if "srt" in fmts and final.get("segments"):
            cues = []
            for idx, seg in enumerate(final["segments"], start=1):
                start = float(seg.get("start", 0.0) or 0.0)
                end = float(seg.get("end", start) or start)
                text = str(seg.get("text_punct") or seg.get("text") or "").strip()
                if not text:
                    continue
                cues.append(f"{idx}\n{_fmt_time_srt(start)} --> {_fmt_time_srt(end)}\n{text}\n")
            (base_dir / srt_name).write_text("\n".join(cues), encoding="utf-8")

        if "vtt" in fmts and final.get("segments"):
            lines = ["WEBVTT", ""]
            for seg in final["segments"]:
                start = float(seg.get("start", 0.0) or 0.0)
                end = float(seg.get("end", start) or start)
                text = str(seg.get("text_punct") or seg.get("text") or "").strip()
                if not text:
                    continue
                lines.append(f"{_fmt_time_vtt(start)} --> {_fmt_time_vtt(end)}")
                lines.append(text)
                lines.append("")
            (base_dir / vtt_name).write_text("\n".join(lines), encoding="utf-8")

    save_json(out_path, final)


if __name__ == "__main__":
    main()
