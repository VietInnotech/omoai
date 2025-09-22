"""
Forced Alignment with wav2vec2 CTC models
Self-contained implementation for phoneme alignment
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Audio constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token

# Language configurations
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]
PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

# Default alignment models
DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
}

# Type definitions
class SingleWordSegment(TypedDict):
    word: str
    start: float
    end: float
    score: float

class SingleCharSegment(TypedDict):
    char: str
    start: float
    end: float
    score: float

class SingleSegment(TypedDict):
    start: float
    end: float
    text: str

class SegmentData(TypedDict):
    clean_char: list[str]
    clean_cdx: list[int]
    clean_wdx: list[int]
    sentence_spans: list[tuple[int, int]]

class SingleAlignedSegment(TypedDict):
    start: float
    end: float
    text: str
    words: list[SingleWordSegment]
    chars: list[SingleCharSegment] | None

class AlignedTranscriptionResult(TypedDict):
    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]

# Data structures for alignment
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Path:
    points: list[Point]
    score: float

@dataclass
class BeamState:
    token_index: int
    time_index: int
    score: float
    path: list[Point]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def exact_div(x, y):
    if x % y != 0:
        raise ValueError(f"exact_div: {x} is not divisible by {y}")
    return x // y

def interpolate_nans(x, method='nearest'):
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()

def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def load_align_model(language_code: str, device: str, model_name: str | None = None, model_dir=None):
    """
    Load alignment model for specified language

    Args:
        language_code: Language code (e.g., 'en', 'fr', 'vi')
        device: Device to load model on ('cpu', 'cuda', etc.)
        model_name: Optional specific model name
        model_dir: Optional model directory (defaults to /home/cetech/omoai/models)

    Returns:
        tuple: (model, metadata)
    """
    # Set default model directory if not provided
    if model_dir is None:
        model_dir = "/home/cetech/omoai/models"
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
        except Exception as e:
            raise ValueError(f'Failed to load model "{model_name}" from huggingface: {e}') from e
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char, code in labels.items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}
    return align_model, align_metadata

def get_trellis(emission, tokens, blank_id=0):
    """
    Build trellis for Viterbi alignment

    Args:
        emission: Emission probabilities (T, V)
        tokens: Token sequence
        blank_id: Blank token ID

    Returns:
        trellis: Trellis matrix (T, N)
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis

def get_wildcard_emission(frame_emission, tokens, blank_id):
    """Processing token emission scores containing wildcards (vectorized version)"""
    if not (0 <= blank_id < len(frame_emission)):
        raise ValueError(f"blank_id {blank_id} out of range [0, {len(frame_emission)})")

    tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
    wildcard_mask = (tokens == -1)

    regular_scores = frame_emission[tokens.clamp(min=0).long()]

    max_valid_score = frame_emission.clone()
    max_valid_score[blank_id] = float('-inf')
    max_valid_score = max_valid_score.max()

    result = torch.where(wildcard_mask, max_valid_score, regular_scores)
    return result

def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Standard CTC backtracking

    Args:
        trellis: Trellis matrix
        emission: Emission probabilities
        tokens: Token sequence
        blank_id: Blank token ID

    Returns:
        path: List of Point objects
    """
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        if t <= 0:
            raise ValueError(f"Invalid time index t={t}, must be > 0")

        p_stay = emission[t - 1, blank_id]
        p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        t -= 1
        if changed > stayed:
            j -= 1

        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    """
    Beam search CTC backtracking

    Args:
        trellis: Trellis matrix
        emission: Emission probabilities
        tokens: Token sequence
        blank_id: Blank token ID
        beam_width: Beam search width

    Returns:
        path: List of Point objects or None if failed
    """
    t_max, j_max = trellis.size(0) - 1, trellis.size(1) - 1

    init_state = BeamState(
        token_index=j_max,
        time_index=t_max,
        score=trellis[t_max, j_max],
        path=[Point(j_max, t_max, emission[t_max, blank_id].exp().item())]
    )

    beams = [init_state]

    while beams and any(beam.token_index > 0 for beam in beams):
        next_beams = []

        for beam in beams:
            t, j = beam.time_index, beam.token_index

            if t <= 0:
                continue

            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

            # Properly compute transition scores
            stay_score = trellis[t - 1, j] + p_stay
            change_score = trellis[t - 1, j - 1] + p_change if j > 0 else float('-inf')

            # Stay
            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(BeamState(
                    token_index=j,
                    time_index=t - 1,
                    score=stay_score,
                    path=new_path
                ))

            # Change
            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(BeamState(
                    token_index=j - 1,
                    time_index=t - 1,
                    score=change_score,
                    path=new_path
                ))

        if not next_beams:
            break

        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]

    if not beams:
        return None

    best_beam = beams[0]
    t = best_beam.time_index
    j = best_beam.token_index

    # Complete the path by adding remaining blank transitions
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1

    return best_beam.path[::-1]

def merge_repeats(path, transcript):
    """Merge consecutive repeated tokens in the alignment path"""
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    """Merge character segments into word segments"""
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def align(
    transcript: list[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: str | np.ndarray | torch.Tensor,
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.

    Args:
        transcript: List of transcript segments
        model: Alignment model
        align_model_metadata: Model metadata
        audio: Audio data (path or array)
        device: Device to run on
        interpolate_method: Interpolation method for missing values
        return_char_alignments: Whether to return character alignments
        print_progress: Whether to print progress
        combined_progress: Whether to use combined progress reporting

    Returns:
        AlignedTranscriptionResult with aligned segments and word segments
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    # max_duration = audio.shape[1] / SAMPLE_RATE  # Unused variable

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    segment_data: dict[int, SegmentData] = {}

    for sdx, segment in enumerate(transcript):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # Split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            # Ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            else:
                # Add placeholder
                clean_char.append('*')
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)
            else:
                # Index for placeholder
                clean_wdx.append(wdx)

        # Simple sentence splitting (NLTK not available, using basic approach)
        sentence_spans = [(0, len(text))]  # Single sentence for now

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    aligned_segments: list[SingleAlignedSegment] = []

    # Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # Check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= audio.shape[1] / SAMPLE_RATE:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        tokens = [model_dictionary.get(c, -1) for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1
        if duration <= 0:
            print(f'Failed to align segment ("{segment["text"]}"): invalid duration {duration}, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if trellis.size(0) <= 1:
            print(f'Failed to align segment ("{segment["text"]}"): insufficient trellis frames {trellis.size(0)}, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # Assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append({
                "char": char,
                "start": start,
                "end": end,
                "score": score,
                "word-idx": word_idx,
            })

            # Increment word_idx
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1

        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # Assign sentence_idx to each character index
        char_segments_arr["sentence-idx"] = None
        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
            curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
            char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx2

            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != ' ']
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue

                # Don't use space character for alignment
                word_chars = word_chars[word_chars["char"] != " "]

                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                # -1 indicates unalignable
                word_segment = {"word": word_text}

                if not pd.isna(word_start):
                    word_segment["start"] = word_start
                if not pd.isna(word_end):
                    word_segment["end"] = word_end
                if not pd.isna(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)

            aligned_subsegments.append({
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            })

            if return_char_alignments:
                curr_chars = curr_chars[["char", "start", "end", "score"]]
                curr_chars.fillna(-1, inplace=True)
                curr_chars = curr_chars.to_dict("records")
                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                aligned_subsegments[-1]["chars"] = curr_chars

        aligned_subsegments = pd.DataFrame(aligned_subsegments)
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)

        # Concatenate sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')
        aligned_segments += aligned_subsegments

    # Create word_segments list
    word_segments: list[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}

# Compatibility functions for existing API
def to_whisperx_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map OmoAI segments to the format expected by the alignment library."""
    def _parse_time_to_seconds(time_str):
        """Parse time strings in HH:MM:SS:MMM format to seconds."""
        if time_str is None:
            return 0.0
        if isinstance(time_str, (int, float)):
            return float(time_str)

        # Handle string formats like "00:00:00:081" (HH:MM:SS:MMM)
        if isinstance(time_str, str) and ':' in time_str:
            try:
                parts = time_str.split(':')
                if len(parts) == 4:  # HH:MM:SS:MMM format
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    milliseconds = int(parts[3])
                    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
                elif len(parts) == 3:  # HH:MM:SS format
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS format
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            except (ValueError, TypeError):
                pass

        # Fallback: try direct float conversion
        try:
            return float(time_str)
        except (ValueError, TypeError):
            return 0.0

    p2s = _parse_time_to_seconds

    out: list[dict[str, Any]] = []
    for s in segments:
        text = (s.get("text_raw") or s.get("text") or "").strip()
        if not text:
            continue

        start_val = p2s(s.get("start")) if s.get("start") is not None else 0.0
        end_val = p2s(s.get("end")) if s.get("end") is not None else start_val

        # Ensure we have valid start and end times
        start = float(start_val) if start_val is not None else 0.0
        end = float(end_val) if end_val is not None else start

        # Ensure end is not before start
        if end < start:
            end = start

        out.append({"start": start, "end": end, "text": text})
    return out

def load_alignment_model(language: str, device: str, model_name: str | None = None):
    """Load the alignment model."""
    return load_align_model(language_code=language, device=device, model_name=model_name)

def align_segments(
    wx_segments: list[dict[str, Any]],
    audio_path_or_array: Any,
    model: Any,
    metadata: dict,
    device: str,
    *,
    return_char_alignments: bool,
    interpolate_method: str,
    print_progress: bool,
) -> dict:
    """Run alignment using the self-contained implementation."""
    # Convert dict segments to TypedDict format
    typed_segments: list[SingleSegment] = []
    for seg in wx_segments:
        typed_segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"]
        })

    result = align(
        typed_segments,
        model,
        metadata,
        audio_path_or_array,
        device,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
        print_progress=print_progress,
    )
    return dict(result)

def merge_alignment_back(
    original_segments: list[dict[str, Any]],
    aligned_result: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge the aligned words/chars back into our original segment objects."""
    aligned_segments = aligned_result.get("segments", []) or []
    word_segments = aligned_result.get("word_segments", []) or []

    out = [dict(s) for s in original_segments]
    # Attach words/chars to corresponding non-empty text segments in order
    j = 0
    for i, s in enumerate(out):
        text = (s.get("text_raw") or s.get("text") or "").strip()
        if not text:
            continue
        if j >= len(aligned_segments):
            break
        al = aligned_segments[j] or {}
        # Preserve existing fields and add enrichments
        if al.get("words"):
            s["words"] = al["words"]
        if "chars" in al:
            s["chars"] = al["chars"]
        out[i] = s
        j += 1
    return out, list(word_segments)
