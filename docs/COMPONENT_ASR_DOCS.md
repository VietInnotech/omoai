# Component: ASR

## 1. Introduction

This document provides a detailed description of the Automatic Speech Recognition (ASR) component of the OmoAI system. This is the core component responsible for converting audio to text.

## 2. Responsibilities

The main responsibilities of the ASR component are:

*   **Transcription:** To generate a raw text transcript from the input audio.
*   **Timestamp Generation:** To produce segment-level timestamps for the transcription.

## 3. Technology Stack

*   **Model:** ChunkFormer, a transformer-based model for streaming ASR.
*   **Framework:** PyTorch

## 4. Process

The ASR process is orchestrated by the `scripts/asr.py` script. The script performs the following steps:

1.  **Model Initialization:** Loads the pre-trained ChunkFormer model and character dictionary.
2.  **Audio Processing:** Loads the audio file and converts it to a log-mel filterbank feature representation.
3.  **Decoding:** The script iterates through the audio features in chunks and uses the ChunkFormer model to decode the speech into text.
4.  **Timestamp Calculation:** The output of the model includes timestamps for each transcribed segment.
5.  **Output Generation:** The script generates a JSON file containing the raw transcript and the segment-level timestamps.

## 5. Configuration

The ASR component is configured in the `asr` section of the `config.yaml` file. The following options are available:

```yaml
asr:
  total_batch_duration_s: 3600
  chunk_size: 64
  left_context_size: 128
  right_context_size: 128
  device: auto
  autocast_dtype: fp16
```

## 6. Related Classes and Files

*   `scripts/asr.py`: The main script for running the ASR process.
*   `src/chunkformer/`: The directory containing the ChunkFormer model implementation.
*   `models/chunkformer/`: The directory where the pre-trained ChunkFormer model checkpoint is stored.
