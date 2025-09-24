# Component: Alignment

## 1. Introduction

This document provides a detailed description of the Alignment component of the OmoAI system. This component is responsible for refining the timestamps from the ASR model to provide word-level accuracy.

## 2. Responsibilities

The main responsibilities of the Alignment component are:

*   **Forced Alignment:** To align the ASR transcript with the audio on a word-by-word basis.
*   **Word-level Timestamps:** To generate accurate start and end times for each word in the transcript.

## 3. Technology Stack

*   **Model:** A wav2vec2-based model.
*   **Framework:** PyTorch

## 4. Process

The alignment process is integrated into the `scripts/asr.py` script and is executed if `alignment.enabled` is set to `true` in the `config.yaml` file. The process involves the following steps:

1.  **Model Loading:** The appropriate wav2vec2 model is loaded based on the specified language.
2.  **Alignment:** The `align_segments` function from `src/omoai/integrations/alignment.py` is called to perform the forced alignment.
3.  **Result Merging:** The word-level timestamps are merged back into the main ASR output.

## 5. Configuration

The Alignment component is configured in the `alignment` section of the `config.yaml` file. The following options are available:

```yaml
alignment:
  enabled: true
  language: vi
  device: auto
  return_char_alignments: false
  interpolate_method: nearest
  print_progress: true
```

## 6. Related Classes and Files

*   `src/omoai/integrations/alignment.py`: The file containing the core logic for the alignment process.
*   `scripts/asr.py`: The script that orchestrates the alignment process.
