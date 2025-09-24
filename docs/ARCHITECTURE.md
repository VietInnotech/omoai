# OmoAI Architecture

## 1. Introduction

This document outlines the architecture of the OmoAI project, a comprehensive Automatic Speech Recognition (ASR) pipeline designed with a primary focus on the Vietnamese language. It details the system's components, their responsibilities, and the interactions between them.

## 2. High-Level Architecture

The OmoAI system is designed as a modular, multi-stage pipeline that processes audio files to produce punctuated and summarized transcripts with word-level timestamps. The architecture is built around a central API server that orchestrates the entire workflow.

```
                               +---------------------+
                               |     API Server      |
                               |    (Litestar)       |
                               +---------------------+
                                         |
                                         v
+------------------+      +--------------------------+      +----------------------+
|  Preprocessing   |----->|      ASR (ChunkFormer)   |----->|       Alignment      |
| (ffmpeg, VAD)    |      |                          |      |      (wav2vec2)      |
+------------------+      +--------------------------+      +----------------------+
                                         |
                                         v
                               +----------------------+
                               |    Post-processing   |
                               | (LLM - Punctuation & |
                               |      Summarization)  |
                               +----------------------+
                                         |
                                         v
                               +---------------------+
                               |   Output Generation |
                               | (JSON, SRT, VTT, etc.)|
                               +---------------------+
```

## 3. Component Breakdown

### 3.1. API Server

*   **Description:** The API server is the main entry point for the OmoAI system. It exposes a RESTful API for submitting audio files for processing and retrieving the results.
*   **Technology:** [Litestar](https://litestar.dev/), a modern and fast ASGI framework.
*   **Responsibilities:**
    *   Handling incoming HTTP requests.
    *   Validating request payloads.
    *   Orchestrating the execution of the ASR pipeline.
    *   Formatting and returning the final output to the client.
    *   Providing health check and metrics endpoints.
*   **Location:** `src/omoai/api/app.py`

### 3.2. Preprocessing

*   **Description:** This component is responsible for preparing the audio for the ASR model.
*   **Technology:**
    *   `ffmpeg` for audio format conversion.
    *   `silero-vad`, `webrtcvad` for Voice Activity Detection (VAD).
*   **Responsibilities:**
    *   Converting input audio to a standardized format (16kHz mono WAV).
    *   Detecting speech segments in the audio to remove silence and split long audio files into manageable chunks.
*   **Location:** The VAD logic is integrated within `scripts/asr.py`.

### 3.3. ASR (ChunkFormer)

*   **Description:** The core ASR component responsible for transcribing audio into text.
*   **Technology:** ChunkFormer, a transformer-based model for streaming ASR.
*   **Responsibilities:**
    *   Generating a raw transcript from the preprocessed audio chunks.
    *   Producing segment-level timestamps.
*   **Location:** The ASR process is orchestrated by `scripts/asr.py`, which utilizes the ChunkFormer model located in `src/chunkformer`.

### 3.4. Alignment

*   **Description:** This component refines the timestamps from the ASR model to provide word-level accuracy.
*   **Technology:** A wav2vec2-based model.
*   **Responsibilities:**
    *   Performing forced alignment between the audio and the generated transcript.
    *   Generating accurate start and end times for each word.
*   **Location:** `src/omoai/integrations/alignment.py`

### 3.5. Post-processing (LLM)

*   **Description:** This component enhances the raw transcript using a Large Language Model (LLM).
*   **Technology:** `vLLM` for efficient LLM inference. The model used is `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`.
*   **Responsibilities:**
    *   **Punctuation:** Adding correct punctuation and capitalization to the transcript.
    *   **Summarization:** Generating a concise summary of the transcript.
    *   **Timestamped Summary:** Creating a list of key topics with their corresponding timestamps.
*   **Location:** The post-processing logic is handled by `scripts/post.py`, with prompts and configurations defined in `config.yaml`. The API uses `src/omoai/api/scripts/postprocess_wrapper.py` to manage the script execution with proper working directory and environment settings.

### 3.6. Configuration

*   **Description:** The configuration of the entire application is managed through a central YAML file and validated using Pydantic models.
*   **Technology:** `PyYAML` for parsing the YAML file and `Pydantic` for data validation.
*   **Responsibilities:**
    *   Providing a centralized and easily editable way to configure all components of the pipeline.
    *   Ensuring that the configuration is valid at runtime.
*   **Location:** `config.yaml` and `src/omoai/config/schemas.py`.

### 3.7. Logging

*   **Description:** The logging system provides structured and configurable logging for the entire application.
*   **Technology:** [Loguru](https://loguru.readthedocs.io/en/stable/), a library that aims to make logging in Python simple and enjoyable.
*   **Responsibilities:**
    *   Generating structured logs in either JSON or a human-readable format.
    *   Configuring log levels, sinks (console, file), rotation, and retention.
*   **Location:** `src/omoai/logging_system/`.

## 4. Data Flow

1.  A client sends a POST request with an audio file to the `/v1/pipeline` endpoint of the API server.
2.  The API server saves the audio file to a temporary location.
3.  The preprocessing stage is initiated: `ffmpeg` converts the audio to the required format.
4.  The VAD model detects speech segments, which are then passed to the ASR model.
5.  The ChunkFormer ASR model transcribes the audio segments, generating a raw transcript with segment-level timestamps.
6.  The alignment module processes the raw transcript and the audio to produce word-level timestamps.
7.  The post-processing script sends the transcript to the LLM for punctuation, summarization, and timestamped topic extraction.
8.  The final results, including the punctuated transcript, summary, timestamped topics, and detailed timestamp information, are compiled into a JSON object.
9.  The API server returns the JSON object to the client and, if configured, saves the output to disk in various formats (JSON, SRT, VTT, etc.).

## 5. Deployment Considerations

The application is designed to be containerized using Docker for easy deployment and scalability. A `Dockerfile` can be created to package the application and all its dependencies. The containerized application can then be deployed on any cloud provider or on-premise infrastructure that supports Docker.

## 6. Script-Based Pipeline Architecture

The system uses a script-based approach for better isolation and resource management:

- **Preprocessing**: Handled by `scripts/preprocess.py` via API service layer
- **ASR**: Orchestrated by `scripts/asr.py` via `src/omoai/api/scripts/asr_wrapper.py`
- **Post-processing**: Managed by `scripts/post.py` via `src/omoai/api/scripts/postprocess_wrapper.py`

This architecture provides:
- **Process isolation**: Each major component runs in separate processes
- **Resource management**: Independent memory and GPU context for each stage
- **Fault tolerance**: Failures in one component don't affect others
- **CUDA compatibility**: Proper multiprocessing setup with spawn method

## 7. Memory Optimization Features

The system includes advanced CUDA memory management:

- **Expandable segments**: Reduces memory fragmentation (PYTORCH_CUDA_ALLOC_CONF)
- **Garbage collection**: Automatic cleanup at 80% utilization threshold
- **Pinned memory**: Optimized host-device transfers with background threads
- **Multiprocessing**: Spawn method enforced for CUDA compatibility
- **vLLM worker optimization**: Dedicated multiprocessing settings for LLM inference
- **GPU cache clearing**: Automatic VRAM cleanup between pipeline stages
