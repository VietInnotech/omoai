# Component: Post-processing

## 1. Introduction

This document provides a detailed description of the Post-processing component of the OmoAI system. This component uses a Large Language Model (LLM) to enhance the raw ASR transcript.

## 2. Responsibilities

The main responsibilities of the Post-processing component are:

*   **Punctuation:** To add correct punctuation and capitalization to the transcript.
*   **Summarization:** To generate a concise summary of the transcript.
*   **Timestamped Summary:** To create a list of key topics with their corresponding timestamps.

## 3. Technology Stack

*   **LLM Inference Engine:** `vLLM`
*   **Model:** `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`

## 4. Process

The post-processing is handled by the `scripts/post.py` script. The script takes the raw ASR output as input and performs the following steps:

1.  **Punctuation:** The raw transcript is sent to the LLM with a specific prompt to add punctuation and capitalization.
2.  **Summarization:** The punctuated transcript is sent to the LLM with a different prompt to generate a summary.
3.  **Timestamped Summary:** The transcript with word-level timestamps is sent to the LLM to extract key topics and their start times.

## 5. Configuration

The Post-processing component is configured in the `punctuation`, `summarization`, and `timestamped_summary` sections of the `config.yaml` file. These sections define the LLM to use, the prompts, and other parameters.

## 6. Script Wrapper Architecture

The post-processing is executed through `src/omoai/api/scripts/postprocess_wrapper.py` which provides:

- **Working directory management**: Ensures scripts run from project root
- **Environment configuration**: Sets up CUDA and multiprocessing environment
- **Memory optimization**: Configures PyTorch CUDA allocator for reduced fragmentation
- **Process isolation**: Runs the post-processing script as a subprocess
- **Error handling**: Proper exception handling and error reporting

Key environment variables configured by the wrapper:
- `MULTIPROCESSING_START_METHOD=spawn`: CUDA-compatible multiprocessing
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: vLLM worker optimization
- `PYTORCH_CUDA_ALLOC_CONF`: Memory optimization settings
- `CUDA_VISIBLE_DEVICES`: GPU device isolation

## 7. Related Classes and Files

*   `scripts/post.py`: The main script for running the post-processing tasks.
*   `src/omoai/api/scripts/postprocess_wrapper.py`: The wrapper that manages script execution with proper environment setup.
*   `config.yaml`: The file containing the prompts and configurations for the LLM.
