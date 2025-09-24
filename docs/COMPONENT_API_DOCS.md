# Component: API

## 1. Introduction

This document provides a detailed description of the API component of the OmoAI system. The API is the primary interface for interacting with the ASR pipeline.

## 2. Responsibilities

The main responsibilities of the API component are:

*   **Request Handling:** To receive and process incoming HTTP requests for audio transcription.
*   **Pipeline Orchestration:** To manage the flow of data through the different stages of the ASR pipeline.
*   **Script Management:** To execute and monitor the external scripts that handle ASR and post-processing.
*   **Response Generation:** To format and deliver the final transcription results to the client.
*   **System Monitoring:** To provide endpoints for health checks and performance metrics.

## 3. Technology Stack

*   **Framework:** [Litestar](https://litestar.dev/)
*   **Server:** [Uvicorn](https://www.uvicorn.org/)

## 4. Endpoints

### 4.1. `POST /v1/pipeline`

*   **Description:** This is the main endpoint for submitting an audio file for transcription.
*   **Request:** The request should be a `multipart/form-data` request containing the audio file.
*   **Response:** The response is a JSON object containing the transcription results. The structure of the JSON object can be customized using query parameters.

### 4.2. `GET /v1/health`

*   **Description:** This endpoint is used to check the health of the API server and its dependencies.
*   **Response:** A JSON object indicating the status of the service.

### 4.3. `GET /v1/metrics`

*   **Description:** This endpoint provides performance metrics for the API server, such as request latency and throughput.
*   **Response:** A Prometheus-compatible text response.

## 4. Script-Based Pipeline Execution

The API uses a script-based approach for pipeline execution:

- **Preprocessing**: Managed by service layer calling `scripts/preprocess.py`
- **ASR**: Executed via `src/omoai/api/scripts/asr_wrapper.py` which runs `scripts/asr.py`
- **Post-processing**: Executed via `src/omoai/api/scripts/postprocess_wrapper.py` which runs `scripts/post.py`

This approach provides:
- **Process isolation**: Each component runs in separate processes
- **Resource management**: Independent GPU context for each stage
- **Fault tolerance**: Failures are contained within individual processes
- **CUDA compatibility**: Proper multiprocessing setup with spawn method

## 5. Configuration

The API component is configured in the `api` section of the `config.yaml` file. The following options are available:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_body_size_mb: 200
  request_timeout_seconds: 1800
  temp_dir: "/tmp"
  cleanup_temp_files: true
  stream_subprocess_output: true
  verbose_scripts: true
  enable_progress_output: true
  default_response_format: json
  allow_accept_override: true
  allow_query_format_override: true
  health_check_dependencies:
    - ffmpeg
    - config_file
```

### Configuration Options Explained:

- **stream_subprocess_output**: When true, subprocess output is streamed to console for real-time monitoring
- **verbose_scripts**: Enables verbose output from the external scripts
- **enable_progress_output**: Shows progress bars and status updates during processing
- **default_response_format**: Controls the default API response format (json/text)
- **allow_accept_override**: Allows clients to override response format via Accept header
- **allow_query_format_override**: Allows clients to override response format via query parameter

## 6. Related Classes and Files

*   `src/omoai/api/app.py`: The main application file that initializes the Litestar app.
*   `src/omoai/api/main_controller.py`: The controller for the main `/v1/pipeline` endpoint.
*   `src/omoai/api/health.py`: The controller for the `/v1/health` endpoint.
*   `src/omoai/api/metrics_middleware.py`: The middleware for collecting and exposing metrics.
*   `src/omoai/api/services.py`: The service layer that orchestrates the ASR pipeline.
*   `src/omoai/api/scripts/asr_wrapper.py`: Wrapper for executing the ASR script with proper environment.
*   `src/omoai/api/scripts/postprocess_wrapper.py`: Wrapper for executing the post-processing script with CUDA optimization.
