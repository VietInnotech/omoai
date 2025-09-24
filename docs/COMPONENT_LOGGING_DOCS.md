# Component: Logging

## 1. Introduction

This document provides a detailed description of the Logging component of the OmoAI system. The system uses a structured and configurable logging system to provide insights into the application's behavior.

## 2. Responsibilities

The main responsibilities of the Logging component are:

*   **Structured Logging:** To generate logs in a structured format (JSON) for easy parsing and analysis.
*   **Configurable Output:** To allow for the configuration of log levels, output formats (JSON or plain text), and destinations (console or file).
*   **Log Rotation:** To automatically rotate log files to prevent them from growing too large.

## 3. Technology Stack

*   **Library:** [Loguru](https://loguru.readthedocs.io/en/stable/)

## 4. Process

The logging system is initialized at the start of the application in `src/omoai/logging_system/logger.py`. The `setup_logging` function reads the logging configuration from the `config.yaml` file and configures the Loguru logger accordingly.

## 5. Configuration

The Logging component is configured in the `logging` section of the `config.yaml` file. The following options are available:

```yaml
logging:
  level: INFO
  format_type: structured # structured | json | simple
  enable_console: true
  enable_file: true
  log_file: "@logs/api_server.jsonl"
  enable_text_file: true
  text_log_file: "@logs/api_server.log"
  max_file_size: 10485760 # 10 MB
  backup_count: 5
  rotation: "10 MB"
  retention: "14 days"
  compression: "gz"
  enqueue: true
  debug_mode: true
  quiet_mode: false
```

## 6. Related Classes and Files

*   `src/omoai/logging_system/logger.py`: The file containing the main logic for setting up the logger.
*   `src/omoai/logging_system/config.py`: The file containing the Pydantic models for the logging configuration.
*   `src/omoai/logging_system/serializers.py`: The file containing the serializers for the structured logs.
