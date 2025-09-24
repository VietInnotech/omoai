# Component: Configuration

## 1. Introduction

This document provides a detailed description of the Configuration component of the OmoAI system. The entire application is configured through a single YAML file, which is validated by Pydantic models.

## 2. Responsibilities

The main responsibilities of the Configuration component are:

*   **Centralized Configuration:** To provide a single source of truth for all configuration parameters.
*   **Validation:** To ensure that the configuration is valid and complete at application startup.
*   **Easy Management:** To allow for easy modification of the system's behavior without changing the code.

## 3. Technology Stack

*   **Configuration Language:** YAML
*   **Validation Library:** [Pydantic](https://docs.pydantic.dev/)

## 4. Process

1.  The `config.yaml` file is loaded at application startup.
2.  The content of the YAML file is parsed and loaded into Pydantic models.
3.  The Pydantic models validate the configuration data. If any required fields are missing or have incorrect types, a validation error is raised.
4.  The validated configuration is then made available to the rest of the application as a Python object.

## 5. Configuration File Structure

The `config.yaml` file is organized into sections, with each section corresponding to a specific component of the system. For example:

*   `paths`: For file system paths.
*   `llm`: For the base LLM configuration.
*   `asr`: For the ASR component.
*   `punctuation`: For the punctuation model.
*   `summarization`: For the summarization model.
*   `output`: For output formats and file names.
*   `api`: For the API server.
*   `logging`: For the logging system.
*   `alignment`: For the alignment component.
*   `vad`: For the Voice Activity Detection component.

## 6. Related Classes and Files

*   `config.yaml`: The main configuration file.
*   `src/omoai/config/schemas.py`: The file containing the Pydantic models used for configuration validation.
