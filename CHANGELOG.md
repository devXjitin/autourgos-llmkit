# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-01

### Changed
- Renamed `generate_response()` to `invoke()` across all provider classes.
- Renamed `generate_response_stream()` to `stream()` across all provider classes.
- Separated vision model initialization into `init_vision_llm()` factory function.
- Updated `init_llm()` to only support text-based models.
- Updated license to Proprietary License.

## [1.0.1] - 2026-02-22

### Changed
- Updated version and licensing information.
- Re-releasing due to package metadata updates.

## [1.0.0] - 2026-02-22

### Added
- Initial release of Autourgos LLM Kit.
- Support for Google Gemini (Text & Vision).
- Support for Anthropic Claude.
- Support for OpenAI GPT.
- Support for xAI Grok.
- Support for Microsoft Azure OpenAI & Foundry.
- Support for Vertex AI (Gemini & Model Garden).
- Support for Ollama (Local & Cloud).
- Unified `init_llm` factory.
- Exponential backoff and retry logic.
- Standardized error handling across all providers.
