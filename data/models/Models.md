# Supported LLM Models

This document describes the LLM models available for benchmarking and labeling in this project.

## Overview

We support **29 models** across **7 providers**, all accessible via the LiteLLM unified API. Models are configured in `benchmark_models.json` and `llm_labeling_models.json`.

## Model Providers

### Google AI Studio
**Free Tier Limits:** Varies by model (see model-specific limits below)

**Models:**
- `google/gemini-2.5-pro` - Gemini 2.5 Pro (2 RPM, 125K TPM, 50 RPD)
- `google/gemini-2.5-flash` - Gemini 2.5 Flash (10 RPM, 250K TPM, 250 RPD)
- `google/gemini-2.5-flash-lite` - Gemini 2.5 Flash Lite (15 RPM, 250K TPM, 1K RPD)
- `google/gemini-2.0-flash` - Gemini 2.0 Flash (15 RPM, 1M TPM, 200 RPD)
- `google/gemini-2.0-flash-exp` - Gemini 2.0 Flash Exp (10 RPM, 250K TPM, 50 RPD)
- `google/gemini-2.0-flash-lite` - Gemini 2.0 Flash Lite (30 RPM, 1M TPM, 200 RPD)

**API Key:** Set `GOOGLE_AI_API_KEY` in your `.env` file. Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Groq
**Free Tier Limits:** 30 RPM (most models), 1-14.4K requests per day (varies by model)

**Models:**
- `groq/llama-3.3-70b-versatile` - Llama 3.3 70B Versatile (30 RPM, 12K TPM, 1K RPD)
- `groq/llama-3.1-8b-instant` - Llama 3.1 8B Instant (30 RPM, 6K TPM, 14.4K RPD)
- `groq/openai/gpt-oss-20b` - GPT-OSS 20B (30 RPM, 8K TPM, 1K RPD)
- `groq/openai/gpt-oss-120b` - GPT-OSS 120B (30 RPM, 8K TPM, 1K RPD)
- `groq/qwen/qwen3-32b` - Qwen 3 32B (60 RPM, 6K TPM, 1K RPD)
- `groq/meta-llama/llama-4-maverick-17b-128e-instruct` - Llama 4 Maverick 17B 128E (30 RPM, 6K TPM, 1K RPD)
- `groq/meta-llama/llama-4-scout-17b-16e-instruct` - Llama 4 Scout 17B 16E (30 RPM, 30K TPM, 1K RPD)
- `groq/compound` - Compound (30 RPM, 70K TPM, 250 RPD)
- `groq/compound-mini` - Compound Mini (30 RPM, 70K TPM, 250 RPD)
- `groq/moonshotai/kimi-k2-instruct` - Kimi K2 Instruct (60 RPM, 10K TPM, 1K RPD)

**API Key:** Set `GROQ_API_KEY` in your `.env` file. Get your key from [Groq Console](https://console.groq.com/keys).

### Hugging Face
**Free Tier Limits:** 1,000 requests per day

**Models:**
- `huggingface/qwen-2.5-72b` - Qwen 2.5 72B Instruct
- `huggingface/phi-3.5-mini` - Phi-3.5 Mini Instruct

**API Key:** Set `HUGGINGFACE_API_KEY` in your `.env` file. Get your key from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Cerebras
**Free Tier Limits:** 1M tokens per day

**Models:**
- `cerebras/llama-3.1-70b` - Llama 3.1 70B Instruct

**API Key:** Set `CEREBRAS_API_KEY` in your `.env` file. Get your key from [Cerebras Console](https://console.cerebras.ai/).

### Mistral AI
**Free Tier Limits:** 100 RPM

**Models:**
- `mistral/mistral-small-latest` - Mistral Small
- `mistral/mistral-medium-latest` - Mistral Medium
- `mistral/mistral-large-latest` - Mistral Large
- `mistral/magistral-medium-2506` - Magistral Medium

**API Key:** Set `MISTRAL_API_KEY` in your `.env` file. Get your key from [Mistral Console](https://console.mistral.ai/).

### Cohere
**Free Tier Limits:** Varies by model

**Models:**
- `cohere/command-r-plus-08-2024` - Command R+ (08-2024)
- `cohere/command-r-08-2024` - Command R (08-2024)
- `cohere/command-a-03-2025` - Command A (03-2025)

**API Key:** Set `COHERE_API_KEY` in your `.env` file. Get your key from [Cohere Dashboard](https://dashboard.cohere.com/api-keys).

### OpenRouter
**Free Tier Limits:** 20 RPM

**Models:**
- `openrouter/deepseek/deepseek-chat-v3-0324:free` - DeepSeek Chat v3 (free)
- `openrouter/z-ai/glm-4.5-air:free` - GLM-4.5 Air (free)
- `openrouter/nvidia/nemotron-nano-9b-v2:free` - Nemotron Nano 9B v2 (free)
- `openrouter/tngtech/deepseek-r1t2-chimera:free` - DeepSeek R1T2 Chimera (free)

**API Key:** Set `OPENROUTER_API_KEY` in your `.env` file. Get your key from [OpenRouter](https://openrouter.ai/keys).

## Model Configuration

Models are configured in JSON files with the following structure:

```json
{
  "id": "provider/model-id",
  "label": "Human-readable model name",
  "provider": "Provider Name",
  "concurrency": 3
}
```

**Fields:**
- `id`: Model identifier used in code (matches LiteLLM format)
- `label`: Display name for the model
- `provider`: Provider name for organization
- `concurrency`: Maximum concurrent requests for this model

## Rate Limiting

Rate limiting is handled with custom retry logic that extracts `Retry-After` headers and rate limit information from API responses. The system uses LiteLLM for API access but implements custom retry mechanisms that:
- Extract retry times from HTTP headers (`retry-after`, `x-ratelimit-reset-requests`, etc.)
- Parse retry instructions from error messages (e.g., Gemini's "Please retry in X.XXXs")
- Handle provider-specific rate limit formats (e.g., Groq's time format headers like "2m59.56s")
- Fall back to exponential backoff when header information is unavailable

Concurrency is configured per-model in the JSON configuration files to respect provider limits.

## Adding New Models

To add a new model:

1. Add the model entry to both `benchmark_models.json` and `llm_labeling_models.json`
2. Ensure the `id` field matches LiteLLM's expected format (e.g., `provider/model-name`)
3. Set appropriate `concurrency` based on provider limits
4. Update this documentation with the new model information

## Model ID Format

Model IDs follow the pattern: `provider/model-name` where:
- `provider` is the LiteLLM provider prefix (e.g., `google`, `groq`, `openrouter`)
- `model-name` matches the provider's model identifier

For OpenRouter models, the full path includes the organization: `openrouter/org/model-name:tag`

## Implementation Details

- All models are accessed via the unified LiteLLM API (`src/benchmark/providers/litellm_provider.py`)
- API keys are passed directly to LiteLLM calls (no environment variable mutation)
- Rate limiting and retries are handled automatically by LiteLLM
- Model selection is configured via environment variables (`LLM_MODEL`, `LLM_SCORER_MODEL`) or runtime arguments
