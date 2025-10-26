# Free Model Providers Research

## Current State
Supporting **OpenRouter only** with 6 free models (rate limited to 20 RPM).

## Recommended Models

**Total: 20 unique models across 7 providers**

| Provider | Models | Free Tier Limits | Models |
|----------|--------|------------------|--------|
| **Google AI Studio** | 4 | 15 RPM, 1500 RPD | Gemini 2.5 Pro, 2.0 Flash, 1.5 Flash, 1.5 Pro |
| **Groq** | 5 | 30 RPM, 1-14.4K RPD | Llama 3.3 70B, Llama 3.1 8B, GPT-OSS 20B/120B, Qwen 3 32B |
| **Hugging Face** | 2 | 1,000 req/day | Qwen 2.5 72B, Phi-3.5 Mini |
| **Cerebras** | 1 | 1M tokens/day | Llama 3.1 70B |
| **Mistral AI** | 2 | 100 RPM | Mistral Small, Mistral Nemo |
| **Cohere** | 2 | 100 RPM | Command R+, Command R |
| **OpenRouter** | 4 | 20 RPM | DeepSeek v3, GLM-4.5 Air, Nemotron 9B, Gemini 2.0 Flash |

**Removed Duplicates:**
- Llama 3.3 70B from OpenRouter (use Groq for faster)
- GPT-OSS 20B from OpenRouter (use Groq)

## Implementation

### Use Litellm SDK
Covers all 7 providers. Install: `poetry add litellm`

### Changes Required

**Delete:**
- `src/benchmark/providers/openrouter_client.py` (389 lines)
- `src/benchmark/core/rate_limit.py` (53 lines)

**Create:**
- `src/benchmark/providers/litellm_provider.py` (wraps litellm, ~50-100 lines)

**Update:**
- `data/models/benchmark_models.json` - Add 20 new models
- `data/models/llm_labeling_models.json` - Add 20 new models
- `.env.example` - Add API keys (GOOGLE_AI_API_KEY, GROQ_API_KEY, etc.)

**Result:** 26 models total, -350 lines of code, automatic rate limiting & retries

### What Litellm Replaces
- Rate limiting (TokenBucket, 100 lines)
- HTTP client management (20 lines)
- Retry logic (200 lines)
- Response parsing (50 lines)
- Custom logging in provider (100 lines)

**Total:** ~450 lines → replaced by litellm's built-in features

### What to Keep
- `ChatProvider` protocol (wrap litellm)
- Core types (`ChatMessage`, `ModelResponse`, etc.)
- Model config files
- All business logic (scoring, prompts, runners)
