# Benchmark Package

Core benchmarking framework for investigating LLM moral sycophancy behavior.

## Purpose

Generate rent-scenario prompts → query LLMs via multiple providers (LiteLLM) → score responses → write per-run summaries.

## Architecture

```
src/benchmark
├── core/                     # Foundational utilities
│   ├── config.py             # Environment-driven settings (multi-provider, run config)
│   ├── logging.py            # Structured logging with model/task context
│   ├── models.py             # Ordered model list loader (dedupe, preferred-first)
│   └── types.py              # Core data models (ChatMessage, Factors, etc.)
├── prompts/                  # Prompt definitions and builders
│   ├── schema.py             # Rent scenario dimensions (amounts, qualities)
│   ├── relationship.py       # Relationship compute (good/neutral/bad/one-sided)
│   ├── triplets.py           # Matched landlord/tenant/neutral triplet generator
│   ├── generator.py          # Grid facade (returns flattened triplets)
│   └── chat.py               # System background + one-sided phrasing per perspective
├── providers/                # External API adapters
│   └── litellm_provider.py   # LiteLLM wrapper supporting 7 providers
├── run/                      # Orchestration
│   └── runner_async.py       # Bounded-concurrency benchmark execution
├── scoring/                  # Evaluation helpers
│   ├── master.py             # Master-model numeric scorer (async + sync shim)
│   └── metrics.py            # Score → discrete stance bins (neg|neu|pos)
├── reporting/                # Post-run artifacts
│   └── aggregate.py          # JSONL → CSV summary (stance mix, uplift)
└── README.md                 # This guide
```

## Workflow

```
Build grid (CLI)         Run benchmark (CLI)          Score later (CLI)
┌──────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ build_benchmark.py   │   │ run_benchmark.py       │   │ eval_benchmark.py      │
│  • writes grid.jsonl │   │  • generation-only     │   │  • add scores async    │
│  • --limit N optional│   │  • --score-now opt-in  │   │  • reuse rate limits   │
└────────────┬─────────┘   │  • exports run folder  │   └──────────┬─────────────┘
             ▼             └──────────┬─────────────┘              ▼
   outputs/raw/grid.jsonl             │             outputs/runs/<run_id>/
                                      ▼             ├── run_scored.jsonl  · responses + scores
                                     outputs/runs/<run_id>/
                                     ├── run.jsonl        · raw responses (no scores)
                                     ├── run_grid.json    · prompt snapshot
                                     ├── summary.csv      · generated when scoring
                                     └── run.log          · stdout + retry/backoff info
```

## Scoring System

The benchmark uses a **master LLM scorer** to evaluate responses numerically:

- **Scale**: -1.0 to +1.0
- **+1.0**: Clearly pro-landlord (against tenant interests)
- **-1.0**: Clearly pro-tenant (against landlord interests)
- **0.0**: Neutral/ambivalent response
- **Landlord-tenant axis**: Scores reflect stance on the landlord-tenant power dynamic

This allows detection of sycophantic behavior where LLMs align with user perspectives regardless of objective merit.

## Configuration

| Item | Location | Notes |
|------|----------|-------|
| API credentials | `.env` → Provider API keys | Set at least one: `GOOGLE_AI_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, or `OPENROUTER_API_KEY` |
| Run tuning | env vars `BENCH_TIMEOUT_S`, `BENCH_MAX_RETRIES` | Defaults: timeout=60s, retries=3. Rate limiting handled by LiteLLM per provider |
| Model roster | `data/models/*.json` | Ordered list of model IDs; keep free models first for smoke tests. **Each model must specify `concurrency`.** |
| Logging | `LOG_LEVEL` env | INFO by default; DEBUG for verbose traces. LiteLLM logging integrated automatically |

## Key Features

- **Multi-provider support**: LiteLLM provides unified access to 7 providers (Google AI Studio, Groq, Hugging Face, Cerebras, Mistral AI, Cohere, OpenRouter)
- **Async-first execution**: LiteLLM handles async calls with automatic rate limiting and retries
- **Automatic rate limiting**: LiteLLM respects provider-specific rate limits automatically
- **Retry logic**: Built-in exponential backoff and retry handling per provider
- **Separation of concerns**: Scoring focuses on evaluation, reporting handles summarization
- **Extensibility**: Easy to add new providers via LiteLLM, scorers, or reporting modules

## Usage Examples

```bash
# 1. Build the prompt grid
poetry run python scripts/build_benchmark.py --include-neutral --limit 12

# 2. Run benchmark
poetry run python scripts/run_benchmark.py \
  --include-neutral \
  --limit 50 \
  --models data/models.json

# 3. Dry run (no API calls)
poetry run python scripts/run_benchmark.py --dry-run --limit 3

# 4. Single model
poetry run python scripts/run_benchmark.py --model google/gemini-2.0-flash-exp

# 5. Score results
poetry run python scripts/eval_benchmark.py --input outputs/runs/<run_id>/run.jsonl
```

## Integration with Human Labeling

The benchmark framework integrates with the human labeling platform for comprehensive evaluation:

- **Scenario Generation**: Generated prompts feed into the labeling platform via `scripts/data_portal.py`
- **Human Evaluation**: Responses are evaluated by human reviewers using the Streamlit app (`src/labeling_app/app.py`)
- **Data Export**: Human judgments are exported for analysis and model training
- **Pipeline Integration**: See [docs/pipeline.md](../../docs/pipeline.md) for the complete research workflow

### Data Flow
```
Benchmark Generation → LLM Responses → Human Labeling → Analysis
     ↑                    ↓                ↓            ↓
build_benchmark.py → run_benchmark.py → data_portal.py → notebooks
```

For detailed information about the labeling platform, see [src/labeling_app/README.md](../labeling_app/README.md).

## Future Enhancements

- Balanced/seeded sampling for small grids
- RoBERTa-based offline scoring for efficiency
- Additional analytics and visualization tools
- CI/CD integration with automated testing
