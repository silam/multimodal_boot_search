## Recommended structure (src layout)
```
boot01/
├── pyproject.toml          # deps, tooling config (uv/poetry + ruff + mypy + pytest)
├── .env.example            # never commit .env
├── src/
│   └── my_ai_app/
│       ├── __init__.py
│       ├── config.py       # pydantic-settings: keys, model names, endpoints
│       ├── api/            # FastAPI routers, request/response schemas
│       ├── core/           # domain models & business logic (no LLM imports)
│       ├── llm/            # thin provider clients/adapters (Anthropic, OpenAI, Azure)
│       ├── prompts/        # versioned prompt templates (.md/.jinja/.yaml)
│       ├── agents/         # graphs/workflows: nodes, state, routing
│       │   ├── state.py
│       │   ├── nodes/
│       │   └── graph.py
│       ├── tools/          # tool definitions the agent can call
│       ├── retrieval/      # RAG: loaders, chunking, embeddings, retrievers, rerank
│       ├── stores/         # vector DB / DB repositories (Cosmos, Qdrant, Postgres)
│       ├── services/       # orchestration that ties core + llm + retrieval together
│       └── observability/  # logging, tracing (LangSmith/OTel), token/cost tracking
├── pipelines/              # offline jobs: ingestion, indexing, batch embedding
├── evals/                  # eval datasets, graders, regression runs
├── notebooks/              # exploration only, never imported by src
├── tests/
│   ├── unit/               # mock the LLM
│   ├── integration/        # real stores, recorded/stubbed LLM responses
│   └── fixtures/
└── scripts/                # one-off CLIs

```
