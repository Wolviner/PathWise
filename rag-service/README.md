---
title: PathWise RAG
emoji: 💸
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: RAG service behind the PathWise Copilot
---

# PathWise RAG service

The retrieval-augmented generation (RAG) service behind **PathWise Copilot**, which answers questions about a user's own money with cited sources.
It's a small FastAPI app that runs as a Docker Space on Hugging Face (free) and uses:

- **Gemini** (Google AI Studio, free tier) for the LLM calls and embeddings
- **pgvector** in PathWise's existing **Supabase Postgres** as the vector store (no extra database)

It's built from week 5 of the LLM-engineering course (`pro_implementation`), adapted for personal finance.

## How it works

```
Ingestion (background, per user)
  PathWise tables (transactions, accounts, budgets)
    -> markdown knowledge base: monthly summaries with pre-computed totals,
       statements per account per month, budget, recurring, accounts
    -> LLM chunking (headline + summary + original text)
    -> Gemini embeddings (768-d)
    -> rag_chunks table (pgvector), one set of rows per user_id

Question (POST /ask)
  question + history
    -> rewrite query (LLM)
    -> embed original + rewritten -> top 20 each, filtered to this user_id + shared guides
    -> merge (dedupe) -> LLM rerank -> top 10
    -> answer (LLM) + source chips
```

**Free tier notes**
- All steps default to `gemini-flash-lite-latest` (which pointed to `gemini-3.5-flash-lite` when this was written, at about 1 s per call).
  `gemini-3-flash-preview` allows only **20 requests per day** on the free tier, and PathWise's receipt
  scanner already uses it, so the Copilot leaves it alone. Set `LLM_MODEL` to use a different answer model.
- A built-in limiter keeps each model under its requests-per-minute limit (`RPM_LIMITS`), and
  background work (indexing, evaluation) only uses 60% of it, so chat always has room.
- When the daily quota is used up, `/ask` returns 429 straight away with a friendly message instead of waiting.
- Gemini 3 models run with `thinking_level=minimal`: about 4 s per call instead of 90-200 s.

### Endpoints (all except `/` need the `X-RAG-Secret` header)

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Health check |
| POST | `/ask` | `{user_id, question, history}` → `{answer, sources, rewritten_query}` |
| POST | `/reindex/{user_id}` | Queue a rebuild of that user's knowledge base (returns 202) |
| GET | `/reindex/{user_id}` | Status of the last rebuild |
| POST | `/reindex-shared?only_if_missing=true` | Index the shared budgeting guides |
| POST | `/maintenance/purge-orphans` | Delete chunks of users that no longer exist |

PathWise calls `/ask` from a server action (the user id comes from Clerk, never from the browser),
and Inngest calls `/reindex` 2 minutes after a user's data changes and every night.

## Environment variables

| Name | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | Same Supabase URL as PathWise. The pooler URL works (Prisma-only params are stripped). |
| `GEMINI_API_KEY` | yes | From Google AI Studio |
| `RAG_SHARED_SECRET` | yes | Long random string, same value as in PathWise |
| `LLM_MODEL`, `FAST_MODEL`, `CHUNK_MODEL`, `JUDGE_MODEL` | no | Model overrides |
| `RPM_LIMITS` | no | e.g. `gemini-3-flash-preview=5,gemini-flash-lite-latest=10` |
| `THINKING_LEVEL` | no | `minimal` (default), `low`, `high` or `default` |

## Deploy to Hugging Face Spaces

1. Create a Space: **New Space**, SDK **Docker**, template **Blank**, hardware **CPU basic (free)**.
2. In the Space's **Settings → Variables and secrets**, add the secrets `DATABASE_URL`, `GEMINI_API_KEY` and `RAG_SHARED_SECRET`.
3. Push this folder as the Space's repo, from the PathWise repo root. The Space starts with a template
   commit, so the first push replaces it with `--force`:
   ```powershell
   $sha = git subtree split --prefix rag-service HEAD
   git push https://huggingface.co/spaces/<hf-user>/<space-name> "${sha}:refs/heads/main" --force
   ```
   Git asks for your Hugging Face username and, as the password, an access token with **write** permission.
   Run the same two lines again to deploy later changes.
4. When the build finishes, open `https://<hf-user>-<space-name>.hf.space/`. It should return `{"status":"ok"}`.

Free Spaces sleep after a period without traffic. The first request after that takes longer while the Space wakes up.

## Run locally

```bash
docker build -t pathwise-rag .
docker run --rm -p 7860:7860 --env-file .env pathwise-rag
```

Offline unit tests (no database or API calls):

```bash
docker run --rm -v "$PWD:/app" -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt pytest && pytest -q"
```

## Evaluation (numbers for the portfolio)

Use a demo user (for example the one PathWise's `/api/seed` fills). The test questions and expected
answers are generated from that user's data, so they always match the database:

```bash
# in the running container, or locally with the same .env
python -m evaluation.make_tests --user-id <users.id>
python -m evaluation.eval --user-id <users.id> --mode basic --reindex   # text-splitter baseline
python -m evaluation.eval --user-id <users.id> --mode llm --reindex     # + LLM chunking
python -m evaluation.eval --user-id <users.id> --mode full              # + rewrite, merge, rerank
```

Each run prints MRR, nDCG, keyword coverage and LLM-judge scores (accuracy, completeness and relevance, out of 5), and saves
`evaluation/results/<mode>.json`. On the free tier, a full run takes several minutes because of the rate limits.
