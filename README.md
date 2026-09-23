# Nekomimi-Waifu-Seeker

An **Nekomimi character guesser** for **anime, manga, comics and games**, plus the original one-shot feature matcher. Both are decided by **[Laya](https://huggingface.co/convaiinnovations/laya)** — a non-autoregressive decision model that answers typed questions (`choice` / `score` / `noul`) with calibrated probabilities. Laya never generates text; it only decides. Candidates come from Playwright (headless Chromium) first, then DuckDuckGo fills remaining slots. The game is **Nekomimi** at `/nekomimi` and `/api/nekomimi/*`.

## Nekomimi mode

Think of a character. The engine asks **Yes / No** and **multiple-choice** questions (hair colour, eye colour) — or you can type a detail instead of answering — and searches for matching characters after every answer. The runtime does not read `data/catalog.json`.

```bash
python -m waifu_engine.web
# open http://127.0.0.1:7860/nekomimi
```

The decision tree first establishes the medium, then selects questions by
expected information gain over retrieved candidates. Empty searches lead to
more questions, up to the turn limit. Laya evaluates each eligible
character independently and also judges readiness:

| Laya question | Type | Decides |
|---|---|---|
| `ready_to_guess` | `noul` | whether the evidence is enough to name a character |
| `match` | `noul` | whether one character satisfies the trait just asked |
| `match` | `choice` | which option of a multiple-choice question fits one character; `probabilities[picked]` is the evidence |
| `focus` | `choice` | which confirmed facts are most distinctive, so they lead the search query |

Search uses concise positive clues, with shorter query variants when a search
is too restrictive. Negative answers remain in Laya's evidence instead of
becoming misleading positive search keywords. New results are evaluated against
the full answer history, including the seed and free-text details. An early
guess needs model evidence; being the only search result is insufficient.

Every eligible candidate is scored, regardless of its current rank. Successful
model judgments are cached for the session; new arrivals receive the earlier
questions too. Scores sum log-likelihoods with a capped popularity prior.
Only confirmed medium contradictions and rejected guesses eliminate identities;
uncertain scores can recover. Mined tags guide question selection but are not
presented as facts to Laya. This costs one model call per eligible candidate per
new trait, so large search result pools take longer than small ones.
It guesses at 80% posterior, or
after 20 questions, and gets up to 3 guesses. Questions live in
`waifu_engine/nekomimi/traits.py` (~110 ACG traits) and are topped up with traits
mined from the search snippets of the current pool.

JSON API:

```bash
curl -X POST http://127.0.0.1:7860/api/nekomimi/start  -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:7860/api/nekomimi/answer -H "Content-Type: application/json" -d '{"session_id":"<id>","answer":"yes"}'
# multiple-choice questions ("kind": "choice") take an option key instead:
curl -X POST http://127.0.0.1:7860/api/nekomimi/answer -H "Content-Type: application/json" -d '{"session_id":"<id>","answer":"white"}'
curl -X POST http://127.0.0.1:7860/api/nekomimi/guess -H "Content-Type: application/json" -d '{"session_id":"<id>","correct":false}'
curl http://127.0.0.1:7860/api/nekomimi/state/<id>
```

Runs without weights too — every Laya decision has a tag/entropy fallback.

## One-shot pipeline

1. **Input** — free-text features (`silver hair tsundere genius`)
2. **Search** — online character discovery and relevance ranking
3. **Decide** — Laya picks the best match among the shortlist (or keyword fallback)
4. **Output** — winner + runners-up + confidence

Laya does **not** generate text. It only answers typed decision questions.

## Install

```bash
cd waifu-determination-engine
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
# optional: headless Chromium search (falls back to DuckDuckGo if missing)
pip install -e ".[playwright]"
playwright install chromium
```

First Laya load downloads ~800MB of weights from Hugging Face. For a quick demo without that:

```bash
python -m waifu_engine "silver hair mage calm reflective" --fallback
```

## CLI

```bash
python -m waifu_engine "pink hair playful cosplay"
python -m waifu_engine "stoic swordswoman loyal protector" --json
python -m waifu_engine "blue hair ice general" --fallback
```

## Web UI

```bash
python -m waifu_engine.web
```

Open http://127.0.0.1:7860

## Candidate discovery

Both modes use online search. The historical catalog file is unused by the
runtime. Disabling online search returns no candidates; it does not fall back
to a fixed character list. In Nekomimi, supply a series, appearance, occupation,
or another distinguishing detail to refine the next search.

### Optional query LLM

Search strings come from templates by default. A small chat model can rewrite
the player's confirmed facts into better search phrases instead. It writes
**search queries only**: never question text, and never decisions. Laya still
makes every decision. The model only sees what the player typed or confirmed;
scraped pages are never sent. Any failure falls back to the templates.

Any OpenAI-compatible `/v1/chat/completions` endpoint works. The default is a
local server hosting `Qwen/Qwen3.6-35B-A3B`:

```bash
# GPU: vLLM
vllm serve Qwen/Qwen3.6-35B-A3B --port 8000
# CPU (no CUDA): a Q4 GGUF under llama.cpp. ~3B active params, ~20-24 GB RAM
llama-server --model Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8000

WAIFU_QUERY_LLM=1 python -m waifu_engine.web
```

To use OpenAI, set `WAIFU_QUERY_LLM_BASE_URL=https://api.openai.com/v1`,
`WAIFU_QUERY_LLM_MODEL=<model>` and `OPENAI_API_KEY`. The last queries used are
listed under `queries` in `web_search.last_search_meta()`.

## Notes

- Scope covers anime, manga, comics and games.
- High-cardinality Laya choice sets are weaker; we shortlist first (~8) then decide.
- Prefer `USE_TF=0` if Transformers hangs while probing TensorFlow.
- CPU-only here (AMD RDNA2, no CUDA): ~26s one-time model load, ~1.2s per batch of 10 questions.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

Offline: Laya, Playwright, and DuckDuckGo are stubbed. Tests never launch Chromium.


## Docker

Slim image (keyword fallback, fast build):

```bash
docker compose up --build -d
```

Open http://127.0.0.1:7860

Full Laya image. The weights are downloaded **at build time** and baked into
the image, so the first build is large and slow. Containers then start offline.
Laya loads (and runs one warm-up pass) before uvicorn opens the port, so no
player waits for the model. If Laya fails to load, the container fails to
start (`WAIFU_LAYA_REQUIRED=1`) instead of quietly running on heuristics.

```bash
docker compose --profile laya up --build -d
curl http://127.0.0.1:7860/healthz   # {"status":"ok","laya":{"loaded":true,...}}
```

The image's `HEALTHCHECK` reports healthy only once `laya.loaded` is true.

Stop:

```bash
docker compose down
```

## Native Laya (local pip / venv)

Already set up under `.venv` on this machine.

```bat
cd E:\code_project\OSS\CatOSSWorks\waifu-determination-engine
run-cli.bat "Mika Misono"
run-web.bat
```

Or manually:

```bat
cd E:\code_project\OSS\CatOSSWorks\waifu-determination-engine
set USE_TF=0
set WAIFU_FORCE_FALLBACK=0
.venv\Scripts\activate
python -m waifu_engine "silver hair mage" 
python -m waifu_engine.web
```

Open http://127.0.0.1:7860 — stop the Docker container first if that port is taken:

```bat
docker compose down
```

Weights cache at `%USERPROFILE%\.cache\huggingface\`.

`python -m waifu_engine.web` loads Laya at startup, taking about 26 s on this
box, before it prints "Application startup complete". Set
`WAIFU_LAYA_PRELOAD=0` to load on the first request instead, or run
`python -m waifu_engine.nekomimi.laya_client` to download and check the weights
without starting the server.
