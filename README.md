# Nekomimi-Waifu-Seeker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooLguNxDD/Nekomimi-Waifu-Seeker/blob/main/notebooks/nekomimi_colab.ipynb)

An **Nekomimi character guesser** for **anime, manga, comics, games, movies and TV series**, plus the original one-shot feature matcher. Both are decided by **[Laya](https://huggingface.co/convaiinnovations/laya)** — a non-autoregressive decision model that answers typed questions (`choice` / `score` / `noul`) with calibrated probabilities. Laya never generates text; it only decides. Candidates come from Playwright (headless Chromium) first, then DuckDuckGo fills remaining slots. The game is **Nekomimi** at `/nekomimi` and `/api/nekomimi/*`.

## Try it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooLguNxDD/Nekomimi-Waifu-Seeker/blob/main/notebooks/nekomimi_colab.ipynb)

`notebooks/nekomimi_colab.ipynb` runs this repo on a Colab **L4**. It clones `main`, installs the Python dependencies, builds the SolidJS UI, optionally rewrites search queries with Ollama, and publishes the app with ngrok.

Use an L4 runtime, then run the cells from the top. The server cell prints a public `/nekomimi` URL.

Before the first cell, add secrets in the Colab sidebar (the key icon). The notebook loads them with `google.colab.userdata`. Tokens stay in Colab secrets.

| Secret | Required | Purpose |
|---|---|---|
| `NGROK_TOKEN` | yes | ngrok authtoken for the public URL |
| `HF_TOKEN` | no | Hugging Face token; faster download of the Laya weights |
| `GOOGLE_API_KEY` | no | Gemini search grounding when the key is set (`WAIFU_GEMINI_SEARCH`) |

The web page is the built bundle in `waifu_engine/webui_dist`. The notebook runs `cd webui && npm ci && npm run build` (the Docker UI stage; `npm install && npm run build` is the same step from a checkout) before `python -m waifu_engine.web`. Until that bundle exists, `/nekomimi` returns 503.

Laya asks the questions and makes the guesses. Ollama is only the optional query rewriter (`WAIFU_QUERY_LLM=1`). The notebook edits the cloned `query_llm.py` so that copy sends `reasoning_effort=none` when `WAIFU_QUERY_LLM_THINKING=0`. Ollama honours that field; the `chat_template_kwargs` switch in this repo is the one vLLM honours.

## Nekomimi mode

Think of a character. The engine asks **Yes / No** and **multiple-choice**
questions — including **where it's from** (anime/manga, game, comic, movie, TV)
and **which series** (built from the leading candidates) when those split the
pool best — or you can type a detail instead of answering — and searches for
matching characters after every answer. The runtime does not read
`data/catalog.json`.

```bash
python -m waifu_engine.web
# open http://127.0.0.1:7860/nekomimi
```

The decision tree selects questions by expected information gain over retrieved
candidates; medium and series compete in that ranking rather than being forced
first. Empty searches lead to more questions, up to the turn limit. Laya
evaluates each eligible character independently and also judges readiness:

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
`waifu_engine/nekomimi/question_bank.json` (~110 ACG traits, expanded by
`traits.py`) and are topped up with traits mined from the search snippets of
the current pool.

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

The browser app is a SolidJS SPA in `webui/` (TanStack Router + Query, Zustand, Tailwind). `npm run build` writes `waifu_engine/webui_dist`, which is what FastAPI serves (503 until that bundle exists) and what `pip install` ships.

```bash
cd webui && npm install && npm run build && cd ..
python -m waifu_engine.web
```

Open http://127.0.0.1:7860. For frontend HMR, run `npm run dev` in `webui/` while the API is already on port 7860.

## Candidate discovery

Both modes use online search. The historical catalog file is unused by the
runtime. Disabling online search returns no candidates; it does not fall back
to a fixed character list. In Nekomimi, supply a series, appearance, occupation,
or another distinguishing detail to refine the next search.

### Configuration file

Settings can live in a `.env` file instead of the shell. Copy `.env.example`
to `.env` and fill it in. It is git-ignored and loaded at startup, and real
environment variables always win. The Google block:

```bash
GOOGLE_API_KEY=...                      # one key for both Gemini features
WAIFU_GEMINI_MODEL=gemini-2.5-flash     # model for both
WAIFU_GEMINI_SEARCH=1                   # search: grounded candidate search
WAIFU_GEMINI_LLM=1                      # llm: query rewriting with Gemini
# WAIFU_GEMINI_SEARCH_MODEL=...         # per-feature model overrides
# WAIFU_GEMINI_LLM_MODEL=gemini-2.5-flash-lite
```

`/healthz` shows which Gemini features are on and which model each uses.

### Optional Gemini search grounding

Gemini with the Google Search tool can be added as a search engine. It is
asked to list characters matching the player's facts, and it runs the Google
searches itself. Unlike the name searches, it finds characters from traits
alone ("female, video game, silver hair"), so it helps most in rounds with no
seed. Its answers are ordinary candidates that Laya scores. It never picks
questions or guesses.

```bash
pip install -e .[gemini]            # google-genai
export GEMINI_API_KEY=...           # or GOOGLE_API_KEY
WAIFU_GEMINI_SEARCH=1 python -m waifu_engine.web
```

Each grounded call is billed and takes a few seconds, so Gemini:
- only runs when Laya's `pool_fits` says the current candidates don't fit;
- runs in the background once the round has candidates (hits join the next
  turn; log line `gemini background ... found=N`), inline only when nothing is
  in play;
- caches results for 15 minutes per set of facts, and is skipped with no facts.

`WAIFU_GEMINI_MODEL` (default `gemini-2.5-flash`) picks the model. `/healthz`
shows `gemini.enabled`, and the timing line shows `hits=gemini:N` /
`gemini_bg:N` and any `err=gemini: ...` (e.g. quota). The Docker images don't
install `google-genai`; add it there if you want Gemini in a container.

### Optional query LLM

The rewriter has two backends. `WAIFU_GEMINI_LLM=1` uses Gemini through the
same key as Gemini search (JSON output, thinking kept to the minimum). Otherwise
it uses any OpenAI-compatible server, as below. Either way the gating is the
same: typed text only, only when Laya says search is stuck, and in the
background.

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

**Laya keeps the game fast; the LLM is called rarely and never waited on.**

- It only ever sees what the player *typed* (the seed and details). Button
  answers already have fixed search wording, so a round played only with
  buttons makes **zero** LLM calls.
- It is called only when Laya says search is stuck. One fast `pool_fits` noul
  (~0.3 s) asks whether any current leader fits the facts. If one does, the
  LLM stays idle.
- Each distinct set of typed text is rewritten at most once, on one background
  worker. Turns never wait: the queries are used by the next search once they
  land, usually one answer later. The first search of a round always uses the
  templates.
- Set `WAIFU_QUERY_LLM_WAIT=1.5` to let a turn wait up to that many seconds for
  a fast (GPU) server. The default `0` never waits.

A typical round makes 0–2 LLM calls instead of one per answer. `/healthz`
reports `query_llm.calls`, `last_ms` and `inflight`.

For the unsloth GGUF on CPU, turn thinking off (it dominates latency):

```bash
llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M --jinja --reasoning-budget 0 -c 2048 --port 8000
```

To use OpenAI, set `WAIFU_QUERY_LLM_BASE_URL=https://api.openai.com/v1`,
`WAIFU_QUERY_LLM_MODEL=<model>` and `OPENAI_API_KEY`. That key is only ever
sent to `https://api.openai.com`; other endpoints use `WAIFU_QUERY_LLM_API_KEY`.
The last queries used are listed under `queries` in
`web_search.last_search_meta()`.

## Finding the bottleneck

Every Nekomimi request logs one line naming its steps, slowest first:

```
[waifu] answer found=5 new=2 turn=4 cands=38/42 total=3412ms | search 2980ms, search.fetch 2950ms, fetch.playwright 2100ms, fetch.wikipedia 610ms, score 420ms, laya.match 410ms/38x, pick 80ms, laya.ready_to_guess 60ms
```

- `search` is candidate discovery, and `fetch.*` splits it by source
  (`playwright`, `wikipedia`, `anilist`, `ddg`, `enrich`). `search.rescore_new`
  scores newly found candidates against the earlier answers.
- `score` is judging every candidate against the answer just given.
- `laya.<question>` is model time, with a call count (`/38x` = 38 forward
  passes). `laya.lock_wait` is time spent queued behind other players' model calls.
- `pick` is choosing the next question; `laya.ready_to_guess` is part of it.
- Background LLM calls log their own `query_llm <model> <ms>` line, since
  turns never wait for them.

- `fetch.ddg_background` is only the time to queue DuckDuckGo. The search
  itself logs its own `ddg background ... found=N` line, and its hits join the
  next turn. DuckDuckGo runs only when Laya's `pool_fits` says the pool is
  missing the answer (`search.stuck`), makes at most 3 requests within 4 s, and
  reuses one client.

- `hits=` counts new candidates per source (`popular`, `wikipedia`, `anilist`,
  `playwright`, `ddg`, `ddg_bg` from the previous turn's background search).
  `err=` shows network failures, so a blocked or rate-limited source is not
  mistaken for "no results".
- With no seed or typed detail, searches for broad traits can't match names.
  The game starts from popular characters instead (`fetch.popular`, filtered by
  medium once known) and the page asks for a series or detail when nothing is
  in play.

Dotted names are part of their parent (`fetch.wikipedia` is inside `search`).
The same numbers come back in each API response as `timing`. Set
`WAIFU_TIMING_LOG=0` to silence the log line.

## Notes

- Scope covers anime, manga, comics, games, movies and TV series. Film and TV
  coverage is thinner (no AniList; Wikipedia categories and Gemini search).
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
