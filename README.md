# Nekomimi-Waifu-Seeker

An **Nekomimi character guesser** for **anime, manga, comics and games**, plus the original one-shot feature matcher. Both are decided by **[Laya](https://huggingface.co/convaiinnovations/laya)** — a non-autoregressive decision model that answers typed questions (`choice` / `score` / `noul`) with calibrated probabilities. Laya never generates text; it only decides. Candidates come from DuckDuckGo.

## Nekomimi mode

Think of a character. The engine asks **Yes / No** questions — or you can type a detail instead of answering — and narrows a live candidate pool until it guesses.

```bash
python -m waifu_engine.web
# open http://127.0.0.1:7860/nekomimi
```

Each turn is two batched Laya forward passes:

| Laya question | Type | Decides |
|---|---|---|
| `next_question` | `choice` | which trait to ask, from ~8 entropy-prefiltered candidates |
| `ready_to_guess` | `noul` | whether the evidence is enough to name a character |
| `match_<i>` | `noul` | whether candidate *i* satisfies the answer just given |

Evidence accumulates as log-odds (`logodds += weight * logit(noul)`); candidates
4.0 log-odds behind the leader are eliminated. It guesses at 80% posterior, or
after 20 questions, and gets up to 3 guesses. Questions live in
`waifu_engine/nekomimi/traits.py` (~110 ACG traits) and are topped up with traits
mined from the search snippets of the current pool.

JSON API:

```bash
curl -X POST http://127.0.0.1:7860/api/nekomimi/start  -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:7860/api/nekomimi/answer -H "Content-Type: application/json" -d '{"session_id":"<id>","answer":"yes"}'
curl -X POST http://127.0.0.1:7860/api/nekomimi/guess -H "Content-Type: application/json" -d '{"session_id":"<id>","correct":false}'
curl http://127.0.0.1:7860/api/nekomimi/state/<id>
```

Runs without weights too — every Laya decision has a tag/entropy fallback.

## One-shot pipeline

1. **Input** — free-text features (`silver hair tsundere genius`)
2. **Search** — keyword shortlist from `data/catalog.json`
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

## Extend the catalog

Edit `data/catalog.json` — add objects with `id`, `name`, `series`, `tags`, `blurb`. No code changes required.

## Notes

- Catalog is SFW adult characters only. Scope covers anime, manga, comics and games.
- High-cardinality Laya choice sets are weaker; we shortlist first (~8) then decide.
- Prefer `USE_TF=0` if Transformers hangs while probing TensorFlow.
- CPU-only here (AMD RDNA2, no CUDA): ~26s one-time model load, ~1.2s per batch of 10 questions.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

Offline: Laya and DuckDuckGo are both stubbed.


## Docker

Slim image (keyword fallback, fast build):

```bash
docker compose up --build -d
```

Open http://127.0.0.1:7860

Full Laya image (large; downloads model weights on first request):

```bash
docker compose --profile laya up --build -d
```

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
