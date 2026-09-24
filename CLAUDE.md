# Nekomimi-Waifu-Seeker — ACG Character Guesser

Python + FastAPI. Two modes over the same candidate machinery:

1. **Determine** (`/`, CLI) — one-shot: free-text features → online shortlist → Laya picks a winner.
2. **Nekomimi** (`/nekomimi`) — interactive: engine asks yes/no questions, narrows a live candidate pool, guesses.

Scope is **ACG plus film and TV**: Anime, Manga, Comics, Games (including visual
novels and gacha), Movies and TV series. Not anime-only. Media values are
`traits.MEDIUM_VALUES` (`anime`, `manga`, `comic`, `game`, `movie`, `tv`).

## The one constraint that shapes everything

**Laya is not a generator.** `convaiinnovations/laya` is a non-autoregressive
ModernBERT-large decision head (421M). `agent.predict(state, questions)` returns
typed answers only:

| type | field | use here |
|---|---|---|
| `choice` | `choice` + `probabilities` + `confidence` | which question to ask; which candidate wins |
| `noul` | `noul` ∈ [0,1] + `confidence` | does this candidate satisfy this fact; ready to guess |
| `score` | `score` (expected ordinal) + `probabilities` | one-shot fit strength |

Every answer also carries `action.act_probability`.

Consequences, in order of how often they get forgotten:

- **Question text never comes from the model.** Question wording lives in
 `waifu_engine/nekomimi/question_bank.json`. `traits.py` expands that template
 into the runtime bank. Do not put question wording in prompt strings
 scattered through the code, and do not add a generative model to write them.
- `criteria` keys are the option labels; `choice` returns one of those keys.
- The result key is **`probabilities`**, not `probs`.
- Option strings are packed into the decision head. If a `choice` has many long
  options, `system_one` raises `ValueError("question ... options exceed
  head_max_len")`. `laya_client` raises the budget to 480; keep `choice` sets
  narrow anyway (~8) — Laya is weak on wide choice sets.

## Module map

| Path | Role |
|---|---|
| `waifu_engine/nekomimi/laya_client.py` | Process-wide `Agent` singleton. `ask(state, questions)` → answers or `None`. Never raises. `preload()` (load + warm-up, run by `web.py`'s lifespan before the port opens), `status()` (read-only, served at `/healthz`). `python -m` it to bake weights. |
| `waifu_engine/nekomimi/question_bank.json` | Question template: yes/no rows, choice rows, and `trait_block` groups |
| `waifu_engine/nekomimi/traits.py` | Expands the JSON template, plus `ANSWER_WEIGHT` and `make_dynamic()` for mined traits |
| `waifu_engine/nekomimi/session.py` | `Candidate`, `GuessSession`, log-odds pool, in-process store + TTL |
| `waifu_engine/nekomimi/engine.py` | The turn loop: `start`, `submit_answer`, `submit_guess_result`, `state_payload` |
| `waifu_engine/nekomimi_page.py` | Static HTML/JS for `/nekomimi` |
| `waifu_engine/browser_search.py` | Process-wide headless Chromium. `search` / `enrich` / `available()`. Never raises. Optional extra. |
| `waifu_engine/web_search.py` | Playwright then DuckDuckGo fill. `search_characters_multiround` (one-shot) + `search_by_constraints` (guessing loop) + `mine_trait_slugs` |
| `waifu_engine/sources/` | Playwright HTML indexes, then AniList + Wikipedia; DuckDuckGo (`web_search.ddg_quick`, capped) fills remaining slots, in the background per session (`background_key`) and only when `ddg_gate()` agrees |
| `waifu_engine/google_config.py` | Google GenAI settings + the one shared `genai.Client`: key, per-feature switches (`search_enabled`, `llm_enabled`), models (`WAIFU_GEMINI_MODEL`, per-feature overrides), minimal `thinking_config` |
| `waifu_engine/sources/gemini.py` | Optional Gemini + Google Search grounding source. `search_characters(facts, medium)` → candidates. Never raises. Off unless `WAIFU_GEMINI_SEARCH=1` and `GEMINI_API_KEY`/`GOOGLE_API_KEY` |
| `waifu_engine/envfile.py` | Dependency-free `.env` loader, run from `waifu_engine/__init__.py`; fills only unset variables. `.env.example` lists the settings. Tests set `WAIFU_ENV_FILE=0` (`tests/conftest.py`) |
| `waifu_engine/query_llm.py` | Optional query rewriter: Gemini (`WAIFU_GEMINI_LLM=1`) or any OpenAI-compatible server (default local `Qwen/Qwen3.6-35B-A3B`). `prefetch` (one background worker) / `peek` (non-blocking) / `rewrite` (blocking, tooling only). Never raises. Off unless `WAIFU_QUERY_LLM=1` |
| `waifu_engine/timing.py` | Per-request spans. `@traced` on `start`/`submit_answer`/`submit_guess_result`/`determine` logs one `[waifu]` line (slowest first) and sets `payload["timing"]`. `span()` is a no-op outside a trace |
| `waifu_engine/decide.py` | One-shot `determine()` pipeline |
| `waifu_engine/search.py`, `catalog.py` | Online shortlist ranking; catalog helpers remain for tooling but runtime never loads the catalog |

## The turn contract

1. `_pick_question` chooses by mutual information (answer entropy minus
   within-candidate uncertainty); Laya only answers `ready_to_guess`. The
   `medium` choice ("Where is your character from?": anime/manga, game, comic,
   movie, TV, something else) and dynamic "Which series?" compete in that
   ranking like any other question — neither is forced first. A medium pick is
   a hard filter (`traits.MEDIUM_ACCEPTS`; movie and TV accept each other,
   "other" accepts no known medium; unknown media are never removed). Empty
   searches keep asking until the turn limit.
2. `score_candidates` records evidence and evaluates **every eligible candidate**
   with an independent `match` noul call. Never gate evidence on `scoring_pool()`;
   that top-10 list is only a readiness summary.
3. Successful probabilities are cached by candidate/question. Replay history for
   new search results, apply known medium constraints before inference, and
   rebuild scores from capped popularity priors plus answer log-likelihoods.
4. Search again after **every answer**, replaying history before the next
   question/guess. Use compact positive clues and shorter fallback queries;
   negatives stay in model evidence. Preserve provider relevance before fame.
   Never seed from `catalog.json`, including in the one-shot mode.

The loop does not call `prune()`: soft evidence must remain recoverable. Only
medium contradictions and rejected guesses eliminate candidates. Heuristic
fallback probabilities are capped to [0.4, 0.6], except a free-text visual
combination (pink hair, halo, wings, horns) and the separate halo / wings /
horns questions: when the profile clearly has or lacks the trait, that
likelihood is used even if Laya's noul is mushy. Never write predictions into
candidate tags or feed noisy mined tags to Laya as confirmed identity facts.
`posterior()` softmaxes these scores. Cost scales with eligible candidates per
new trait; there is no longer a ten-forward-pass evidence budget.

Multiple-choice questions (`kind: "choice"`, built with `traits._choice`, ≤8
options including `other`) are answered with an option key. Each candidate gets
its own Laya `choice` call; `probabilities[picked]` is the likelihood (cached in
`choice_cache`). Heuristic fallback weights a tag hit at most 1.5× uniform.
Early-guess support for choice evidence is `p_pick / (p_pick + best_other)`.

"Which series?" (`engine._series_question`, wording in `traits.series_question`)
is a dynamic choice over the leading candidates' series (`names.series_key`,
up to 6 plus "Another series"), ranked by information gain like any question,
asked at most twice (the second time only after "Another series"). Known
medium/series are scored from data (`_known_choice`), no model call; only
candidates without them go to Laya. A picked series is a search fact that sets
`specific=True`. Option labels are scraped text: `_series_label` sanitises
them, and the page renders them with `textContent`.

Before each search, one Laya `choice` call (`focus`) ranks the positive facts;
the top three lead the query. A near-uniform answer (< 1.5/n) is ignored in
favour of rarity order (details first, broad medium/gender facts last).

The query LLM is gated by Laya: `_llm_queries` sends only the player's typed
text (`_free_text`: seed + details), at most once per distinct text, and only
when `_search_stuck` says so (one `pool_fits` noul over the top five; heuristic
without Laya). It runs in the background; searches reuse the last good rewrite
until a newer one lands. The first search of a round never uses it.

DuckDuckGo is the slow source (uncapped it made ~35 sequential requests, ~40 s
a turn). It is capped (`WAIFU_DDG_MAX_REQUESTS` generic queries within
`WAIFU_DDG_BUDGET` seconds, one shared client), gated by the same memoised
`_search_stuck` call as the query LLM (one `pool_fits` per search at most), and
runs on a background worker whenever the session already has candidates; its
hits join the next search. It runs inline only when nothing else was found.

Only the player's typed text (or an LLM rewrite of it) can match names, so
`find_candidates(specific=False)` when the facts are broad button answers:
Playwright/Wikipedia/AniList name searches are skipped, DuckDuckGo never runs
inline, and the pool is topped up from `sources.popular_characters` (AniList
top characters for anime/manga, Wikipedia game/comic character categories;
live, never `catalog.json`) up to `WAIFU_POPULAR_POOL` candidates in play.
`_http` records network failures (`last_search_meta()["errors"]`); the timing
line shows per-source `hits=` and `err=`.

Gemini search grounding (`sources.gemini`) finds characters from the facts
themselves, so it works on broad button facts where name searches cannot. It
shares the memoised `ddg_gate` (Laya `pool_fits`), runs in the background
(`_GEMINI_BG`, its own worker, capped by `WAIFU_GEMINI_BG_MAX_PENDING`) whenever
the session has candidates, and inline only when the pool would be empty.
Results are cached 15 min per facts. With no facts it is skipped (billed calls).

Answers to yes/no questions are **`yes` / `no` / `detail`**. `detail` does not answer the current
yes/no trait. Instead, the text becomes a separate `clue_question` for Laya and
refines search. The initial seed is evaluated the same way.

Guess when any of: top posterior ≥ 0.80 after ≥ 5 questions; `ready_to_guess.noul`
≥ 0.75 with `act_probability` ≥ 0.6; or turn ≥ `MAX_TURNS`. Early guesses also
require at least two model judgments with mean answer likelihood ≥ 0.6. A lone
search hit is not sufficient evidence. Up to 3 guesses.

**Every Laya path has a heuristic fallback** (`_tag_match`, `_split_quality`), so
the loop plays with no weights installed — less sharply. Never let a Laya failure
raise out of a request.

## Session store

Plain `dict` in `session.py`, 30-minute TTL, guarded by a lock. **Single process
only.** Running uvicorn with more than one worker splits sessions across workers;
swap in Redis/SQLite before doing that.

## Hardware

CPU-only. Development machine is AMD RDNA2 on Windows: no CUDA, and neither
`torch-directml` nor ROCm has a wheel for `torch 2.14` / Python 3.14. Measured on
this box: **~26 s one-time load, ~0.3 s for one question, ~1.2 s for a batch of
10.** The load now happens at app startup (`preload()` in the lifespan), never on a
player's request. That is why the agent is a singleton and why a turn is two batched calls
rather than one call per candidate. A future ONNX Runtime + DirectML export is
the plausible GPU path; not built.

## Safety notes

Candidate names, blurbs and image URLs are **scraped web content**. Never
interpolate them into HTML unescaped — `web.py` uses `html.escape`, and the
`/nekomimi` page builds DOM nodes with `textContent` only.

## Env vars

| Var | Default | Effect |
|---|---|---|
| `WAIFU_FORCE_FALLBACK` | `0` | Skip Laya entirely (heuristics only) |
| `WAIFU_ONLINE_SEARCH` | `1` | Allow online search (Playwright + DuckDuckGo) |
| `WAIFU_SEARCH_BACKEND` | `auto` | `auto` (Playwright then DDG), `playwright`, or `ddg` |
| `WAIFU_PLAYWRIGHT_ENRICH` | `1` | Visit character pages to fill blurb/tags/image |
| `WAIFU_PLAYWRIGHT_ENRICH_LIMIT` | `8` | Max candidates to enrich per search |
| `WAIFU_SEARCH_ROUNDS` | `3` | Rounds for one-shot determine |
| `WAIFU_NEKOMINI_MAX_TURNS` | `20` | Hard question cap |
| `WAIFU_NEKOMINI_MAX_GUESSES` | `3` | Guesses before giving up |
| `WAIFU_NEKOMINI_TTL` | `1800` | Session lifetime, seconds |
| `WAIFU_NEKOMINI_CHOICE_WIDTH` | `8` | Questions offered to Laya per turn |
| `WAIFU_NEKOMINI_GUESS_CONFIDENCE` | `0.80` | Posterior needed to guess |
| `WAIFU_LAYA_HEAD_MAX_LEN` | `480` | Option-token budget |
| `WAIFU_QUERY_LLM` | `0` | Let an LLM rewrite search queries (search strings only) |
| `WAIFU_QUERY_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint (`https://api.openai.com/v1` for OpenAI) |
| `WAIFU_QUERY_LLM_MODEL` | `Qwen/Qwen3.6-35B-A3B` | Model name sent to that endpoint |
| `WAIFU_QUERY_LLM_API_KEY` | — | Bearer key for the query endpoint; blank for local. Falls back to `OPENAI_API_KEY` only for `https://api.openai.com` |
| `WAIFU_QUERY_LLM_TIMEOUT` | `20` | Seconds per rewrite call |
| `WAIFU_QUERY_LLM_WAIT` | `0` | Seconds a turn may wait for the LLM (`0` = never block) |
| `WAIFU_DDG_MAX_REQUESTS` | `3` | DuckDuckGo requests per search |
| `WAIFU_DDG_BUDGET` | `4` | Seconds after which no new DuckDuckGo request starts |
| `WAIFU_DDG_TIMEOUT` | `5` | Per-request DuckDuckGo timeout |
| `WAIFU_DDG_BG_MAX_PENDING` | `4` | Background DuckDuckGo fills queued or running at once; extra ones are dropped |
| `WAIFU_DDG_BACKGROUND` | `1` | Fill with DuckDuckGo off the request thread (`0` = inline, capped) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key for both Gemini features; both stay off without one |
| `WAIFU_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for search and llm |
| `WAIFU_GEMINI_SEARCH` | `0` | Use Gemini with Google Search grounding as a candidate source (billed) |
| `WAIFU_GEMINI_SEARCH_MODEL` | `WAIFU_GEMINI_MODEL` | Search model; must support the Google Search tool |
| `WAIFU_GEMINI_LLM` | `0` | Use Gemini as the query LLM instead of the OpenAI-compatible server (billed) |
| `WAIFU_GEMINI_LLM_MODEL` | `WAIFU_GEMINI_MODEL` | Query-LLM model (a lite model is enough) |
| `WAIFU_GEMINI_TIMEOUT` | `15` | Seconds per grounded request |
| `WAIFU_GEMINI_BACKGROUND` | `1` | Run Gemini off the request thread when the pool has candidates (`0` = inline) |
| `WAIFU_GEMINI_BG_MAX_PENDING` | `2` | Background Gemini searches queued or running at once; extra ones are dropped |
| `WAIFU_POPULAR_POOL` | `24` | Popular candidates kept in play on a cold start (each costs a `match` call per answer) |
| `WAIFU_TIMING_LOG` | `1` | Log one timing line per request (`payload["timing"]` is always filled) |
| `WAIFU_LAYA_PRELOAD` | `1` | Load Laya at app startup (`0` = on first request) |
| `WAIFU_LAYA_REQUIRED` | `0` | Fail startup if Laya does not load (set in the Laya Docker image) |
| `WAIFU_ENV_FILE` | `.env` | Env file loaded at import (`0` = off); real env vars win |
| `USE_TF` | — | Set `0`; Transformers hangs probing TensorFlow |
| `HF_HOME` | — | Weight cache (`/data/hf` in Docker) |

## Dev

```bash
python -m pytest tests -q          # offline tests, no weights, no network, no Chromium
python -m waifu_engine.web         # http://127.0.0.1:7860  (+ /nekomimi)
python -m waifu_engine "silver hair mage" --fallback
```

Tests stub `laya_client.ask`, `web_search.search_by_constraints` and the
query LLM's `urlopen`. Keep them
offline — do not add a test that downloads weights, launches Chromium, or hits
DuckDuckGo.

## Rules

1. New questions go in `question_bank.json`, nowhere else. `traits.py` only expands that template.
2. Tag slugs in `question_bank.json` and `web_search.TRAIT_PATTERNS` share one vocabulary
   — add to both or evidence and questions stop lining up.
3. Candidate identity is `names.name_keys` everywhere (search merge, DDG merge,
   session pool): names match in either word order, because Japanese names
   come family-first ("Shimoe Koharu") and given-first ("Koharu Shimoe"), and
   a duplicate splits its own probability. Don't compare raw name strings.
4. Trait matching is whole-word regex. Plain substring matching once made `"he"`
   fire on `"the"` and tagged every character male.
5. Laya calls go through `laya_client.ask`. Do not construct `Router` or call
   `laya.load` anywhere else.
6. The query LLM (`query_llm.py`, either backend: Gemini or OpenAI-compatible) writes **search strings only** — never
   question text, never decisions. Its input is player facts only; never send
   it scraped names or blurbs. Tests stub its HTTP; never call a real endpoint.
7. The query LLM must never block a turn by default, and must not be called
   per answer: only for new typed text, and only when Laya says search is stuck.
8. Gemini (`sources/gemini.py`) is a **candidate source only**: it lists
   characters, never writes question text, never makes decisions. Its reply is
   untrusted web content (parse defensively, render with `textContent`). Its
   prompt holds player facts only, never scraped names or blurbs -- the one
   exception is a series label the player confirmed, after `_series_label`
   sanitising. Tests use a
   fake client; never call the real API.
9. Every function you add or change gets a docstring, including private
   helpers: one line saying what it returns or does, plus the non-obvious
   *why* (a measured failure, a constraint) when there is one. CodeRabbit's
   pre-merge check requires 80% docstring coverage over the functions a PR
   touches. Tests are exempt. Don't restate the signature; `"""Return x."""`
   on `def x()` adds nothing.
