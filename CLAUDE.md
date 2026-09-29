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
| `waifu_engine/nekomimi/memory.py` | Rebuilds normalized player facts from the transcript, scored answers, and rejected identities |
| `waifu_engine/nekomimi/context.py` | Fits Laya state to each question window and records included or omitted fact IDs |
| `waifu_engine/nekomimi/engine.py` | The turn loop: `start`, `submit_answer`, `submit_guess_result`, `state_payload` |
| `webui/` | SolidJS UI (Vite). File routes for `/` and `/nekomimi`. `npm run build` writes `waifu_engine/webui_dist`, which setuptools ships; FastAPI 503s until that bundle exists |
| `waifu_engine/browser_search.py` | Process-wide headless Chromium. `search` / `enrich` / `available()`. Never raises. Optional extra. |
| `waifu_engine/web_search.py` | Playwright then DuckDuckGo fill. `search_characters_multiround` (one-shot) + `search_by_constraints` (guessing loop) + `mine_trait_slugs` |
| `waifu_engine/sources/` | Playwright HTML indexes, then AniList + Wikipedia; DuckDuckGo (`web_search.ddg_quick`, capped) fills remaining slots, in the background per session (`background_key`) and only when `ddg_gate()` agrees |
| `waifu_engine/google_config.py` | Google GenAI settings + the one shared `genai.Client`: key, per-feature switches (`search_enabled`, `llm_enabled`), models (`WAIFU_GEMINI_MODEL`, per-feature overrides), minimal `thinking_config` |
| `waifu_engine/sources/gemini.py` | Optional Gemini + Google Search grounding source. `search_characters(facts, medium)` → candidates. Never raises. Off unless `WAIFU_GEMINI_SEARCH=1` and `GEMINI_API_KEY`/`GOOGLE_API_KEY` |
| `waifu_engine/envfile.py` | Dependency-free `.env` loader, run from `waifu_engine/__init__.py`; fills only unset variables. `.env.example` lists the settings. Tests set `WAIFU_ENV_FILE=0` (`tests/conftest.py`) |
| `waifu_engine/query_llm.py` | Optional query rewriter and name proposer: Gemini (`WAIFU_GEMINI_LLM=1`) or any OpenAI-compatible server (default local Ollama `hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q3_K_M`; Qwen via env). `prefetch` (one background worker) / `peek` (non-blocking) / `rewrite` (blocking, tooling only); `propose_characters` (blocking, called from a source worker). Thinking off by default (`reasoning_effort=none` on Ollama, chat-template switch on Qwen). Never raises. Off unless `WAIFU_QUERY_LLM=1` |
| `waifu_engine/sources/llm_names.py` | LLM hypotheses: `propose_characters` names, each resolved on AniList/Wikipedia; unresolved names are dropped. Runs on `sources._LLM_BG`; hits join the next search. Never raises |
| `waifu_engine/timing.py` | Per-request spans. `@traced` on `start`/`submit_answer`/`submit_guess_result`/`determine` logs one `[waifu]` line (slowest first) and sets `payload["timing"]`. `span()` is a no-op outside a trace |
| `waifu_engine/decide.py` | One-shot `determine()` pipeline |
| `waifu_engine/search.py`, `catalog.py` | Online shortlist ranking; catalog helpers remain for tooling but runtime never loads the catalog |

## The turn contract

1. `_pick_question` chooses by mutual information (answer entropy minus
   within-candidate uncertainty). Unpinned questions are re-ranked by
   `_lookahead_eig`: one Laya call per top candidate
   (`WAIFU_NEKOMINI_EIG_CANDIDATES`) carrying every shortlisted yes/no as a
   `noul`, so the gain uses Laya's likelihoods, not tag heuristics (BED-LLM's
   prior x likelihood). Those judgments sit in `sess.lookahead` with the trait
   rows they were made under and join `match_cache` only if the rows still
   match at answer time (`_promote_lookahead`). Without Laya the heuristic
   order stands. Laya never picks a question id; it also answers `ready_to_guess`. The
   `medium` choice ("Where is your character from?": anime/manga, game, comic,
   movie, TV, something else) and dynamic "Which series?" compete in that
   ranking like any other question — neither is forced first. Traits the seed
   already named (hair colour, halo, wings, horns) are pulled in front of that
   ranking, and the same appearance questions lead once a series is confirmed,
   because school/uniform/teen do not separate students from one school. A medium pick is
   a hard filter (`traits.MEDIUM_ACCEPTS`; movie and TV accept each other;
   "other" / "Something else" is an empty accept-set and is **not** a hard
   filter — it must not wipe known media; unknown media are never removed). Empty
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

**Silent profiles score near the base rate.** Laya's judgment of a trait the
profile never mentions is drift, not evidence: on real AniList/Wikipedia pages
"has a halo" scored 0.63-0.73 (prior 0.05) and Tifa's black hair 0.06, and the
2B/Makima/Megumin/Tifa misses lost ~0.5 per "no" to described rivals. A yes/no
noul on a profile with no tag and no whole-word mention of the trait
(`_grounded`) is pulled toward the question's `prior` (`_calibrated_noul`,
`WAIFU_NEKOMINI_SILENT_WEIGHT`, 0.3 kept; 1 = raw). A Laya choice distribution
whose top option is under `WAIFU_NEKOMINI_CHOICE_SILENT_MAX` (0.5; silent pages
measured <= 0.40, stated 0.97) is replaced by the options' base rates
(`_silent_choice`). Caches stay raw; lookahead gain uses the same calibration.
The cost: a thin profile pays the base rate on a rare "yes" it never states.
A row with no series (`""`, `Unknown`, `Web result`) never asks Laya "Which
series?": a listed work its page names gets 0.9, else "Another series" 0.6.
`scripts/replay_trace.py` replays a bench trace over real profiles under the
raw and the calibrated setting.

Multiple-choice questions (`kind: "choice"`, built with `traits._choice`, ≤8
options including `other`) are answered with an option key. Each candidate gets
its own Laya `choice` call; `probabilities[picked]` is the likelihood (cached in
`choice_cache`). Heuristic fallback weights a tag hit at most 1.5× uniform.
Early-guess support for choice evidence is `p_pick / (p_pick + best_other)`.

"Which series?" (`engine._series_question`, wording in `traits.series_question`)
is a dynamic choice over the leading candidates' series (`names.series_key`,
up to 6 plus "Another series"), ranked by information gain like any question,
asked at most twice (the second time only after "Another series"). From
turn `SERIES_PIN_TURN` (6) it is pinned first once the heaviest unoffered
series holds ≥ 0.35 posterior (`_series_pin_id`): Tifa's round surfaced a
Final Fantasy cluster that information gain never asked about. Known
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
until a newer one lands. The first search starts that prefetch but still uses
the template queries, so a turn never waits for it.

DuckDuckGo is the slow source (uncapped it made ~35 sequential requests, ~40 s
a turn). It is capped (`WAIFU_DDG_MAX_REQUESTS` generic queries within
`WAIFU_DDG_BUDGET` seconds, one shared client), gated by the same memoised
`_search_stuck` call as the query LLM (one `pool_fits` per search at most), and
runs on a background worker whenever the session already has candidates; its
hits join the next search. It runs inline only when nothing else was found.

Name proposals are a separate background path. `proposal_facts()` rebuilds a
bounded prompt from the seed, answered buttons (including the labels offered
for an `other` choice), typed details, and the confirmed medium or series;
rejected names and scraped profiles never enter that prompt. A proposal can
run for an empty or flat pool, a new meaningful seed/detail/medium/series/rare
answer, or a newly rejected identity. Unchanged facts use a turn cooldown and
the session cap (`WAIFU_LLM_NAMES_COOLDOWN_TURNS` and
`WAIFU_LLM_NAMES_MAX_PER_SESSION`). Results carry the evidence revision that
requested them and are replayed against the current evidence before they affect
the posterior.

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
(`_GEMINI_BG`, its own worker, capped by `WAIFU_GEMINI_BG_MAX_PENDING`) when
the pool already has a candidate who shows the seed's visual traits, and
inline when the pool is empty or the only hits are junk (a Wikipedia page
that merely shares a word with the seed, or a name in `exclude_names` that
does not show the combination). Results are cached 15 min per facts. With no
facts it is skipped (billed calls). AniList also runs for `medium=game`.
The medium it reports is kept: a game hint does not relabel anime rows.

Answers to yes/no questions are **`yes` / `no` / `detail`**. `detail` does not answer the current
yes/no trait. Instead, the text becomes a separate `clue_question` for Laya and
refines search. The initial seed is evaluated the same way.

Guess when any of: top posterior ≥ 0.80 after ≥ 5 questions; the same candidate
has led for ≥ 2 checks (`WAIFU_NEKOMINI_LEADER_STREAK`) at posterior ≥ 0.50
(`WAIFU_NEKOMINI_LEADER_POSTERIOR`) and ≥ 0.15 ahead of the runner-up
(`WAIFU_NEKOMINI_LEADER_MARGIN`); `ready_to_guess.noul`
≥ 0.75 with `act_probability` ≥ 0.6 and posterior ≥ 0.45; the best lookahead
gain is under `EIG_EXHAUSTED` (0.05 bits) over ≥ 80% of the mass with the
leader at posterior ≥ 0.3 (nothing left separates the top); or turn ≥
`sess.turn_cap()`.
Early guesses also require at least two model judgments with mean answer
likelihood ≥ 0.6. A lone search hit is not sufficient evidence. Up to 3 guesses.
The 0.80 bar stays the single-check gate; the stable-leader path is what commits
a crowded empty-seed pool that otherwise sits at ~0.50–0.70 until the turn cap.

`turn_cap()` is `MAX_TURNS + RECOVERY_TURNS × wrong guesses`. With a fixed cap,
every guess after the first fired back to back on the same evidence (the live
Megumin run guessed Arue, Yunyun, Funifura at turn 20). Now a wrong guess
buys `RECOVERY_TURNS` (2) questions, and recovery and near-twin defer work
inside them. When a guess would commit, near-twin defer asks one rare look
that splits the leader from a same-work runner (≥ 0.10). With no rare look
it asks the unasked yes/no (bank or a slug mined from only some blurbs) that
best splits the leader's same-work cluster of ≤ 4, if that split is worth
≥ 0.6 bits (`_cluster_split_question`). Once per leader/runner pair. Recovery
after a same-work miss falls back to the same cluster split.

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
interpolate them into HTML unescaped. The Solid UI renders those fields as
text nodes (`{name}`), never `innerHTML`.

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
| `WAIFU_NEKOMINI_GUESS_CONFIDENCE` | `0.80` | Posterior that guesses on a single check |
| `WAIFU_NEKOMINI_LEADER_POSTERIOR` | `0.50` | Posterior a stable leader may guess at, below the single-check bar |
| `WAIFU_NEKOMINI_LEADER_MARGIN` | `0.15` | How far that leader must lead the runner-up |
| `WAIFU_NEKOMINI_LEADER_STREAK` | `2` | Consecutive guess-checks the same candidate must have led |
| `WAIFU_NEKOMINI_RECOVERY_TURNS` | `2` | Questions added to the turn cap per wrong guess |
| `WAIFU_NEKOMINI_EIG_CANDIDATES` | `8` | Top candidates Laya judges ahead of each question pick (one forward pass each; `0` = heuristic gain only) |
| `WAIFU_NEKOMINI_SILENT_WEIGHT` | `0.3` | Share of Laya's yes/no drift from the base rate kept when the profile never mentions the trait (`1` = raw noul) |
| `WAIFU_NEKOMINI_CHOICE_SILENT_MAX` | `0.5` | A Laya choice answer whose top option is below this is treated as silent and replaced by option base rates (`0` = off) |
| `WAIFU_HTTP_429_RETRIES` | `2` | Extra attempts after HTTP 429 before the host cools down |
| `WAIFU_HTTP_429_BACKOFF` | `0.8` | Base wait (seconds) when Retry-After is absent; doubles each try |
| `WAIFU_HTTP_429_CAP` | `8` | Max seconds to honour from one Retry-After or backoff |
| `WAIFU_LAYA_HEAD_MAX_LEN` | `480` | Option-token budget |
| `WAIFU_QUERY_LLM` | `0` | Let an LLM rewrite search queries (search strings only) |
| `WAIFU_QUERY_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint (Ollama; `http://localhost:8000/v1` for vLLM, `https://api.openai.com/v1` for OpenAI) |
| `WAIFU_QUERY_LLM_MODEL` | `hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q3_K_M` | Model name sent to that endpoint (`Qwen/Qwen3.6-35B-A3B` still works) |
| `WAIFU_QUERY_LLM_THINKING` | `0` | `1` leaves the model's reasoning on (raise `WAIFU_QUERY_LLM_MAX_TOKENS` with it) |
| `WAIFU_QUERY_LLM_MAX_TOKENS` | `96` | Rewrite reply cap; name proposals use at least 384 |
| `WAIFU_LLM_NAMES` | follows `WAIFU_QUERY_LLM` | `0` stops the LLM proposing candidate names |
| `WAIFU_LLM_NAMES_BG_MAX_PENDING` | `2` | Name-proposal jobs queued or running at once |
| `WAIFU_LLM_NAMES_MAX_PER_SESSION` | `4` | Maximum name-proposal requests in one session |
| `WAIFU_LLM_NAMES_COOLDOWN_TURNS` | `2` | Minimum turns between proposals with unchanged evidence |
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

Bench post-mortems: `GET /api/nekomimi/trace/{session_id}?target=<name>`
(`engine.trace_payload`) returns the target's rank after every answer and,
per guess, the evidence gap to the target (`engine.evidence_breakdown` over
`sess.contrib`). Fetch it before the 30-minute TTL. On Colab the `[waifu]`
lines are in `/content/server.log`: `eig_calls`/`eig_qs` split `pick.eig`,
`dup=` counts source hits already in play, and the `llm_names background`
line lists `proposed/resolved/unresolved` names.

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
6. The query LLM (`query_llm.py`, either backend: Gemini or OpenAI-compatible) writes **search strings** and
   proposes **candidate names** (`propose_characters`), never question text and never decisions. Proposed
   names are leads: `sources.llm_names` keeps one only if a real AniList/Wikipedia page matches it, and
   the profile comes from that page. Its input is player facts only; never send it scraped names or
   blurbs (a rejected guess's name is left out for that reason). Tests stub its HTTP; never call a real endpoint.
7. The query LLM must never block a turn by default. Rewrites run only for new
   typed text when Laya says search is stuck. Name proposals use normalized
   player facts, including button answers and details, and run on a background
   source worker when the pool is empty or flat, meaningful new facts arrive,
   or an identity is rejected. Unchanged facts obey the turn cooldown and
   per-session request cap.
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
