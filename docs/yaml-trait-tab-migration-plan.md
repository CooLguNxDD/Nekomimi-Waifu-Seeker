# YAML trait / tab migration plan

Research on `main` at `9bf1f7d` (Vocaloid / list-page and franchise near-miss merges). This document does not change runtime behavior. Snippets below are proposals, not files the engine loads.

## Executive answer

**Partial.** Static mappings — synonym lists, medium accept sets, ordered franchise markers, category crumbs, color words, publisher buckets — can become YAML. Decision algorithms cannot.

There is no UI tab layer to templatize. The Solid app renders one question at a time (`webui/src/components/nekomimi/QuestionCard.tsx`) and only forwards `category` as an optional string (`webui/src/types/game.ts`). “Tabs” in this codebase are question **categories** plus a few category sets the engine uses to rank or skip questions.

Question wording is already a template, in JSON: `waifu_engine/nekomimi/question_bank.json` (58 rows → 98 runtime questions, 21 categories), expanded by `traits.load_question_bank` / `_expand_entry`. Converting that file to YAML is a format swap. The duplication that actually hurts is the second copy of the same vocabulary in Python (`web_search.TRAIT_PATTERNS` and the visual/franchise tables).

## Current architecture sketch

```
question_bank.json
    └─ traits._expand_entry → QUESTION_BANK / QUESTIONS_BY_ID / CATEGORIES
           ├─ engine.candidate_questions
           │     info-gain (_split_quality) over the bank
           │     skip settled categories (GuessSession.settled_categories: a "yes")
           │     pin appearance qids (_pinned_question_ids, _SERIES_APPEARANCE)
           │     dynamic "Which series?" (traits.series_question, ≤6 + other)
           └─ engine.score_candidates
                 yes/no: Laya noul, else _tag_match / _profile_likelihood
                 choice: Laya choice, else _tag_choice; known medium/series via _known_choice
                 then _apply_identity_priors (franchise lead, protagonist, namesake)

search hits
    web_search._guess_medium / franchise_label / is_aggregate_page / mine_trait_slugs
    wikipedia._medium_of / _is_character / _series_of
    names.series_key / is_publisher_series / same_character
    session.add_candidates drops aggregate pages before they enter the pool

guess
    engine._should_guess (posterior, leader streak, ready_to_guess, turn cap)
```

Medium is special. `traits.MEDIUM_ACCEPTS` is a hard filter in `engine._eliminate_by_medium`, except the empty set for `"other"`, which means **no filter**. `engine._known_choice` then puts most of the probability on the candidate’s own medium and a smaller share on crossovers (movie ↔ TV, anime covering manga).

Series is two different tables that must not be merged blindly:

| Table | Role |
|---|---|
| `web_search._SERIES_MARKERS` | Ordered phrase → canonical label on scraped text. Longer Vocaloid aliases first. |
| `engine._FRANCHISE_ALIASES` + `_EXACT_ALIASES` | What the **player typed**. `"mario"` counts only as a whole clue (`_typed_franchise`). |

`web_search.SERIES_BLOCK` is both a name blocklist (`_ok_person_name`) and an aggregate rule (`is_aggregate_page`: a plural whose stem is in the block, so `vocaloids` dies because `vocaloid` is blocked).

## Inventory

Sizes counted on this commit. “YAML?” is easy / partial / hard.

| Cluster | Where | Size | Kind | YAML? | Risk if moved |
|---|---|---|---|---|---|
| Question catalog | `question_bank.json`; `traits._q`, `_choice`, `_trait_block`, `load_question_bank` | 50 yes/no rows, 3 choice rows, 5 `trait_block`s (hair 4, eyes 1, pers 13, job 15, power 12) → 98 questions, 21 categories | Data, already templated. Criteria text is still built in Python (`noul_criteria`) | Partial. Keep JSON or swap format later. Do not re-encode the expander | Choice width is capped at 8 (`_choice` assert, Laya `head_max_len`). Duplicate ids already fail load. A YAML port that evaluates the JSON `"criteria_from": "noul_criteria(...)"` string would be a new interpreter |
| Category policy (the “tabs”) | `traits.CATEGORIES` (derived). Engine sets: `_BROAD_CATEGORIES` {medium, gender, meta}, `_HAIR_CATEGORIES` {hair_color, hair}, `_SERIES_APPEARANCE` (4 qids). `session.settled_categories` | 3 small sets | Data for the sets; control flow for skip-on-yes and pin order | Easy for the sets. Hard for the skip/pin algorithms | Pin order is load-bearing (hair, then halo/wings/horns). A yes on `look_halo` must not settle `look_wings` (separate categories on purpose) |
| Medium accept / values | `traits.MEDIUM_VALUES`, `MEDIUM_ACCEPTS`. Consumed by `_medium_hint`, `_eliminate_by_medium`, `_known_choice` | 6 media, 6 accept rows | Data, with one sentinel | Easy | `"other": []` is “do not filter”, not “accept nothing”. Encoding it as a missing key would wipe known media (the bug fixed in the guess-quality work) |
| Medium classifiers | `web_search.MEDIUM_DOMAIN_HINTS` (11), `MEDIUM_TEXT_HINTS` (6 media, 36 phrases). `wikipedia.MEDIUM_CATEGORY_HINTS` (30 phrases), `MEDIUM_TEXT_HINTS` (26). `sources._MEDIUM_SUFFIX` (6) | ~80 literals, two independent copies | Data. **Order is the algorithm**: ACG hints before movie/TV so an anime TV series stays anime | Easy as ordered lists. Partial if the two copies are unified | Substring checks in Wikipedia (`m in joined`) are weaker than `web_search._marker_re` whole-word matches. Unifying the matcher changes labels. Domain hints are first-match |
| Franchise markers | `web_search._SERIES_MARKERS` (56, Vocaloid aliases first), `franchise_label`, `franchise_mentioned`, `_phrase_re` | 56 pairs | Ordered data + whole-phrase matcher | Easy for the list. Matcher stays code | Dict YAML would drop order and let a later broader marker beat Crypton / Project Diva / Sekai. Whole-word boundaries exist because `persona` matched `personality` and `bleach` matched `bleached` |
| Typed franchise aliases | `engine._FRANCHISE_ALIASES` (9), `_EXACT_ALIASES` {mario}, `_typed_franchise` | 10 strings | Data + a whole-clue exception | Easy for the rows. The exception is a flag, not a second algorithm | Folding this into `_SERIES_MARKERS` would treat a typed `mario` like a scraped `super mario` and penalise “Mario Rossi” |
| Series crumbs | `web_search._SERIES_CRUMBS` (13), `series_is_crumb`. Used by `wikipedia._series_of` | 13 | Data | Easy | Exact normalized string match. Adding regex here is how crumbs start eating real titles |
| Aggregate / list pages | `is_aggregate_page`, `_AGGREGATE_NAME`, `_ROSTER_URL`, `_AGGREGATE_EXACT` (4), plus `SERIES_BLOCK` plural stem. `wikipedia.BAD_TITLE`, `NOT_CHARACTER`, `IS_CHARACTER`, `CATEGORY_HINT`. `browser_search._FANDOM_SKIP` | ~15 regexes, 4 exact names | Control flow with a few exact-name hooks | Partial. Exact names and URL tokens can be lists. The decision tree stays code | Roster vs profile: `/characters` is aggregate, `/characters/hatsune-miku` is not (`test_profiles_under_a_characters_route_are_not_aggregates`). A looser YAML regex would guess VOCALOIDs again |
| Name blocklist | `SERIES_BLOCK` (63), `STOP` (101), `NON_NAME_WORDS` (17), `JUNK_DOMAINS` (5), `LISTICLE`, `NON_NAME`, `_ok_person_name` | ~190 literals | Mixed. Lists are data; length/case/colon rules are code | Partial. Move the sets. Leave `_ok_person_name` | `SERIES_BLOCK` is shared with aggregate detection. Splitting the set without a test that `vocaloids` is still aggregate will regress list pages |
| Color words | `web_search._COLOR_WORDS` (28), `is_color_phrase`. Hair choice tags live in the question bank; mined slugs in `TRAIT_PATTERNS` (blonde…purple, eyes-*) | 28 words + 9 hair slugs + 5 eye slugs | Data | Easy | `is_color_phrase` only suppresses a **detail** on a hair question (`_fact_entries`). A seed that is only a color is still searched as a name (`test_color_word_seed_is_still_a_name_query`) |
| Trait mining vocabulary | `web_search.TRAIT_PATTERNS` (42 slugs, 126 markers), `mine_trait_slugs`. Must match question tags (`CLAUDE.md` rule 2). Visual override: `traits.mined_visual_slugs` / `has_body_wings` | 42 + wing/halo/horn special case | Data for markers. Wing-vs-halo is an algorithm (`_sentence_has_body_wings`, `_BODY_WING`, `_WINGS_NEG`, `_HALO_DECOR`) | Partial. Markers easy. Wing sentences hard | Whole-word compiler is `_marker_re` because `"he"` matched `"the"`. Gender markers are the highest-risk rows. Tests: `test_colour_option_tags_are_mined_from_snippets`, `test_mika_seed.py` |
| Visual player phrases | `traits._VISUAL_PHRASES` (pink hair, halo, wings, horns), `visual_search_phrases`, `clue_likelihood`, `yesno_visual_likelihood` | 4 phrases; likelihoods 0.92 / 0.10 | Mixed | Partial. Phrase list easy. Likelihoods and negation windows stay code | These bypass the [0.4, 0.6] heuristic cap on purpose. Putting `0.92` in YAML without the “profile long enough” rule (`_APPEARANCE_MIN` 80) makes thin blurbs decisive |
| Publisher / generic series | `wikipedia.GENERIC_SERIES` (11), `_PUBLISHER_SERIES` (5), `names._PUBLISHER_KEYS` (6), `is_publisher_series` | ~22 | Data, duplicated (display string vs `series_key`) | Easy, one table with both forms | Keys are alphanumeric (`marvelcomics`). A YAML author who writes `Marvel Comics` only on one side desyncs Wikipedia and the series question |
| Popular / catalog categories | `sources.POPULAR_CATEGORIES` (4 media × 2). Tooling only: `scripts/build_catalog.py` `GAME_CATEGORIES` (16), `COMIC_CATEGORIES` | 8 live + ~20 offline | Data | Easy, low value | Live path never reads `catalog.json`. Do not point runtime YAML at the build script’s lists |
| Identity priors | `_SERIES_LEAD_BONUS` 0.9, `_PROTAGONIST_LEAD_BONUS` 0.8, `_SIDE_CHARACTER_PENALTY` 0.7, `_NAMESAKE_PENALTY` 2.2. `_is_series_lead` (protagonist/mascot tag, else ≥3× popularity and ≥1000). `_apply_identity_priors`, `_in_franchise` | 4 numbers + branching | Control flow. Numbers are calibrated against a trait miss (~1.7 nats) | Hard | A “bonus table” without the `protag == "no"` guard re-boosts the face of the series after the player said they are not the lead |
| Guess gates | `GUESS_CONFIDENCE`, `LEADER_*`, `MIN_QUESTIONS_BEFORE_GUESS`, `READY_MIN_POSTERIOR`, `_should_guess`, `_note_leader` | Already env vars | Control flow | Hard. Leave as env | YAML would fork a second way to set the same knobs |
| Name identity | `names.name_keys`, `same_character`, `series_key`, `same_series` (prefix ≥5), `longer_namesake` | Algorithms | Control flow | Hard | Japanese given/family order is why keys exist. Not a synonym list |
| Search / HTTP | `sources.find_candidates`, DDG caps, 429 backoff, Gemini, query LLM | Orchestration | Control flow | Hard | Out of scope |

## Proposed YAML schemas

Keep one concern per file under the package, not the repo root. `pyproject.toml` only ships `waifu_engine.nekomimi` → `question_bank.json`. A `config/` directory at the repo root would be missing from the wheel and the Docker image. Proposed home: `waifu_engine/nekomimi/lexicon/` (or keep questions in JSON and add sibling YAML). Load once at import; do not parse YAML per turn.

Validate on load: unknown keys fail, lists that must be ordered are sequences, duplicate ids fail, choice options stay ≤8, `"other"` medium is present and tagged `filter: false`.

### Trait synonyms (the slice that removes real duplication)

`TRAIT_PATTERNS` and question-bank tags are the same vocabulary written twice. A lexicon row can feed `mine_trait_slugs` without becoming the question.

```yaml
# waifu_engine/nekomimi/lexicon/traits.yml
traits:
  - id: pink
    group: hair_color
    # mined from blurbs; question_bank hair_color option "pink" already tags this slug
    markers: ["pink hair"]
  - id: eyes-blue
    group: eye_color
    markers: ["blue eyes", "blue-eyed"]
  - id: wings
    group: look_wings
    markers: ["angel wings", "feathered wings"]
    # mine_trait_slugs still calls traits.mined_visual_slugs; this flag is not a regex
    reconcile: body_wings
```

Question prompts stay in `question_bank.json`. The loader checks that every `tags_true` / option tag that has a lexicon row uses the same id. Slugs with no markers (most personality questions) stay JSON-only.

### Category policy (not UI tabs)

```yaml
# waifu_engine/nekomimi/lexicon/categories.yml
categories:
  medium:      { broad: true }
  gender:      { broad: true }
  meta:        { broad: true }
  hair_color:  { color_detail: true }
  hair:        { color_detail: true }
  look_halo:   { series_appearance: true }
  look_wings:  { series_appearance: true }
  look_horns:  { series_appearance: true }
# hair_color is pinned by appearance_question_ids(), not by this flag.
# settled_categories() remains: a "yes" answer retires that category.
```

### Medium accept map

```yaml
# waifu_engine/nekomimi/lexicon/medium.yml
media: [anime, manga, comic, game, movie, tv]
# filter: false is the "other" / "Something else" sentinel.
accepts:
  anime: { media: [anime, manga] }
  game:  { media: [game] }
  comic: { media: [comic] }
  movie: { media: [movie, tv] }
  tv:    { media: [tv, movie] }
  other: { filter: false }
search_suffix:
  anime: "anime character"
  manga: "manga character"
  game: "video game character"
  comic: "comic book character"
  movie: "film character"
  tv: "TV series character"
```

Domain and text hints stay ordered sequences in the same file or in `medium_hints.yml`, one list per source (`search_domains`, `search_text`, `wikipedia_categories`, `wikipedia_text`). Do not merge the Wikipedia substring matcher and the search whole-word matcher in the first PR that moves the lists.

### Franchise markers, crumbs, aggregate names

```yaml
# waifu_engine/nekomimi/lexicon/franchises.yml
# First match wins. Longer aliases before the bare name.
markers:
  - { phrase: "project sekai", label: "Vocaloid" }
  - { phrase: "colorful stage", label: "Vocaloid" }
  - { phrase: "project diva", label: "Vocaloid" }
  - { phrase: "crypton future", label: "Vocaloid" }
  - { phrase: "vocaloid", label: "Vocaloid" }
  - { phrase: "blue archive", label: "Blue Archive" }

# Player text only. exact: true → the whole clue must be the phrase.
typed_aliases:
  - { phrase: "super mario", label: "Super Mario" }
  - { phrase: "spider-man", label: "Spider-Man" }
  - { phrase: "spiderman", label: "Spider-Man" }
  - { phrase: "mario", label: "Super Mario", exact: true }

crumbs:
  - "internet meme"
  - "internet memes"
  - "software"
  - "voice bank"
  - "voice banks"

aggregate_exact: ["vocaloids", "fanloid", "fanloids", "vocaloid characters"]

publishers:
  - { label: "Marvel Comics", key: "marvelcomics" }
  - { label: "DC Comics", key: "dccomics" }
```

`is_aggregate_page` keeps its regexes in Python and reads `aggregate_exact` plus the blocklist. `SERIES_BLOCK` can move as `name_block` in the same file only in the phase that updates both call sites together.

### Color words

```yaml
# waifu_engine/nekomimi/lexicon/colors.yml
# is_color_phrase: every token is in this set.
words:
  - teal
  - aqua
  - turquoise
  - cyan
  - pink
  - blonde
  - blond
```

## What must stay in code

- Scoring math: `_split_quality`, `_entropy_n`, `session.posterior`, `logit`, log-odds updates in `_rescore_candidates`.
- Guess policy: `_should_guess`, `_note_leader`, `_leader_support`. Thresholds stay environment variables.
- When priors apply: `_apply_identity_priors`, `_is_series_lead`, `_in_franchise` (known series beats a blurb mention). The four bonus constants can be named in YAML later; the branches cannot.
- Matchers: `_marker_re`, `_phrase_re`, `series_key`, `same_series_key`, `name_keys`.
- Wing vs halo sentence logic and clue negation (`traits._sentence_has_body_wings` and callers).
- Aggregate decision tree, Wikipedia `_is_character` / `_series_of` category regex, Fandom skip regex, character-URL allow lists.
- Search orchestration, HTTP/429, DuckDuckGo and Gemini workers, query LLM.
- Laya criteria construction (`noul_criteria`, `_q`, `_choice`). The JSON `templates` block documents shapes; it is not executed.
- Sanitising scraped labels (`_series_label`) and rendering them as text (`QuestionCard` uses `{opt.label}`).

## Migration phases

Each phase ships behind the current Python constants until a parity test says the loaded tables match. Rollback is reverting the loader call so the module-level literals are authoritative again. No phase deletes a literal until the following phase’s tests have been green on `main`.

### Phase 0 — Loader and parity freeze

Scope: a YAML loader (PyYAML is not a dependency today; add it or parse a JSON subset — decide here, do not smuggle a second parser). Load the proposed files in tests only. Assert equality with `MEDIUM_ACCEPTS`, `_COLOR_WORDS`, `_SERIES_MARKERS`, `_SERIES_CRUMBS`, `_AGGREGATE_EXACT`, `_FRANCHISE_ALIASES`. Runtime still uses the Python objects.

Success: `pytest tests -q` unchanged; a new parity test fails if a literal drifts from the fixture YAML.

Rollback: delete the test and the unused files.

### Phase 1 — Pure maps the engine already treats as data

Scope: switch readers to the loader for medium accepts, search suffixes, color words, crumbs, aggregate exact names, publisher labels/keys, popular-category titles. Leave matchers and `is_aggregate_page` in place.

Success: existing tests listed below pass without edits except import paths. `test_a_medium_pick_removes_known_other_media_but_never_unknowns` still treats `"other"` as a no-op filter.

Rollback: constants reassigned from the old literals; YAML ignored.

### Phase 2 — Trait marker catalog, not the question bank

Scope: `TRAIT_PATTERNS` generated from `traits.yml`. `question_bank.json` stays. A unit test fails if a bank tag and a lexicon id disagree, or if a lexicon slug has no bank tag and is not listed as mine-only.

Success: `test_colour_option_tags_are_mined_from_snippets`, `test_halo_wings_and_horns_are_separate_questions`, `test_body_wings_are_not_a_wing_shaped_halo`.

Rollback: restore the tuple. Do not convert `question_bank.json` to YAML in this phase.

### Phase 3 — Franchise and aggregate data

Scope: ordered markers, typed aliases (with `exact`), `SERIES_BLOCK` / `name_block` once both `_ok_person_name` and the plural-stem rule read it. Wikipedia generic-series words. Regexes stay Python.

Success: `test_vocaloid_series_beats_category_crumbs`, `test_list_pages_never_lead_the_pool`, `test_franchise_markers_match_whole_words`, `test_mario_alias_needs_the_whole_clue`, `test_a_known_series_beats_a_blurb_mention_of_the_franchise`, `test_wikipedia_series_skips_the_publisher_bucket`.

Rollback: previous tuples. Order bugs show up as the wrong `franchise_label`, so the parity test must compare sequences, not sets.

### Phase 4 — Delete dead Python tables

Scope: remove literals that no production function reads. Keep a short comment pointing at the YAML file and the matcher that interprets it.

Success: grep shows one definition per table; package data includes the new files (`pyproject.toml` `[tool.setuptools.package-data]`); Docker image still answers `/healthz` and a stubbed nekomimi turn.

Rollback: the phase is a deletion commit; revert it.

Question-bank JSON → YAML is an optional later format cleanup, not a phase above. It does not remove manual checks.

## Testing strategy

Offline tests already pin behavior at the functions, not at “this line is a Python set”. Keep them as the contract while storage moves.

| Behavior | Tests that must stay green |
|---|---|
| Medium hard filter, movie/TV crossover, `"other"` | `tests/test_target_questions.py` (`test_medium_*`, `test_a_medium_pick_*`, `test_movie_and_tv_stay_apart_by_soft_evidence`) |
| Vocaloid label, crumbs, list pages, color details | `test_vocaloid_series_beats_category_crumbs`, `test_category_crumbs_are_not_a_series`, `test_list_pages_never_lead_the_pool`, `test_color_detail_is_not_a_name_query`, `test_typed_vocaloid_confirms_series_and_is_offered`, `test_profiles_under_a_characters_route_are_not_aggregates` |
| Near-miss priors, Mario exact alias, publishers | `tests/test_franchise_nearmiss.py` |
| Mined tags line up with bank tags | `tests/test_search_decisions.py` `test_colour_option_tags_are_mined_from_snippets`; `tests/test_mika_seed.py` |
| Category skip, choice width, guess gates | `tests/test_nekomimi.py` (`test_settled_category_is_not_asked_again`, `test_choice_questions_are_narrow_and_replace_colour_yes_nos`, leader-streak tests) |
| Name identity | `tests/test_names.py` |

Add one parity test per phase: load YAML, compare to the structure the function used last release (order, empty `other`, `exact` aliases). Do not assert on private regex source. Tests stay offline: no weights, Chromium, or network. New tests stub `laya_client.ask` the same way.

A loader failure (missing file, bad type, duplicate id) should raise at import, the way duplicate question ids already raise in `load_question_bank`. Silent fallback to an empty map would drop every franchise marker and look like a search bug.

## Non-goals / anti-patterns

- No Python expressions, `eval`, or `!!python` tags in YAML. The unused `templates.yesno.criteria_from` string in `question_bank.json` is documentation; do not make it executable.
- No control flow in YAML: no “if blurb contains wings and halo then 0.1” rules. That logic stays in `traits.py`.
- No single mega-file. Medium, traits, franchises, and colors are separate so a Vocaloid alias edit does not churn the question catalog.
- No UI tabs invented to match the schema. The client does not group questions.
- No change to guess thresholds, log-odds, or search order in the migration PRs.
- No runtime read of `data/catalog.json`.
- Do not unify `_SERIES_MARKERS` and `_FRANCHISE_ALIASES` until `exact: true` exists and `test_mario_alias_needs_the_whole_clue` covers it.

## Recommendation

**Worth doing, narrowly. Do not YAML-ify the engine.**

Start with Phase 0 + Phase 1: medium accepts (including the `other` sentinel), color words, crumbs, aggregate exact names, publisher keys. Those are small, ordered or set-shaped, and already covered by tests. Phase 2 is the payoff (one trait vocabulary). Phase 3 is where regressions hide (Vocaloid order, Mario exactness, `SERIES_BLOCK` double duty); do it only after parity tests compare sequences.

Leave `question_bank.json` as the question catalog. Leave scoring, guess gates, wing/halo parsing, name keys, and HTTP in Python.

Effort: Phase 0–1 is one small PR (loader, four YAML files, packaging, parity test). Phase 2 is a second PR. Phase 3–4 are a third, and the risky one. This is not a rewrite of `engine.py`.

**Go / no-go:** go on Phase 0–1. Revisit Phase 3 only if adding a franchise alias still means editing two Python tuples and a blocklist.
