# Build instructions (Nekomimi-Waifu-Seeker)

- **Architecture:** Python + FastAPI backend (`waifu_engine/`) with SolidJS (Vite) web UI in `webui/`.
- **Test suite (offline only):** `python -m pytest tests -q`. All tests must stay completely offline — never add tests that download Hugging Face weights, launch Chromium, or make live network requests. Tests stub `laya_client.ask`, `web_search.search_by_constraints`, and query LLM HTTP.
- **Frontend build:** `cd webui && npm ci && npm run build`. This compiles the bundle into `waifu_engine/webui_dist`, which FastAPI serves.
- **Docstring requirement:** Every function you add or modify gets a docstring (including private helpers): one line stating what it returns/does, plus the non-obvious *why* (measured failure, constraints). Tests are exempt.
- **Question bank contract:** New questions go in `waifu_engine/nekomimi/question_bank.json` only. `traits.py` expands that template. Tag slugs in `question_bank.json` and `web_search.TRAIT_PATTERNS` must share one vocabulary.
- **Candidate identity:** Candidate identity is `names.name_keys` everywhere (search merge, DDG merge, session pool) — names match in either word order (family-first vs given-first). Never compare raw name strings.
- **Trait matching:** Trait matching is strictly whole-word regex to avoid accidental substring matching.
- **Model boundaries:**
  - Laya is a non-autoregressive decision model (`choice`, `score`, `noul`). Laya calls must always go through `laya_client.ask`. Never generate text from Laya.
  - The query LLM (`query_llm.py`) writes search queries only — never questions or decisions. Never block turns by default.
  - Gemini (`sources/gemini.py`) is a candidate source only.
- **Scraped content safety:** Scraped web content (names, blurbs, image URLs) is untrusted. Never interpolate into HTML unescaped; SolidJS renders via text nodes (`{name}`), never `innerHTML`.
- **Git workflow:** Never push directly to `main`. All changes land via PR from a feature or fix branch.
- **CI Fixes:** Prefer minimal diffs and include `[pullfrog-ci-fix]` in the commit message.
- **Secrets & environment:** Do not commit secrets, `.env` files, or weight caches (`HF_HOME`).
