from __future__ import annotations

import asyncio
import html
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from .nekomimi import engine as nekomimi_engine
from .nekomimi import laya_client
from .nekomimi import session as nekomimi_session
from .nekomimi_page import NEKOMIMI_PAGE
from . import query_llm
from .decide import determine

_YES = {"1", "true", "yes"}


def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in _YES


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load Laya before uvicorn opens the port, so no player waits for it.

    ``WAIFU_LAYA_PRELOAD=0`` restores load-on-first-request. A failed load runs
    the app on heuristics, unless ``WAIFU_LAYA_REQUIRED=1`` asks startup to fail.
    """
    # Required means required: load (and check) even when preload is off.
    required = _env_on("WAIFU_LAYA_REQUIRED", "0")
    if required or (_env_on("WAIFU_LAYA_PRELOAD", "1") and not _env_on("WAIFU_FORCE_FALLBACK", "0")):
        info = await asyncio.to_thread(laya_client.preload)
        if not info["loaded"] and required:
            raise RuntimeError("WAIFU_LAYA_REQUIRED=1 but Laya failed to load: %s" % info["error"])
    yield


app = FastAPI(title="Nekomimi-Waifu-Seeker", version="0.2.0", lifespan=lifespan)

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Nekomimi-Waifu-Seeker</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { max-width: 720px; margin: 2rem auto; padding: 0 1rem; background: #0f1115; color: #e8eaed; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    .sub { color: #9aa0a6; margin-bottom: 1.5rem; }
    textarea { width: 100%; min-height: 96px; padding: 0.75rem; border-radius: 10px; border: 1px solid #333; background: #1a1d24; color: inherit; }
    button { margin-top: 0.75rem; padding: 0.6rem 1rem; border: 0; border-radius: 10px; background: #7c5cff; color: white; font-weight: 600; cursor: pointer; }
    .card { margin-top: 1.25rem; padding: 1rem; border-radius: 12px; background: #171a21; border: 1px solid #2a2f3a; }
    .mode { font-size: 0.85rem; color: #9aa0a6; }
    .winner { font-size: 1.25rem; font-weight: 700; margin: 0.35rem 0; }
    ul { padding-left: 1.1rem; }
    label.row { display:flex; gap:0.5rem; align-items:center; margin-top:0.5rem; color:#c5c8ce; font-size:0.9rem;}
    .shot { margin-top: 0.75rem; border-radius: 10px; overflow: hidden; border: 1px solid #2a2f3a; background:#0b0d11; }
    .shot img { display:block; width:100%; max-height:360px; object-fit:contain; background:#0b0d11; }
    .runners { display:grid; grid-template-columns: repeat(auto-fill,minmax(140px,1fr)); gap:0.75rem; margin-top:0.75rem; }
    .runner { background:#12151b; border:1px solid #2a2f3a; border-radius:10px; overflow:hidden; }
    .runner img { width:100%; height:120px; object-fit:cover; display:block; background:#0b0d11; }
    .runner .cap { padding:0.5rem; font-size:0.8rem; }
  </style>
</head>
<body>
  <h1>Nekomimi-Waifu-Seeker</h1>
  <p class="sub"><a href="/nekomimi" style="color:#9db7ff">Play Nekomimi &rarr;</a></p>
  <p class="sub">Describe features or a name → online character search → Laya decides (fallback only if Laya fails).</p>
  <form method="post">
    <textarea name="query" placeholder="e.g. silver hair tsundere genius mage">__QUERY__</textarea>
    <label class="row"><input type="checkbox" name="fallback" value="1" __FB__/> Force keyword fallback (skip Laya)</label>
    <label class="row"><input type="checkbox" name="online" value="1" __ON__/> Online character search (required — local catalog is empty)</label>
    <label class="row">Rounds <input type="number" name="rounds" min="1" max="5" value="__ROUNDS__" style="width:3.5rem;margin-left:0.35rem;background:#1a1d24;color:inherit;border:1px solid #333;border-radius:6px;padding:0.2rem 0.35rem;"/></label>
    <div><button type="submit">Determine</button></div>
  </form>
  __RESULT__
</body>
</html>"""


def render_page(query: str = "", fb: str = "", online: str = "checked", rounds: str = "3", result: str = "") -> str:
    return (
        PAGE.replace("__QUERY__", query)
        .replace("__FB__", fb)
        .replace("__ON__", online)
        .replace("__ROUNDS__", rounds)
        .replace("__RESULT__", result)
    )


def render_result(result: dict) -> str:
    """Render a determine() result.

    Every field here comes from a scraped web page, so it is HTML-escaped
    before it reaches the DOM.
    """
    if not result or not result.get("winner"):
        return ""
    e = lambda v: html.escape(str(v or ""), quote=True)  # noqa: E731
    w = result["winner"]
    img = e(w.get("image_url"))
    shot = (
        f'<div class="shot"><img src="{img}" alt="{e(w.get("name"))}" loading="lazy" referrerpolicy="no-referrer"/></div>'
        if img
        else ""
    )
    src = e(w.get("source_url"))
    src_html = (
        f'<div class="mode"><a href="{src}" target="_blank" rel="noopener" style="color:#9db7ff">source</a></div>'
        if src
        else ""
    )
    cards = []
    for r in (result.get("runners_up") or []):
        rimg = e(r.get("image_url"))
        img_html = (
            f'<img src="{rimg}" alt="{e(r.get("name"))}" loading="lazy" referrerpolicy="no-referrer"/>'
            if rimg
            else '<div style="height:120px;display:flex;align-items:center;justify-content:center;color:#666;font-size:0.75rem;">no image</div>'
        )
        cards.append(
            f'<div class="runner">{img_html}<div class="cap"><b>{e(r.get("name"))}</b><br/>{e(r.get("confidence"))}</div></div>'
        )
    runners = ('<h3>Runners-up</h3><div class="runners">' + "".join(cards) + "</div>") if cards else ""
    return f"""
    <div class="card">
      <div class="mode">Mode: {e(result.get('mode'))}</div>
      <div class="winner">{e(w.get('name'))} <span style="font-weight:500;color:#9aa0a6">({e(w.get('series'))})</span></div>
      <div>Confidence: <b>{e(w.get('confidence'))}</b></div>
      {shot}
      <p>{e(w.get('blurb'))}</p>
      {src_html}
      <div class="mode">{e(result.get('notes'))}</div>
      {runners}
    </div>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_page()


@app.post("/", response_class=HTMLResponse)
def decide_form(
    query: str = Form(""),
    fallback: str | None = Form(None),
    online: str | None = Form(None),
    rounds: str = Form("3"),
):
    q = query.strip()
    if not q:
        return render_page(result='<div class="card">Enter some features.</div>')
    use_online = True  # catalog cleared — always search
    try:
        n_rounds = max(1, min(5, int(rounds or 3)))
    except ValueError:
        n_rounds = 3
    result = determine(q, force_fallback=bool(fallback), online=use_online, rounds=n_rounds)
    safe_q = html.escape(q, quote=True)
    search = result.get("search") or {}
    extra = ""
    if search:
        extra = (
            f"<div class=\"mode\">DDG online={search.get('online_used')} "
            f"hits={search.get('online_count')} catalog={search.get('catalog_size')} "
            f"err={search.get('online_error')}</div>"
        )
        logs = search.get("rounds") or []
        if logs:
            items = "".join(
                f"<li>r{log.get('round')}: <code>{html.escape(str(log.get('query') or ''))}</code> "
                f"— raw {log.get('raw_hits')}, new {log.get('new_candidates')}"
                f"{(' — ' + html.escape(str(log.get('error')))) if log.get('error') else ''}</li>"
                for log in logs
            )
            extra += f"<div class=\"card\"><div class=\"mode\">Search rounds</div><ul>{items}</ul></div>"
    body = render_result(result) + extra
    return render_page(
        query=safe_q,
        fb="checked" if fallback else "",
        online="checked",
        rounds=str(n_rounds),
        result=body,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "laya": laya_client.status(), "query_llm": query_llm.status()}


@app.get("/nekomimi", response_class=HTMLResponse)
def nekomimi_page() -> str:
    return NEKOMIMI_PAGE


@app.post("/api/nekomimi/start")
def nekomimi_start(payload: dict):
    seed = str(payload.get("seed") or "").strip()[:200]
    return nekomimi_engine.start(seed)


@app.post("/api/nekomimi/answer")
def nekomimi_answer(payload: dict):
    sess = nekomimi_session.get_session(str(payload.get("session_id") or ""))
    if sess is None:
        return {"error": "session not found or expired -- start a new round"}
    answer = str(payload.get("answer") or "")
    detail = str(payload.get("detail") or "")[:300]
    return nekomimi_engine.submit_answer(sess, answer, detail)


@app.post("/api/nekomimi/guess")
def nekomimi_guess(payload: dict):
    sess = nekomimi_session.get_session(str(payload.get("session_id") or ""))
    if sess is None:
        return {"error": "session not found or expired -- start a new round"}
    return nekomimi_engine.submit_guess_result(sess, bool(payload.get("correct")))


@app.get("/api/nekomimi/state/{session_id}")
def nekomimi_state(session_id: str):
    sess = nekomimi_session.get_session(session_id)
    if sess is None:
        return {"error": "session not found or expired"}
    return nekomimi_engine.state_payload(sess)


@app.post("/api/determine")
def api_determine(payload: dict):
    q = str(payload.get("query") or "").strip()
    force = bool(payload.get("fallback"))
    online = payload.get("online")
    if online is None:
        online = True
    rounds = int(payload.get("rounds") or 3)
    return determine(q, force_fallback=force, online=bool(online), rounds=rounds)


def main() -> None:
    import uvicorn

    uvicorn.run("waifu_engine.web:app", host="127.0.0.1", port=7860, reload=False)


if __name__ == "__main__":
    main()
