from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .nekomimi import engine as nekomimi_engine
from .nekomimi import laya_client
from .nekomimi import session as nekomimi_session
from . import query_llm
from .sources import gemini
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

def _bundle_dir() -> Path | None:
    """Return a built Solid bundle, preferring files shipped inside the package.

    ``pip install`` only includes ``waifu_engine*``, so a sibling ``webui/`` tree
    is absent in site-packages. The Vite build writes ``waifu_engine/webui_dist``.
    A checkout that still has ``webui/dist`` is accepted so an older build works.
    The uncompiled Vite ``index.html`` is never served: its ``/src/main.tsx``
    entry cannot run under FastAPI.
    """
    packaged = Path(__file__).resolve().parent / "webui_dist"
    sibling = Path(__file__).resolve().parent.parent / "webui" / "dist"
    for candidate in (packaged, sibling):
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _spa_response() -> FileResponse:
    """Serve the built shell, or 503 until ``npm run build`` has produced one."""
    bundle = _bundle_dir()
    if bundle is None:
        raise HTTPException(status_code=503, detail="webui is not built; run npm run build in webui/")
    return FileResponse(bundle / "index.html")


_BUNDLE = _bundle_dir()
if _BUNDLE is not None and (_BUNDLE / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_BUNDLE / "assets"), name="assets")


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    """Serve the Solid app at the one-shot route."""
    return _spa_response()


@app.get("/nekomimi", response_class=FileResponse)
def nekomimi_page() -> FileResponse:
    """Serve the same shell; the client router opens the guessing game."""
    return _spa_response()


@app.get("/healthz")
def healthz():
    """Liveness plus the state of each model-backed piece (Laya, query LLM, Gemini)."""
    return {"status": "ok", "laya": laya_client.status(), "query_llm": query_llm.status(),
            "gemini": gemini.status()}


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
