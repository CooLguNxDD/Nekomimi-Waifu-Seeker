"""Colab cold start: shallow checkout, then overlapping installs and downloads.

The notebook embeds this file (Colab downloads the ipynb only) and calls
``bootstrap``. Independent work shares the wall clock: the Ollama installer
and the GGUF pull, the Node/UI build, pip, and the Laya weight prefetch.

The GGUF pull starts as soon as ``ollama serve`` is up. ``bootstrap`` does not
wait for it. The warmup cell joins the pull and loads Qwen onto the GPU
before uvicorn, so Laya's preload does not take VRAM the 17 GB model needs
on an L4.

``python notebooks/colab_bootstrap.py`` prints the plan and does not install
anything. ``--run`` is the notebook's ``bootstrap`` path from a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Kept in sync with laya.Agent's snapshot_download allow-list (laya 0.3.x).
# An unfiltered snapshot also pulls sibling checkpoints.
_LAYA_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
_LAYA_FLOORS = {
    "transformers": "4.48.0",
    "safetensors": "0.4.0",
    "huggingface-hub": "0.20.0",
    "numpy": "1.20.0",
}
# uvicorn[standard] is satisfied only when these are installed too.
_UVICORN_STANDARD = ("uvloop", "httptools", "watchfiles", "websockets")
COLAB_EXTRAS = (
    "pyngrok",
    "playwright>=1.49",
    "google-genai>=1.51.0",
)
STATE_PATH = "/tmp/nekomimi-colab-bootstrap.json"
_NODE_INSTALL = """
set -euo pipefail
if ! command -v xz >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xz-utils
fi
ver=$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt \\
  | awk '/node-v22\\..*-linux-x64\\.tar\\.xz$/{print $2; exit}')
case "$ver" in
  node-v22.*) ;;
  *) echo "could not resolve a Node 22 tarball: ${ver}" >&2; exit 1 ;;
esac
curl -fsSL "https://nodejs.org/dist/latest-v22.x/${ver}" \\
  | sudo tar -xJ -C /usr/local --strip-components=1
"""


@dataclass
class Config:
    """Inputs the notebook config cell already collected."""

    repo_url: str
    ref: str
    app_dir: str
    hf_repo: str
    quant: str
    ctx: int
    model_name: str
    ollama_url: str
    laya_model: str = "convaiinnovations/laya"
    laya_subfolder: str = "typed-decisions"


@dataclass
class Snapshot:
    """What is already on the machine, so a rerun can skip finished work."""

    head: str | None
    at_ref: bool
    ollama_bin: bool
    ollama_up: bool
    model_present: bool
    node_major: int
    ui_fresh: bool
    pip_specs: list[str]
    laya_spec: str | None
    playwright: bool
    defer_laya_until_pip: bool
    cuda_torch: bool


@dataclass
class Decisions:
    """The steps ``bootstrap`` will run. Pure data so tests can pin the plan."""

    fetch: bool
    install_ollama: bool
    start_serve: bool
    pull: bool
    install_node: bool
    build_ui: bool
    pip_specs: list[str] = field(default_factory=list)
    laya_spec: str | None = None
    playwright: bool = True
    prefetch_laya: bool = True
    defer_laya_until_pip: bool = True


def is_commit_sha(ref: str) -> bool:
    """True when ``ref`` is a 7–40 character hex commit id.

    Benches pin a commit by pasting it into ``BRANCH``. A name that is only
    hex is fetched as a commit; branch names like ``main`` stay branches.
    """
    text = ref.strip()
    if len(text) < 7 or len(text) > 40:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in text)


def dist_name(spec: str) -> str:
    """Return the normalized distribution name from a requirement specifier."""
    base = spec.split("#", 1)[0].split(";", 1)[0].strip()
    name = ""
    for ch in base:
        if ch in "<>=!~[":
            break
        name += ch
    return name.strip().lower().replace("_", "-")


def parse_requirement_specs(text: str) -> list[str]:
    """Return requirement specifiers from a requirements file.

    Comments and blank lines are skipped. Options lines (``-r``, ``-e``) are
    not specs; Colab installs extras itself.
    """
    specs = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        specs.append(line)
    return specs


def _version_below(installed: str, floor: str) -> bool:
    """True when ``installed`` is older than ``floor``.

    A missing packaging install treats the version as old so pip still runs.
    """
    try:
        from packaging.version import Version
    except ImportError:
        return True
    try:
        return Version(installed) < Version(floor)
    except Exception:
        return True


def select_install_specs(
    specs: list[str],
    versions: dict[str, str],
    *,
    drop_torch: bool,
) -> tuple[list[str], str | None]:
    """Return ``(pip specs, laya --no-deps spec or None)``.

    Names already new enough are omitted so a warm runtime does not pay for
    a resolver. When ``drop_torch`` is set, Colab's CUDA torch stays: ``laya``
    is installed with ``--no-deps`` and its other floors are checked here,
    because a normal ``laya`` install would let pip replace that wheel.
    """
    wanted: list[str] = []
    laya_spec: str | None = None
    seen: set[str] = set()
    for spec in specs:
        name = dist_name(spec)
        if not name or name in seen:
            continue
        seen.add(name)
        if name == "torch" and drop_torch:
            continue
        if name == "laya" and drop_torch:
            if _spec_unsatisfied(spec, versions):
                laya_spec = spec
            continue
        if _spec_unsatisfied(spec, versions):
            wanted.append(spec)
    if drop_torch:
        for name, floor in _LAYA_FLOORS.items():
            have = versions.get(name)
            if have is not None and not _version_below(have, floor):
                wanted = [spec for spec in wanted if dist_name(spec) != name]
                continue
            pin = f"{name}>={floor}"
            wanted = [spec for spec in wanted if dist_name(spec) != name]
            wanted.append(pin)
        if laya_spec is None and "laya" not in versions:
            laya_spec = "laya>=0.3.0"
    return wanted, laya_spec


def _spec_unsatisfied(spec: str, versions: dict[str, str]) -> bool:
    """True when this interpreter does not already meet ``spec``."""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        return True
    try:
        req = Requirement(spec)
    except Exception:
        return True
    key = req.name.lower().replace("_", "-")
    have = versions.get(key)
    if have is None:
        return True
    if req.specifier:
        try:
            if Version(have) not in req.specifier:
                return True
        except Exception:
            return True
    if req.name.lower() == "uvicorn" and "standard" in req.extras:
        for dep in _UVICORN_STANDARD:
            if dep not in versions:
                return True
    return False


def laya_allow_patterns(subfolder: str) -> list[str]:
    """Return the ``snapshot_download`` allow-list ``laya.Agent`` uses.

    Matching that list is what makes the server preload a cache hit. A full
    repo download also pulls the sibling checkpoints the library skips.
    """
    prefix = f"{subfolder}/" if subfolder else ""
    return [prefix + name for name in _LAYA_FILES]


def ui_stamp_text(app_dir: Path, head: str) -> str:
    """Return the stamp body for this checkout and lockfile.

    The stamp is the commit plus the lockfile hash, so a new bench SHA rebuilds
    the UI even when ``package-lock.json`` did not change.
    """
    lock = (app_dir / "webui" / "package-lock.json").read_bytes()
    digest = hashlib.sha256(lock).hexdigest()
    return f"{head} {digest}"


def ui_is_fresh(app_dir: Path, head: str) -> bool:
    """True when ``webui_dist`` was built for this commit and lockfile.

    ``npm ci`` on every rerun costs a minute and does not change the bundle
    when the checkout is the same.
    """
    dist = app_dir / "waifu_engine" / "webui_dist"
    index = dist / "index.html"
    stamp = dist / ".nekomimi-ui-stamp"
    if not head or not index.is_file() or not stamp.is_file():
        return False
    try:
        expected = ui_stamp_text(app_dir, head)
    except OSError:
        return False
    return stamp.read_text().strip() == expected


def chromium_present() -> bool:
    """True when a Playwright Chromium binary is already on disk.

    ``playwright install chromium`` contacts the CDN even when the browser
    is present. The executable is enough to skip that.
    """
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    # Playwright uses only PLAYWRIGHT_BROWSERS_PATH when it is set. "0" means
    # the package default, which is the home cache below.
    if env and env != "0":
        roots = [Path(env)]
    else:
        roots = [
            Path.home() / ".cache" / "ms-playwright",
            Path.home() / "ms-playwright",
        ]
    patterns = ("chromium-*/chrome-linux/chrome", "chromium-*/chrome-linux64/chrome")
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for chrome in root.glob(pattern):
                if chrome.is_file() and os.access(chrome, os.X_OK):
                    return True
    return False


def decide(snap: Snapshot) -> Decisions:
    """Choose setup steps from a snapshot.

    ``fetch``, the UI build, pip, and the GGUF pull are independent once the
    checkout exists. ``pull`` stays true until the model is listed so the
    download can run beside pip instead of after it.
    """
    return Decisions(
        fetch=not snap.at_ref,
        install_ollama=not snap.ollama_bin,
        start_serve=not snap.ollama_up,
        pull=not snap.model_present,
        install_node=snap.node_major < 22,
        build_ui=not snap.ui_fresh,
        pip_specs=list(snap.pip_specs),
        laya_spec=snap.laya_spec,
        playwright=snap.playwright,
        prefetch_laya=True,
        defer_laya_until_pip=snap.defer_laya_until_pip,
    )


def format_plan(cfg: Config, snap: Snapshot, decisions: Decisions) -> str:
    """Return a multi-line plan of the lanes ``bootstrap`` is about to run."""
    lines = [
        f"[colab] ref {cfg.ref}",
        "[colab] checkout: "
        + (f"at {snap.head}" if snap.at_ref and snap.head else f"fetch {cfg.ref}"),
        "[colab] lanes in parallel: ollama (install, serve, start GGUF pull) | "
        "ui (node, npm) | python (pip, chromium) | laya weights",
    ]
    if snap.cuda_torch:
        lines.append("[colab] CUDA torch is already installed; pip will not replace it")
    if decisions.pip_specs:
        lines.append("[colab] pip: " + ", ".join(decisions.pip_specs))
    else:
        lines.append("[colab] pip: nothing missing")
    if decisions.laya_spec:
        lines.append(f"[colab] pip --no-deps {decisions.laya_spec}")
    if not decisions.pull:
        lines.append("[colab] GGUF already present; pull skipped")
    lines.append("[colab] apt-get update is not part of this setup")
    return "\n".join(lines)


def _run(args: list[str], *, cwd: str | None = None, capture: bool = False) -> str:
    """Run a command. Return stdout when ``capture`` is set.

    Progress stays on the console so a long pip or npm does not look stuck.
    A non-zero exit raises ``CalledProcessError``.
    """
    print("[colab] $", " ".join(args), flush=True)
    proc = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if proc.returncode != 0:
        detail = ""
        if capture:
            detail = (proc.stderr or proc.stdout or "").strip()
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, detail)
    if capture:
        return proc.stdout or ""
    return ""


def _head_or_none(dest: str) -> str | None:
    """Return the checkout's HEAD, or None when the repo has no commits yet."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dest,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip().lower() or None


def _remote_branch_sha(dest: str, branch: str) -> str:
    """Return the upstream SHA for ``branch``, or ``""`` when it is missing."""
    out = _run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=dest,
        capture=True,
    )
    line = out.strip().splitlines()
    if not line or not line[0].strip():
        return ""
    return line[0].split()[0].lower()


def sync_repo(repo_url: str, dest: str, ref: str) -> str:
    """Point ``dest`` at ``ref`` and return the checked-out commit.

    Clones are shallow. ``git fetch --all`` on every rerun downloaded every
    branch and dominated the checkout cell. A hex ref is fetched by SHA and
    left detached so ``origin/<sha>`` does not have to exist. When HEAD is
    already that commit, the fetch is skipped.
    """
    path = Path(dest)
    if path.exists() and not (path / ".git").is_dir():
        if any(path.iterdir()):
            raise RuntimeError(f"{dest} exists and is not a git checkout")
    if not (path / ".git").is_dir():
        path.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], cwd=dest)
        _run(["git", "remote", "add", "origin", repo_url], cwd=dest)
    head = _head_or_none(dest)
    if is_commit_sha(ref):
        want = ref.strip().lower()
        if head and head.startswith(want):
            print(f"[colab] already at {head}")
            return head
        _run(["git", "fetch", "--depth", "1", "origin", ref.strip()], cwd=dest)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    else:
        if head and _remote_branch_sha(dest, ref) == head:
            print(f"[colab] already at origin/{ref} ({head[:12]})")
            return head
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest)
        _run(["git", "checkout", "-B", ref, "FETCH_HEAD"], cwd=dest)
    resolved = _head_or_none(dest)
    if not resolved:
        raise RuntimeError(f"checkout of {ref} did not produce a commit")
    print("[colab] HEAD", resolved[:12])
    return resolved


def _installed_versions() -> dict[str, str]:
    """Return lowercase distribution name → version for this interpreter."""
    from importlib.metadata import distributions

    found: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name and dist.version:
            found[name] = dist.version
    return found


def _cuda_torch_ready() -> bool:
    """True when this interpreter already has CUDA torch 2.x.

    Import errors mean torch still has to be installed. A successful CUDA
    import means pip must not fetch another torch build.
    """
    try:
        import torch
    except Exception:
        return False
    raw = getattr(torch, "__version__", "") or ""
    numbers = []
    for piece in raw.split("+", 1)[0].split(".")[:2]:
        if not piece.isdigit():
            return False
        numbers.append(int(piece))
    if tuple(numbers) < (2, 0):
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _node_major() -> int:
    """Major version of ``node`` on PATH, or 0 when Node is missing."""
    try:
        out = subprocess.check_output(
            ["node", "-p", "process.versions.node"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out.split(".")[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0


def _hub_importable() -> bool:
    """True when ``huggingface_hub`` can be imported without installing it."""
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return False
    return True


def _ollama_up(url: str) -> bool:
    """True when Ollama's HTTP port accepts a connection."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status < 500
    except Exception:
        return False


def _ollama_list() -> str:
    """Return ``ollama list`` text, or ``""`` when the binary is missing."""
    try:
        return subprocess.check_output(
            ["ollama", "list"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def model_ref(hf_repo: str, quant: str) -> str:
    """Return the Ollama tag the pull cell uses for this quant."""
    return f"hf.co/{hf_repo}:{quant}"


def model_present(hf_repo: str, quant: str, model_name: str) -> bool:
    """True when the pulled tag or the wrapped model name is already listed.

    A warm runtime should not download the GGUF again. Either name counts:
    the pull tag before ``ollama create``, and ``MODEL_NAME`` after it.
    """
    listed = _ollama_list()
    if not listed:
        return False
    return model_ref(hf_repo, quant) in listed or model_name in listed


def _requirements_text(app_dir: str) -> str:
    """Return ``requirements.txt`` from the checkout, or from this repo.

    ``bootstrap`` reads the checkout after ``sync_repo``. Dry-run prints a plan
    before any clone, so it uses the copy next to this file when that exists.
    """
    path = Path(app_dir) / "requirements.txt"
    if path.is_file():
        return path.read_text()
    try:
        sibling = Path(__file__).resolve().parents[1] / "requirements.txt"
    except NameError:
        return ""
    if sibling.is_file():
        return sibling.read_text()
    return ""


def take_snapshot(cfg: Config, head: str | None) -> Snapshot:
    """Probe the machine and the checkout. This does not install anything."""
    app = Path(cfg.app_dir)
    versions = _installed_versions()
    text = _requirements_text(cfg.app_dir)
    cuda = _cuda_torch_ready()
    pip_specs, laya_spec = select_install_specs(
        parse_requirement_specs(text) + list(COLAB_EXTRAS),
        versions,
        drop_torch=cuda,
    )
    # Any pip install can upgrade huggingface_hub as a dependency. Wait for
    # that to finish before snapshot_download so the two do not share the module.
    defer = bool(pip_specs or laya_spec) or not _hub_importable()
    return Snapshot(
        head=head,
        at_ref=bool(head),
        ollama_bin=shutil.which("ollama") is not None,
        ollama_up=_ollama_up(cfg.ollama_url),
        model_present=model_present(cfg.hf_repo, cfg.quant, cfg.model_name),
        node_major=_node_major(),
        ui_fresh=bool(head) and ui_is_fresh(app, head),
        pip_specs=pip_specs,
        laya_spec=laya_spec,
        playwright=not chromium_present(),
        defer_laya_until_pip=defer,
        cuda_torch=cuda,
    )


def _read_state() -> dict[str, Any]:
    """Return the pull/serve state file, or an empty dict when it is absent."""
    try:
        data = json.loads(Path(STATE_PATH).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _update_state(**kwargs: Any) -> dict[str, Any]:
    """Merge ``kwargs`` into the state file and return the stored dict."""
    data = _read_state()
    data.update(kwargs)
    Path(STATE_PATH).write_text(json.dumps(data))
    return data


def _install_ollama() -> None:
    """Install the Ollama binary with the upstream script.

    The script apt-installs what it needs. This setup does not run its own
    ``apt-get update`` first; that update was on the critical path and the
    stock Colab image already has curl.
    """
    _run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])


def _start_serve(url: str) -> None:
    """Start ``ollama serve`` when the port is closed.

    Flash attention, q8 KV, and keep-alive match the previous notebook cell.
    A server that is already up is left alone so a rerun does not kill a pull.
    """
    if _ollama_up(url):
        print("[colab] Ollama already listening")
        return
    env = os.environ.copy()
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    log = open("ollama.log", "w")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=log,
        stderr=log,
        env=env,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Ollama exited during startup — see ollama.log")
        if _ollama_up(url):
            print("[colab] Ollama up, pid", proc.pid)
            _update_state(serve_pid=proc.pid)
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama didn't start — check ollama.log")


def _start_pull(cfg: Config) -> int:
    """Start ``ollama pull`` in the background and return its pid.

    The log goes to ``ollama-pull.log`` so pip and npm can share the console.
    ``wait_for_model_pull`` joins this pid before ``ollama create``.
    """
    ref = model_ref(cfg.hf_repo, cfg.quant)
    log_path = "ollama-pull.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["ollama", "pull", ref],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    _update_state(
        pull_pid=proc.pid,
        pull_log=log_path,
        pull_reaped=False,
        pull_exit=None,
        model_ref=ref,
    )
    print(f"[colab] GGUF pull started pid {proc.pid} ({ref})")
    return proc.pid


def _mark_pull_skipped() -> None:
    """Record that no pull is in flight because the model is already listed."""
    _update_state(pull_pid=None, pull_reaped=True, pull_exit=0)


def _lane_ollama(cfg: Config, decisions: Decisions) -> str:
    """Install Ollama if needed, serve, and start the GGUF pull."""
    if decisions.install_ollama:
        _install_ollama()
    else:
        print("[colab] Ollama already installed")
    if decisions.start_serve or not _ollama_up(cfg.ollama_url):
        _start_serve(cfg.ollama_url)
    else:
        print("[colab] Ollama already listening")
    if decisions.pull:
        return f"pull:{_start_pull(cfg)}"
    print("[colab] GGUF already present")
    _mark_pull_skipped()
    return "pull:skip"


def _install_node22() -> None:
    """Install Node 22 under ``/usr/local`` when the major version is older.

    Vite 7 needs a current Node. The Docker UI stage uses node:22. ``xz`` is
    installed only when it is missing, without a full ``apt-get update``.
    """
    _run(["bash", "-lc", _NODE_INSTALL])
    print("[colab] node", _run(["node", "--version"], capture=True).strip())


def _build_ui(app_dir: str, head: str) -> None:
    """Run ``npm ci`` and ``npm run build``, then stamp the bundle.

    ``--no-audit`` skips the registry advisory request. The stamp is written
    after the build because Vite empties ``webui_dist`` first.
    """
    ui = os.path.join(app_dir, "webui")
    _run(["npm", "ci", "--no-audit", "--no-fund"], cwd=ui)
    _run(["npm", "run", "build"], cwd=ui)
    dist = Path(app_dir) / "waifu_engine" / "webui_dist"
    index = dist / "index.html"
    if not index.is_file():
        raise RuntimeError(f"UI build did not write {index}")
    (dist / ".nekomimi-ui-stamp").write_text(ui_stamp_text(Path(app_dir), head) + "\n")
    print("[colab] UI bundle:", index)


def _lane_ui(cfg: Config, decisions: Decisions, head: str) -> str:
    """Install Node when required, then build the Solid bundle."""
    if decisions.install_node or _node_major() < 22:
        _install_node22()
    else:
        print("[colab] node", _run(["node", "--version"], capture=True).strip())
    if decisions.build_ui:
        _build_ui(cfg.app_dir, head)
    else:
        print("[colab] UI bundle is current; skipping npm")
    return "ui"


def _pip_install(specs: list[str], *, no_deps: bool = False) -> None:
    """Pip-install ``specs`` with the already-satisfied strategy.

    ``only-if-needed`` is pip's default; it is passed explicitly so a future
    ``-U`` does not slip in and upgrade Colab's CUDA torch.
    """
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
    ]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(specs)
    _run(cmd)


def _lane_python(cfg: Config, decisions: Decisions, pip_done: threading.Event) -> str:
    """Install missing Python packages, then Chromium.

    The event is set as soon as pip returns so the Laya prefetch can overlap
    the browser download. Chromium is optional: search still has Wikipedia,
    AniList, and DuckDuckGo when the install fails.
    """
    try:
        if decisions.pip_specs:
            _pip_install(decisions.pip_specs)
        else:
            print("[colab] Python packages already satisfied")
        if decisions.laya_spec:
            _pip_install([decisions.laya_spec], no_deps=True)
    finally:
        pip_done.set()
    if not decisions.playwright:
        print("[colab] Chromium already installed")
        return "python"
    proc = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=False,
    )
    if proc.returncode != 0:
        print(
            "[colab] Playwright Chromium did not install. "
            "Search still uses Wikipedia, AniList, and DuckDuckGo."
        )
    return "python"


def prefetch_laya(model_id: str, subfolder: str) -> str:
    """Download the Laya checkpoint into the Hugging Face cache.

    Uses the same allow-list and token variable as ``laya.Agent``, so uvicorn's
    preload reads the cache instead of blocking the port on an 800 MB download.
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    path = snapshot_download(
        model_id,
        token=token,
        allow_patterns=laya_allow_patterns(subfolder),
    )
    print("[colab] Laya weights:", path)
    return path


def _lane_laya(cfg: Config, decisions: Decisions, pip_done: threading.Event) -> str:
    """Prefetch Laya weights, waiting for pip only when hub itself is changing."""
    if decisions.defer_laya_until_pip:
        pip_done.wait()
    if not decisions.prefetch_laya:
        return "laya:skip"
    sub = os.environ.get("WAIFU_LAYA_SUBFOLDER", cfg.laya_subfolder).strip()
    model = os.environ.get("WAIFU_LAYA_MODEL", cfg.laya_model).strip() or cfg.laya_model
    try:
        prefetch_laya(model, sub)
    except Exception as exc:  # noqa: BLE001 — the server can still download
        print(f"[colab] Laya prefetch failed ({exc}). The server will download on startup.")
        return "laya:failed"
    return "laya"


def gpu_line() -> str:
    """Return the GPU name, or a note when ``nvidia-smi`` shows nothing.

    The notebook metadata requests an L4. This does not change the runtime;
    a CPU session is printed before the GGUF pull starts.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "no NVIDIA GPU visible (this notebook expects a Colab L4)"
    if not out:
        return "no NVIDIA GPU visible (this notebook expects a Colab L4)"
    return out


def run_lanes(lanes: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Run each callable at the same time. Return name → result.

    Pip, npm, and the Ollama installer do not share a lock. Running them one
    after another made cold start the sum of those waits. The first failure
    is re-raised after the other lanes finish, so one error does not leave
    ``apt`` or ``ollama`` half-killed mid-command.
    """
    if not lanes:
        return {}
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {pool.submit(fn): name for name, fn in lanes.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except BaseException as exc:  # noqa: BLE001 — collected, then re-raised
                errors[name] = exc
    if errors:
        names = ", ".join(sorted(errors))
        print(f"[colab] lane failed: {names}", flush=True)
        raise next(iter(errors.values()))
    return results


def _wrap_timed(name: str, fn: Callable[[], Any], timed: dict[str, float]) -> Callable[[], Any]:
    """Return a callable that records ``name``'s wall time in ``timed``."""

    def inner() -> Any:
        """Run one lane and store its elapsed seconds under ``name``."""
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timed[name] = round(time.perf_counter() - started, 1)
            print(f"[colab] lane {name} {timed[name]}s", flush=True)

    return inner


def bootstrap(cfg: Config) -> dict[str, Any]:
    """Check out ``cfg.ref`` and run the setup lanes. Return a small status dict.

    The returned dict's ``pull_pid`` is set while the GGUF download is still
    running. Call ``wait_for_model_pull`` before creating the Ollama model.
    ``USE_TF=0`` is set when unset so Transformers does not probe TensorFlow
    during the weight prefetch.
    """
    os.environ.setdefault("USE_TF", "0")
    started = time.perf_counter()
    print("[colab] GPU:", gpu_line())
    head = sync_repo(cfg.repo_url, cfg.app_dir, cfg.ref)
    oneline = _run(["git", "log", "-1", "--oneline"], cwd=cfg.app_dir, capture=True).strip()
    print("[colab]", oneline)
    snap = take_snapshot(cfg, head)
    decisions = decide(snap)
    print(format_plan(cfg, snap, decisions))
    pip_done = threading.Event()
    if not decisions.defer_laya_until_pip:
        pip_done.set()
    timed: dict[str, float] = {}
    lanes = {
        "ollama": _wrap_timed("ollama", lambda: _lane_ollama(cfg, decisions), timed),
        "ui": _wrap_timed("ui", lambda: _lane_ui(cfg, decisions, head), timed),
        "python": _wrap_timed("python", lambda: _lane_python(cfg, decisions, pip_done), timed),
        "laya": _wrap_timed("laya", lambda: _lane_laya(cfg, decisions, pip_done), timed),
    }
    run_lanes(lanes)
    elapsed = round(time.perf_counter() - started, 1)
    state = _read_state()
    print(f"[colab] setup lanes finished in {elapsed}s")
    return {
        "head": head,
        "seconds": elapsed,
        "lanes": timed,
        "pull_pid": state.get("pull_pid"),
        "pull_reaped": bool(state.get("pull_reaped")),
    }


def _pull_log_tail(log_path: str, limit: int = 20) -> str:
    """Return the last ``limit`` lines of the pull log, or a short placeholder."""
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return "(no pull log)"
    return "\n".join(lines[-limit:])


def wait_for_model_pull(poll_seconds: float = 15.0) -> None:
    """Block until the background ``ollama pull`` exits 0.

    The pull overlaps pip and the UI build. This is the join before
    ``ollama create``. A state file that already records success returns
    immediately, which is the warm-runtime path.
    """
    state = _read_state()
    if state.get("pull_reaped"):
        code = state.get("pull_exit")
        if code in (0, None) and state.get("pull_pid") is None:
            print("[colab] GGUF pull already finished")
            return
        if code == 0:
            print("[colab] GGUF pull already finished")
            return
        tail = _pull_log_tail(str(state.get("pull_log") or "ollama-pull.log"))
        raise RuntimeError(f"ollama pull failed (exit {code})\n{tail}")
    pid = state.get("pull_pid")
    if not pid:
        raise RuntimeError("no ollama pull was started; re-run the setup cell")
    log_path = str(state.get("pull_log") or "ollama-pull.log")
    ref = str(state.get("model_ref") or "")
    while True:
        try:
            got, status = os.waitpid(int(pid), os.WNOHANG)
        except ChildProcessError:
            if ref and ref in _ollama_list():
                _update_state(pull_reaped=True, pull_exit=0)
                print("[colab] GGUF pull finished")
                return
            tail = _pull_log_tail(log_path)
            raise RuntimeError(f"ollama pull is gone and {ref or 'the model'} is not listed\n{tail}")
        if got == 0:
            print("[colab] pull", _pull_log_tail(log_path, limit=1), flush=True)
            time.sleep(poll_seconds)
            continue
        code = os.waitstatus_to_exitcode(status)
        _update_state(pull_reaped=True, pull_exit=code)
        if code != 0:
            raise RuntimeError(f"ollama pull failed (exit {code})\n{_pull_log_tail(log_path)}")
        print("[colab] GGUF pull finished")
        return


def create_fixed_context_model(hf_repo: str, quant: str, ctx: int, model_name: str) -> None:
    """Register an Ollama model named ``model_name`` with ``num_ctx`` fixed.

    The pulled tag keeps Ollama's default context. The rewriter calls
    ``model_name``. Creating the wrapper is metadata over the blob that just
    finished downloading.
    """
    Path("Modelfile").write_text(f"FROM hf.co/{hf_repo}:{quant}\nPARAMETER num_ctx {ctx}\n")
    _run(["ollama", "create", model_name, "-f", "Modelfile"])
    print(_run(["ollama", "list"], capture=True))


def _default_config() -> Config:
    """Return the same defaults as the notebook config cell, for ``--dry-run``."""
    return Config(
        repo_url="https://github.com/CooLguNxDD/Nekomimi-Waifu-Seeker.git",
        ref=os.environ.get("BRANCH", "main"),
        app_dir=os.environ.get("APP_DIR", "Nekomimi-Waifu-Seeker"),
        hf_repo=os.environ.get("HF_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF"),
        quant=os.environ.get("QUANT", "UD-Q3_K_M"),
        ctx=int(os.environ.get("CTX", "10000")),
        model_name=os.environ.get("MODEL_NAME", "Qwen3.6-35B-A3B-GGUF"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    )


def main(argv: list[str] | None = None) -> int:
    """Print the setup plan, or run it when ``--run`` is passed.

    The default is a dry run so executing this file on a laptop does not
    install Ollama or start a 17 GB download. The notebook calls ``bootstrap``.
    """
    parser = argparse.ArgumentParser(description="Colab cold-start helper")
    parser.add_argument(
        "--run",
        action="store_true",
        help="clone, install, and start the GGUF pull (the notebook path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan for this machine and exit (the default)",
    )
    args = parser.parse_args(argv)
    cfg = _default_config()
    if args.run:
        bootstrap(cfg)
        return 0
    print("[colab] GPU:", gpu_line())
    # Local probe only. A missing checkout still plans from installed packages.
    app = Path(cfg.app_dir)
    head = _head_or_none(cfg.app_dir) if (app / ".git").is_dir() else None
    snap = take_snapshot(cfg, head)
    if head is None:
        snap.at_ref = False
        snap.ui_fresh = False
    print(format_plan(cfg, snap, decide(snap)))
    print("[colab] dry run only — pass --run to install and download")
    return 0


def _running_as_script() -> bool:
    """True when this file is the process entry point, not a notebook cell.

    Notebook cells run with ``__name__ == '__main__'`` too. The argv check
    keeps ``main`` from starting during that exec.
    """
    return Path(sys.argv[0]).name == "colab_bootstrap.py"


if _running_as_script():
    raise SystemExit(main())
