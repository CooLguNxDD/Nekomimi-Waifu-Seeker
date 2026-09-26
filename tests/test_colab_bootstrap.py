"""Colab setup planner: skips, shallow checkout, and overlapping lanes.

Offline. Git remotes are local bare repos. Nothing here installs packages,
talks to Hugging Face, or starts Ollama.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "notebooks" / "colab_bootstrap.py"
    spec = importlib.util.spec_from_file_location("colab_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the class module via sys.modules during decoration.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boot = _load()


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    return env


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env or _git_env(),
    )
    return proc.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")


def _commit_file(repo: Path, name: str, body: str) -> str:
    target = repo / name
    target.write_text(body)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _bare_remote(src: Path, bare: Path) -> None:
    subprocess.run(["git", "clone", "--bare", "-q", str(src), str(bare)], check=True)
    # Local SHA fetches need this; GitHub enables it for the notebook path.
    _git(bare, "config", "uploadpack.allowReachableSHA1InWant", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")


def test_is_commit_sha_accepts_full_and_short_hex():
    assert boot.is_commit_sha("a" * 40)
    assert boot.is_commit_sha("abc1234")
    assert not boot.is_commit_sha("main")
    assert not boot.is_commit_sha("abc123")  # 6 chars stays a branch name
    assert not boot.is_commit_sha("deadbeef-fix")


def test_laya_allow_patterns_match_agent_subfolder():
    patterns = boot.laya_allow_patterns("typed-decisions")
    assert "typed-decisions/model.safetensors" in patterns
    assert "typed-decisions/rl_agent_config.json" in patterns
    assert "typed-decisions/tokenizer/*" in patterns
    assert "typed-decisions/encoder/*" in patterns
    assert boot.laya_allow_patterns("")[0] == "rl_agent_config.json"


def test_requirements_file_parses_without_option_lines():
    specs = boot.parse_requirement_specs((ROOT / "requirements.txt").read_text())
    names = {boot.dist_name(spec) for spec in specs}
    assert "torch" in names
    assert "laya" in names
    assert "fastapi" in names


def test_cuda_path_drops_torch_and_pins_laya_floors():
    specs = boot.parse_requirement_specs((ROOT / "requirements.txt").read_text())
    specs = specs + list(boot.COLAB_EXTRAS)
    versions = {
        "transformers": "4.45.0",
        "safetensors": "0.4.5",
        "huggingface-hub": "0.26.0",
        "numpy": "2.0.0",
    }
    main, laya = boot.select_install_specs(specs, versions, drop_torch=True)
    assert all(boot.dist_name(spec) != "torch" for spec in main)
    assert any(spec.startswith("transformers>=4.48") for spec in main)
    assert laya is not None and boot.dist_name(laya) == "laya"
    assert not any(boot.dist_name(spec) == "laya" for spec in main)


def test_warm_versions_install_nothing():
    specs = boot.parse_requirement_specs((ROOT / "requirements.txt").read_text())
    specs = specs + list(boot.COLAB_EXTRAS)
    versions = {
        "fastapi": "0.115.0",
        "uvicorn": "0.32.0",
        "uvloop": "0.21.0",
        "httptools": "0.6.4",
        "watchfiles": "1.0.0",
        "websockets": "13.0",
        "python-multipart": "0.0.12",
        "laya": "0.3.20",
        "torch": "2.6.0+cu124",
        "transformers": "4.48.0",
        "safetensors": "0.4.5",
        "huggingface-hub": "0.26.0",
        "numpy": "2.1.0",
        "ddgs": "9.5.0",
        "pyyaml": "6.0.2",
        "pyngrok": "7.2.0",
        "playwright": "1.49.1",
        "google-genai": "1.52.0",
    }
    main, laya = boot.select_install_specs(specs, versions, drop_torch=True)
    assert main == []
    assert laya is None


def test_torch_local_version_satisfies_lower_bound():
    main, laya = boot.select_install_specs(
        ["torch>=2.0.0"],
        {"torch": "2.6.0+cu124"},
        drop_torch=False,
    )
    assert main == []
    assert laya is None


def test_uvicorn_standard_extra_is_missing_until_deps_exist():
    main, _laya = boot.select_install_specs(
        ["uvicorn[standard]>=0.27"],
        {"uvicorn": "0.34.0"},
        drop_torch=False,
    )
    assert main == ["uvicorn[standard]>=0.27"]
    full = {
        "uvicorn": "0.34.0",
        "uvloop": "0.21.0",
        "httptools": "0.6.0",
        "watchfiles": "1.0.0",
        "websockets": "13.0",
    }
    main, _laya = boot.select_install_specs(
        ["uvicorn[standard]>=0.27"],
        full,
        drop_torch=False,
    )
    assert main == []


def test_cpu_path_keeps_torch_in_the_main_install():
    main, laya = boot.select_install_specs(
        ["torch>=2.0.0", "laya>=0.3.0"],
        {},
        drop_torch=False,
    )
    assert any(boot.dist_name(spec) == "torch" for spec in main)
    assert any(boot.dist_name(spec) == "laya" for spec in main)
    assert laya is None


def test_cold_plan_pulls_beside_installs():
    snap = boot.Snapshot(
        head=None,
        at_ref=False,
        ollama_bin=False,
        ollama_up=False,
        model_present=False,
        node_major=18,
        ui_fresh=False,
        pip_specs=["pyngrok"],
        laya_spec="laya>=0.3.0",
        playwright=True,
        defer_laya_until_pip=True,
        cuda_torch=True,
    )
    plan = boot.decide(snap)
    assert plan.fetch
    assert plan.install_ollama
    assert plan.pull
    assert plan.install_node
    assert plan.build_ui
    assert plan.pip_specs == ["pyngrok"]
    assert plan.laya_spec == "laya>=0.3.0"
    assert plan.playwright
    text = boot.format_plan(
        boot.Config(
            repo_url="https://example.invalid/repo.git",
            ref="main",
            app_dir="app",
            hf_repo="org/model",
            quant="Q",
            ctx=1000,
            model_name="m",
            ollama_url="http://127.0.0.1:11434",
        ),
        snap,
        plan,
    )
    assert "apt-get update is not part of this setup" in text
    assert "CUDA torch" in text


def test_warm_plan_skips_download_and_rebuild():
    snap = boot.Snapshot(
        head="abc123abc123",
        at_ref=True,
        ollama_bin=True,
        ollama_up=True,
        model_present=True,
        node_major=22,
        ui_fresh=True,
        pip_specs=[],
        laya_spec=None,
        playwright=False,
        defer_laya_until_pip=False,
        cuda_torch=True,
    )
    plan = boot.decide(snap)
    assert not plan.fetch
    assert not plan.install_ollama
    assert not plan.pull
    assert not plan.install_node
    assert not plan.build_ui
    assert plan.pip_specs == []
    assert not plan.playwright


def test_ui_stamp_tracks_commit_and_lockfile(tmp_path: Path):
    app = tmp_path / "app"
    (app / "webui").mkdir(parents=True)
    lock = app / "webui" / "package-lock.json"
    lock.write_text("{}\n")
    dist = app / "waifu_engine" / "webui_dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")
    head = "a" * 40
    assert not boot.ui_is_fresh(app, head)
    (dist / ".nekomimi-ui-stamp").write_text(boot.ui_stamp_text(app, head) + "\n")
    assert boot.ui_is_fresh(app, head)
    assert not boot.ui_is_fresh(app, "b" * 40)
    lock.write_text('{"lock": 2}\n')
    assert not boot.ui_is_fresh(app, head)


def test_chromium_present_uses_browser_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "ms-playwright"
    chrome = root / "chromium-1148" / "chrome-linux64" / "chrome"
    chrome.parent.mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n")
    chrome.chmod(0o755)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(root))
    assert boot.chromium_present()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "empty"))
    assert not boot.chromium_present()


def test_node_install_script_is_valid_bash():
    proc = subprocess.run(
        ["bash", "-n"],
        input=boot._NODE_INSTALL,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_sync_repo_branch_then_sha_then_skip(tmp_path: Path):
    src = tmp_path / "src"
    _init_repo(src)
    first = _commit_file(src, "a.txt", "one\n")
    second = _commit_file(src, "b.txt", "two\n")
    bare = tmp_path / "origin.git"
    _bare_remote(src, bare)
    dest = tmp_path / "checkout"

    head = boot.sync_repo(str(bare), str(dest), "main")
    assert head == second
    assert (dest / "b.txt").is_file()
    again = boot.sync_repo(str(bare), str(dest), "main")
    assert again == second

    pinned = boot.sync_repo(str(bare), str(dest), first)
    assert pinned == first
    assert not (dest / "b.txt").exists()
    detached = subprocess.run(
        ["git", "-C", str(dest), "symbolic-ref", "-q", "HEAD"],
        check=False,
    )
    assert detached.returncode != 0
    skipped = boot.sync_repo(str(bare), str(dest), first)
    assert skipped == first


def test_sync_repo_rejects_a_non_git_directory(tmp_path: Path):
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "file").write_text("x")
    with pytest.raises(RuntimeError, match="not a git checkout"):
        boot.sync_repo(str(tmp_path / "missing.git"), str(dest), "main")


def test_run_lanes_overlap_and_surface_the_first_error():
    def slow(name: str):
        def _inner():
            time.sleep(0.4)
            return name

        return _inner

    started = time.perf_counter()
    out = boot.run_lanes({"a": slow("a"), "b": slow("b"), "c": slow("c")})
    assert time.perf_counter() - started < 1.0
    assert out == {"a": "a", "b": "b", "c": "c"}

    finished: list[str] = []

    def ok():
        time.sleep(0.2)
        finished.append("ok")
        return 1

    def bad():
        raise RuntimeError("pip failed")

    with pytest.raises(RuntimeError, match="pip failed"):
        boot.run_lanes({"ok": ok, "bad": bad})
    assert finished == ["ok"]


def test_bootstrap_runs_the_four_lanes_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(boot, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(boot, "sync_repo", lambda *_a, **_k: "abc123abc123")
    monkeypatch.setattr(boot, "gpu_line", lambda: "NVIDIA L4, 23034 MiB")
    monkeypatch.setattr(boot, "_run", lambda *_a, **_k: "abc123 message\n")
    snap = boot.Snapshot(
        head="abc123abc123",
        at_ref=True,
        ollama_bin=True,
        ollama_up=False,
        model_present=False,
        node_major=22,
        ui_fresh=True,
        pip_specs=["pyngrok"],
        laya_spec=None,
        playwright=False,
        defer_laya_until_pip=True,
        cuda_torch=True,
    )
    monkeypatch.setattr(boot, "take_snapshot", lambda *_a, **_k: snap)

    def slow(label: str):
        def _inner(*_a, **_k):
            time.sleep(0.45)
            return label

        return _inner

    monkeypatch.setattr(boot, "_lane_ollama", slow("ollama"))
    monkeypatch.setattr(boot, "_lane_ui", slow("ui"))
    monkeypatch.setattr(boot, "_lane_python", slow("python"))
    monkeypatch.setattr(boot, "_lane_laya", slow("laya"))
    cfg = boot.Config(
        repo_url="https://example.invalid/repo.git",
        ref="main",
        app_dir=str(tmp_path / "app"),
        hf_repo="org/model",
        quant="Q",
        ctx=1000,
        model_name="m",
        ollama_url="http://127.0.0.1:11434",
    )
    started = time.perf_counter()
    result = boot.bootstrap(cfg)
    assert time.perf_counter() - started < 1.3
    assert result["head"] == "abc123abc123"
    assert set(result["lanes"]) == {"ollama", "ui", "python", "laya"}


def test_wait_for_pull_honors_a_finished_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(boot, "STATE_PATH", str(tmp_path / "state.json"))
    boot._update_state(pull_reaped=True, pull_exit=0, pull_pid=None)
    boot.wait_for_model_pull()
    log = tmp_path / "pull.log"
    log.write_text("error: disk full\n")
    boot._update_state(pull_reaped=True, pull_exit=1, pull_pid=99, pull_log=str(log))
    with pytest.raises(RuntimeError, match="disk full"):
        boot.wait_for_model_pull()


def test_dry_run_does_not_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_DIR", "Nekomimi-Waifu-Seeker")
    code = boot.main(["--dry-run"])
    assert code == 0
    assert not (tmp_path / "Nekomimi-Waifu-Seeker").exists()


def test_notebook_embeds_helper_and_keeps_url_order():
    nb = json.loads((ROOT / "notebooks" / "nekomimi_colab.ipynb").read_text())
    assert nb["metadata"]["colab"]["gpuType"] == "L4"
    assert nb["metadata"]["accelerator"] == "GPU"
    helper = (ROOT / "notebooks" / "colab_bootstrap.py").read_text()
    cells = nb["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    blob = "\n".join(code)
    assert helper.strip() in blob
    for src in code:
        if src.lstrip().startswith("!") or "!ollama" in src or "!nvidia" in src:
            continue
        ast.parse(textwrap.dedent(src))
    assert blob.index("bootstrap(Config(") < blob.index("wait_for_model_pull()")
    assert blob.index("wait_for_model_pull()") < blob.index("ngrok.connect")
    assert blob.index("ngrok.connect") < blob.index("colab_query_smoke")
    assert 'BRANCH     = "main"' in blob or 'BRANCH = "main"' in blob
    joined_md = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    assert "full commit SHA" in joined_md or "commit SHA" in joined_md
