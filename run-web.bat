@echo off
cd /d "%~dp0"
set USE_TF=0
set WAIFU_FORCE_FALLBACK=0
set WAIFU_ONLINE_SEARCH=1
echo Starting native Laya web UI on http://127.0.0.1:7860
".venv\Scripts\python.exe" -m waifu_engine.web