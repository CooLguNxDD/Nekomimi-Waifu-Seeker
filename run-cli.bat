@echo off
cd /d "%~dp0"
set USE_TF=0
set WAIFU_FORCE_FALLBACK=0
set WAIFU_ONLINE_SEARCH=1
".venv\Scripts\python.exe" -m waifu_engine %*