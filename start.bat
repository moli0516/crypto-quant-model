@echo off
REM 本地快捷啟動腳本 (Windows)

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python cli.py %*
