@echo off
rem ===========================================================================
rem  Gold Desk - one-click launcher for the web dashboard.
rem  Pure ASCII on purpose: cmd.exe parses this file with the OEM codepage,
rem  so non-ASCII comments can corrupt the script on a Chinese Windows box.
rem
rem  Usage:
rem    start.bat                     default: http://127.0.0.1:8686, equity 50000
rem    start.bat --port 8689         any dashboard flag is passed through
rem    start.bat --equity 20000 --risk 0.25
rem ===========================================================================
setlocal
chcp 65001 >nul
title Gold Desk Dashboard
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
set "GOLDEXE=.venv\Scripts\tradingagents-gold.exe"

if not exist "%PYEXE%" goto bootstrap
if not exist "%GOLDEXE%" goto bootstrap
goto run

:bootstrap
echo.
echo [setup] No usable virtualenv in this folder. Bootstrapping (runs once).
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo [error] Python not found in PATH. Install Python 3.10+ first.
  goto end
)
if not exist "%PYEXE%" (
  echo [setup] Creating .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [error] Could not create .venv.
    goto end
  )
)
echo [setup] Installing dependencies. This can take several minutes.
"%PYEXE%" -m pip install --upgrade pip -q
"%PYEXE%" -m pip install -e . -q
if errorlevel 1 (
  echo [error] Dependency install failed. Scroll up for the reason.
  goto end
)

:run
if not exist ".env" (
  echo [warn] No .env found: the board still works, but "run model review"
  echo [warn] will fail until you add OPENAI_API_KEY / DEEPSEEK key there.
  echo.
)
echo [run] Dashboard:  http://127.0.0.1:8686
echo [run] Press Ctrl+C or close this window to stop the server.
echo.
"%GOLDEXE%" dashboard %*

:end
if "%GOLD_NO_PAUSE%"=="1" exit /b 0
echo.
pause
