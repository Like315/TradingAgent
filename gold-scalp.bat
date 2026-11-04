@echo off
rem Gold scalp launcher: intraday board for a scalping session.
rem
rem   gold-scalp.bat              live monitor, refresh every 60s (free; Ctrl-C to stop)
rem   gold-scalp.bat once         print the board once and exit
rem   gold-scalp.bat --llm        board + a written card (about 4 model calls)
rem   gold-scalp.bat --watch 30    custom refresh interval in seconds
rem   gold-scalp.bat --json        machine-readable board
rem
rem Account size and per-trade risk live in DEFAULTS below (0.25-0.5% for scalping).
rem This file is ASCII only on purpose: cmd.exe parses .bat files in the OEM
rem codepage, and UTF-8 Chinese comments break the parser before chcp takes effect.
chcp 65001 >nul
cd /d "%~dp0"

set DEFAULTS=scalp --equity 50000 --risk 0.3

if not exist ".venv\Scripts\tradingagents-gold.exe" (
  echo [error] .venv not found in "%~dp0".
  echo Follow section 2 of GOLD.md to create it, then run this again.
  goto :end
)

if "%~1"=="" (
  ".venv\Scripts\tradingagents-gold.exe" %DEFAULTS% --watch 60
  goto :end
)

if /i "%~1"=="once" (
  ".venv\Scripts\tradingagents-gold.exe" %DEFAULTS%
  goto :end
)

".venv\Scripts\tradingagents-gold.exe" %DEFAULTS% %*

:end
echo.
echo Reports and cache: %USERPROFILE%\.tradingagents\gold
if not defined GOLD_NO_PAUSE pause
