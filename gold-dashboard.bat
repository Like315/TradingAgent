@echo off
rem Gold intraday web dashboard launcher (pure ASCII: cmd parses OEM codepage).
chcp 65001 >nul
title Gold Intraday Dashboard
cd /d "%~dp0"

if not exist ".venv\Scripts\tradingagents-gold.exe" (
  echo [error] .venv not found in "%~dp0".
  echo Follow section 2 of GOLD.md to create it, then run this again.
  goto :end
)

set DEFAULTS=--equity 50000 --risk 0.3

if "%1"=="" (
    ".venv\Scripts\tradingagents-gold.exe" dashboard %DEFAULTS%
) else (
    ".venv\Scripts\tradingagents-gold.exe" dashboard %*
)

:end
if "%GOLD_NO_PAUSE%"=="1" exit /b 0
echo.
pause
