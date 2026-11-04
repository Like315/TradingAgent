@echo off
rem Gold desk launcher. Double-click runs a full analysis; pass flags to change it:
rem   gold.bat --fast          one debate round per side (cheaper, quicker)
rem   gold.bat data            print the four data panels, no model calls
rem   gold.bat doctor          environment check
rem   gold.bat --show          also print the final decision
rem Set GOLD_NO_PAUSE=1 to skip the closing prompt (for scripts).
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\tradingagents-gold.exe" (
  echo [error] .venv not found in "%~dp0".
  echo Follow section 2 of GOLD.md to create it, then run this again.
  goto :end
)

".venv\Scripts\tradingagents-gold.exe" %*

:end
echo.
echo Reports: %USERPROFILE%\.tradingagents\gold\logs\reports
if not defined GOLD_NO_PAUSE pause
