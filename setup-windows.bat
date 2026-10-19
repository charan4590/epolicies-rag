@echo off
REM ============================================================
REM  ePolicies - one-click setup for Windows (safe to run again)
REM  1. creates the Python environment (backend\.venv)
REM  2. installs the backend libraries
REM  3. creates .env and points MARKDOWN_DIR at the "output" folder
REM  4. checks every Markdown file (offline, nothing is changed)
REM ============================================================
setlocal
cd /d "%~dp0"
echo.
echo === ePolicies setup ===
echo Project folder: %CD%
echo.

set "PY="
where py >nul 2>nul && set "PY=py -3.12"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python was not found. Install Python 3.12 from python.org
  echo         and tick "Add python.exe to PATH" during installation.
  goto :end
)

if not exist "backend\.venv\Scripts\python.exe" (
  echo [1/4] Creating Python environment in backend\.venv ...
  %PY% -m venv backend\.venv
  if errorlevel 1 (
    echo [ERROR] Could not create the Python environment.
    goto :end
  )
) else (
  echo [1/4] Python environment already exists.
)
set "VPY=%CD%\backend\.venv\Scripts\python.exe"

echo [2/4] Installing backend libraries (first time takes a few minutes) ...
"%VPY%" -m pip install --disable-pip-version-check -q -r backend\requirements.txt
if errorlevel 1 (
  echo [ERROR] pip install failed. Check the internet connection or company proxy.
  goto :end
)

if not exist ".env" (
  echo [3/4] Creating .env from .env.example ...
  copy /y ".env.example" ".env" >nul
) else (
  echo [3/4] .env already exists - keeping your settings.
)
if exist "output\" (
  "%VPY%" -c "import re,pathlib;p=pathlib.Path('.env');s=p.read_text(encoding='utf-8');p.write_text(re.sub(r'(?m)^MARKDOWN_DIR=.*$','MARKDOWN_DIR=./output',s),encoding='utf-8')"
  echo       MARKDOWN_DIR set to the "output" folder.
) else (
  echo [WARN] No "output" folder found here. Copy your policy .md folder into
  echo        %CD% and name it "output", then run this file again.
)

echo [4/4] Checking your Markdown files ...
echo.
cd backend
"%VPY%" -m app.cli check-files

:end
echo.
echo Done. Take a photo of this window if anything says ERROR or FAILED.
pause
