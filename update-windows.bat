@echo off
REM ============================================================
REM  ePolicies - weekly policy update for Windows
REM  1. starts your own PostgreSQL database if it is not running
REM  2. checks the Google Cloud login (needed for embeddings)
REM  3. backs up the output folder and embedding.jsonl
REM  4. downloads the latest policies with scrape_epolicies.py --update
REM  5. checks the files and shows what changed, then asks before updating
REM  6. re-indexes only new and changed files, then shows the status
REM  The backend and website can keep running; no restart is needed.
REM ============================================================
setlocal
cd /d "%~dp0"

REM ---- Settings: keep these the same as in start-windows.bat ----
set "PG_BIN=C:\Program Files\PostgreSQL\16\bin"
set "PG_DATA=%USERPROFILE%\epolicies-pgdata"
set "PG_PORT=5433"

echo.
echo === ePolicies weekly update ===
echo Project folder: %CD%
echo.

REM Python environment: backend\.venv, .venv here, or .venv one folder up
set "VPY="
for %%V in ("%CD%\backend\.venv" "%CD%\.venv" "%CD%\..\.venv") do (
  if not defined VPY if exist "%%~V\Scripts\python.exe" set "VPY=%%~fV\Scripts\python.exe"
)
if not defined VPY (
  echo [ERROR] No Python environment found in backend\.venv, .venv or ..\.venv
  goto :fail
)
echo Python: %VPY%
if not exist "scrape_epolicies.py" (
  echo [ERROR] scrape_epolicies.py is not in this folder. Put this file next to it.
  goto :fail
)

REM ---- 1. Database ----
echo [1/6] Database
if exist "%PG_DATA%\PG_VERSION" (
  "%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" status >nul 2>nul
  if errorlevel 1 (
    echo       Starting PostgreSQL on port %PG_PORT% ...
    "%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" -o "-p %PG_PORT%" -l "%PG_DATA%\log.txt" -w -t 60 start >nul
    if errorlevel 1 (
      echo [ERROR] PostgreSQL did not start. See %PG_DATA%\log.txt
      goto :fail
    )
  )
  echo       Running.
) else (
  echo       No database folder at %PG_DATA% - using DATABASE_URL from .env as it is.
)

REM ---- 2. Google Cloud login ----
echo [2/6] Google Cloud login
where gcloud >nul 2>nul
if errorlevel 1 (
  echo       gcloud not found - skipping the check.
  goto :backup
)
call gcloud auth application-default print-access-token >nul 2>nul
if not errorlevel 1 (
  echo       Login is valid.
  goto :backup
)
echo       Your login has expired. Sign in with your work Google account in the browser window.
call gcloud auth application-default login
if errorlevel 1 (
  echo [ERROR] Google Cloud sign-in did not finish.
  goto :fail
)

REM ---- 3. Backup ----
:backup
echo [3/6] Backing up the current files
if exist "output\" (
  robocopy "output" "output_backup" /MIR /NFL /NDL /NJH /NJS /NP >nul
  if errorlevel 8 (
    echo [ERROR] Could not back up the output folder.
    goto :fail
  )
  echo       output  -^>  output_backup
)
if exist "artifacts\embedding.jsonl" (
  copy /y "artifacts\embedding.jsonl" "artifacts\embedding-backup.jsonl" >nul
  echo       artifacts\embedding.jsonl  -^>  artifacts\embedding-backup.jsonl
)

REM ---- 4. Download the latest policies ----
echo [4/6] Downloading the latest policies - close Microsoft Edge first if it is open
choice /c YN /m "Is Edge closed and ready to continue"
if errorlevel 2 goto :fail
"%VPY%" scrape_epolicies.py --update
if errorlevel 1 (
  echo [ERROR] The scraper stopped with an error. Nothing in the app was changed.
  goto :fail
)

REM ---- 5. Check and preview ----
echo [5/6] Checking the files and previewing the changes
pushd backend
"%VPY%" -m app.cli check-files
"%VPY%" -m app.cli ingest --dry-run
if errorlevel 1 (
  popd
  echo [ERROR] The preview failed - read the lines above.
  goto :fail
)
echo.
echo Check the numbers above: new / modified / unchanged / deleted.
echo If MANY files show as deleted, the download probably failed - answer N.
choice /c YN /m "Update the app with these changes"
if errorlevel 2 (
  popd
  echo Update cancelled. The app still uses last week's index.
  goto :fail
)

REM ---- 6. Update ----
echo [6/6] Updating the index - only new and changed files are sent to Gemini
"%VPY%" -m app.cli ingest
set "INGEST_STATUS=%errorlevel%"
"%VPY%" -m app.cli status
popd
if not "%INGEST_STATUS%"=="0" (
  echo [WARN] Some files failed. Run this file again later, or in backend:
  echo        python -m app.cli ingest --retry-failed
  goto :fail
)
echo.
echo Update finished. The website uses the new policies from the next question.
pause
exit /b 0

:fail
echo.
echo Update did not finish. Take a photo of this window and send it.
echo To undo, see section 8 of RUN_GUIDE.pdf.
pause
exit /b 1
