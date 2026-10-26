@echo off
REM ============================================================
REM  ePolicies - one-click start for Windows (safe to run again)
REM  1. starts your own PostgreSQL database if it is not running
REM  2. checks the Google Cloud login (Vertex AI); asks you to sign in if it expired
REM  3. runs "doctor" to check the database and Gemini
REM  4. opens the backend and the website in their own windows
REM  5. opens http://localhost:3000 in your browser
REM  Stop everything with stop-windows.bat
REM ============================================================
setlocal
cd /d "%~dp0"

REM ---- Settings: change these three lines if your setup is different ----
set "PG_BIN=C:\Program Files\PostgreSQL\16\bin"
set "PG_DATA=%USERPROFILE%\epolicies-pgdata"
set "PG_PORT=5433"

echo.
echo === Starting ePolicies ===
echo Project folder: %CD%
echo.

REM Python environment: backend\.venv (made by setup-windows.bat), or .venv here, or .venv one folder up
set "VPY="
for %%V in ("%CD%\backend\.venv" "%CD%\.venv" "%CD%\..\.venv") do (
  if not defined VPY if exist "%%~V\Scripts\python.exe" set "VPY=%%~fV\Scripts\python.exe"
)
if not defined VPY (
  echo [ERROR] No Python environment found in backend\.venv, .venv or ..\.venv
  echo         Run setup-windows.bat first.
  goto :fail
)
echo Python: %VPY%

REM ---- 1. Database ----
echo [1/5] Database
if not exist "%PG_DATA%\PG_VERSION" (
  echo       No database folder at %PG_DATA% - skipping.
  echo       The app will use DATABASE_URL from .env as it is.
  goto :gcloud
)
if not exist "%PG_BIN%\pg_ctl.exe" (
  echo [ERROR] pg_ctl.exe was not found in %PG_BIN%
  echo         Edit PG_BIN at the top of this file.
  goto :fail
)
"%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" status >nul 2>nul
if not errorlevel 1 (
  echo       Already running.
  goto :gcloud
)
echo       Starting PostgreSQL on port %PG_PORT% ...
"%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" -o "-p %PG_PORT%" -l "%PG_DATA%\log.txt" -w -t 60 start >nul
if errorlevel 1 (
  echo [ERROR] PostgreSQL did not start. Last lines of its log:
  powershell -NoProfile -Command "Get-Content -Tail 15 '%PG_DATA%\log.txt'"
  goto :fail
)
echo       Started.

REM ---- 2. Google Cloud login (only when .env uses Vertex AI) ----
:gcloud
echo [2/5] Google Cloud login
set "VERTEX="
for %%F in (".env" "backend\.env") do (
  if exist %%F findstr /i /r /c:"^ *GOOGLE_GENAI_USE_VERTEXAI *= *true" %%F >nul 2>nul && set "VERTEX=1"
)
if not defined VERTEX (
  echo       Not using Vertex AI - skipping.
  goto :doctor
)
where gcloud >nul 2>nul
if errorlevel 1 (
  echo [WARN] gcloud was not found, so the login cannot be checked.
  goto :doctor
)
call gcloud auth application-default print-access-token >nul 2>nul
if not errorlevel 1 (
  echo       Login is valid.
  goto :doctor
)
echo       Your login has expired. A browser window will open:
echo       sign in with your work Google account and click Allow.
call gcloud auth application-default login
if errorlevel 1 (
  echo [ERROR] Google Cloud sign-in did not finish.
  goto :fail
)

REM ---- 3. Health check ----
:doctor
echo [3/5] Checking the database and Gemini
pushd backend
"%VPY%" -m app.cli doctor
set "DOCTOR_STATUS=%errorlevel%"
popd
if not "%DOCTOR_STATUS%"=="0" (
  echo.
  echo [WARN] doctor found problems - read the FAIL lines above.
  choice /c YN /m "Start the app anyway"
  if errorlevel 2 goto :fail
)

REM ---- 4. Backend and website ----
echo [4/5] Starting the backend and the website
netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul
if not errorlevel 1 (
  echo       Backend is already running on port 8000.
) else (
  start "ePolicies backend" /d "%CD%\backend" cmd /k ""%VPY%" -m uvicorn app.main:app --port 8000"
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm was not found. Install Node.js LTS from nodejs.org, then run this file again.
  goto :fail
)
if not exist "frontend\node_modules\" (
  echo       Installing website libraries - first time only, takes a few minutes ...
  pushd frontend
  call npm install
  popd
)
netstat -ano | findstr /r /c:":3000 .*LISTENING" >nul
if not errorlevel 1 (
  echo       Website is already running on port 3000.
) else (
  start "ePolicies website" /d "%CD%\frontend" cmd /k "npm run dev"
)

REM ---- 5. Open the browser when both are ready ----
echo [5/5] Waiting for the backend and the website ...
set /a TRIES=0
:wait
set /a TRIES+=1
curl -s -o nul http://localhost:8000/health >nul 2>nul
if errorlevel 1 goto :not_ready
curl -s -o nul http://localhost:3000 >nul 2>nul
if errorlevel 1 goto :not_ready
goto :open
:not_ready
if %TRIES% geq 90 (
  echo [WARN] Not ready after 3 minutes. Check the "ePolicies backend" and "ePolicies website" windows.
  goto :open
)
timeout /t 2 /nobreak >nul
goto :wait

:open
start "" http://localhost:3000
echo.
echo ePolicies is running at http://localhost:3000
echo Keep the "ePolicies backend" and "ePolicies website" windows open.
echo To stop everything, double-click stop-windows.bat
echo.
pause
exit /b 0

:fail
echo.
echo Start did not finish. Take a photo of this window and send it.
pause
exit /b 1
