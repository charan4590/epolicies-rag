@echo off
REM ============================================================
REM  ePolicies - one-click stop for Windows
REM  Stops the website (port 3000), the backend (port 8000) and your own PostgreSQL database.
REM ============================================================
setlocal
cd /d "%~dp0"

REM ---- Settings: keep these the same as in start-windows.bat ----
set "PG_BIN=C:\Program Files\PostgreSQL\16\bin"
set "PG_DATA=%USERPROFILE%\epolicies-pgdata"

echo.
echo === Stopping ePolicies ===

echo [1/3] Website
call :stop_port 3000
taskkill /fi "WINDOWTITLE eq ePolicies website*" /t /f >nul 2>nul

echo [2/3] Backend
call :stop_port 8000
taskkill /fi "WINDOWTITLE eq ePolicies backend*" /t /f >nul 2>nul

echo [3/3] Database
if not exist "%PG_DATA%\PG_VERSION" (
  echo       No database folder at %PG_DATA% - skipping.
  goto :done
)
"%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" status >nul 2>nul
if errorlevel 1 (
  echo       Not running.
  goto :done
)
"%PG_BIN%\pg_ctl.exe" -D "%PG_DATA%" stop -m fast >nul
if errorlevel 1 (
  echo [WARN] PostgreSQL did not stop cleanly.
) else (
  echo       Stopped.
)

:done
echo.
echo Done.
pause
exit /b 0

REM Stops whatever is listening on the given port.
:stop_port
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%1 .*LISTENING"') do (
  taskkill /pid %%P /t /f >nul 2>nul
  set "FOUND=1"
)
if defined FOUND (echo       Stopped port %1.) else (echo       Not running.)
exit /b 0
