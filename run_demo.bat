@echo off
setlocal
cd /d "%~dp0"

echo === ISQTrafic demo ===
where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do if exist "%%D\python.exe" set "PYTHON=%%D\python.exe"
if not defined PYTHON (
    echo Python 3 is not installed or not available in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
if errorlevel 1 goto :error

echo Starting demonstration...
.venv\Scripts\python.exe demo.py %*
if errorlevel 1 goto :error

echo.
echo Demo finished. Results are in the artifacts folder.
pause
exit /b 0

:error
echo.
echo Demo failed. Check the error above.
pause
exit /b 1
