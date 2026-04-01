@echo off
setlocal
cd /d "%~dp0"
title MSA AI Studio Launcher

set "APP_DIR=%~dp0"
set "CACHE_ROOT=%LOCALAPPDATA%\MSA_AI_Studio"
set "VENV_DIR=%CACHE_ROOT%\venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=%APP_DIR%requirements.txt"
set "PY_LAUNCHER=py"
set "PY_VERSION_ARG="

call :detect_python
if errorlevel 1 goto :error

call :ensure_venv
if errorlevel 1 goto :error

call :ensure_requirements
if errorlevel 1 goto :error

echo ===================================================
echo   Starting MSA AI Studio
echo ===================================================
echo.
echo Please wait. Your browser will open automatically.
echo Close this window to stop the server.
echo.
"%PYTHON_EXE%" -m streamlit run "%APP_DIR%app.py" --server.fileWatcherType=none
pause
exit /b %errorlevel%

:detect_python
%PY_LAUNCHER% -3.14 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PY_VERSION_ARG=-3.14"
    exit /b 0
)

%PY_LAUNCHER% -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PY_VERSION_ARG=-3"
    exit /b 0
)

%PY_LAUNCHER% -c "import sys" >nul 2>&1
if not errorlevel 1 exit /b 0

echo ===================================================
echo   Python launcher not found
echo ===================================================
echo.
echo Install Python first, then run this launcher again.
echo Recommended: Python 3.14 or newer
echo.
pause
exit /b 1

:ensure_venv
if not exist "%CACHE_ROOT%" mkdir "%CACHE_ROOT%"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import sys" >nul 2>&1
    if not errorlevel 1 exit /b 0

    echo Existing virtual environment is invalid.
    echo Rebuilding project environment...
    rmdir /s /q "%VENV_DIR%"
)

echo ===================================================
echo   Creating project environment
echo ===================================================
echo.
echo The virtual environment will be created on this PC:
echo   %VENV_DIR%
echo This may take a few minutes on the first run.
echo.
%PY_LAUNCHER% %PY_VERSION_ARG% -m venv --without-scm-ignore-files "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Virtual environment was created, but python.exe is missing.
    pause
    exit /b 1
)

exit /b 0

:ensure_requirements
if not exist "%REQUIREMENTS_FILE%" exit /b 0

"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Restoring pip in the virtual environment...
    "%PYTHON_EXE%" -m ensurepip --default-pip
    if errorlevel 1 (
        echo.
        echo Failed to restore pip in the virtual environment.
        pause
        exit /b 1
    )
)

echo ===================================================
echo   Installing required packages
echo ===================================================
echo.
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo.
    echo Package installation failed.
    pause
    exit /b 1
)

exit /b 0

:error
echo.
echo Launcher stopped because setup did not complete successfully.
pause
exit /b 1
