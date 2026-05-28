@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%..\.."

echo =========================================
echo   GraphRAG Benchmark Runner (CMD)
echo =========================================
echo.

if not exist "%PROJECT_DIR%\app.py" (
  echo [ERROR] Cannot find app.py under project root:
  echo %PROJECT_DIR%
  pause
  exit /b 1
)

set /p PYTHON_EXE=Enter python path (default: D:\Anaconda3\envs\jsjsj\python.exe): 
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=D:\Anaconda3\envs\jsjsj\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

echo.
set /p SILICONFLOW_API_KEY=Enter SILICONFLOW_API_KEY:
set /p SILICONFLOW_MODEL=Enter SILICONFLOW_MODEL (default: Qwen/Qwen3-8B):
if "%SILICONFLOW_MODEL%"=="" set "SILICONFLOW_MODEL=Qwen/Qwen3-8B"

set /p NEO4J_URI=Enter NEO4J_URI:
set /p NEO4J_DATABASE=Enter NEO4J_DATABASE:
set /p NEO4J_USER=Enter NEO4J_USER:
set /p NEO4J_PASSWORD=Enter NEO4J_PASSWORD:

if "%SILICONFLOW_API_KEY%"=="" (
  echo [ERROR] SILICONFLOW_API_KEY is required.
  pause
  exit /b 1
)

if "%NEO4J_URI%"=="" (
  echo [ERROR] NEO4J_URI is required.
  pause
  exit /b 1
)

if "%NEO4J_DATABASE%"=="" (
  echo [ERROR] NEO4J_DATABASE is required.
  pause
  exit /b 1
)

if "%NEO4J_USER%"=="" (
  echo [ERROR] NEO4J_USER is required.
  pause
  exit /b 1
)

if "%NEO4J_PASSWORD%"=="" (
  echo [ERROR] NEO4J_PASSWORD is required.
  pause
  exit /b 1
)

set "SILICONFLOW_API_KEY=%SILICONFLOW_API_KEY%"
set "SILICONFLOW_MODEL=%SILICONFLOW_MODEL%"
set "NEO4J_URI=%NEO4J_URI%"
set "NEO4J_DATABASE=%NEO4J_DATABASE%"
set "NEO4J_USER=%NEO4J_USER%"
set "NEO4J_PASSWORD=%NEO4J_PASSWORD%"

echo.
echo [1/2] Running experiment...
"%PYTHON_EXE%" "%SCRIPT_DIR%run_experiment.py"
if errorlevel 1 (
  echo [ERROR] run_experiment.py failed.
  pause
  exit /b 1
)

echo.
echo [2/2] Aggregating scores...
"%PYTHON_EXE%" "%SCRIPT_DIR%aggregate_manual_scores.py" --input "%SCRIPT_DIR%manual_scores_template.csv"
if errorlevel 1 (
  echo [ERROR] aggregate_manual_scores.py failed.
  pause
  exit /b 1
)

echo.
echo Done. Check:
echo - %SCRIPT_DIR%manual_scores_template.csv
echo - %SCRIPT_DIR%outputs
pause
exit /b 0

