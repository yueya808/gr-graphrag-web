@echo off
setlocal
chcp 65001 >nul

rem Copy this file to: run_benchmark_config.cmd
rem Then fill your own values below and double-click to run.

set "PYTHON_EXE=D:\Anaconda3\envs\jsjsj\python.exe"
set "SILICONFLOW_API_KEY=sk-chbavturojicqcgvxuidraaizsxtyysxblfmsapiwmepebcn"
set "SILICONFLOW_MODEL=Qwen/Qwen3-8B"

set "NEO4J_URI=neo4j+s://01a0e5bf.databases.neo4j.io"
set "NEO4J_DATABASE=01a0e5bf"
set "NEO4J_USER=01a0e5bf"
set "NEO4J_PASSWORD=JZ920NcZWJmZe3Cc3WjYNouz7hOvk1Qxr8XfPSPRjXU"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

echo [1/2] Running experiment...
"%PYTHON_EXE%" "%SCRIPT_DIR%run_experiment.py"
if errorlevel 1 (
  echo [ERROR] run_experiment.py failed.
  popd
  pause
  exit /b 1
)

echo [2/2] Aggregating scores...
"%PYTHON_EXE%" "%SCRIPT_DIR%aggregate_manual_scores.py" --input "%SCRIPT_DIR%manual_scores_template.csv"
if errorlevel 1 (
  echo [ERROR] aggregate_manual_scores.py failed.
  popd
  pause
  exit /b 1
)

echo Done.
popd
pause
exit /b 0
