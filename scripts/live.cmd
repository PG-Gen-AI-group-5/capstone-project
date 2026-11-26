@echo off
rem The live suggester for the Windows Task Scheduler: runs "python -m src.live <command>" from the project
rem folder with the project's virtual environment, appending the output to data\live\logs\live.log.
cd /d "%~dp0.."
if not exist "data\live\logs" mkdir "data\live\logs"
".venv\Scripts\python.exe" -m src.live %* >> "data\live\logs\live.log" 2>&1
