@echo off
REM Script wrapper to run pipeline in slither-env

echo Activating slither-env...
call conda activate slither-env

echo Running HF pipeline...
python scripts\run_hf_pipeline.py

echo.
echo Pipeline finished!
pause
