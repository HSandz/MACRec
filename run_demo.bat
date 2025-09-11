@echo off
REM MACRec Demo Launcher
REM Try to activate conda environment
call conda activate macrec
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'macrec'
    pause
    exit /b 1
)
streamlit run web_demo.py
pause