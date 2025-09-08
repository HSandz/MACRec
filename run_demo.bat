@echo off
REM MACRec Demo Launcher
echo Starting MACRec demo...
echo Activating conda environment 'macrec'...

REM Try to activate conda environment
call conda activate macrec
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'macrec'
    echo Please make sure conda is installed and the environment exists
    pause
    exit /b 1
)

echo Environment activated successfully!
echo Starting Streamlit demo...
streamlit run web_demo.py
echo Demo finished.
pause