@echo off
setlocal

:: Automatically set working directory to where this batch file is located
cd /d "%~dp0"

:: Ensure Python can find the taggui module
set PYTHONPATH=.

echo ==================================================
echo Starting TagGUI
echo ==================================================
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv\
    echo.
    echo Please create the virtual environment first:
    echo   python -m venv venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Check if run_gui.py exists
if not exist "taggui\run_gui.py" (
    echo ERROR: run_gui.py not found at taggui\run_gui.py
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Launching TagGUI...
echo Close the GUI window to exit
echo ==================================================
echo.

:: Run the GUI application and wait for it to exit
python taggui\run_gui.py

echo.
echo ==================================================
echo TagGUI closed. Deactivating virtual environment...
echo ==================================================

:: Deactivate virtual environment
call deactivate

echo Done.
timeout /t 1 /nobreak >nul

endlocal