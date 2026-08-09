@echo off
call "E:\ProgramData\miniconda3\Scripts\activate.bat"
call conda activate mortal
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"

echo ============================================================
echo   Local Mortal Review
echo ============================================================
echo.

echo [1/2] Converting to tenhou format...
python review\reverse_mjai_review.py json\test.json --tenhou -o json\tenhou.json
if %errorlevel% neq 0 (
    echo ERROR: Conversion failed!
    pause
    exit /b 1
)

set /p PID=<json\tenhou.pid
echo   Player ID: %PID%

echo.
echo [2/2] Running Mortal analysis...
"E:\Program Files\JetBrains\IdeaProjects\mjai-reviewer\target\release\mjai-reviewer.exe" -e mortal -i json\tenhou.json -a %PID% --mortal-exe scripts\mortal.bat --mortal-cfg config.toml --lang zh --show-rating --without-log-viewer -o json\review.html
if %errorlevel% neq 0 (
    echo ERROR: Analysis failed!
    pause
    exit /b 1
)

echo.
echo Opening report...
start "" "json\review.html"

echo Done!
pause
