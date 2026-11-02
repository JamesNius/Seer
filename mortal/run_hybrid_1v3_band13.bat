@echo off
title Hybrid 1v3 BAND13 (early + turn^>=13 window)
set PYTHONPATH=
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if errorlevel 1 goto :badcd

echo ============================================================
echo  v5 = v2.1 + late window (turn GE 13, before South-4 only)
echo  Opponent:  Seer 1.0 (1v3)
echo  Games:     2500 seeds x 4 = 10000   (about 3.5 hours)
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --gate-mode early --late-from 13 --base v21 --seed-key 10380103433202888458 --log-dir 1v3_logs_hybrid_band13 %*
set RC=%errorlevel%

echo.
echo ------------------------------------------------------------
echo  finished. exit code = %RC%
echo ------------------------------------------------------------
pause
exit /b %RC%

:badcd
echo [ERROR] cannot enter directory:
echo   E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal
pause
exit /b 1
