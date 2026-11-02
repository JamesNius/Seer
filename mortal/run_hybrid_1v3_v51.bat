@echo off
title Hybrid 1v3 V5.1  (v2.1 base + late window 13)
set PYTHONPATH=
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if errorlevel 1 goto :badcd

echo ============================================================
echo  V5.1 = v2.1 base + late window (turn GE 13, before South-4)
echo  Opponent:  Seer 1.0 (1v3)
echo  Games:     2500 seeds x 4 = 10000   (about 3.5 hours)
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --gate-mode early --late-from 13 --base v21 --seed-key 10380103433202888458 --log-dir 1v3_logs_hybrid_v51 %*
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
