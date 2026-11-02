@echo off
title Hybrid 1v3 Test  (10000 games)
set PYTHONPATH=
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if errorlevel 1 goto :badcd

echo ============================================================
echo  Hybrid challenger:  early turns = SeerX , rest = Seer
echo  Opponent:           Seer 1.0 (1v3)
echo  Gate:  turn ^<= 6  AND no riichi  AND not South-4-or-later
echo  Games: 2500 seeds x 4 = 10000   (about 3.5 hours)
echo ============================================================
echo.

rem seed-key pinned for pairing with the control run. Override with appended args.
"E:\ProgramData\miniconda3\envs\mortal\python.exe" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --seed-key 10380103433202888458 %*
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
