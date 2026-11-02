@echo off
title Hybrid 1v3 WIDE Test  (10000 games)
set PYTHONPATH=
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if errorlevel 1 goto :badcd

echo ============================================================
echo  Hybrid WIDE challenger:  SeerX covers everything before South-4
echo    before South-4 : no opponent riichi -> SeerX (whole hand)
echo    South-4 or later: first N turns, no opp riichi, no meld -> SeerX
echo    otherwise        : Seer
echo  Opponent:           Seer 1.0 (1v3)
echo  Games: 2500 seeds x 4 = 10000   (about 3.5 hours)
echo ============================================================
echo.

"E:\ProgramData\miniconda3\envs\mortal\python.exe" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --gate-mode wide --seed-key 10380103433202888458 %*
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
