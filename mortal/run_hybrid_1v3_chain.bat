@echo off
title Hybrid 1v3 CHAIN  (wide 10k then early 10k)
set PYTHONPATH=
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if errorlevel 1 goto :badcd

set PY=E:\ProgramData\miniconda3\envs\mortal\python.exe
set KEY=10380103433202888458

echo ==============================================================
echo  STEP 1/2 : WIDE   (SeerX covers everything before South-4)
echo            10000 games, about 3.5 hours
echo ==============================================================
echo.
"%PY%" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --gate-mode wide --seed-key %KEY% --log-dir 1v3_logs_hybrid_wide
set RC1=%errorlevel%
echo.
if not "%RC1%"=="0" goto :fail1

echo ==============================================================
echo  STEP 2/2 : EARLY  (v3: SeerX only in first N turns,
echo            South-4 needs no meld/kan)
echo            10000 games, about 3.5 hours
echo ==============================================================
echo.
"%PY%" -u one_vs_three_hybrid.py --seeds 2500 --chunk 125 --gate-turn 6 --gate-mode early --seed-key %KEY% --log-dir 1v3_logs_hybrid_early
set RC2=%errorlevel%
echo.
if not "%RC2%"=="0" goto :fail2

echo ==============================================================
echo  ALL DONE   wide rc=0   early rc=0
echo  summaries:
echo    1v3_logs_hybrid_wide\hybrid_summary.txt
echo    1v3_logs_hybrid_early\hybrid_summary.txt
echo ==============================================================
pause
exit /b 0

:fail1
echo.
echo [ABORT] STEP 1 (wide) failed with exit code %RC1%
echo         STEP 2 not started.
pause
exit /b %RC1%

:fail2
echo.
echo [WARN] STEP 1 (wide) ok, but STEP 2 (early) failed with exit code %RC2%
pause
exit /b %RC2%

:badcd
echo [ERROR] cannot enter directory:
echo   E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal
pause
exit /b 1
