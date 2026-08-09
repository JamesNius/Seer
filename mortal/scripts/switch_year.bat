@echo off
if "%1"=="" (
    echo Usage: switch_year.bat ^<year^>  e.g. switch_year.bat 2021
    exit /b 1
)

set YEAR=%1
set CHECKPOINT=checkpoints\mortal.pth

echo ============================================
echo   Switching to year %YEAR%
echo ============================================

echo [1/3] Updating config.toml globs to point to year %YEAR%...
python -c "import toml; c=toml.load('config.toml'); c['dataset']['globs']=[f'E:/天凤训练数据/%YEAR%/**/*.mjson']; toml.dump(c, open('config.toml','w'))" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to update config
    exit /b 1
)
echo   done

echo [2/3] Clearing resume info from checkpoint...
python -c "import torch; s=torch.load('%CHECKPOINT%', weights_only=True, map_location='cpu'); s.pop('file_list',None); s.pop('file_list_consumed',None); torch.save(s,'%CHECKPOINT%'); print('  done')" 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Failed to clear checkpoint
    exit /b 1
)

if exist file_index.pth (
    del file_index.pth
    echo   deleted file_index.pth
)

echo [3/3] Done. Ready to train on year %YEAR%.
echo.
echo   Run: python train.py
echo ============================================
