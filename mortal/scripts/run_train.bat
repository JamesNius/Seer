@echo off
call E:\ProgramData\miniconda3\Scripts\activate.bat mortal
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
if exist file_index.pth del /f file_index.pth
python train.py
pause
