@echo off
call "E:\ProgramData\miniconda3\Scripts\activate.bat"
call conda activate mortal
cd /d "E:\Program Files\JetBrains\IdeaProjects\Mortal\mortal"
python mortal.py 1%
