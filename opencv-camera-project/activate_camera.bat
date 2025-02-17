@echo off
echo Starting Drone Security Camera...
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0srcs\components\camera\camera_test.py"
pause
