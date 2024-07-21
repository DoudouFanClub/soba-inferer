@echo off

:: Launch 3 python3 inferer servers
taskkill /IM python.exe /F

:: Wait a moment to ensure both Services start
timeout /t 5 /nobreak

:: Wait for the user to press a key before closing the script
pause