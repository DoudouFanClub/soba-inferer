@echo off

:: Launch 3 python inferer servers
:: first argument: localhost for llm
:: second argument: port for self (listening from GOlang server)
:: third arugument: port for sending llm data
start python3 tcp_core.py "127.0.0.1" 7060 11434
start python3 tcp_core.py "127.0.0.1" 7061 11435
start python3 tcp_core.py "127.0.0.1" 7062 11436

:: Wait a moment to ensure both Services start
timeout /t 5 /nobreak

:: Wait for the user to press a key before closing the script
pause