@echo off
echo 停止 PokieTicker 服务...
echo.

echo 停止后端 (端口 8000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

echo 停止前端 (端口 5173)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

echo 停止前端 (端口 7777)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7777 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

echo.
echo 所有服务已停止
pause
