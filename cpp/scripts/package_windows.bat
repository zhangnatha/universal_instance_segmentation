@echo off
rem 一键打包 Windows C++ 推理可执行程序与运行时 DLL 批处理脚本
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%package_windows.ps1" %*
exit /b %ERRORLEVEL%
