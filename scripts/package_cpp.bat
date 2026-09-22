@echo off
rem 一键打包 C++ 推理可执行程序部署包（Windows x64）
setlocal
set "ROOT_DIR=%~dp0.."
echo Starting C++ inference executable packaging for Windows...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT_DIR%\cpp\scripts\package_windows.ps1" %*
exit /b %ERRORLEVEL%
