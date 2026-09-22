param([string]$Version="4.10.0")
if ($Version -ne "4.10.0") { throw "This script pins OpenCV 4.10.0 Windows binary" }
$Root=Split-Path $PSScriptRoot -Parent; $Third=Join-Path $Root "3rdparty"; New-Item -ItemType Directory -Force $Third | Out-Null
$Zip=Join-Path $Third "opencv-$Version-windows.exe"; $Url="https://github.com/opencv/opencv/releases/download/$Version/opencv-$Version-windows.exe"
Invoke-WebRequest $Url -OutFile $Zip; Start-Process $Zip -ArgumentList "/S","/D=$Third\opencv" -Wait; Remove-Item $Zip
Write-Host "Use OpenCV_DIR=$Third\opencv\build"
