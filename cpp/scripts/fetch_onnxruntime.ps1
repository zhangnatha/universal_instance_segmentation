param([string]$Version="1.20.0", [string]$Sha256="")
if ($Version -ne "1.20.0") { throw "This script pins ORT 1.20.0" }
$Root=Split-Path $PSScriptRoot -Parent; $Out=Join-Path $Root "3rdparty"; New-Item -ItemType Directory -Force $Out | Out-Null
$Dir=Join-Path $Out "onnxruntime-win-x64-$Version"; $Zip=Join-Path $Out "ort-$Version.zip"
if (!(Test-Path (Join-Path $Dir "include\onnxruntime_cxx_api.h"))) { Invoke-WebRequest "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip" -OutFile $Zip; if($Sha256 -ne "" -and (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower() -ne $Sha256.ToLower()){Remove-Item $Zip;throw "SHA256 mismatch"}; Expand-Archive $Zip -DestinationPath $Out -Force; Remove-Item $Zip }
Write-Host "ORT_ROOT=$Dir"