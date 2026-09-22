#requires -Version 5.1
# 一键打包已编译的 Windows C++ 推理可执行程序及其依赖 DLL。
# 生成的部署包为自包含独立包，支持直接复制到 Windows 10 与 Windows 11 新机器上运行。
# 目标机器无需安装 Visual Studio 或配置复杂环境。
[CmdletBinding()]
param(
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$Target = "",
    [string]$Mode = "",
    [string]$OrtRoot = "",
    [string]$OpenCVRoot = "",
    [string]$CudaRoot = "",
    [string[]]$RuntimeDir = @(),
    [string]$Model = "",
    [switch]$NoModels,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CppDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ProjectRoot = (Resolve-Path (Join-Path $CppDir "..")).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot "dist\universal_instance_segmentation_windows-x64"
}

# 读取 CMakeCache.txt 中的键值辅助函数
function Read-CacheValue([string]$Cache, [string]$Key) {
    if (-not (Test-Path -LiteralPath $Cache -PathType Leaf)) { return "" }
    $line = Select-String -LiteralPath $Cache -Pattern ("^" + [regex]::Escape($Key) + ":[^=]*=(.*)$") |
        Select-Object -Last 1
    if ($null -eq $line) { return "" }
    return ([regex]::Match($line.Line, ("^" + [regex]::Escape($Key) + ":[^=]*=(.*)$"))).Groups[1].Value
}

# 自动发现 CMake 构建目录
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    foreach ($candidate in @("build", "build-windows", "build-cuda", "cpp\build")) {
        $path = Join-Path $ProjectRoot $candidate
        if (Test-Path (Join-Path $path "CMakeCache.txt") -PathType Leaf) {
            $BuildDir = $path
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) { throw "CMakeCache.txt not found; please specify -BuildDir DIR" }
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
$Cache = Join-Path $BuildDir "CMakeCache.txt"
if (-not (Test-Path -LiteralPath $Cache -PathType Leaf)) { throw "CMakeCache.txt not found: $Cache" }

# 已知的可执行程序名称列表
$KnownTargets = @(
    "detectron2_maskrcnn_infer",
    "detectron2_maskrcnn_infer_cuda",
    "ultralytics_yolo_infer",
    "ultralytics_yolo_infer_cuda",
    "rfdetr_infer",
    "rfdetr_infer_cuda"
)

# 搜索已编译生成的 .exe 文件
$FoundExes = @()
if (-not [string]::IsNullOrWhiteSpace($Target) -and ($Target -ne "all")) {
    $tName = $Target
    if (-not $tName.EndsWith(".exe", [System.StringComparison]::OrdinalIgnoreCase)) {
        $tName = $tName + ".exe"
    }
    $f = Get-ChildItem -LiteralPath $BuildDir -Filter $tName -File -Recurse | Select-Object -First 1
    if ($null -ne $f) { $FoundExes += $f }
} else {
    foreach ($kt in $KnownTargets) {
        $f = Get-ChildItem -LiteralPath $BuildDir -Filter ($kt + ".exe") -File -Recurse | Select-Object -First 1
        if ($null -ne $f) { $FoundExes += $f }
    }
}

if ($FoundExes.Count -eq 0) {
    # 兜底查找构建目录下所有 .exe
    $allExes = Get-ChildItem -LiteralPath $BuildDir -Filter "*.exe" -File -Recurse
    foreach ($candidate in $allExes) {
        if ($candidate.Name -notmatch "(compiler|test|check)") {
            $FoundExes += $candidate
        }
    }
}

if ($FoundExes.Count -eq 0) {
    throw "No inference executables found under ${BuildDir}. Please build the project first."
}

Write-Host "Found executables to package: $($FoundExes.Count)"
foreach ($e in $FoundExes) {
    Write-Host "  - $($e.Name)"
}

# 解析运行模式（CPU 或 CUDA）
if ([string]::IsNullOrWhiteSpace($Mode)) {
    $cachedGpu = Read-CacheValue $Cache "BUILD_GPU"
    if ($cachedGpu -match "^(ON|TRUE|1)$") {
        $Mode = "cuda"
    } else {
        $hasCuda = $false
        foreach ($e in $FoundExes) {
            if ($e.Name -match "_cuda") { $hasCuda = $true; break }
        }
        if ($hasCuda) { $Mode = "cuda" } else { $Mode = "cpu" }
    }
}

# 自动推导或回退 ONNX Runtime 根目录
if ([string]::IsNullOrWhiteSpace($OrtRoot)) {
    $OrtRoot = Read-CacheValue $Cache "ONNXRUNTIME_ROOT"
    if ($Mode -eq "cuda" -and [string]::IsNullOrWhiteSpace($OrtRoot)) {
        $OrtRoot = Read-CacheValue $Cache "ONNXRUNTIME_GPU_ROOT"
    }
}
if ([string]::IsNullOrWhiteSpace($OrtRoot) -or -not (Test-Path -LiteralPath $OrtRoot -PathType Container)) {
    $ortCands = Get-ChildItem -LiteralPath (Join-Path $CppDir "3rdparty") -Directory -Filter "onnxruntime*" -ErrorAction SilentlyContinue
    if ($ortCands.Count -gt 0) {
        $OrtRoot = $ortCands[0].FullName
    }
}

# 自动推导或回退 OpenCV 根目录
if ([string]::IsNullOrWhiteSpace($OpenCVRoot)) {
    $opencvDirVal = Read-CacheValue $Cache "OpenCV_DIR"
    if (-not [string]::IsNullOrWhiteSpace($opencvDirVal)) {
        $cand = Split-Path (Split-Path (Split-Path $opencvDirVal -Parent) -Parent) -Parent
        if (Test-Path -LiteralPath $cand -PathType Container) { $OpenCVRoot = $cand }
    }
}
if ([string]::IsNullOrWhiteSpace($OpenCVRoot) -or -not (Test-Path -LiteralPath $OpenCVRoot -PathType Container)) {
    $cand = Join-Path $CppDir "3rdparty\opencv"
    if (Test-Path -LiteralPath $cand -PathType Container) {
        $OpenCVRoot = $cand
    }
}

# 自动推导 CUDA 运行时库目录
if ([string]::IsNullOrWhiteSpace($CudaRoot) -and ($Mode -eq "cuda")) {
    $cachedCuda = Read-CacheValue $Cache "ONNXRUNTIME_GPU_RUNTIME_ROOT"
    if (-not [string]::IsNullOrWhiteSpace($cachedCuda) -and (Test-Path -LiteralPath $cachedCuda -PathType Container)) {
        $CudaRoot = $cachedCuda
    } elseif (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH) -and (Test-Path -LiteralPath $env:CUDA_PATH -PathType Container)) {
        $CudaRoot = $env:CUDA_PATH
    }
}

Write-Host "Packaging configuration:"
Write-Host "  Mode:        $Mode"
Write-Host "  ORT Root:    $OrtRoot"
Write-Host "  OpenCV Root: $OpenCVRoot"
Write-Host "  Output:      $OutputDir"

# 准备输出目录结构
if (Test-Path -LiteralPath $OutputDir) {
    if (-not $Force) { throw "Output directory exists; use -Force to overwrite: $OutputDir" }
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}
$null = New-Item -ItemType Directory -Path (Join-Path $OutputDir "bin") -Force
$null = New-Item -ItemType Directory -Path (Join-Path $OutputDir "models") -Force

# 拷贝已编译生成的可执行文件
foreach ($e in $FoundExes) {
    Copy-Item -LiteralPath $e.FullName -Destination $OutputDir
    Copy-Item -LiteralPath $e.FullName -Destination (Join-Path $OutputDir "bin")
}

# 递归收集指定目录中的所有动态链接库 (.dll) 并拷贝至部署包根目录与 bin\ 目录
function Copy-Dlls([string]$Root) {
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) { return }
    $destRoot = $OutputDir
    $destBin = Join-Path $OutputDir "bin"
    Get-ChildItem -LiteralPath $Root -Filter "*.dll" -File -Recurse | ForEach-Object {
        $t1 = Join-Path $destRoot $_.Name
        if (-not (Test-Path -LiteralPath $t1 -PathType Leaf)) {
            Copy-Item -LiteralPath $_.FullName -Destination $t1
        }
        $t2 = Join-Path $destBin $_.Name
        if (-not (Test-Path -LiteralPath $t2 -PathType Leaf)) {
            Copy-Item -LiteralPath $_.FullName -Destination $t2
        }
    }
}

# 收集可执行文件同级目录生成的依赖 DLL
foreach ($e in $FoundExes) {
    Copy-Dlls (Split-Path -Parent $e.FullName)
}
if (-not [string]::IsNullOrWhiteSpace($OrtRoot)) { Copy-Dlls $OrtRoot }
if (-not [string]::IsNullOrWhiteSpace($OpenCVRoot)) { Copy-Dlls $OpenCVRoot }
if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) { Copy-Dlls $CudaRoot }
foreach ($dir in $RuntimeDir) { Copy-Dlls $dir }

$runtimeDlls = @(Get-ChildItem -LiteralPath (Join-Path $OutputDir "bin") -Filter "*.dll" -File)
Write-Host "Collected runtime DLLs: $($runtimeDlls.Count)"

# 拷贝模型文件（若存在）
if (-not $NoModels) {
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
        if (Test-Path -LiteralPath $Model -PathType Leaf) {
            Copy-Item -LiteralPath $Model -Destination (Join-Path $OutputDir "models")
            Write-Host "Bundled model: $Model"
        } else {
            Write-Warning "Model file not found: $Model"
        }
    } else {
        $benchModel = Join-Path $ProjectRoot "benchmark_model\model.onnx"
        if (Test-Path -LiteralPath $benchModel -PathType Leaf) {
            Copy-Item -LiteralPath $benchModel -Destination (Join-Path $OutputDir "models")
            Write-Host "Bundled default benchmark model."
        }
    }
}

# 拷贝类别定义文件（若存在）
$classes = Join-Path $ProjectRoot "classes.names"
if (Test-Path -LiteralPath $classes -PathType Leaf) {
    Copy-Item -LiteralPath $classes -Destination $OutputDir
}

# 拷贝配套后处理与真值评估 Python 脚本
foreach ($script in @("compare_json.py", "convert_to_labelme.py")) {
    $scriptPath = Join-Path (Join-Path $ProjectRoot "scripts") $script
    if (Test-Path -LiteralPath $scriptPath -PathType Leaf) {
        Copy-Item -LiteralPath $scriptPath -Destination $OutputDir
    }
}

# 生成 Windows PowerShell 启动脚本 run.ps1
$runPs1 = @'
param(
    [Parameter(Position=0)][string]$Target = "",
    [Parameter(Position=1,ValueFromRemainingArguments=$true)][string[]]$Arguments
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $PSScriptRoot).Path
$env:PATH = "$Root;$Root\bin;$env:PATH"
Set-Location $Root

function Show-Usage {
    Write-Host "================================================================================"
    Write-Host "Universal Instance Segmentation - Windows C++ Runtime Launcher"
    Write-Host "================================================================================"
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 <target_executable> [arguments...]"
    Write-Host ""
    Write-Host "Available executables in this package:"
    Get-ChildItem -LiteralPath $Root -Filter "*.exe" -File | ForEach-Object {
        Write-Host "  - $($_.BaseName)"
    }
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 detectron2_maskrcnn_infer --model models\model.onnx --input images --output results\d2 --classes class1,class2"
    Write-Host "  .\run.ps1 ultralytics_yolo_infer --model models\yolo.onnx --input images --output results\yolo --classes class1,class2"
    Write-Host "  .\run.ps1 rfdetr_infer --model models\rfdetr.onnx --input images --output results\rfdetr --classes class1,class2"
}

if ([string]::IsNullOrWhiteSpace($Target) -or ($Target -eq "-h") -or ($Target -eq "--help")) {
    Show-Usage
    exit 0
}

$exeName = $Target
if (-not $exeName.EndsWith(".exe", [System.StringComparison]::OrdinalIgnoreCase)) {
    $exeName = $exeName + ".exe"
}

$exe = Join-Path $Root $exeName
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    $exe = Join-Path $Root ("bin\" + $exeName)
}
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    Write-Error "Executable not found in package: $exeName"
    Show-Usage
    exit 1
}

& $exe @Arguments
exit $LASTEXITCODE
'@
Set-Content -LiteralPath (Join-Path $OutputDir "run.ps1") -Value $runPs1 -Encoding UTF8

# 生成 Windows CMD 启动脚本 run.bat
$runBat = @'
@echo off
setlocal
set "ROOT=%~dp0"
set "PATH=%ROOT%;%ROOT%bin;%PATH%"

if "%~1"=="" goto usage
if "%~1"=="-h" goto usage
if "%~1"=="--help" goto usage

set "TARGET=%~1"
shift

set "EXE=%ROOT%%TARGET%"
if exist "%EXE%.exe" set "EXE=%EXE%.exe"
if exist "%EXE%" goto execute

set "EXE=%ROOT%bin\%TARGET%"
if exist "%EXE%.exe" set "EXE=%EXE%.exe"
if exist "%EXE%" goto execute

echo [ERROR] Executable "%TARGET%" not found in package.
goto usage

:execute
"%EXE%" %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:usage
echo ================================================================================
echo Universal Instance Segmentation - Windows C++ Runtime Launcher
echo ================================================================================
echo Usage:
echo   run.bat ^<target_executable^> [options...]
echo.
echo Examples:
echo   run.bat detectron2_maskrcnn_infer --model models\model.onnx --input images --output results\d2 --classes class1,class2
echo   run.bat ultralytics_yolo_infer --model models\yolo.onnx --input images --output results\yolo --classes class1,class2
echo   run.bat rfdetr_infer --model models\rfdetr.onnx --input images --output results\rfdetr --classes class1,class2
echo ================================================================================
exit /b 0
'@
Set-Content -LiteralPath (Join-Path $OutputDir "run.bat") -Value $runBat -Encoding ASCII

# 生成部署包详细说明文档
$readme = @"
================================================================================
通用实例分割 (Universal Instance Segmentation) C++ Windows 独立部署包
================================================================================

本部署包包含编译完成的 C++ 高性能推理可执行程序、ONNX Runtime 运行时 DLL、OpenCV 核心 DLL、
CUDA/cuDNN 运行时库（若包含 GPU 目标）以及一键运行批处理脚本与评估工具。
支持在 Windows 10 与 Windows 11 系统上解压即用，无需配置 Visual Studio 或编译环境。

系统要求：
  - Windows 10 或 Windows 11 (64-bit)
  - 目标机器已安装 Visual C++ 可再发行软件包 (MSVC Redistributable x64)
  - GPU 模式要求显卡驱动版本兼容所包含的 CUDA/cuDNN 版本

--------------------------------------------------------------------------------
一、快速运行指南
--------------------------------------------------------------------------------

部署包根目录已放置所有依赖的 DLL 动态链接库。
您可以通过 run.bat（CMD 终端）、run.ps1（PowerShell 终端）调用，或直接双击/在命令行运行 .exe：

1. Detectron2 Mask R-CNN 推理：
   .\run.bat detectron2_maskrcnn_infer ^
     --model models\model.onnx ^
     --input C:\path\to\images ^
     --output results\pred_d2 ^
     --classes class1,class2,class3 ^
     --score-threshold 0.50 ^
     --class-conf class1=0.60,class2=0.50 ^
     --class-iou class1=0.50,class2=0.50 ^
     --device cpu

2. Ultralytics YOLO-seg 推理：
   .\run.bat ultralytics_yolo_infer ^
     --model models\yolo.onnx ^
     --input C:\path\to\images ^
     --output results\pred_yolo ^
     --classes class1,class2,class3 ^
     --score-threshold 0.25 ^
     --iou-threshold 0.45 ^
     --device cpu

3. RF-DETR 推理：
   .\run.bat rfdetr_infer ^
     --model models\rfdetr.onnx ^
     --input C:\path\to\images ^
     --output results\pred_rfdetr ^
     --classes class1,class2,class3 ^
     --score-threshold 0.40 ^
     --iou-threshold 0.50 ^
     --device cpu

4. GPU 加速模式（若包含 *_cuda 可执行程序）：
   将上述目标名称替换为 *_infer_cuda，并将 --device 参数设为 cuda。

--------------------------------------------------------------------------------
二、评估与格式转换脚本
--------------------------------------------------------------------------------
- compare_json.py：将推理输出的 JSON 预测与 LabelMe 真值标注对标，计算 Precision/Recall/F1 并生成报告。
- convert_to_labelme.py：将 C++ 推理 JSON 一键转换为标准 LabelMe 多边形格式，便于人工复核修正。
"@
Set-Content -LiteralPath (Join-Path $OutputDir "README.txt") -Value $readme -Encoding UTF8

# 生成清单文件
$manifest = @"
packaged_utc=$([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"))
mode=$Mode
build_dir=$BuildDir
ort_root=$OrtRoot
opencv_root=$OpenCVRoot
executables=$($FoundExes.Name -join ' ')
"@
Set-Content -LiteralPath (Join-Path $OutputDir "manifest.txt") -Value $manifest -Encoding UTF8

Write-Host "================================================================================"
Write-Host "Windows package created successfully: $OutputDir"
Write-Host "================================================================================"
