param (
    [switch]$Clean,
    [switch]$UseNinja,
    [switch]$EnableCuda,
    [string]$Configuration = "Release",
    [string]$GGMLDir = ""
)

# 1. Load Visual Studio environment (for cl/link and Ninja+MSVC cases)
function Import-VSEnv {
    Write-Host "Loading Visual Studio environment..."
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return }

    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsroot) { return }

    $vcvars = Join-Path $vsroot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { return }

    $envDump = cmd /c "call `"$vcvars`" > nul && set PATH && set INCLUDE && set LIB"
    $envDump | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}
Import-VSEnv

# 2. Paths
$ScriptDir = $PSScriptRoot
$BuildDir = Join-Path $ScriptDir "build"

if (-not $GGMLDir) {
    $GGMLDir = Join-Path $ScriptDir "ggml"
}

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Path $BuildDir -Recurse -Force
}

if (!(Test-Path $BuildDir)) {
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
}

if (!(Test-Path (Join-Path $GGMLDir "CMakeLists.txt"))) {
    Write-Host "GGML directory is missing or invalid: $GGMLDir" -ForegroundColor Red
    Write-Host "Run: git -C $ScriptDir submodule update --init --recursive" -ForegroundColor Yellow
    exit 1
}

# 3. Generator
$GeneratorArgs = @()
if ($UseNinja) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        Write-Host "Using Ninja generator..."
        $GeneratorArgs += @("-G", "Ninja")
    } else {
        Write-Host "Ninja not found; falling back to Visual Studio generator."
        $GeneratorArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
    }
} else {
    Write-Host "Using Visual Studio 2022 generator..."
    $GeneratorArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
}

# 4. Configure
Set-Location $BuildDir
Write-Host "Configuring CMake..."

$CudaFlag = if ($EnableCuda) { "ON" } else { "OFF" }

$configureArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildDir
) + $GeneratorArgs + @(
    "-DCMAKE_CXX_STANDARD=17",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DQWEN3_TTS_COREML=OFF",
    "-DQWEN3_TTS_EMBED_GGML=ON",
    "-DQWEN3_TTS_GGML_DIR=$GGMLDir",
    "-DGGML_CUDA=$CudaFlag"
)

& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# 5. Build
Write-Host "Building qwen3-tts-cli..."
$buildArgs = @("--build", $BuildDir, "--target", "qwen3-tts-cli", "--config", $Configuration, "--parallel")
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# 6. Locate artifact
$exeVS = Join-Path $BuildDir "$Configuration\qwen3-tts-cli.exe"
$exeFlat = Join-Path $BuildDir "qwen3-tts-cli.exe"
$exeBin = Join-Path $BuildDir "bin\$Configuration\qwen3-tts-cli.exe"

# 7. Copy runtime DLLs next to executable (Windows loader convenience)
$dllSourceDir = Join-Path $BuildDir "bin\$Configuration"
$exeDir = ""
if (Test-Path $exeVS) {
    $exeDir = Split-Path -Parent $exeVS
} elseif (Test-Path $exeBin) {
    $exeDir = Split-Path -Parent $exeBin
} elseif (Test-Path $exeFlat) {
    $exeDir = Split-Path -Parent $exeFlat
}

if ($exeDir -and (Test-Path $dllSourceDir)) {
    $dlls = Get-ChildItem -Path $dllSourceDir -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($dlls) {
        Write-Host "Copying runtime DLLs to $exeDir ..."
        Copy-Item -Path (Join-Path $dllSourceDir "*.dll") -Destination $exeDir -Force
    }
}

Write-Host "Build success!" -ForegroundColor Green
if (Test-Path $exeVS) {
    Write-Host "Executable: $exeVS"
} elseif (Test-Path $exeBin) {
    Write-Host "Executable: $exeBin"
} elseif (Test-Path $exeFlat) {
    Write-Host "Executable: $exeFlat"
} else {
    Write-Host "Executable built, but path was not auto-detected. Check build output tree." -ForegroundColor Yellow
}
