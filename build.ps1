[CmdletBinding()]
param (
    [switch]$Clean,
    [switch]$UseNinja,
    [switch]$EnableCuda,
    [switch]$EnableCudaGraphs,
    [switch]$BuildAll,
    [switch]$RunSmokeTest,
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$BuildDir = "",
    [string]$GGMLDir = "",
    [string]$Target = "qwen3-tts-cli",
    [string]$ModelDir = "models",
    [string]$SmokeText = "This is a test synthesis from build.ps1."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-VSEnv {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${Env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio Build Tools with C++ workload."
    }

    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ([string]::IsNullOrWhiteSpace($vsroot)) {
        throw "Visual Studio C++ toolchain not found."
    }

    $vcvars = Join-Path $vsroot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat not found at $vcvars"
    }

    Write-Host "Importing Visual Studio environment from: $vcvars"
    $envDump = cmd /s /c "`"$vcvars`" > nul && set"
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Find-FirstExisting([string[]]$paths) {
    foreach ($path in $paths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }
    return $null
}

function Get-ConfiguredGenerator([string]$buildDir) {
    $cachePath = Join-Path $buildDir "CMakeCache.txt"
    if (-not (Test-Path $cachePath)) {
        return $null
    }

    $line = Select-String -Path $cachePath -Pattern "^CMAKE_GENERATOR:INTERNAL=(.+)$" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($line) {
        return $line.Matches[0].Groups[1].Value
    }
    return $null
}

$ScriptDir = $PSScriptRoot
$resolvedBuildDir = if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    Join-Path $ScriptDir "build"
} elseif ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir
} else {
    Join-Path $ScriptDir $BuildDir
}

$resolvedGGMLDir = if ([string]::IsNullOrWhiteSpace($GGMLDir)) {
    Join-Path $ScriptDir "ggml"
} elseif ([System.IO.Path]::IsPathRooted($GGMLDir)) {
    $GGMLDir
} else {
    Join-Path $ScriptDir $GGMLDir
}

if (-not (Test-Path (Join-Path $resolvedGGMLDir "CMakeLists.txt"))) {
    throw "GGML directory is missing or invalid: $resolvedGGMLDir"
}

Import-VSEnv

if ($Clean -and (Test-Path $resolvedBuildDir)) {
    Write-Host "Cleaning build directory: $resolvedBuildDir"
    Remove-Item -Path $resolvedBuildDir -Recurse -Force
}
New-Item -Path $resolvedBuildDir -ItemType Directory -Force | Out-Null

$GeneratorArgs = @()
$isNinja = $false
if ($UseNinja) {
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        throw "Ninja was requested with -UseNinja but was not found on PATH."
    }
    $GeneratorArgs += @("-G", "Ninja")
    $isNinja = $true
    Write-Host "Generator: Ninja"
} else {
    $GeneratorArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
    Write-Host "Generator: Visual Studio 17 2022"
}

$configuredGenerator = Get-ConfiguredGenerator -buildDir $resolvedBuildDir
$expectedGenerator = if ($isNinja) { "Ninja" } else { "Visual Studio 17 2022" }
if ($configuredGenerator -and $configuredGenerator -ne $expectedGenerator) {
    Write-Host "Build directory generator mismatch detected: '$configuredGenerator' -> '$expectedGenerator'" -ForegroundColor Yellow
    Write-Host "Cleaning $resolvedBuildDir to keep using the same build directory..."
    Remove-Item -Path $resolvedBuildDir -Recurse -Force
    New-Item -Path $resolvedBuildDir -ItemType Directory -Force | Out-Null
}

$cudaFlag = if ($EnableCuda) { "ON" } else { "OFF" }
$cudaGraphsFlag = if ($EnableCudaGraphs -or $EnableCuda) { "ON" } else { "OFF" }

$configureArgs = @(
    "-S", $ScriptDir,
    "-B", $resolvedBuildDir
) + $GeneratorArgs + @(
    "-DCMAKE_CXX_STANDARD=17",
    "-DQWEN3_TTS_COREML=OFF",
    "-DQWEN3_TTS_EMBED_GGML=ON",
    "-DQWEN3_TTS_GGML_DIR=$resolvedGGMLDir",
    "-DQWEN3_TTS_BUILD_SHARED=OFF",
    "-DQWEN3_TTS_CUDA=$cudaFlag",
    "-DGGML_CUDA=$cudaFlag",
    "-DGGML_CUDA_GRAPHS=$cudaGraphsFlag"
)

if ($isNinja) {
    $configureArgs += "-DCMAKE_BUILD_TYPE=$Configuration"
}

Write-Host "Configuring CMake in: $resolvedBuildDir"
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed."
}

$resolvedTarget = if ($BuildAll) { "ALL_BUILD" } else { $Target }
Write-Host "Building target: $resolvedTarget ($Configuration)"

$buildArgs = @("--build", $resolvedBuildDir, "--target", $resolvedTarget, "--parallel")
if (-not $isNinja) {
    $buildArgs += @("--config", $Configuration)
}

& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Build failed."
}

$exePath = Find-FirstExisting @(
    (Join-Path $resolvedBuildDir "$Configuration\qwen3-tts-cli.exe"),
    (Join-Path $resolvedBuildDir "qwen3-tts-cli.exe"),
    (Join-Path $resolvedBuildDir "bin\$Configuration\qwen3-tts-cli.exe"),
    (Join-Path $resolvedBuildDir "bin\qwen3-tts-cli.exe")
)

$dllSourceDir = Find-FirstExisting @(
    (Join-Path $resolvedBuildDir "bin\$Configuration"),
    (Join-Path $resolvedBuildDir "bin")
)
if ($exePath -and $dllSourceDir) {
    $exeDir = Split-Path -Parent $exePath
    $dlls = Get-ChildItem -Path $dllSourceDir -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($dlls) {
        Write-Host "Copying runtime DLLs to $exeDir"
        Copy-Item -Path (Join-Path $dllSourceDir "*.dll") -Destination $exeDir -Force
    }
}

Write-Host "Build success." -ForegroundColor Green
if ($exePath) {
    Write-Host "CLI executable: $exePath"
} else {
    Write-Host "CLI executable not found (target might not include qwen3-tts-cli)." -ForegroundColor Yellow
}

if ($RunSmokeTest) {
    if (-not $exePath) {
        throw "RunSmokeTest requested, but qwen3-tts-cli.exe was not found."
    }

    $resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $ScriptDir $ModelDir }
    if (-not (Test-Path $resolvedModelDir)) {
        throw "RunSmokeTest requested, model directory not found: $resolvedModelDir"
    }

    $outWav = Join-Path $resolvedBuildDir "smoke_test.wav"
    Write-Host "Running smoke test synthesis..."
    & $exePath -m $resolvedModelDir -t $SmokeText -o $outWav --temperature 0 --top-k 0 --max-tokens 64
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed."
    }
    Write-Host "Smoke test output: $outWav" -ForegroundColor Green
}
