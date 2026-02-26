<#
    build-ninja.ps1
    --------------------
    Builds qwen3-tts.cpp using the Ninja generator for faster compilation.
    Automatically loads the MSVC environment variables required by Ninja.
#>

[CmdletBinding()]
param(
    [switch]$RunTest
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Function to load MSVC environment variables
function Import-VSEnv {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Is Visual Studio installed?" }

    $vsroot = & $vswhere -latest -products * `
               -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
               -property installationPath 2>$null
    
    if ([string]::IsNullOrWhiteSpace($vsroot)) { throw "VS Build Tools with C++ workload not found." }

    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    Write-Host "-> Importing MSVC environment from $vcvars"
    $envDump = cmd /s /c "`"$vcvars`" && set"
    foreach ($line in $envDump -split "`r?`n") {
        if ($line -match '^(.*?)=(.*)$') {
            $name,$value = $Matches[1],$Matches[2]
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

# 1. Load the MSVC compiler environment
Import-VSEnv

# 2. Check for Ninja
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "Ninja not found on PATH. Please ensure it is installed."
}

$BuildDir = Join-Path $ScriptRoot 'build-cuda-ninja'

# 3. Configure with CMake
Write-Host "`n-> Configuring CMake with Ninja generator in $BuildDir..."
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Set-Location $BuildDir

cmake .. -G Ninja `
    -DQWEN3_TTS_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release

# 4. Build
Write-Host "`n-> Building project..."
cmake --build . --config Release --parallel

# 5. Copy DLLs to the executable directory
Write-Host "`n-> Copying runtime DLLs to build directory..."
Copy-Item -Path "bin\*.dll" -Destination "." -Force -ErrorAction SilentlyContinue

Write-Host "`n-> Build complete! Binaries are in $BuildDir"
Set-Location $ScriptRoot

if ($RunTest) {
    Write-Host "`n-> Running test synthesis..."
    $exe = Join-Path $BuildDir "qwen3-tts-cli.exe"
    & $exe -m models -t "This is a test to verify the CUDA backend is working." -o test_cuda.wav
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Test successful! Output saved to test_cuda.wav" -ForegroundColor Green
    } else {
        Write-Host "Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}
