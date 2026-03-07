param(
    [string]$ModelDir = "models",
    [string]$ReferenceDir = "reference",
    [string]$ReferenceAudio = "",
    [switch]$GenerateMissing,
    [switch]$ForceRegenerate,
    [switch]$InstallPythonDeps,
    [bool]$UseVenv = $true,
    [bool]$AutoCreateVenv = $true,
    [string]$VenvPath = ".venv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-AllExist([string[]]$paths) {
    foreach ($p in $paths) {
        if (-not (Test-Path $p)) {
            return $false
        }
    }
    return $true
}

function Find-FirstExisting([string[]]$paths) {
    foreach ($p in $paths) {
        if ([string]::IsNullOrWhiteSpace($p)) {
            continue
        }
        if (Test-Path $p) {
            return $p
        }
    }
    return $null
}

function Resolve-SystemPythonRunner() {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return [PSCustomObject]@{
            Exe  = "python"
            Args = @()
            Name = "python"
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return [PSCustomObject]@{
            Exe  = "py"
            Args = @("-3")
            Name = "py -3"
        }
    }

    return $null
}

function Resolve-PythonRunner([string]$repoRoot, [bool]$useVenv, [bool]$autoCreateVenv, [string]$venvPath) {
    if ($useVenv) {
        $resolvedVenvDir = if ([System.IO.Path]::IsPathRooted($venvPath)) { $venvPath } else { Join-Path $repoRoot $venvPath }
        $venvPython = Join-Path $resolvedVenvDir "Scripts\python.exe"

        if (-not (Test-Path $venvPython) -and $autoCreateVenv) {
            $bootstrap = Resolve-SystemPythonRunner
            if (-not $bootstrap) {
                throw "Cannot auto-create venv. No system Python runner found ('py' or 'python')."
            }

            Write-Host "Creating virtual environment at: $resolvedVenvDir"
            $createArgs = @() + $bootstrap.Args + @("-m", "venv", $resolvedVenvDir)
            & $bootstrap.Exe @createArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create venv at $resolvedVenvDir"
            }
        }

        if (Test-Path $venvPython) {
            return [PSCustomObject]@{
                Exe  = $venvPython
                Args = @()
                Name = "venv python ($venvPython)"
            }
        }

        throw "Venv Python not found at $venvPython. Set -AutoCreateVenv `$true or disable -UseVenv."
    }

    $runner = Resolve-SystemPythonRunner
    if ($runner) {
        return $runner
    }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        return [PSCustomObject]@{
            Exe  = "uv"
            Args = @("run", "python")
            Name = "uv run python"
        }
    }

    return $null
}

function Invoke-PythonCommand([object]$runner, [string[]]$pythonArgs, [switch]$captureOutput) {
    $cmdArgs = @() + $runner.Args + $pythonArgs
    if ($captureOutput) {
        return (& $runner.Exe @cmdArgs 2>&1)
    }
    & $runner.Exe @cmdArgs
    return $LASTEXITCODE
}

function Ensure-PythonDependencies([object]$runner, [switch]$installDeps) {
    $requiredModules = @("numpy", "soundfile", "torch", "qwen_tts")
    $checkCode = @"
import importlib.util
mods = ["numpy", "soundfile", "torch", "qwen_tts"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(",".join(missing))
"@

    $output = Invoke-PythonCommand -runner $runner -pythonArgs @("-c", $checkCode) -captureOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to probe Python modules with $($runner.Name). Output: $output"
    }

    $missingText = (($output | Out-String).Trim())
    $missingModules = @()
    if (-not [string]::IsNullOrWhiteSpace($missingText)) {
        $missingModules = @($missingText.Split(",") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    if ($missingModules.Count -eq 0) {
        return
    }

    $moduleToPackage = @{
        "numpy"    = "numpy"
        "soundfile"= "soundfile"
        "torch"    = "torch"
        "qwen_tts" = "qwen-tts"
    }
    $packages = @()
    foreach ($m in $missingModules) {
        if ($moduleToPackage.ContainsKey($m)) {
            $packages += $moduleToPackage[$m]
        } else {
            $packages += $m
        }
    }
    $packages = @($packages | Select-Object -Unique)

    if ($installDeps) {
        Write-Host "Installing missing Python packages: $($packages -join ', ')" -ForegroundColor Yellow

        Invoke-PythonCommand -runner $runner -pythonArgs (@("-m", "pip", "install") + $packages) | Out-Null

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python dependencies: $($packages -join ', ')"
        }
        return
    }

    $installCmd = "$($runner.Exe) $($runner.Args -join ' ') -m pip install " + ($packages -join " ")

    throw "Missing Python modules: $($missingModules -join ', '). Install with: $installCmd"
}

function Write-GroupStatus([string]$name, [string[]]$paths) {
    Write-Host ""
    Write-Host "${name}:"
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Write-Host "  [OK]      $p" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $p" -ForegroundColor Yellow
        }
    }
}

function Invoke-PythonScript([string]$repoRoot, [string]$scriptRelativePath, [object]$runner) {
    $scriptPath = Join-Path $repoRoot $scriptRelativePath
    if (-not (Test-Path $scriptPath)) {
        throw "Script not found: $scriptPath"
    }

    Write-Host "Running: $($runner.Name) $scriptRelativePath"
    Invoke-PythonCommand -runner $runner -pythonArgs @($scriptPath) | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed while running $scriptRelativePath"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
$resolvedReferenceDir = if ([System.IO.Path]::IsPathRooted($ReferenceDir)) { $ReferenceDir } else { Join-Path $repoRoot $ReferenceDir }
$resolvedReferenceAudio = if ([string]::IsNullOrWhiteSpace($ReferenceAudio)) {
    Find-FirstExisting @(
        (Join-Path $repoRoot "clone.wav"),
        (Join-Path $repoRoot "examples/readme_clone_input.wav"),
        (Join-Path $repoRoot "my_voice_ref.wav")
    )
} elseif ([System.IO.Path]::IsPathRooted($ReferenceAudio)) {
    $ReferenceAudio
} else {
    Join-Path $repoRoot $ReferenceAudio
}

$requiredModelFiles = @(
    (Join-Path $resolvedModelDir "qwen3-tts-0.6b-f16.gguf"),
    (Join-Path $resolvedModelDir "qwen3-tts-tokenizer-f16.gguf")
)

$requiredDeterministicRefs = @(
    (Join-Path $resolvedReferenceDir "det_text_tokens.bin"),
    (Join-Path $resolvedReferenceDir "det_speaker_embedding.bin"),
    (Join-Path $resolvedReferenceDir "det_prefill_embedding.bin"),
    (Join-Path $resolvedReferenceDir "det_first_frame_logits.bin"),
    (Join-Path $resolvedReferenceDir "det_speech_codes.bin"),
    (Join-Path $resolvedReferenceDir "det_decoded_audio.bin")
)

$legacyRefs = @(
    (Join-Path $resolvedReferenceDir "speech_codes.bin"),
    (Join-Path $resolvedReferenceDir "decoded_audio.bin"),
    (Join-Path $resolvedReferenceDir "ref_audio_embedding.bin")
)

Write-Host "Preparing test assets in:"
Write-Host "  Repo:      $repoRoot"
Write-Host "  ModelDir:  $resolvedModelDir"
Write-Host "  RefDir:    $resolvedReferenceDir"
Write-Host "  RefAudio:  $resolvedReferenceAudio"

$runner = Resolve-PythonRunner -repoRoot $repoRoot -useVenv $UseVenv -autoCreateVenv $AutoCreateVenv -venvPath $VenvPath
if (-not $runner) {
    throw "No Python runner found. Ensure a venv exists, or install system Python."
}
Write-Host "Using Python runner: $($runner.Name)"
Ensure-PythonDependencies -runner $runner -installDeps:$InstallPythonDeps

if ($GenerateMissing -or $ForceRegenerate) {
    if (-not $resolvedReferenceAudio -or -not (Test-Path $resolvedReferenceAudio)) {
        throw "Reference audio not found. Provide -ReferenceAudio or place examples/readme_clone_input.wav."
    }
    $env:QWEN3_TTS_REF_AUDIO = $resolvedReferenceAudio

    if ($ForceRegenerate -or -not (Test-AllExist $requiredDeterministicRefs)) {
        Invoke-PythonScript -repoRoot $repoRoot -scriptRelativePath "scripts/generate_deterministic_reference.py" -runner $runner
    }

    if ($ForceRegenerate -or -not (Test-AllExist $legacyRefs)) {
        Invoke-PythonScript -repoRoot $repoRoot -scriptRelativePath "scripts/generate_reference_outputs.py" -runner $runner
    }
}

Write-GroupStatus -name "Required model files" -paths $requiredModelFiles
Write-GroupStatus -name "Required deterministic refs" -paths $requiredDeterministicRefs
Write-GroupStatus -name "Optional legacy refs" -paths $legacyRefs

$missingModels = -not (Test-AllExist $requiredModelFiles)
$missingDetRefs = -not (Test-AllExist $requiredDeterministicRefs)

if ($missingModels -or $missingDetRefs) {
    Write-Host ""
    Write-Host "Asset preparation is incomplete." -ForegroundColor Red
    Write-Host "Hints:"
    Write-Host "  1. Ensure models exist under '$resolvedModelDir'."
    Write-Host "  2. Generate references:"
    Write-Host "     pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_test_assets.ps1 -GenerateMissing"
    Write-Host "  3. Force full regeneration:"
    Write-Host "     pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_test_assets.ps1 -ForceRegenerate"
    exit 1
}

Write-Host ""
Write-Host "All required test assets are present." -ForegroundColor Green
exit 0
