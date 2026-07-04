param(
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$ModelDir = "models",
    [string]$ModelName06 = "qwen-talker-0.6b-base-Q8_0.gguf",
    [string]$ModelName17Base = "qwen-talker-1.7b-base-Q8_0.gguf",
    [string]$ModelName17Custom = "qwen-talker-1.7b-customvoice-Q8_0.gguf",
    [string]$Model17Speaker = "vivian",
    [string]$ReferenceAudio = "examples/readme_clone_input.wav",
    [string]$OutputDir = "test_output",
    [switch]$BuildFirst,
    [switch]$BuildMissingTargets,
    [switch]$RequireComponentTests,
    [switch]$RequireIclSmoke,
    [switch]$RequireParityFixtures,
    [switch]$ParityFixturesOnly,
    [switch]$Skip17B,
    [switch]$PrepareAssets,
    [switch]$GenerateMissingAssets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$PASS_COUNT = 0
$FAIL_COUNT = 0
$SKIP_COUNT = 0

function Write-Section([string]$title) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host $title -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
}

function Add-Pass([string]$msg) {
    $script:PASS_COUNT++
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Add-Fail([string]$msg) {
    $script:FAIL_COUNT++
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

function Add-Skip([string]$msg) {
    $script:SKIP_COUNT++
    Write-Host "[SKIP] $msg" -ForegroundColor Yellow
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

function Find-StudioModelDir() {
    $candidate = Join-Path $env:USERPROFILE ".qwen-tts-studio\models"
    if ((Test-Path -LiteralPath (Join-Path $candidate "qwen-talker-0.6b-base-Q8_0.gguf")) -and
        (Test-Path -LiteralPath (Join-Path $candidate "qwen-talker-1.7b-base-Q8_0.gguf")) -and
        (Test-Path -LiteralPath (Join-Path $candidate "qwen-tokenizer-12hz-Q8_0.gguf"))) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    return $null
}

function Resolve-BinaryPath([string]$name, [string]$buildDir, [string]$config) {
    $candidates = @(
        (Join-Path $buildDir "$config\$name.exe"),
        (Join-Path $buildDir "$name.exe"),
        (Join-Path $buildDir "bin\$config\$name.exe"),
        (Join-Path $buildDir "bin\$name.exe")
    )
    return Find-FirstExisting $candidates
}

function Get-AllBuildTarget([string]$buildDir) {
    if (Test-Path (Join-Path $buildDir "build.ninja")) {
        return "all"
    }
    return "ALL_BUILD"
}

function Write-PreflightStatus([string]$label, [string]$path, [bool]$required, [string]$hint) {
    $exists = $path -and (Test-Path $path)
    $tag = if ($exists) { "[OK]" } else { "[MISSING]" }
    $color = if ($exists) { "Green" } else { "Yellow" }
    $reqText = if ($required) { "required" } else { "optional" }
    $displayPath = if ($path) { $path } else { "<not resolved>" }
    Write-Host "$tag $label ($reqText): $displayPath" -ForegroundColor $color
    if (-not $exists -and $hint) {
        Write-Host "  -> $hint" -ForegroundColor DarkYellow
    }
    return $exists
}

function Write-OutputTail([string]$output, [int]$maxLines = 20) {
    if ([string]::IsNullOrWhiteSpace($output)) {
        return
    }
    $lines = $output -split "`r?`n"
    $start = [Math]::Max(0, $lines.Length - $maxLines)
    Write-Host "  Last output lines:"
    for ($i = $start; $i -lt $lines.Length; $i++) {
        Write-Host ("    " + $lines[$i])
    }
}

function Invoke-CommandCapture([string]$exe, [string[]]$commandArgs) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $exe @commandArgs 2>&1
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $exitCode = $LASTEXITCODE
    $text = (($output | ForEach-Object { $_.ToString() }) | Out-String).TrimEnd()
    return [PSCustomObject]@{
        ExitCode = $exitCode
        Output   = $text
    }
}

function ConvertTo-InvariantText([object]$value) {
    return [System.Convert]::ToString($value, [Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-CheckedTest(
    [string]$name,
    [string]$exe,
    [string[]]$commandArgs,
    [string]$successRegex = "",
    [string]$forbiddenRegex = "FAIL:"
) {
    Write-Host ""
    Write-Host "--- $name ---"
    $res = Invoke-CommandCapture -exe $exe -commandArgs $commandArgs

    $hasForbidden = $false
    if (-not [string]::IsNullOrWhiteSpace($forbiddenRegex)) {
        $hasForbidden = $res.Output -match $forbiddenRegex
    }

    $hasSuccess = $true
    if (-not [string]::IsNullOrWhiteSpace($successRegex)) {
        $hasSuccess = $res.Output -match $successRegex
    }

    if ($res.ExitCode -eq 0 -and -not $hasForbidden -and $hasSuccess) {
        Add-Pass $name
        return $true
    }

    Add-Fail "$name (exit code: $($res.ExitCode))"
    Write-OutputTail -output $res.Output
    return $false
}

function Save-DebugDumpEnv() {
    return [PSCustomObject]@{
        Dir = $env:QWEN3_TTS_DEBUG_DUMP_DIR
        MaxFrames = $env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES
        MaxCodeSteps = $env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS
    }
}

function Disable-DebugDumpEnv() {
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES -ErrorAction SilentlyContinue
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS -ErrorAction SilentlyContinue
}

function Restore-DebugDumpEnv([object]$snapshot) {
    if ($null -eq $snapshot) {
        return
    }

    if ($null -ne $snapshot.Dir) {
        $env:QWEN3_TTS_DEBUG_DUMP_DIR = $snapshot.Dir
    } else {
        Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_DIR -ErrorAction SilentlyContinue
    }

    if ($null -ne $snapshot.MaxFrames) {
        $env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES = $snapshot.MaxFrames
    } else {
        Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES -ErrorAction SilentlyContinue
    }

    if ($null -ne $snapshot.MaxCodeSteps) {
        $env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS = $snapshot.MaxCodeSteps
    } else {
        Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS -ErrorAction SilentlyContinue
    }
}

function Get-WavInfo([string]$path, [int]$maxFramesForRms = 48000) {
    if (-not (Test-Path $path)) {
        return $null
    }

    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -lt 44) {
        return $null
    }

    $ascii = [System.Text.Encoding]::ASCII
    if ($ascii.GetString($bytes, 0, 4) -ne "RIFF" -or $ascii.GetString($bytes, 8, 4) -ne "WAVE") {
        return $null
    }

    $pos = 12
    $audioFormat = 0
    $numChannels = 0
    $sampleRate = 0
    $bitsPerSample = 0
    $dataOffset = -1
    $dataSize = 0

    while ($pos + 8 -le $bytes.Length) {
        $chunkId = $ascii.GetString($bytes, $pos, 4)
        $chunkSize = [BitConverter]::ToUInt32($bytes, $pos + 4)
        $chunkDataPos = $pos + 8
        if ($chunkDataPos + $chunkSize -gt $bytes.Length) {
            break
        }

        if ($chunkId -eq "fmt ") {
            if ($chunkSize -ge 16) {
                $audioFormat = [BitConverter]::ToUInt16($bytes, $chunkDataPos)
                $numChannels = [BitConverter]::ToUInt16($bytes, $chunkDataPos + 2)
                $sampleRate = [BitConverter]::ToUInt32($bytes, $chunkDataPos + 4)
                $bitsPerSample = [BitConverter]::ToUInt16($bytes, $chunkDataPos + 14)
            }
        } elseif ($chunkId -eq "data") {
            $dataOffset = $chunkDataPos
            $dataSize = [int]$chunkSize
            break
        }

        $pad = if (($chunkSize % 2) -eq 1) { 1 } else { 0 }
        $pos = [int]($chunkDataPos + $chunkSize + $pad)
    }

    if ($dataOffset -lt 0 -or $sampleRate -le 0 -or $numChannels -le 0 -or $bitsPerSample -le 0) {
        return $null
    }

    $bytesPerSample = [int]($bitsPerSample / 8)
    $bytesPerFrame = $bytesPerSample * $numChannels
    if ($bytesPerSample -le 0 -or $bytesPerFrame -le 0) {
        return $null
    }

    $totalFrames = [int64][Math]::Floor($dataSize / $bytesPerFrame)
    $durationSec = [double]$totalFrames / [double]$sampleRate

    $framesForRms = [int64][Math]::Min($totalFrames, [int64]$maxFramesForRms)
    $sumSq = 0.0

    for ($f = 0; $f -lt $framesForRms; $f++) {
        $frameStart = $dataOffset + ($f * $bytesPerFrame)
        $mono = 0.0

        for ($c = 0; $c -lt $numChannels; $c++) {
            $sampleOffset = $frameStart + ($c * $bytesPerSample)
            $sample = 0.0
            $fmtKey = "$audioFormat/$bitsPerSample"

            if ($fmtKey -eq "1/16") {
                $sample = [double]([BitConverter]::ToInt16($bytes, $sampleOffset)) / 32768.0
            } elseif ($fmtKey -eq "1/32") {
                $sample = [double]([BitConverter]::ToInt32($bytes, $sampleOffset)) / 2147483648.0
            } elseif ($fmtKey -eq "3/32") {
                $sample = [double]([BitConverter]::ToSingle($bytes, $sampleOffset))
            } elseif ($fmtKey -eq "3/64") {
                $sample = [double]([BitConverter]::ToDouble($bytes, $sampleOffset))
            } else {
                return $null
            }

            $mono += $sample
        }

        $mono /= [double]$numChannels
        $sumSq += $mono * $mono
    }

    $rms = 0.0
    if ($framesForRms -gt 0) {
        $rms = [Math]::Sqrt($sumSq / [double]$framesForRms)
    }

    return [PSCustomObject]@{
        SampleRate    = [int]$sampleRate
        Channels      = [int]$numChannels
        BitsPerSample = [int]$bitsPerSample
        AudioFormat   = [int]$audioFormat
        DataBytes     = [int]$dataSize
        Frames        = [int64]$totalFrames
        DurationSec   = [double]$durationSec
        RMS           = [double]$rms
    }
}

function Validate-WavOutput(
    [string]$testName,
    [string]$wavPath,
    [int]$expectedSampleRate = 24000,
    [double]$minDurationSec = 0.20,
    [double]$maxDurationSec = 60.0,
    [double]$minRms = 0.001
) {
    if (-not (Test-Path $wavPath)) {
        Add-Fail "$testName - output file missing: $wavPath"
        return $false
    }

    $info = Get-WavInfo -path $wavPath
    if ($null -eq $info) {
        Add-Fail "$testName - invalid/unsupported WAV format"
        return $false
    }

    if ($info.SampleRate -ne $expectedSampleRate) {
        Add-Fail "$testName - unexpected sample rate ($($info.SampleRate), expected $expectedSampleRate)"
        return $false
    }

    if ($info.DurationSec -lt $minDurationSec -or $info.DurationSec -gt $maxDurationSec) {
        Add-Fail "$testName - suspicious duration ($([Math]::Round($info.DurationSec, 3)) sec)"
        return $false
    }

    if ($info.RMS -lt $minRms) {
        Add-Fail "$testName - near-silent output (RMS=$([Math]::Round($info.RMS, 6)))"
        return $false
    }

    Add-Pass "$testName - valid WAV ($([Math]::Round($info.DurationSec, 2)) sec, RMS=$([Math]::Round($info.RMS, 4)))"
    return $true
}

function Write-FinalSummaryAndExit() {
    Write-Section "Summary"
    Write-Host "PASS: $PASS_COUNT"
    Write-Host "FAIL: $FAIL_COUNT"
    Write-Host "SKIP: $SKIP_COUNT"
    Write-Host "Outputs: $resolvedOutputDir"

    Restore-DebugDumpEnv -snapshot $debugDumpEnvSnapshot

    if ($FAIL_COUNT -gt 0) {
        exit 1
    }

    exit 0
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
if ($ParityFixturesOnly) {
    $RequireParityFixtures = $true
}
$run17B = (-not $Skip17B) -and (-not $ParityFixturesOnly)

$debugDumpEnvSnapshot = Save-DebugDumpEnv
$hadDebugDumpEnv = ($null -ne $debugDumpEnvSnapshot.Dir) -or
                   ($null -ne $debugDumpEnvSnapshot.MaxFrames) -or
                   ($null -ne $debugDumpEnvSnapshot.MaxCodeSteps)
Disable-DebugDumpEnv
if ($hadDebugDumpEnv) {
    Write-Host "Debug trace dump environment variables detected and disabled for this test run." -ForegroundColor DarkYellow
}

$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repoRoot $BuildDir }
$studioModelDir = Find-StudioModelDir
if ($ModelDir -eq "models" -and $studioModelDir) {
    $resolvedModelDir = $studioModelDir
} else {
    $resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
}
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }

if ($BuildFirst) {
    $buildScript = Join-Path $repoRoot "build.ps1"
    if (-not (Test-Path $buildScript)) {
        throw "build.ps1 not found at $buildScript"
    }

    Write-Host "Running build step via build.ps1 ..."
    & $buildScript -Configuration $Configuration -BuildAll
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed via build.ps1"
    }
}

if ($PrepareAssets) {
    $assetScript = Join-Path $repoRoot "scripts/prepare_test_assets.ps1"
    if (-not (Test-Path $assetScript)) {
        throw "prepare_test_assets.ps1 not found at $assetScript"
    }
    Write-Host "Preparing test assets via prepare_test_assets.ps1 ..."
    $assetArgs = @("-ModelDir", $resolvedModelDir, "-ReferenceDir", (Join-Path $repoRoot "reference"))
    if ($GenerateMissingAssets) {
        $assetArgs += "-GenerateMissing"
    }
    & $assetScript @assetArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Asset preparation failed."
    }
}

Write-Host "Running Windows test suite from $repoRoot"
Write-Host "BuildDir: $resolvedBuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "ModelDir: $resolvedModelDir"

New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null

$tokenizerExe = Resolve-BinaryPath -name "test_tokenizer" -buildDir $resolvedBuildDir -config $Configuration
$encoderExe = Resolve-BinaryPath -name "test_encoder" -buildDir $resolvedBuildDir -config $Configuration
$transformerExe = Resolve-BinaryPath -name "test_transformer" -buildDir $resolvedBuildDir -config $Configuration
$decoderExe = Resolve-BinaryPath -name "test_decoder" -buildDir $resolvedBuildDir -config $Configuration
$cliExe = Resolve-BinaryPath -name "qwen3-tts-cli" -buildDir $resolvedBuildDir -config $Configuration

if (-not $tokenizerExe -or -not $encoderExe -or -not $transformerExe -or -not $decoderExe -or -not $cliExe) {
    if ($BuildMissingTargets) {
        Write-Host ""
        Write-Host "Some binaries are missing. Attempting to build full target ..." -ForegroundColor Yellow
        $allTarget = Get-AllBuildTarget -buildDir $resolvedBuildDir
        Write-Host "Using build target: $allTarget"
        $buildArgs = @("--build", $resolvedBuildDir, "--target", $allTarget, "--parallel")
        if ($allTarget -eq "ALL_BUILD") {
            $buildArgs += @("--config", $Configuration)
        }
        & cmake @buildArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build ALL_BUILD for test binaries"
        }

        $tokenizerExe = Resolve-BinaryPath -name "test_tokenizer" -buildDir $resolvedBuildDir -config $Configuration
        $encoderExe = Resolve-BinaryPath -name "test_encoder" -buildDir $resolvedBuildDir -config $Configuration
        $transformerExe = Resolve-BinaryPath -name "test_transformer" -buildDir $resolvedBuildDir -config $Configuration
        $decoderExe = Resolve-BinaryPath -name "test_decoder" -buildDir $resolvedBuildDir -config $Configuration
        $cliExe = Resolve-BinaryPath -name "qwen3-tts-cli" -buildDir $resolvedBuildDir -config $Configuration
    } else {
        Write-Host ""
        Write-Host "Some component test binaries are missing. Re-run with -BuildMissingTargets to auto-build all targets." -ForegroundColor Yellow
    }
}

$ttsModel = Join-Path $resolvedModelDir $ModelName06
$ttsModel17Base = Join-Path $resolvedModelDir $ModelName17Base
$ttsModel17Custom = Join-Path $resolvedModelDir $ModelName17Custom
$tokModel = Join-Path $resolvedModelDir "qwen-tokenizer-12hz-Q8_0.gguf"
$resolvedRefAudioArg = if ([System.IO.Path]::IsPathRooted($ReferenceAudio)) {
    $ReferenceAudio
} else {
    Join-Path $repoRoot $ReferenceAudio
}
$refAudio = Find-FirstExisting @(
    $resolvedRefAudioArg,
    (Join-Path $repoRoot "clone.wav"),
    (Join-Path $repoRoot "examples/readme_clone_input.wav")
)
$encoderRef = Find-FirstExisting @(
    (Join-Path $repoRoot "reference/ref_audio_embedding.bin"),
    (Join-Path $repoRoot "reference/det_speaker_embedding.bin")
)
$decoderCodes = Find-FirstExisting @(
    (Join-Path $repoRoot "reference/speech_codes.bin"),
    (Join-Path $repoRoot "reference/det_speech_codes.bin")
)
$decoderRef = Find-FirstExisting @(
    (Join-Path $repoRoot "reference/decoded_audio.bin"),
    (Join-Path $repoRoot "reference/det_decoded_audio.bin")
)

$transformerReq = @(
    (Join-Path $repoRoot "reference/det_text_tokens.bin"),
    (Join-Path $repoRoot "reference/det_speaker_embedding.bin"),
    (Join-Path $repoRoot "reference/det_prefill_embedding.bin"),
    (Join-Path $repoRoot "reference/det_first_frame_logits.bin"),
    (Join-Path $repoRoot "reference/det_speech_codes.bin")
)
$missingTransformerRefs = @()
foreach ($f in $transformerReq) {
    if (-not (Test-Path $f)) {
        $missingTransformerRefs += $f
    }
}
$hasTransformerRefs = $missingTransformerRefs.Count -eq 0

Write-Section "Section 0: Preflight"

$strictMissingCount = 0
$mainModelRequired = -not $ParityFixturesOnly

$preflightChecks = @(
    @{ Label = "CLI binary";                 Path = $cliExe;         Required = $false; Hint = "Run: pwsh -File .\build.ps1 -BuildAll -Configuration $Configuration" },
    @{ Label = "Tokenizer test binary";      Path = $tokenizerExe;   Required = $RequireComponentTests; Hint = "Run: pwsh -File .\build.ps1 -BuildAll -Configuration $Configuration" },
    @{ Label = "Encoder test binary";        Path = $encoderExe;     Required = $RequireComponentTests; Hint = "Run: pwsh -File .\build.ps1 -BuildAll -Configuration $Configuration" },
    @{ Label = "Transformer test binary";    Path = $transformerExe; Required = $RequireComponentTests; Hint = "Run: pwsh -File .\build.ps1 -BuildAll -Configuration $Configuration" },
    @{ Label = "Decoder test binary";        Path = $decoderExe;     Required = $RequireComponentTests; Hint = "Run: pwsh -File .\build.ps1 -BuildAll -Configuration $Configuration" },
    @{ Label = "TTS model (0.6B)";           Path = $ttsModel;       Required = $mainModelRequired;  Hint = "Place $ModelName06 under '$resolvedModelDir'." },
    @{ Label = "TTS model (1.7B Base)";      Path = $ttsModel17Base; Required = $run17B; Hint = "Place $ModelName17Base under '$resolvedModelDir'." },
    @{ Label = "TTS model (1.7B Custom)";    Path = $ttsModel17Custom; Required = $run17B; Hint = "Place $ModelName17Custom under '$resolvedModelDir'." },
    @{ Label = "Tokenizer model";            Path = $tokModel;       Required = $mainModelRequired;  Hint = "Place qwen-tokenizer-12hz-Q8_0.gguf under '$resolvedModelDir'." },
    @{ Label = "Reference audio";            Path = $refAudio;       Required = $mainModelRequired; Hint = "Use -ReferenceAudio or place examples/readme_clone_input.wav." },
    @{ Label = "Encoder reference embedding";Path = $encoderRef;     Required = $RequireComponentTests; Hint = "Run: pwsh -File .\scripts\prepare_test_assets.ps1 -GenerateMissing" },
    @{ Label = "Decoder codes";              Path = $decoderCodes;   Required = $RequireComponentTests; Hint = "Run: pwsh -File .\scripts\prepare_test_assets.ps1 -GenerateMissing" },
    @{ Label = "Decoder reference audio";    Path = $decoderRef;     Required = $RequireComponentTests; Hint = "Run: pwsh -File .\scripts\prepare_test_assets.ps1 -GenerateMissing" }
)

foreach ($check in $preflightChecks) {
    $ok = Write-PreflightStatus -label $check.Label -path $check.Path -required ([bool]$check.Required) -hint $check.Hint
    if ((-not $ok) -and [bool]$check.Required) {
        $strictMissingCount++
    }
}

if ($RequireComponentTests) {
    if ($missingTransformerRefs.Count -eq 0) {
        Write-Host "[OK] Transformer deterministic references (required): all files present" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] Transformer deterministic references (required): $($missingTransformerRefs.Count) file(s) missing" -ForegroundColor Yellow
        foreach ($f in $missingTransformerRefs) {
            Write-Host "  -> $f" -ForegroundColor DarkYellow
        }
        Write-Host "  -> Run: pwsh -File .\scripts\prepare_test_assets.ps1 -GenerateMissing" -ForegroundColor DarkYellow
        $strictMissingCount += $missingTransformerRefs.Count
    }
}

$strictPreflightFailed = ($strictMissingCount -gt 0)
if ($strictPreflightFailed) {
    Add-Fail "Strict preflight failed: $strictMissingCount required prerequisite(s) missing."
    Write-Host "Fix commands:" -ForegroundColor DarkYellow
    Write-Host "  pwsh -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -Configuration $Configuration -BuildAll" -ForegroundColor DarkYellow
    Write-Host "  pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_test_assets.ps1 -GenerateMissing" -ForegroundColor DarkYellow
}

Write-Section "Section 1: Component Tests"

if ($tokenizerExe -and (Test-Path $ttsModel)) {
    [void](Invoke-CheckedTest -name "Tokenizer" -exe $tokenizerExe -commandArgs @("--model", $ttsModel) -successRegex "All tests passed")
} else {
    Add-Skip "Tokenizer (binary or model missing)"
}

if ($encoderExe -and (Test-Path $ttsModel) -and $refAudio) {
    $encoderArgs = @("--tokenizer", $ttsModel, "--audio", $refAudio)
    if ($encoderRef) {
        $encoderArgs += @("--reference", $encoderRef)
    }
    [void](Invoke-CheckedTest -name "Encoder" -exe $encoderExe -commandArgs $encoderArgs -successRegex "All tests passed")
} else {
    Add-Skip "Encoder (binary/model/reference audio missing)"
}

if ($transformerExe -and (Test-Path $ttsModel) -and $hasTransformerRefs) {
    [void](Invoke-CheckedTest -name "Transformer deterministic" `
        -exe $transformerExe `
        -commandArgs @("--model", $ttsModel, "--ref-dir", (Join-Path $repoRoot "reference")) `
        -successRegex "=== All tests passed|=== All tests passed with warnings ===" `
        -forbiddenRegex "(?m)^\s*\|\s*FAIL:\s*[1-9]")
} else {
    Add-Skip "Transformer deterministic (binary/model/reference artifacts missing)"
}

if ($decoderExe -and (Test-Path $tokModel) -and $decoderCodes) {
    $decoderArgs = @("--tokenizer", $tokModel, "--codes", $decoderCodes)
    if ($decoderRef) {
        $decoderArgs += @("--reference", $decoderRef)
    }
    [void](Invoke-CheckedTest -name "Decoder" `
        -exe $decoderExe `
        -commandArgs $decoderArgs `
        -successRegex "All tests completed|PASS: Decoded" `
        -forbiddenRegex "FAIL:")
} else {
    Add-Skip "Decoder (binary/model/codes missing)"
}

Write-Section "Section 1b: Python Parity Fixtures"

$parityScript = Join-Path $repoRoot "scripts/run_speaker_parity_fixture.ps1"
$parityPythonPath = Join-Path (Split-Path $repoRoot -Parent) "Qwen3-TTS"
$parityPythonModel = Join-Path (Split-Path $repoRoot -Parent) "audio.cpp/models/Qwen3-TTS-12Hz-1.7B-Base"
$parityCppModelDir = Join-Path $repoRoot "benchmark_output/bf16_parity_modeldir"
$paritySpeakerEmbedding = Join-Path $repoRoot "benchmark_output/python_parity/python_speaker_embedding.json"
$parityReferenceText = Join-Path $repoRoot "benchmark_output/parity_serveurperso_seed/ref.txt"
$parityReferenceCodes = Join-Path $repoRoot "benchmark_output/python_parity/python_reference_codes.json"
$parityExpectationsFile = Join-Path $repoRoot "tests/fixtures/python_parity_expectations.json"
$parityOutputRoot = Join-Path $resolvedOutputDir "python_parity"

$parityAssets = @(
    $parityScript,
    $parityExpectationsFile,
    $parityPythonPath,
    $parityPythonModel,
    $parityCppModelDir,
    $paritySpeakerEmbedding,
    $parityReferenceText,
    $parityReferenceCodes
)
$missingParityAssets = @()
foreach ($asset in $parityAssets) {
    if (-not (Test-Path $asset)) {
        $missingParityAssets += $asset
    }
}

if (-not $cliExe) {
    if ($RequireParityFixtures) {
        Add-Fail "Python parity fixtures (qwen3-tts-cli binary missing)"
    } else {
        Add-Skip "Python parity fixtures (qwen3-tts-cli binary missing)"
    }
} elseif ($missingParityAssets.Count -gt 0) {
    if ($RequireParityFixtures) {
        Add-Fail "Python parity fixtures ($($missingParityAssets.Count) asset(s) missing)"
        foreach ($asset in $missingParityAssets) {
            Write-Host "  -> $asset" -ForegroundColor DarkYellow
        }
    } else {
        Add-Skip "Python parity fixtures (local model/parity assets missing)"
    }
} else {
    $parityExpectations = Get-Content -LiteralPath $parityExpectationsFile -Raw | ConvertFrom-Json
    $speakerFixture = $parityExpectations.fixtures.speaker_only
    $speakerExpect = $speakerFixture.expect
    $iclFixture = $parityExpectations.fixtures.icl
    $iclExpect = $iclFixture.expect

    $speakerParityOut = Join-Path $parityOutputRoot "speaker_fixture"
    $speakerParityRes = Invoke-CommandCapture -exe "powershell" -commandArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $parityScript,
        "-PythonPath", $parityPythonPath,
        "-PythonModel", $parityPythonModel,
        "-CppModelDir", $parityCppModelDir,
        "-SpeakerEmbedding", $paritySpeakerEmbedding,
        "-CliExe", $cliExe,
        "-OutputDir", $speakerParityOut,
        "-Text", $speakerFixture.text,
        "-MaxTokens", (ConvertTo-InvariantText $speakerFixture.max_tokens),
        "-MaxFrames", (ConvertTo-InvariantText $speakerFixture.max_frames),
        "-ExpectMatchPercentAtLeast", (ConvertTo-InvariantText $speakerExpect.match_percent_at_least),
        "-ExpectFirstDiffFrame", (ConvertTo-InvariantText $speakerExpect.first_diff_frame),
        "-ExpectFirstDiffCodebook", (ConvertTo-InvariantText $speakerExpect.first_diff_codebook),
        "-ExpectFirstDiffTokenA", (ConvertTo-InvariantText $speakerExpect.first_diff_token_a),
        "-ExpectFirstDiffTokenB", (ConvertTo-InvariantText $speakerExpect.first_diff_token_b),
        "-ExpectFirstDiffCosineAtLeast", (ConvertTo-InvariantText $speakerExpect.first_diff_cosine_at_least),
        "-ExpectFirstDiffMaxAbsAtMost", (ConvertTo-InvariantText $speakerExpect.first_diff_max_abs_at_most),
        "-ExpectFirstDiffCategory", $speakerExpect.first_diff_category,
        "-RequireAssets"
    )
    if ($speakerParityRes.ExitCode -eq 0) {
        Add-Pass "Python parity fixture (speaker-only)"
    } else {
        Add-Fail "Python parity fixture (speaker-only, exit code: $($speakerParityRes.ExitCode))"
        Write-OutputTail -output $speakerParityRes.Output
    }

    $iclParityOut = Join-Path $parityOutputRoot "icl_fixture"
    $iclParityRes = Invoke-CommandCapture -exe "powershell" -commandArgs @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $parityScript,
        "-PythonPath", $parityPythonPath,
        "-PythonModel", $parityPythonModel,
        "-CppModelDir", $parityCppModelDir,
        "-SpeakerEmbedding", $paritySpeakerEmbedding,
        "-CliExe", $cliExe,
        "-OutputDir", $iclParityOut,
        "-Text", $iclFixture.text,
        "-MaxTokens", (ConvertTo-InvariantText $iclFixture.max_tokens),
        "-MaxFrames", (ConvertTo-InvariantText $iclFixture.max_frames),
        "-ReferenceTextFile", $parityReferenceText,
        "-ReferenceCodes", $parityReferenceCodes,
        "-ExpectMatchPercentAtLeast", (ConvertTo-InvariantText $iclExpect.match_percent_at_least),
        "-ExpectFirstDiffFrame", (ConvertTo-InvariantText $iclExpect.first_diff_frame),
        "-ExpectFirstDiffCodebook", (ConvertTo-InvariantText $iclExpect.first_diff_codebook),
        "-ExpectFirstDiffTokenA", (ConvertTo-InvariantText $iclExpect.first_diff_token_a),
        "-ExpectFirstDiffTokenB", (ConvertTo-InvariantText $iclExpect.first_diff_token_b),
        "-ExpectFirstDiffCosineAtLeast", (ConvertTo-InvariantText $iclExpect.first_diff_cosine_at_least),
        "-ExpectFirstDiffMaxAbsAtMost", (ConvertTo-InvariantText $iclExpect.first_diff_max_abs_at_most),
        "-ExpectFirstDiffCategory", $iclExpect.first_diff_category,
        "-RequireAssets"
    )
    if ($iclParityRes.ExitCode -eq 0) {
        Add-Pass "Python parity fixture (ICL)"
    } else {
        Add-Fail "Python parity fixture (ICL, exit code: $($iclParityRes.ExitCode))"
        Write-OutputTail -output $iclParityRes.Output
    }
}

if ($ParityFixturesOnly) {
    Write-FinalSummaryAndExit
}

Write-Section "Section 2: CLI Output Regression Checks (0.6B)"

if (-not $cliExe) {
    Add-Skip "CLI output tests (qwen3-tts-cli binary missing)"
} elseif (-not (Test-Path $ttsModel) -or -not (Test-Path $tokModel)) {
    Add-Skip "CLI output tests (required GGUF model files missing)"
} else {
    $basicOut = Join-Path $resolvedOutputDir "regression_basic.wav"
    $cloneOut = Join-Path $resolvedOutputDir "regression_clone.wav"

    Write-Host ""
    Write-Host "--- CLI basic synthesis ---"
    $basicRes = Invoke-CommandCapture -exe $cliExe -commandArgs @(
        "-m", $resolvedModelDir,
        "-t", "Hello world from qwen3 tts.",
        "-o", $basicOut
    )
    if ($basicRes.ExitCode -eq 0) {
        Validate-WavOutput -testName "CLI basic synthesis" -wavPath $basicOut | Out-Null
    } else {
        Add-Fail "CLI basic synthesis (exit code: $($basicRes.ExitCode))"
        Write-OutputTail -output $basicRes.Output
    }

    if ($refAudio) {
        Write-Host ""
        Write-Host "--- CLI voice cloning ---"
        $cloneRes = Invoke-CommandCapture -exe $cliExe -commandArgs @(
            "-m", $resolvedModelDir,
            "-t", "Hello world from cloned voice.",
            "-r", $refAudio,
            "-o", $cloneOut
        )
        if ($cloneRes.ExitCode -eq 0) {
            Validate-WavOutput -testName "CLI voice cloning" -wavPath $cloneOut | Out-Null
        } else {
            Add-Fail "CLI voice cloning (exit code: $($cloneRes.ExitCode))"
            Write-OutputTail -output $cloneRes.Output
        }
    } else {
        Add-Skip "CLI voice cloning (reference audio missing)"
    }

    if ($RequireIclSmoke) {
        $referenceTextFile = Join-Path $repoRoot "reference_text.txt"
        if ($refAudio -and (Test-Path $referenceTextFile)) {
            $iclOut = Join-Path $resolvedOutputDir "regression_icl_reference_text.wav"
            Write-Host ""
            Write-Host "--- CLI ICL voice cloning with reference text ---"
            $iclRes = Invoke-CommandCapture -exe $cliExe -commandArgs @(
                "-m", $resolvedModelDir,
                "-t", "This is a longer ICL smoke test. The output should contain clear spoken audio, not only noise or silence.",
                "-r", $refAudio,
                "--reference-text-file", $referenceTextFile,
                "--temperature", "0.9",
                "--top-k", "50",
                "--top-p", "1.0",
                "--seed", "42",
                "--max-tokens", "48",
                "-o", $iclOut
            )
            if ($iclRes.ExitCode -eq 0) {
                Validate-WavOutput -testName "CLI ICL voice cloning with reference text" -wavPath $iclOut -minDurationSec 1.0 -maxDurationSec 8.0 | Out-Null
            } else {
                Add-Fail "CLI ICL voice cloning with reference text (exit code: $($iclRes.ExitCode))"
                Write-OutputTail -output $iclRes.Output
            }
        } else {
            Add-Fail "CLI ICL voice cloning with reference text (reference audio/text missing)"
        }
    } else {
        Add-Skip "CLI ICL voice cloning with reference text (enable with -RequireIclSmoke)"
    }
}

Write-Section "Section 3: CLI Output Regression Checks (1.7B)"

if (-not $run17B) {
    Add-Skip "1.7B CLI output tests disabled (-Skip17B)"
} elseif (-not $cliExe) {
    Add-Skip "1.7B CLI output tests (qwen3-tts-cli binary missing)"
} elseif (-not (Test-Path $ttsModel17Base) -or -not (Test-Path $ttsModel17Custom)) {
    Add-Fail "1.7B CLI output tests (required models missing)"
} else {
    $basic17Out = Join-Path $resolvedOutputDir "regression_1p7b_basic.wav"
    $clone17Out = Join-Path $resolvedOutputDir "regression_1p7b_clone.wav"
    $styleWhisperOut = Join-Path $resolvedOutputDir "regression_1p7b_style_whisper.wav"
    $styleAngryOut = Join-Path $resolvedOutputDir "regression_1p7b_style_angry_shout.wav"

    Write-Host ""
    Write-Host "--- CLI 1.7B basic synthesis ---"
    $basic17Res = Invoke-CommandCapture -exe $cliExe -commandArgs @(
        "-m", $resolvedModelDir,
        "--model-name", $ModelName17Custom,
        "--speaker", $Model17Speaker,
        "-t", "Hello world from qwen3 one point seven b.",
        "--temperature", "0",
        "--top-k", "1",
        "--max-tokens", "128",
        "-o", $basic17Out
    )
    if ($basic17Res.ExitCode -eq 0) {
        Validate-WavOutput -testName "CLI 1.7B basic synthesis" -wavPath $basic17Out | Out-Null
    } else {
        Add-Fail "CLI 1.7B basic synthesis (exit code: $($basic17Res.ExitCode))"
        Write-OutputTail -output $basic17Res.Output
    }

    if ($refAudio) {
        Write-Host ""
        Write-Host "--- CLI 1.7B voice cloning ---"
    $clone17Res = Invoke-CommandCapture -exe $cliExe -commandArgs @(
        "-m", $resolvedModelDir,
        "--model-name", $ModelName17Base,
        "-t", "Hello world from cloned voice on one point seven b.",
        "-r", $refAudio,
        "--temperature", "0",
        "--top-k", "1",
            "--max-tokens", "128",
            "-o", $clone17Out
        )
        if ($clone17Res.ExitCode -eq 0) {
            Validate-WavOutput -testName "CLI 1.7B voice cloning" -wavPath $clone17Out | Out-Null
        } else {
            Add-Fail "CLI 1.7B voice cloning (exit code: $($clone17Res.ExitCode))"
            Write-OutputTail -output $clone17Res.Output
        }
    } else {
        Add-Fail "CLI 1.7B voice cloning (reference audio missing)"
    }

    Write-Host ""
    Write-Host "--- CLI 1.7B style: whisper ---"
    $styleWhisperRes = Invoke-CommandCapture -exe $cliExe -commandArgs @(
        "-m", $resolvedModelDir,
        "--model-name", $ModelName17Custom,
        "--speaker", $Model17Speaker,
        "-t", "Please keep this between us.",
        "--instruct", "Whispering, very soft and quiet voice.",
        "--temperature", "0",
        "--top-k", "1",
        "--max-tokens", "128",
        "-o", $styleWhisperOut
    )
    if ($styleWhisperRes.ExitCode -eq 0) {
        Validate-WavOutput -testName "CLI 1.7B style whisper" -wavPath $styleWhisperOut | Out-Null
    } else {
        Add-Fail "CLI 1.7B style whisper (exit code: $($styleWhisperRes.ExitCode))"
        Write-OutputTail -output $styleWhisperRes.Output
    }

    Write-Host ""
    Write-Host "--- CLI 1.7B style: angry shout ---"
    $styleAngryRes = Invoke-CommandCapture -exe $cliExe -commandArgs @(
        "-m", $resolvedModelDir,
        "--model-name", $ModelName17Custom,
        "--speaker", $Model17Speaker,
        "-t", "This is unacceptable. Stop right now.",
        "--instruct", "Angry voice, shouting, high intensity.",
        "--temperature", "0",
        "--top-k", "1",
        "--max-tokens", "128",
        "-o", $styleAngryOut
    )
    if ($styleAngryRes.ExitCode -eq 0) {
        Validate-WavOutput -testName "CLI 1.7B style angry shout" -wavPath $styleAngryOut | Out-Null
    } else {
        Add-Fail "CLI 1.7B style angry shout (exit code: $($styleAngryRes.ExitCode))"
        Write-OutputTail -output $styleAngryRes.Output
    }
}

Write-FinalSummaryAndExit
