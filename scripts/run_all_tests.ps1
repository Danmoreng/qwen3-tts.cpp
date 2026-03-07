param(
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$ModelDir = "models",
    [string]$ReferenceAudio = "examples/readme_clone_input.wav",
    [string]$OutputDir = "test_output",
    [switch]$BuildFirst,
    [switch]$BuildMissingTargets,
    [switch]$RequireComponentTests
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

function Resolve-BinaryPath([string]$name, [string]$buildDir, [string]$config) {
    $candidates = @(
        (Join-Path $buildDir "$config\$name.exe"),
        (Join-Path $buildDir "$name.exe"),
        (Join-Path $buildDir "bin\$config\$name.exe"),
        (Join-Path $buildDir "bin\$name.exe")
    )
    return Find-FirstExisting $candidates
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repoRoot $BuildDir }
$resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
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
        Write-Host "Some binaries are missing. Attempting to build ALL_BUILD target ..." -ForegroundColor Yellow
        & cmake --build $resolvedBuildDir --config $Configuration --target ALL_BUILD --parallel
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

$ttsModel = Join-Path $resolvedModelDir "qwen3-tts-0.6b-f16.gguf"
$tokModel = Join-Path $resolvedModelDir "qwen3-tts-tokenizer-f16.gguf"
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

Write-Section "Section 1: Component Tests"

if ($tokenizerExe -and (Test-Path $ttsModel)) {
    Invoke-CheckedTest -name "Tokenizer" -exe $tokenizerExe -commandArgs @("--model", $ttsModel) -successRegex "All tests passed"
} else {
    if ($RequireComponentTests) { Add-Fail "Tokenizer (binary or model missing)" } else { Add-Skip "Tokenizer (binary or model missing)" }
}

if ($encoderExe -and (Test-Path $ttsModel) -and $refAudio) {
    $encoderArgs = @("--tokenizer", $ttsModel, "--audio", $refAudio)
    if ($encoderRef) {
        $encoderArgs += @("--reference", $encoderRef)
    }
    Invoke-CheckedTest -name "Encoder" -exe $encoderExe -commandArgs $encoderArgs -successRegex "All tests passed"
} else {
    if ($RequireComponentTests) { Add-Fail "Encoder (binary/model/reference audio missing)" } else { Add-Skip "Encoder (binary/model/reference audio missing)" }
}

$transformerReq = @(
    (Join-Path $repoRoot "reference/det_text_tokens.bin"),
    (Join-Path $repoRoot "reference/det_speaker_embedding.bin"),
    (Join-Path $repoRoot "reference/det_prefill_embedding.bin"),
    (Join-Path $repoRoot "reference/det_first_frame_logits.bin"),
    (Join-Path $repoRoot "reference/det_speech_codes.bin")
)
$hasTransformerRefs = $true
foreach ($f in $transformerReq) {
    if (-not (Test-Path $f)) {
        $hasTransformerRefs = $false
        break
    }
}

if ($transformerExe -and (Test-Path $ttsModel) -and $hasTransformerRefs) {
    Invoke-CheckedTest -name "Transformer deterministic" `
        -exe $transformerExe `
        -commandArgs @("--model", $ttsModel, "--ref-dir", (Join-Path $repoRoot "reference")) `
        -successRegex "=== All tests passed|=== All tests passed with warnings ===" `
        -forbiddenRegex "FAIL:"
} else {
    if ($RequireComponentTests) { Add-Fail "Transformer deterministic (binary/model/reference artifacts missing)" } else { Add-Skip "Transformer deterministic (binary/model/reference artifacts missing)" }
}

if ($decoderExe -and (Test-Path $tokModel) -and $decoderCodes) {
    $decoderArgs = @("--tokenizer", $tokModel, "--codes", $decoderCodes)
    if ($decoderRef) {
        $decoderArgs += @("--reference", $decoderRef)
    }
    Invoke-CheckedTest -name "Decoder" `
        -exe $decoderExe `
        -commandArgs $decoderArgs `
        -successRegex "All tests completed|PASS: Decoded" `
        -forbiddenRegex "FAIL:"
} else {
    if ($RequireComponentTests) { Add-Fail "Decoder (binary/model/codes missing)" } else { Add-Skip "Decoder (binary/model/codes missing)" }
}

Write-Section "Section 2: CLI Output Regression Checks"

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
        "-o", $basicOut,
        "--temperature", "0",
        "--top-k", "0",
        "--top-p", "1.0",
        "--repetition-penalty", "1.0",
        "--max-tokens", "96"
    )
    if ($basicRes.ExitCode -eq 0) {
        Validate-WavOutput -testName "CLI basic synthesis" -wavPath $basicOut -minRms 0.000001 | Out-Null
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
            "-o", $cloneOut,
            "--temperature", "0",
            "--top-k", "0",
            "--top-p", "1.0",
            "--repetition-penalty", "1.0",
            "--max-tokens", "96"
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
}

Write-Section "Summary"
Write-Host "PASS: $PASS_COUNT"
Write-Host "FAIL: $FAIL_COUNT"
Write-Host "SKIP: $SKIP_COUNT"
Write-Host "Outputs: $resolvedOutputDir"

if ($FAIL_COUNT -gt 0) {
    exit 1
}

exit 0
