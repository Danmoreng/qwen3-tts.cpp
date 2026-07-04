param(
    [string]$PythonModel = "",
    [string]$CppModelDir = "benchmark_output\bf16_parity_modeldir",
    [string]$SpeakerEmbedding = "benchmark_output\python_parity\python_speaker_embedding.json",
    [string]$ReferenceText = "",
    [string]$ReferenceTextFile = "",
    [string]$ReferenceCodes = "",
    [string]$CliExe = "",
    [string]$PythonExe = "",
    [string]$PythonPath = "",
    [string]$OutputDir = "benchmark_output\python_parity\speaker_fixture",
    [string]$Text = "This is a short parity check for speaker embedding voice cloning.",
    [string]$PythonLanguage = "English",
    [string]$CppLanguage = "en",
    [ValidateSet("cpu", "cuda")]
    [string]$PythonDevice = "cpu",
    [ValidateSet("float32", "bfloat16", "float16")]
    [string]$PythonDType = "float32",
    [int]$MaxTokens = 12,
    [int]$MaxFrames = 10,
    [switch]$DoSample = $true,
    [double]$ExpectMatchPercentAtLeast = -1.0,
    [int]$ExpectFirstDiffFrame = -1,
    [int]$ExpectFirstDiffCodebook = -1,
    [int]$ExpectFirstDiffTokenA = -1,
    [int]$ExpectFirstDiffTokenB = -1,
    [double]$ExpectFirstDiffCosineAtLeast = -1.0,
    [double]$ExpectFirstDiffMaxAbsAtMost = -1.0,
    [ValidateSet("", "exact_tie", "near_tie_token_swap", "near_tie", "token_swap", "logit_drift")]
    [string]$ExpectFirstDiffCategory = "",
    [double]$ExpectFirstDiffMaxAbsOverMarginAtLeast = -1.0,
    [switch]$TalkerKvCacheF32,
    [switch]$TalkerKvCacheF16,
    [switch]$RequireAssets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-RepoPath([string]$path) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($path)) {
        return [System.IO.Path]::GetFullPath($path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\$path"))
}

function Resolve-FirstExisting([string[]]$paths) {
    foreach ($path in $paths) {
        if ([string]::IsNullOrWhiteSpace($path)) {
            continue
        }
        $candidate = Resolve-RepoPath $path
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return ""
}

function Resolve-PythonExe([string]$requested) {
    if (-not [string]::IsNullOrWhiteSpace($requested)) {
        return (Resolve-Path -LiteralPath (Resolve-RepoPath $requested)).Path
    }

    $venvPython = Resolve-RepoPath ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }

    return "python"
}

function Resolve-CliExe([string]$requested) {
    if (-not [string]::IsNullOrWhiteSpace($requested)) {
        return (Resolve-Path -LiteralPath (Resolve-RepoPath $requested)).Path
    }

    return Resolve-FirstExisting @(
        "build-timing-current\qwen3-tts-cli.exe",
        "build\Release\qwen3-tts-cli.exe",
        "build\qwen3-tts-cli.exe"
    )
}

function Find-DefaultPythonModel() {
    if (-not [string]::IsNullOrWhiteSpace($env:QWEN3_TTS_PYTHON_MODEL)) {
        return $env:QWEN3_TTS_PYTHON_MODEL
    }

    return Resolve-FirstExisting @(
        "models\Qwen3-TTS-12Hz-1.7B-Base",
        "..\audio.cpp\models\Qwen3-TTS-12Hz-1.7B-Base",
        "C:\Development\Qwen3TTSDev\audio.cpp\models\Qwen3-TTS-12Hz-1.7B-Base"
    )
}

function Save-DebugDumpEnv() {
    return [PSCustomObject]@{
        Dir = $env:QWEN3_TTS_DEBUG_DUMP_DIR
        MaxFrames = $env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES
        MaxCodeSteps = $env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS
    }
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

function Save-TalkerKvCacheEnv() {
    return [PSCustomObject]@{
        TalkerKvCacheF32 = $env:QWEN3_TTS_TALKER_KV_F32
        TalkerKvCacheF16 = $env:QWEN3_TTS_TALKER_KV_F16
    }
}

function Restore-TalkerKvCacheEnv([object]$snapshot) {
    if ($null -eq $snapshot) {
        return
    }

    if ($null -ne $snapshot.TalkerKvCacheF32) {
        $env:QWEN3_TTS_TALKER_KV_F32 = $snapshot.TalkerKvCacheF32
    } else {
        Remove-Item Env:QWEN3_TTS_TALKER_KV_F32 -ErrorAction SilentlyContinue
    }

    if ($null -ne $snapshot.TalkerKvCacheF16) {
        $env:QWEN3_TTS_TALKER_KV_F16 = $snapshot.TalkerKvCacheF16
    } else {
        Remove-Item Env:QWEN3_TTS_TALKER_KV_F16 -ErrorAction SilentlyContinue
    }
}

function Invoke-Checked([string]$name, [string]$exe, [string[]]$commandArgs, [string]$logPath) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $exe @commandArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $text = (($output | ForEach-Object { $_.ToString() }) | Out-String).TrimEnd()
    if (-not [string]::IsNullOrWhiteSpace($logPath)) {
        [System.IO.File]::WriteAllText($logPath, $text + "`n")
    }
    if ($exitCode -ne 0) {
        Write-Host $text
        throw "$name failed with exit code $exitCode"
    }
    if (-not [string]::IsNullOrWhiteSpace($text)) {
        $lines = $text -split "`r?`n"
        $start = [Math]::Max(0, $lines.Length - 12)
        for ($i = $start; $i -lt $lines.Length; $i++) {
            Write-Host $lines[$i]
        }
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExeResolved = Resolve-PythonExe $PythonExe
$cliExeResolved = Resolve-CliExe $CliExe
$pythonModelResolved = if ([string]::IsNullOrWhiteSpace($PythonModel)) { Find-DefaultPythonModel } else { Resolve-RepoPath $PythonModel }
$cppModelDirResolved = Resolve-RepoPath $CppModelDir
$speakerEmbeddingResolved = Resolve-RepoPath $SpeakerEmbedding
$referenceTextFileResolved = Resolve-RepoPath $ReferenceTextFile
$referenceCodesResolved = Resolve-RepoPath $ReferenceCodes
$outputDirResolved = Resolve-RepoPath $OutputDir
$isIclFixture = -not [string]::IsNullOrWhiteSpace($ReferenceText) -or
    -not [string]::IsNullOrWhiteSpace($ReferenceTextFile) -or
    -not [string]::IsNullOrWhiteSpace($ReferenceCodes)
$talkerKvCacheF32Enabled = -not [bool]$TalkerKvCacheF16
if ($TalkerKvCacheF32 -and $TalkerKvCacheF16) {
    throw "-TalkerKvCacheF32 and -TalkerKvCacheF16 cannot be combined."
}

$missing = @()
if ([string]::IsNullOrWhiteSpace($pythonModelResolved) -or -not (Test-Path -LiteralPath $pythonModelResolved)) { $missing += "PythonModel" }
if ([string]::IsNullOrWhiteSpace($cliExeResolved) -or -not (Test-Path -LiteralPath $cliExeResolved)) { $missing += "CliExe" }
if (-not (Test-Path -LiteralPath $cppModelDirResolved)) { $missing += "CppModelDir" }
if (-not (Test-Path -LiteralPath $speakerEmbeddingResolved)) { $missing += "SpeakerEmbedding" }
if ($isIclFixture) {
    if ([string]::IsNullOrWhiteSpace($ReferenceCodes) -or -not (Test-Path -LiteralPath $referenceCodesResolved)) { $missing += "ReferenceCodes" }
    if ([string]::IsNullOrWhiteSpace($ReferenceText) -and
        ([string]::IsNullOrWhiteSpace($ReferenceTextFile) -or -not (Test-Path -LiteralPath $referenceTextFileResolved))) {
        $missing += "ReferenceText"
    }
}

if ($missing.Count -gt 0) {
    $message = "Missing parity fixture assets: $($missing -join ', ')."
    if ($RequireAssets) {
        throw $message
    }
    Write-Host "[SKIP] $message" -ForegroundColor Yellow
    Write-Host "       Use -PythonModel, -CppModelDir, -SpeakerEmbedding, and -CliExe as needed."
    exit 0
}

$pythonModelResolved = (Resolve-Path -LiteralPath $pythonModelResolved).Path
$cppModelDirResolved = (Resolve-Path -LiteralPath $cppModelDirResolved).Path
$speakerEmbeddingResolved = (Resolve-Path -LiteralPath $speakerEmbeddingResolved).Path
if (-not [string]::IsNullOrWhiteSpace($ReferenceCodes)) {
    $referenceCodesResolved = (Resolve-Path -LiteralPath $referenceCodesResolved).Path
}
if (-not [string]::IsNullOrWhiteSpace($ReferenceTextFile)) {
    $referenceTextFileResolved = (Resolve-Path -LiteralPath $referenceTextFileResolved).Path
}
New-Item -ItemType Directory -Force -Path $outputDirResolved | Out-Null
$outputDirResolved = (Resolve-Path -LiteralPath $outputDirResolved).Path

$fixtureMode = if ($isIclFixture) { "icl" } else { "speaker" }
$pyTrace = Join-Path $outputDirResolved "python_$PythonDType"
$cppTrace = Join-Path $outputDirResolved "cpp"
$summaryPath = Join-Path $outputDirResolved "summary_python_${PythonDType}_vs_cpp.json"
$metadataPath = Join-Path $outputDirResolved "fixture_metadata.json"
Remove-Item -Recurse -Force $pyTrace, $cppTrace -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $pyTrace, $cppTrace | Out-Null

$fixtureMetadata = [PSCustomObject]@{
    SchemaVersion = 1
    FixtureMode = $fixtureMode
    Python = [PSCustomObject]@{
        Exe = $pythonExeResolved
        Model = $pythonModelResolved
        Path = $PythonPath
        Device = $PythonDevice
        DType = $PythonDType
        Language = $PythonLanguage
        DoSample = [bool]$DoSample
    }
    Cpp = [PSCustomObject]@{
        CliExe = $cliExeResolved
        ModelDir = $cppModelDirResolved
        Language = $CppLanguage
        Temperature = 1.0
        TopK = 1
        TopP = 1.0
        Seed = 0
        TalkerKvCacheF32 = [bool]$talkerKvCacheF32Enabled
    }
    Inputs = [PSCustomObject]@{
        Text = $Text
        SpeakerEmbedding = $speakerEmbeddingResolved
        ReferenceText = if ([string]::IsNullOrWhiteSpace($ReferenceText)) { $null } else { $ReferenceText }
        ReferenceTextFile = if ([string]::IsNullOrWhiteSpace($ReferenceTextFile)) { $null } else { $referenceTextFileResolved }
        ReferenceCodes = if ([string]::IsNullOrWhiteSpace($ReferenceCodes)) { $null } else { $referenceCodesResolved }
        MaxTokens = $MaxTokens
        MaxFrames = $MaxFrames
    }
    Expectations = [PSCustomObject]@{
        MatchPercentAtLeast = if ($ExpectMatchPercentAtLeast -ge 0.0) { $ExpectMatchPercentAtLeast } else { $null }
        FirstDiffFrame = if ($ExpectFirstDiffFrame -ge 0) { $ExpectFirstDiffFrame } else { $null }
        FirstDiffCodebook = if ($ExpectFirstDiffCodebook -ge 0) { $ExpectFirstDiffCodebook } else { $null }
        FirstDiffTokenA = if ($ExpectFirstDiffTokenA -ge 0) { $ExpectFirstDiffTokenA } else { $null }
        FirstDiffTokenB = if ($ExpectFirstDiffTokenB -ge 0) { $ExpectFirstDiffTokenB } else { $null }
        FirstDiffCosineAtLeast = if ($ExpectFirstDiffCosineAtLeast -ge 0.0) { $ExpectFirstDiffCosineAtLeast } else { $null }
        FirstDiffMaxAbsAtMost = if ($ExpectFirstDiffMaxAbsAtMost -ge 0.0) { $ExpectFirstDiffMaxAbsAtMost } else { $null }
        FirstDiffCategory = if ([string]::IsNullOrWhiteSpace($ExpectFirstDiffCategory)) { $null } else { $ExpectFirstDiffCategory }
        FirstDiffMaxAbsOverMarginAtLeast = if ($ExpectFirstDiffMaxAbsOverMarginAtLeast -ge 0.0) { $ExpectFirstDiffMaxAbsOverMarginAtLeast } else { $null }
    }
    Outputs = [PSCustomObject]@{
        PythonTraceDir = $pyTrace
        CppTraceDir = $cppTrace
        Summary = $summaryPath
    }
}
$fixtureMetadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host "$fixtureMode parity fixture" -ForegroundColor Cyan
Write-Host "  Python model:     $pythonModelResolved"
Write-Host "  C++ model dir:    $cppModelDirResolved"
Write-Host "  Speaker embed:    $speakerEmbeddingResolved"
if ($isIclFixture) {
    if (-not [string]::IsNullOrWhiteSpace($ReferenceTextFile)) {
        Write-Host "  Reference text:   $referenceTextFileResolved"
    } else {
        Write-Host "  Reference text:   <inline>"
    }
    Write-Host "  Reference codes:  $referenceCodesResolved"
}
Write-Host "  CLI:              $cliExeResolved"
if ($talkerKvCacheF32Enabled) {
    Write-Host "  Talker KV cache:  F32 (default)"
} else {
    Write-Host "  Talker KV cache:  F16 (QWEN3_TTS_TALKER_KV_F16=1)"
}
Write-Host "  Output dir:       $outputDirResolved"
Write-Host "  Metadata:         $metadataPath"

$previousPythonPath = $env:PYTHONPATH
if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    $env:PYTHONPATH = (Resolve-Path -LiteralPath (Resolve-RepoPath $PythonPath)).Path
}
$debugSnapshot = Save-DebugDumpEnv
$talkerKvCacheSnapshot = Save-TalkerKvCacheEnv

try {
    $dumpArgs = @(
        (Join-Path $repoRoot "scripts\dump_python_trace.py"),
        "--model", $pythonModelResolved,
        "--speaker-embedding", $speakerEmbeddingResolved,
        "--text", $Text,
        "--language", $PythonLanguage,
        "--trace-dir", $pyTrace,
        "--max-new-tokens", "$MaxTokens",
        "--max-frames", "$MaxFrames",
        "--device", $PythonDevice,
        "--dtype", $PythonDType
    )
    if ($isIclFixture) {
        if (-not [string]::IsNullOrWhiteSpace($ReferenceTextFile)) {
            $dumpArgs += @("--reference-text-file", $referenceTextFileResolved)
        } else {
            $dumpArgs += @("--reference-text", $ReferenceText)
        }
        $dumpArgs += @("--reference-codes", $referenceCodesResolved)
    }
    if ($DoSample) {
        $dumpArgs += "--do-sample"
    }
    Invoke-Checked "Python trace" $pythonExeResolved $dumpArgs (Join-Path $outputDirResolved "python_trace.log")

    $env:QWEN3_TTS_DEBUG_DUMP_DIR = $cppTrace
    $env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES = "$MaxFrames"
    $env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS = "15"
    Remove-Item Env:QWEN3_TTS_TALKER_KV_F32 -ErrorAction SilentlyContinue
    if ($talkerKvCacheF32Enabled) {
        Remove-Item Env:QWEN3_TTS_TALKER_KV_F16 -ErrorAction SilentlyContinue
    } else {
        $env:QWEN3_TTS_TALKER_KV_F16 = "1"
    }
    $cppArgs = @(
        "--model", $cppModelDirResolved,
        "--text", $Text,
        "--speaker-embedding", $speakerEmbeddingResolved,
        "--language", $CppLanguage,
        "--temperature", "1.0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "0",
        "--max-tokens", "$MaxTokens",
        "--dump-generated-codes", (Join-Path $cppTrace "cpp_codes.json"),
        "--output", (Join-Path $cppTrace "cpp.wav")
    )
    if ($isIclFixture) {
        if (-not [string]::IsNullOrWhiteSpace($ReferenceTextFile)) {
            $cppArgs += @("--reference-text-file", $referenceTextFileResolved)
        } else {
            $cppArgs += @("--reference-text", $ReferenceText)
        }
        $cppArgs += @("--reference-codes", $referenceCodesResolved)
    }
    Invoke-Checked "C++ trace" $cliExeResolved $cppArgs (Join-Path $outputDirResolved "cpp_trace.log")

    Restore-DebugDumpEnv $debugSnapshot
    $debugSnapshot = $null

    $summaryArgs = @(
        (Join-Path $repoRoot "scripts\parity_trace_summary.py"),
        "--trace-a", $pyTrace,
        "--trace-b", $cppTrace,
        "--label-a", "python_${fixtureMode}_$PythonDType",
        "--label-b", "cpp_$fixtureMode",
        "--output", $summaryPath
    )
    if ($ExpectMatchPercentAtLeast -ge 0.0) {
        $summaryArgs += @("--expect-match-percent-at-least", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ExpectMatchPercentAtLeast)))
    }
    if ($ExpectFirstDiffFrame -ge 0) {
        $summaryArgs += @("--expect-first-diff-frame", "$ExpectFirstDiffFrame")
    }
    if ($ExpectFirstDiffCodebook -ge 0) {
        $summaryArgs += @("--expect-first-diff-codebook", "$ExpectFirstDiffCodebook")
    }
    if ($ExpectFirstDiffTokenA -ge 0) {
        $summaryArgs += @("--expect-first-diff-token-a", "$ExpectFirstDiffTokenA")
    }
    if ($ExpectFirstDiffTokenB -ge 0) {
        $summaryArgs += @("--expect-first-diff-token-b", "$ExpectFirstDiffTokenB")
    }
    if ($ExpectFirstDiffCosineAtLeast -ge 0.0) {
        $summaryArgs += @("--expect-first-diff-cosine-at-least", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ExpectFirstDiffCosineAtLeast)))
    }
    if ($ExpectFirstDiffMaxAbsAtMost -ge 0.0) {
        $summaryArgs += @("--expect-first-diff-max-abs-at-most", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ExpectFirstDiffMaxAbsAtMost)))
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectFirstDiffCategory)) {
        $summaryArgs += @("--expect-first-diff-category", $ExpectFirstDiffCategory)
    }
    if ($ExpectFirstDiffMaxAbsOverMarginAtLeast -ge 0.0) {
        $summaryArgs += @("--expect-first-diff-max-abs-over-margin-at-least", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ExpectFirstDiffMaxAbsOverMarginAtLeast)))
    }
    Invoke-Checked "Trace summary" $pythonExeResolved $summaryArgs (Join-Path $outputDirResolved "summary.log")

    Write-Host ""
    Write-Host "Summary written to: $summaryPath" -ForegroundColor Green
} finally {
    Restore-DebugDumpEnv $debugSnapshot
    Restore-TalkerKvCacheEnv $talkerKvCacheSnapshot
    if ($null -ne $previousPythonPath) {
        $env:PYTHONPATH = $previousPythonPath
    } else {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
}
