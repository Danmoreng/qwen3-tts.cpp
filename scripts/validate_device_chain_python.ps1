param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$CppExe = "build\qwen3-tts-cli.exe",
    [string]$ModelDir = "$env:USERPROFILE\.qwen-tts-studio\models",
    [ValidateSet("0.6B", "1.7B")]
    [string]$ModelSize = "0.6B",
    [string[]]$CppModels = @(),
    [switch]$IncludePrecisionModels,
    [string]$CodecModel = "$env:USERPROFILE\.qwen-tts-studio\models\qwen-tokenizer-12hz-BF16.gguf",
    [string]$HfModel = "",
    [string]$PythonRepo = "..\Qwen3-TTS",
    [string]$ReferenceAudio = "examples\readme_clone_input.wav",
    [string]$ReferenceTextFile = "reference_text.txt",
    [string]$OutputDir = "benchmark_output\python_device_chain_validation",
    [string]$Text = "Hello. This is a deterministic parity check.",
    [int[]]$Lengths = @(32, 64, 96),
    [string]$PythonDevice = "cuda:0",
    [int]$BenchmarkWarmups = 0,
    [int]$BenchmarkRuns = 0,
    [ValidateSet("speaker-only", "icl")]
    [string]$PromptMode = "speaker-only",
    [ValidateSet("greedy", "topk1")]
    [string]$DecodeMode = "greedy",
    [string]$PythonSpeakerEmbedding = "",
    [string]$PythonReferenceCodes = "",
    [switch]$RequireExactPythonCodes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return Join-Path $repoRoot $PathValue
}

if ([string]::IsNullOrWhiteSpace($HfModel)) {
    $snapshotRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-$ModelSize-Base\snapshots"
    $snapshot = Get-ChildItem -LiteralPath $snapshotRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object -Property `
            @{ Expression = "LastWriteTimeUtc"; Descending = $true }, `
            @{ Expression = "FullName"; Descending = $false } |
        Select-Object -First 1
    if (-not $snapshot) {
        throw "No cached official $ModelSize Hugging Face snapshot found under $snapshotRoot"
    }
    $HfModel = $snapshot.FullName
}

if ($CppModels.Count -eq 0) {
    $CppModels = @("qwen-talker-$($ModelSize.ToLower())-base-Q8_0.gguf")
}

if ($IncludePrecisionModels) {
    if ($ModelSize -eq "0.6B") {
        $CppModels += @(
            "qwen-talker-0.6b-base-F16.gguf",
            "qwen-talker-0.6b-base-F32.gguf"
        )
    } else {
        $CppModels += @(
            "qwen-talker-1.7b-base-BF16.gguf",
            "qwen-talker-1.7b-base-F32.gguf"
        )
    }
    $CppModels = @($CppModels | Select-Object -Unique)
}

$pythonPath = Resolve-RepoPath $PythonExe
$cppPath = Resolve-RepoPath $CppExe
$codecPath = Resolve-RepoPath $CodecModel
$pythonRepoPath = Resolve-RepoPath $PythonRepo
$referenceAudioPath = Resolve-RepoPath $ReferenceAudio
$referenceTextPath = Resolve-RepoPath $ReferenceTextFile
$outputPath = Resolve-RepoPath $OutputDir
$pythonSpeakerPath = if ($PythonSpeakerEmbedding) { Resolve-RepoPath $PythonSpeakerEmbedding } else { "" }
$pythonReferenceCodesPath = if ($PythonReferenceCodes) { Resolve-RepoPath $PythonReferenceCodes } else { "" }

$required = @($pythonPath, $cppPath, $ModelDir, $codecPath, $HfModel, $referenceAudioPath, $referenceTextPath)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing validation input: $path"
    }
}
foreach ($model in $CppModels) {
    $modelPath = Join-Path $ModelDir $model
    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "Missing C++ model: $modelPath"
    }
}

$arguments = @(
    (Join-Path $PSScriptRoot "validate_device_chain_python.py"),
    "--repo-root", $repoRoot,
    "--cpp-cli", $cppPath,
    "--cpp-model-dir", $ModelDir,
    "--codec-model", $codecPath,
    "--hf-model", $HfModel,
    "--python-repo", $pythonRepoPath,
    "--python-device", $PythonDevice,
    "--reference-audio", $referenceAudioPath,
    "--reference-text-file", $referenceTextPath,
    "--output-dir", $outputPath,
    "--text", $Text,
    "--benchmark-warmups", "$BenchmarkWarmups",
    "--benchmark-runs", "$BenchmarkRuns"
    "--prompt-mode", $PromptMode
    "--decode-mode", $DecodeMode
)
if ($pythonSpeakerPath) {
    if (-not (Test-Path -LiteralPath $pythonSpeakerPath)) { throw "Missing Python speaker embedding: $pythonSpeakerPath" }
    $arguments += @("--python-speaker-embedding", $pythonSpeakerPath)
}
if ($pythonReferenceCodesPath) {
    if (-not (Test-Path -LiteralPath $pythonReferenceCodesPath)) { throw "Missing Python reference codes: $pythonReferenceCodesPath" }
    $arguments += @("--python-reference-codes", $pythonReferenceCodesPath)
}
if ($RequireExactPythonCodes) {
    $arguments += "--require-exact-python-codes"
}
foreach ($model in $CppModels) {
    $arguments += @("--cpp-model", $model)
}
foreach ($length in $Lengths) {
    $arguments += @("--length", "$length")
}

Write-Host "Running official Python / C++ device-chain validation"
Write-Host "  Python model: $HfModel"
Write-Host "  Prompt mode:  $PromptMode"
Write-Host "  Decode mode:  $DecodeMode"
Write-Host "  C++ models:   $($CppModels -join ', ')"
Write-Host "  Lengths:      $($Lengths -join ', ')"
Write-Host "  Output:       $outputPath"

& $pythonPath @arguments
exit $LASTEXITCODE
