param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$Output = "",

    [int]$SampleRate = 24000,

    [switch]$Loudnorm
)

$ErrorActionPreference = "Stop"

function Resolve-Ffmpeg {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "ffmpeg was not found in PATH. Install ffmpeg and retry."
    }
    return $cmd.Source
}

$ffmpeg = Resolve-Ffmpeg

if (-not (Test-Path $InputPath)) {
    throw "Input file not found: $InputPath"
}

$inPath = (Resolve-Path $InputPath).Path

if ([string]::IsNullOrWhiteSpace($Output)) {
    $dir = Split-Path -Parent $inPath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($inPath)
    $Output = Join-Path $dir ($stem + "_ref.wav")
}

$outDir = Split-Path -Parent $Output
if (-not [string]::IsNullOrWhiteSpace($outDir) -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$audioFilter = "highpass=f=70,lowpass=f=7600"
if ($Loudnorm) {
    $audioFilter = "$audioFilter,loudnorm=I=-20:LRA=11:TP=-1.5"
}

$args = @(
    "-y",
    "-i", $inPath,
    "-vn",
    "-ac", "1",
    "-ar", "$SampleRate",
    "-c:a", "pcm_s16le",
    "-af", $audioFilter,
    $Output
)

Write-Host "Converting:"
Write-Host "  Input:  $inPath"
Write-Host "  Output: $Output"
Write-Host "  Format: mono, ${SampleRate}Hz, PCM s16le WAV"

& $ffmpeg @args
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg conversion failed with exit code $LASTEXITCODE"
}

Write-Host "Done."
Write-Host "Use with:"
Write-Host "  .\build\Release\qwen3-tts-cli.exe -m models -t `"Your text`" -r `"$Output`" -o out.wav"
