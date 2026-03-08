param(
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$ModelDir = "models",
    [string]$ModelName = "qwen3-tts-0.6b-f16.gguf",
    [string]$OutputDir = "benchmark_output\0p6b_f16_cuda",
    [string]$Text = "Hello from the 0.6B F16 CUDA iteration benchmark.",
    [int]$MaxTokens = 128,
    [int]$TopK = 1,
    [double]$Temperature = 0.0,
    [switch]$DisableCudaGraphs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
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

function Get-FirstMatch([string]$text, [string]$pattern, [int]$group = 1) {
    $m = [regex]::Match($text, $pattern)
    if (-not $m.Success) {
        return $null
    }
    return $m.Groups[$group].Value
}

function Parse-Int([string]$s) {
    if ([string]::IsNullOrWhiteSpace($s)) {
        return $null
    }
    $v = 0
    if ([int]::TryParse($s, [ref]$v)) {
        return $v
    }
    return $null
}

function Parse-Double([string]$s) {
    if ([string]::IsNullOrWhiteSpace($s)) {
        return $null
    }
    $v = 0.0
    if ([double]::TryParse(
            $s,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$v)) {
        return $v
    }
    return $null
}

function Parse-Timing([string]$text) {
    $gen = Parse-Int (Get-FirstMatch $text '(?mi)^\s*Code generation:\s*([0-9]+)\s*ms')
    $dec = Parse-Int (Get-FirstMatch $text '(?mi)^\s*Vocoder decode:\s*([0-9]+)\s*ms')
    $tot = Parse-Int (Get-FirstMatch $text '(?mi)^\s*Total:\s*([0-9]+)\s*ms')
    $aud = Parse-Double (Get-FirstMatch $text '(?mi)^\s*Audio duration:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|seconds)')
    $xrt = Parse-Double (Get-FirstMatch $text '(?mi)^\s*Throughput:\s*([0-9]+(?:\.[0-9]+)?)x realtime')
    $rtf = Parse-Double (Get-FirstMatch $text '(?mi)RTF=([0-9]+(?:\.[0-9]+)?)')

    if ($null -eq $rtf -and $null -ne $tot -and $null -ne $aud -and $aud -gt 0) {
        $rtf = ($tot / 1000.0) / $aud
    }
    if ($null -eq $xrt -and $null -ne $rtf -and $rtf -gt 0) {
        $xrt = 1.0 / $rtf
    }

    return [PSCustomObject]@{
        GenerateMs = $gen
        DecodeMs   = $dec
        TotalMs    = $tot
        AudioSec   = $aud
        RTF        = $rtf
        XRealtime  = $xrt
    }
}

function Round-Nullable([object]$value, [int]$digits) {
    if ($null -eq $value) {
        return $null
    }
    return [Math]::Round([double]$value, $digits)
}

function Read-JsonFile([string]$path) {
    if (-not (Test-Path $path)) {
        return $null
    }
    $raw = Get-Content -Path $path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }
    return $raw | ConvertFrom-Json
}

function Show-Delta(
    [string]$label,
    [object]$previous,
    [object]$current,
    [bool]$lowerIsBetter
) {
    if ($null -eq $previous -or $null -eq $current) {
        return
    }

    $prev = [double]$previous
    $cur = [double]$current
    $delta = $cur - $prev
    $pct = if ([Math]::Abs($prev) -gt 1e-9) { ($delta / $prev) * 100.0 } else { $null }
    $improved = if ($lowerIsBetter) { $cur -lt $prev } else { $cur -gt $prev }
    $verdict = if ($improved) { "better" } elseif ($cur -eq $prev) { "same" } else { "worse" }

    if ($null -eq $pct) {
        Write-Host ("  {0}: {1} -> {2} ({3})" -f $label, $prev, $cur, $verdict)
        return
    }

    Write-Host ("  {0}: {1} -> {2} ({3:+0.00;-0.00;0.00}%, {4})" -f $label, $prev, $cur, $pct, $verdict)
}

function Get-GitCommit([string]$repoRoot) {
    try {
        $commit = (& git -C $repoRoot rev-parse --short HEAD 2>$null)
        if ($LASTEXITCODE -eq 0) {
            return (($commit | Out-String).Trim())
        }
    } catch {
    }
    return ""
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repoRoot $BuildDir }
$resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }

$cliExe = Resolve-BinaryPath -name "qwen3-tts-cli" -buildDir $resolvedBuildDir -config $Configuration
if (-not $cliExe) {
    throw "Could not find qwen3-tts-cli.exe in '$resolvedBuildDir'. Build first with .\build.ps1."
}

$modelPath = Join-Path $resolvedModelDir $ModelName
if (-not (Test-Path $modelPath)) {
    throw "Model not found: $modelPath"
}

New-Item -Path $resolvedOutputDir -ItemType Directory -Force | Out-Null
$runsDir = Join-Path $resolvedOutputDir "runs"
New-Item -Path $runsDir -ItemType Directory -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $runsDir $timestamp
New-Item -Path $runDir -ItemType Directory -Force | Out-Null

$wavPath = Join-Path $runDir "out.wav"
$logPath = Join-Path $runDir "run.log"
$latestPath = Join-Path $resolvedOutputDir "latest.json"
$historyPath = Join-Path $resolvedOutputDir "history.csv"
$previous = Read-JsonFile -path $latestPath
$graphsEnabled = -not $DisableCudaGraphs

$args = @(
    "-m", $resolvedModelDir,
    "--model-name", $ModelName,
    "--temperature", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Temperature)),
    "--top-k", "$TopK",
    "--max-tokens", "$MaxTokens",
    "-t", $Text,
    "-o", $wavPath
)

$originalDisableGraphs = [Environment]::GetEnvironmentVariable("GGML_CUDA_DISABLE_GRAPHS", "Process")
try {
    if ($DisableCudaGraphs) {
        $env:GGML_CUDA_DISABLE_GRAPHS = "1"
    } else {
        Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    }

    $commandLine = ('"{0}" {1}' -f $cliExe, (($args | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '))

    $result = Invoke-CommandCapture -exe $cliExe -commandArgs $args
    $timing = Parse-Timing -text $result.Output
} finally {
    if ($null -eq $originalDisableGraphs) {
        Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    } else {
        $env:GGML_CUDA_DISABLE_GRAPHS = $originalDisableGraphs
    }
}

$logBody = @(
    "Timestamp: $timestamp"
    "Command: $commandLine"
    "Graphs: $(if ($graphsEnabled) { "ON" } else { "OFF" })"
    "ExitCode: $($result.ExitCode)"
    ""
    $result.Output
) -join "`r`n"
Set-Content -Path $logPath -Value $logBody -Encoding UTF8

$status = if ($result.ExitCode -eq 0 -and (Test-Path $wavPath)) { "PASS" } else { "FAIL" }
$wavBytes = if (Test-Path $wavPath) { (Get-Item $wavPath).Length } else { 0 }
$commit = Get-GitCommit -repoRoot $repoRoot

$current = [PSCustomObject]([ordered]@{
    Timestamp   = $timestamp
    Commit      = $commit
    Status      = $status
    Graphs      = if ($graphsEnabled) { "ON" } else { "OFF" }
    Model       = $ModelName
    TotalMs     = $timing.TotalMs
    GenerateMs  = $timing.GenerateMs
    DecodeMs    = $timing.DecodeMs
    AudioSec    = Round-Nullable -value $timing.AudioSec -digits 3
    RTF         = Round-Nullable -value $timing.RTF -digits 3
    XRealtime   = Round-Nullable -value $timing.XRealtime -digits 3
    WavBytes    = $wavBytes
    LogPath     = $logPath
    WavPath     = $wavPath
    Text        = $Text
    MaxTokens   = $MaxTokens
    Temperature = $Temperature
    TopK        = $TopK
})

$current | ConvertTo-Json -Depth 6 | Set-Content -Path $latestPath -Encoding UTF8

$historyRow = [PSCustomObject]@{
    Timestamp  = $current.Timestamp
    Commit     = $current.Commit
    Status     = $current.Status
    Graphs     = $current.Graphs
    Model      = $current.Model
    TotalMs    = $current.TotalMs
    GenerateMs = $current.GenerateMs
    DecodeMs   = $current.DecodeMs
    AudioSec   = $current.AudioSec
    RTF        = $current.RTF
    XRealtime  = $current.XRealtime
    WavBytes   = $current.WavBytes
    LogPath    = $current.LogPath
    WavPath    = $current.WavPath
}

if (Test-Path $historyPath) {
    $historyRow | Export-Csv -Path $historyPath -NoTypeInformation -Append -Encoding UTF8
} else {
    $historyRow | Export-Csv -Path $historyPath -NoTypeInformation -Encoding UTF8
}

Write-Host ""
Write-Host "0.6B F16 CUDA Benchmark" -ForegroundColor Cyan
$historyRow | Format-List

if ($null -ne $previous) {
    Write-Host ""
    Write-Host "Compared to previous run" -ForegroundColor Cyan
    Show-Delta -label "TotalMs" -previous $previous.TotalMs -current $current.TotalMs -lowerIsBetter $true
    Show-Delta -label "GenerateMs" -previous $previous.GenerateMs -current $current.GenerateMs -lowerIsBetter $true
    Show-Delta -label "DecodeMs" -previous $previous.DecodeMs -current $current.DecodeMs -lowerIsBetter $true
    Show-Delta -label "RTF" -previous $previous.RTF -current $current.RTF -lowerIsBetter $true
    Show-Delta -label "XRealtime" -previous $previous.XRealtime -current $current.XRealtime -lowerIsBetter $false
    Show-Delta -label "AudioSec" -previous $previous.AudioSec -current $current.AudioSec -lowerIsBetter $false
    Show-Delta -label "WavBytes" -previous $previous.WavBytes -current $current.WavBytes -lowerIsBetter $false
}

Write-Host ""
Write-Host "Latest:  $latestPath" -ForegroundColor Green
Write-Host "History: $historyPath" -ForegroundColor Green
Write-Host "Run dir:  $runDir" -ForegroundColor Green

if ($status -ne "PASS") {
    throw "Benchmark failed. See log: $logPath"
}
