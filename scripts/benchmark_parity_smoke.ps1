param(
    [string]$CliExe = "",
    [string]$ModelDir = "benchmark_output\bf16_parity_modeldir",
    [string]$SpeakerEmbedding = "benchmark_output\python_parity\python_speaker_embedding.json",
    [string]$Text = "This is a short parity check for speaker embedding voice cloning.",
    [string]$OutputDir = "benchmark_output\perf_parity_smoke",
    [string]$Language = "en",
    [int]$MaxTokens = 64,
    [int]$Repeat = 4,
    [double]$Temperature = 1.0,
    [int]$TopK = 1,
    [double]$TopP = 1.0,
    [int]$Seed = 0,
    [string]$BaselineSummary = "",
    [double]$MaxGenerateRegressionPercent = -1.0,
    [double]$MaxPipelineRegressionPercent = -1.0,
    [double]$MaxRtfRegressionPercent = -1.0,
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

function ConvertFrom-InvariantDouble([string]$value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}

function ConvertTo-InvariantText([object]$value) {
    return [System.Convert]::ToString($value, [Globalization.CultureInfo]::InvariantCulture)
}

function Get-Median([double[]]$values) {
    if ($values.Count -eq 0) {
        return $null
    }
    $sorted = @($values | Sort-Object)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[[int]($sorted.Count / 2)]
    }
    return [double](($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2.0)
}

function Get-ObjectNumber([object]$obj, [string]$name) {
    if ($null -eq $obj) {
        return $null
    }
    $property = $obj.PSObject.Properties[$name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return $null
    }
    return [double]$property.Value
}

function New-DeltaMetric([string]$name, [object]$current, [object]$baseline) {
    $currentNumber = $null
    $baselineNumber = $null
    if ($null -ne $current) {
        $currentNumber = [double]$current
    }
    if ($null -ne $baseline) {
        $baselineNumber = [double]$baseline
    }

    $delta = $null
    $deltaPercent = $null
    if ($null -ne $currentNumber -and $null -ne $baselineNumber) {
        $delta = [double]($currentNumber - $baselineNumber)
        if ([Math]::Abs($baselineNumber) -gt 0.0) {
            $deltaPercent = [double](($delta / $baselineNumber) * 100.0)
        }
    }

    return [PSCustomObject]@{
        Name = $name
        Current = $currentNumber
        Baseline = $baselineNumber
        Delta = $delta
        DeltaPercent = $deltaPercent
    }
}

function Add-RegressionFailure(
    [System.Collections.Generic.List[string]]$failures,
    [object]$metric,
    [double]$thresholdPercent
) {
    if ($thresholdPercent -lt 0.0 -or $null -eq $metric -or $null -eq $metric.DeltaPercent) {
        return
    }
    if ($metric.DeltaPercent -gt $thresholdPercent) {
        $failures.Add(("{0} regressed by {1}% (threshold {2}%)" -f `
            $metric.Name,
            (([double]$metric.DeltaPercent).ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)),
            ($thresholdPercent.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture))))
    }
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

function Disable-DebugDumpEnv() {
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES -ErrorAction SilentlyContinue
    Remove-Item Env:QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS -ErrorAction SilentlyContinue
}

function Get-GpuSnapshot() {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return $null
    }

    $args = @(
        "--query-gpu=name,temperature.gpu,power.draw,clocks.gr,clocks.mem,utilization.gpu",
        "--format=csv,noheader,nounits"
    )
    $output = & $nvidiaSmi.Source @args 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
        return $null
    }

    $parts = ($output | Select-Object -First 1).ToString().Split(",") | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 6) {
        return [PSCustomObject]@{ Raw = ($output | Out-String).Trim() }
    }

    return [PSCustomObject]@{
        Name = $parts[0]
        TemperatureC = ConvertFrom-InvariantDouble $parts[1]
        PowerW = ConvertFrom-InvariantDouble $parts[2]
        GraphicsClockMHz = ConvertFrom-InvariantDouble $parts[3]
        MemoryClockMHz = ConvertFrom-InvariantDouble $parts[4]
        UtilizationPercent = ConvertFrom-InvariantDouble $parts[5]
    }
}

function Parse-TimingLog([string]$logPath) {
    $records = @()
    $current = $null
    foreach ($line in (Get-Content -LiteralPath $logPath)) {
        if ($line -match '^Repeat (\d+)/(\d+)') {
            if ($null -ne $current) {
                $records += [PSCustomObject]$current
            }
            $current = @{ Repeat = [int]$matches[1] }
        } elseif ($null -ne $current -and $line -match '^\s*Talker forward_step') {
            $current.InTalker = $true
            $current.InCodePred = $false
        } elseif ($null -ne $current -and $line -match '^\s*Code predictor') {
            $current.InCodePred = $true
            $current.InTalker = $false
        } elseif ($null -ne $current -and $line -match '^\s*Embed lookups') {
            $current.InTalker = $false
            $current.InCodePred = $false
        } elseif ($null -ne $current -and $line -match '^\s*Total generate:\s*([0-9.]+) ms') {
            $current.GenerateMs = ConvertFrom-InvariantDouble $matches[1]
        } elseif ($null -ne $current -and $line -match '^\s*Total:\s*([0-9.]+) ms' -and $current.ContainsKey("InTalker") -and $current.InTalker) {
            $current.TalkerMs = ConvertFrom-InvariantDouble $matches[1]
        } elseif ($null -ne $current -and $line -match '^\s*Total:\s*([0-9.]+) ms' -and $current.ContainsKey("InCodePred") -and $current.InCodePred) {
            $current.CodePredMs = ConvertFrom-InvariantDouble $matches[1]
        } elseif ($null -ne $current -and $line -match '^\s*Total:\s*([0-9.]+) ms\s*$' -and -not $current.ContainsKey("PipelineTotalMs")) {
            $current.PipelineTotalMs = ConvertFrom-InvariantDouble $matches[1]
        } elseif ($null -ne $current -and $line -match 'RTF=([0-9.]+)') {
            $current.RTF = ConvertFrom-InvariantDouble $matches[1]
        }
    }
    if ($null -ne $current) {
        $records += [PSCustomObject]$current
    }
    return @($records)
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cliExeResolved = Resolve-CliExe $CliExe
$modelDirResolved = Resolve-RepoPath $ModelDir
$speakerEmbeddingResolved = Resolve-RepoPath $SpeakerEmbedding
$outputDirResolved = Resolve-RepoPath $OutputDir
$baselineSummaryResolved = Resolve-RepoPath $BaselineSummary

$missing = @()
if ([string]::IsNullOrWhiteSpace($cliExeResolved) -or -not (Test-Path -LiteralPath $cliExeResolved)) { $missing += "CliExe" }
if (-not (Test-Path -LiteralPath $modelDirResolved)) { $missing += "ModelDir" }
if (-not (Test-Path -LiteralPath $speakerEmbeddingResolved)) { $missing += "SpeakerEmbedding" }
if (-not [string]::IsNullOrWhiteSpace($baselineSummaryResolved) -and -not (Test-Path -LiteralPath $baselineSummaryResolved)) {
    $missing += "BaselineSummary"
}

if ($missing.Count -gt 0) {
    $message = "Missing parity benchmark assets: $($missing -join ', ')."
    if ($RequireAssets) {
        throw $message
    }
    Write-Host "[SKIP] $message" -ForegroundColor Yellow
    exit 0
}

$cliExeResolved = (Resolve-Path -LiteralPath $cliExeResolved).Path
$modelDirResolved = (Resolve-Path -LiteralPath $modelDirResolved).Path
$speakerEmbeddingResolved = (Resolve-Path -LiteralPath $speakerEmbeddingResolved).Path
$baselineSummaryObj = $null
if (-not [string]::IsNullOrWhiteSpace($baselineSummaryResolved)) {
    $baselineSummaryResolved = (Resolve-Path -LiteralPath $baselineSummaryResolved).Path
    $baselineSummaryObj = Get-Content -LiteralPath $baselineSummaryResolved -Raw | ConvertFrom-Json
}
New-Item -ItemType Directory -Force -Path $outputDirResolved | Out-Null
$outputDirResolved = (Resolve-Path -LiteralPath $outputDirResolved).Path

$logPath = Join-Path $outputDirResolved "speaker_repeat${Repeat}.log"
$summaryPath = Join-Path $outputDirResolved "summary.json"
$wavPath = Join-Path $outputDirResolved "speaker.wav"

Write-Host "Parity timing smoke" -ForegroundColor Cyan
Write-Host "  CLI:            $cliExeResolved"
Write-Host "  Model dir:      $modelDirResolved"
Write-Host "  Speaker embed:  $speakerEmbeddingResolved"
Write-Host "  Output dir:     $outputDirResolved"
Write-Host "  Repeat:         $Repeat"
if ($null -ne $baselineSummaryObj) {
    Write-Host "  Baseline:       $baselineSummaryResolved"
}

$debugSnapshot = Save-DebugDumpEnv
Disable-DebugDumpEnv
$gpuBefore = Get-GpuSnapshot

try {
    $cliArgs = @(
        "--model", $modelDirResolved,
        "--text", $Text,
        "--speaker-embedding", $speakerEmbeddingResolved,
        "--language", $Language,
        "--temperature", (ConvertTo-InvariantText $Temperature),
        "--top-k", "$TopK",
        "--top-p", (ConvertTo-InvariantText $TopP),
        "--seed", "$Seed",
        "--max-tokens", "$MaxTokens",
        "--repeat", "$Repeat",
        "--output", $wavPath
    )

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $cliExeResolved @cliArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $textOut = (($output | ForEach-Object { $_.ToString() }) | Out-String).TrimEnd()
    [System.IO.File]::WriteAllText($logPath, $textOut + "`n")
    if ($exitCode -ne 0) {
        Write-Host $textOut
        throw "qwen3-tts-cli failed with exit code $exitCode"
    }
} finally {
    Restore-DebugDumpEnv $debugSnapshot
}

$gpuAfter = Get-GpuSnapshot
$records = @(Parse-TimingLog $logPath | Select-Object Repeat, GenerateMs, TalkerMs, CodePredMs, PipelineTotalMs, RTF)
$warmRecords = @($records | Where-Object { $_.Repeat -ge 2 })
if ($warmRecords.Count -eq 0) {
    $warmRecords = $records
}

$warmGenerateMedianMs = Get-Median ([double[]]@($warmRecords | ForEach-Object { $_.GenerateMs }))
$warmTalkerMedianMs = Get-Median ([double[]]@($warmRecords | ForEach-Object { $_.TalkerMs }))
$warmCodePredMedianMs = Get-Median ([double[]]@($warmRecords | ForEach-Object { $_.CodePredMs }))
$warmPipelineTotalMedianMs = Get-Median ([double[]]@($warmRecords | ForEach-Object { $_.PipelineTotalMs }))
$warmRtfMedian = Get-Median ([double[]]@($warmRecords | ForEach-Object { $_.RTF }))
$baselineComparison = $null
$regressionFailures = [System.Collections.Generic.List[string]]::new()

if ($null -ne $baselineSummaryObj) {
    $metricsList = [System.Collections.Generic.List[object]]::new()
    [void]$metricsList.Add((New-DeltaMetric "WarmGenerateMedianMs" $warmGenerateMedianMs (Get-ObjectNumber $baselineSummaryObj "WarmGenerateMedianMs")))
    [void]$metricsList.Add((New-DeltaMetric "WarmTalkerMedianMs" $warmTalkerMedianMs (Get-ObjectNumber $baselineSummaryObj "WarmTalkerMedianMs")))
    [void]$metricsList.Add((New-DeltaMetric "WarmCodePredMedianMs" $warmCodePredMedianMs (Get-ObjectNumber $baselineSummaryObj "WarmCodePredMedianMs")))
    [void]$metricsList.Add((New-DeltaMetric "WarmPipelineTotalMedianMs" $warmPipelineTotalMedianMs (Get-ObjectNumber $baselineSummaryObj "WarmPipelineTotalMedianMs")))
    [void]$metricsList.Add((New-DeltaMetric "WarmRtfMedian" $warmRtfMedian (Get-ObjectNumber $baselineSummaryObj "WarmRtfMedian")))
    $metrics = @($metricsList)

    Add-RegressionFailure $regressionFailures $metrics[0] $MaxGenerateRegressionPercent
    Add-RegressionFailure $regressionFailures $metrics[3] $MaxPipelineRegressionPercent
    Add-RegressionFailure $regressionFailures $metrics[4] $MaxRtfRegressionPercent

    $baselineComparison = [PSCustomObject]@{
        BaselineSummary = $baselineSummaryResolved
        Metrics = $metrics
        Thresholds = [PSCustomObject]@{
            MaxGenerateRegressionPercent = $MaxGenerateRegressionPercent
            MaxPipelineRegressionPercent = $MaxPipelineRegressionPercent
            MaxRtfRegressionPercent = $MaxRtfRegressionPercent
        }
        Failures = @($regressionFailures)
    }
}

$summary = [PSCustomObject]@{
    CliExe = $cliExeResolved
    ModelDir = $modelDirResolved
    SpeakerEmbedding = $speakerEmbeddingResolved
    Text = $Text
    MaxTokens = $MaxTokens
    Repeat = $Repeat
    WarmRepeatStart = if ($records.Count -ge 2) { 2 } else { 1 }
    Runs = $records.Count
    WarmRuns = $warmRecords.Count
    WarmGenerateMedianMs = $warmGenerateMedianMs
    WarmTalkerMedianMs = $warmTalkerMedianMs
    WarmCodePredMedianMs = $warmCodePredMedianMs
    WarmPipelineTotalMedianMs = $warmPipelineTotalMedianMs
    WarmRtfMedian = $warmRtfMedian
    BaselineComparison = $baselineComparison
    GpuBefore = $gpuBefore
    GpuAfter = $gpuAfter
    RunsDetail = $records
    LogPath = $logPath
}

$summary | ConvertTo-Json -Depth 6 | Tee-Object -FilePath $summaryPath
Write-Host ""
Write-Host "Summary written to: $summaryPath" -ForegroundColor Green

if ($regressionFailures.Count -gt 0) {
    foreach ($failure in $regressionFailures) {
        Write-Host "REGRESSION FAILED: $failure" -ForegroundColor Red
    }
    exit 1
}
