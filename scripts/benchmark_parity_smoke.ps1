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
    [double]$MaxGpuUtilizationBeforePercent = -1.0,
    [double]$MaxWarmGenerateRangePercent = -1.0,
    [double]$MaxWarmCodePredRangePercent = -1.0,
    [double]$MaxWarmPipelineRangePercent = -1.0,
    [double]$MaxWarmRtfRangePercent = -1.0,
    [int]$WaitForGpuIdleSeconds = 0,
    [int]$GpuPollIntervalSeconds = 5,
    [int]$MinWarmRuns = 3,
    [switch]$TalkerKvCacheF32,
    [switch]$TalkerKvCacheF16,
    [switch]$RequireComparableBaseline,
    [switch]$SelfTest,
    [switch]$RequireAssets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
if ($WaitForGpuIdleSeconds -lt 0) {
    throw "-WaitForGpuIdleSeconds must be greater than or equal to 0."
}
if ($GpuPollIntervalSeconds -lt 1) {
    throw "-GpuPollIntervalSeconds must be at least 1."
}
if ($MinWarmRuns -lt 1) {
    throw "-MinWarmRuns must be at least 1."
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

function New-WarmMetricStats([string]$name, [double[]]$values) {
    if ($values.Count -eq 0) {
        return [PSCustomObject]@{
            Name = $name
            Count = 0
            Min = $null
            Max = $null
            Median = $null
            Range = $null
            RangePercentOfMedian = $null
        }
    }

    $sorted = @($values | Sort-Object)
    $minValue = [double]$sorted[0]
    $maxValue = [double]$sorted[$sorted.Count - 1]
    $medianValue = Get-Median $values
    $rangeValue = [double]($maxValue - $minValue)
    $rangePercent = $null
    if ($null -ne $medianValue -and [Math]::Abs([double]$medianValue) -gt 0.0) {
        $rangePercent = [double](($rangeValue / [double]$medianValue) * 100.0)
    }

    return [PSCustomObject]@{
        Name = $name
        Count = $values.Count
        Min = $minValue
        Max = $maxValue
        Median = $medianValue
        Range = $rangeValue
        RangePercentOfMedian = $rangePercent
    }
}

function New-BenchmarkWarnings([int]$warmRunCount, [int]$minWarmRuns) {
    $warnings = [System.Collections.Generic.List[string]]::new()
    if ($warmRunCount -lt $minWarmRuns) {
        $warnings.Add("Only $warmRunCount warm run(s) were available; use at least $minWarmRuns for a stronger regression signal.")
    }
    return @($warnings)
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

function Get-ObjectValue([object]$obj, [string]$name) {
    if ($null -eq $obj) {
        return $null
    }
    $property = $obj.PSObject.Properties[$name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function ConvertTo-ComparableText([object]$value) {
    if ($null -eq $value) {
        return $null
    }
    return [System.Convert]::ToString($value, [Globalization.CultureInfo]::InvariantCulture)
}

function New-BaselineCompatibility(
    [object]$current,
    [object]$baseline,
    [string[]]$fields,
    [bool]$required
) {
    $issues = [System.Collections.Generic.List[string]]::new()
    foreach ($field in $fields) {
        $currentValue = Get-ObjectValue $current $field
        $baselineValue = Get-ObjectValue $baseline $field
        if ($null -eq $baselineValue) {
            if ($field -eq "TalkerKvCacheF32" -and $currentValue -eq $false) {
                continue
            }
            $issues.Add("Baseline summary is missing '$field'; compatibility could not be verified.")
            continue
        }

        $currentText = ConvertTo-ComparableText $currentValue
        $baselineText = ConvertTo-ComparableText $baselineValue
        if ($currentText -ne $baselineText) {
            $issues.Add("Baseline '$field' differs: current '$currentText' vs baseline '$baselineText'.")
        }
    }

    return [PSCustomObject]@{
        Required = $required
        Fields = $fields
        IsComparable = ($issues.Count -eq 0)
        Issues = @($issues)
    }
}

function Add-BaselineCompatibilityMessages(
    [System.Collections.Generic.List[string]]$warnings,
    [System.Collections.Generic.List[string]]$failures,
    [object]$compatibility,
    [bool]$required
) {
    if ($null -eq $compatibility -or $compatibility.IsComparable) {
        return
    }

    if ($required) {
        foreach ($issue in $compatibility.Issues) {
            $failures.Add($issue)
        }
    } else {
        $warnings.Add("Baseline comparison is not fully comparable; inspect BaselineComparison.Compatibility.Issues.")
    }
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

function Add-WarmRangeFailure(
    [System.Collections.Generic.List[string]]$failures,
    [object]$metric,
    [double]$thresholdPercent
) {
    if ($thresholdPercent -lt 0.0 -or $null -eq $metric -or $null -eq $metric.RangePercentOfMedian) {
        return
    }
    if ($metric.RangePercentOfMedian -gt $thresholdPercent) {
        $failures.Add(("{0} warm range was {1}% of median (threshold {2}%)" -f `
            $metric.Name,
            (([double]$metric.RangePercentOfMedian).ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)),
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

function Assert-SelfTest([bool]$condition, [string]$message) {
    if (-not $condition) {
        throw "Self-test failed: $message"
    }
}

function Invoke-SelfTest() {
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("qwen3_tts_benchmark_smoke_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    try {
        $logPath = Join-Path $tempDir "timing.log"
        @'
Repeat 1/3
  Talker forward_step (total / per-frame):
    Total:               100.0 ms
  Code predictor (total / per-frame):
    Total:               200.0 ms
  Embed lookups:          10.0 ms
  Total generate:        350.0 ms
  Total:                 400.0 ms
  Throughput:            0.50x realtime (RTF=2.000)
Repeat 2/3
  Talker forward_step (total / per-frame):
    Total:               110.0 ms
  Code predictor (total / per-frame):
    Total:               220.0 ms
  Embed lookups:          10.0 ms
  Total generate:        360.0 ms
  Total:                 410.0 ms
  Throughput:            0.49x realtime (RTF=2.050)
Repeat 3/3
  Talker forward_step (total / per-frame):
    Total:               130.0 ms
  Code predictor (total / per-frame):
    Total:               260.0 ms
  Embed lookups:          10.0 ms
  Total generate:        390.0 ms
  Total:                 430.0 ms
  Throughput:            0.47x realtime (RTF=2.150)
'@ | Set-Content -LiteralPath $logPath -Encoding UTF8

        $records = @(Parse-TimingLog $logPath)
        Assert-SelfTest ($records.Count -eq 3) "expected 3 timing records"
        Assert-SelfTest ([Math]::Abs(([double]$records[1].GenerateMs) - 360.0) -lt 0.001) "repeat 2 generate parse"
        Assert-SelfTest ([Math]::Abs(([double]$records[2].CodePredMs) - 260.0) -lt 0.001) "repeat 3 code predictor parse"
        Assert-SelfTest ([Math]::Abs((Get-Median @([double]$records[1].GenerateMs, [double]$records[2].GenerateMs)) - 375.0) -lt 0.001) "warm generate median"
        $warmGenerateStats = New-WarmMetricStats "GenerateMs" ([double[]]@($records[1].GenerateMs, $records[2].GenerateMs))
        Assert-SelfTest ([Math]::Abs(([double]$warmGenerateStats.Min) - 360.0) -lt 0.001) "warm generate min"
        Assert-SelfTest ([Math]::Abs(([double]$warmGenerateStats.Max) - 390.0) -lt 0.001) "warm generate max"
        Assert-SelfTest ([Math]::Abs(([double]$warmGenerateStats.Range) - 30.0) -lt 0.001) "warm generate range"
        Assert-SelfTest ([Math]::Abs(([double]$warmGenerateStats.RangePercentOfMedian) - 8.0) -lt 0.001) "warm generate range percent"
        Assert-SelfTest (@(New-BenchmarkWarnings 2 3).Count -eq 1) "low warm-run warning"
        Assert-SelfTest (@(New-BenchmarkWarnings 3 3).Count -eq 0) "sufficient warm-run warning"
        $stabilityFailures = [System.Collections.Generic.List[string]]::new()
        Add-WarmRangeFailure $stabilityFailures $warmGenerateStats 5.0
        Assert-SelfTest ($stabilityFailures.Count -eq 1) "warm range threshold failure"
        $stabilityFailures.Clear()
        Add-WarmRangeFailure $stabilityFailures $warmGenerateStats 10.0
        Assert-SelfTest ($stabilityFailures.Count -eq 0) "warm range threshold pass"

        $currentWorkload = [PSCustomObject]@{
            ModelDir = "model-a"
            SpeakerEmbedding = "speaker-a.json"
            Text = "hello"
            Language = "en"
            MaxTokens = 64
            Temperature = 1.0
            TopK = 1
            TopP = 1.0
            Seed = 0
            TalkerKvCacheF32 = $true
        }
        $matchingBaseline = [PSCustomObject]@{
            ModelDir = "model-a"
            SpeakerEmbedding = "speaker-a.json"
            Text = "hello"
            Language = "en"
            MaxTokens = 64
            Temperature = 1.0
            TopK = 1
            TopP = 1.0
            Seed = 0
            TalkerKvCacheF32 = $true
        }
        $mismatchedBaseline = [PSCustomObject]@{
            ModelDir = "model-a"
            SpeakerEmbedding = "speaker-b.json"
            Text = "hello"
        }
        $compatFields = @("ModelDir", "SpeakerEmbedding", "Text", "Language", "MaxTokens", "Temperature", "TopK", "TopP", "Seed", "TalkerKvCacheF32")
        $compatible = New-BaselineCompatibility $currentWorkload $matchingBaseline $compatFields $true
        Assert-SelfTest ($compatible.IsComparable) "matching baseline compatibility"
        $incompatible = New-BaselineCompatibility $currentWorkload $mismatchedBaseline $compatFields $true
        Assert-SelfTest (-not $incompatible.IsComparable) "mismatched baseline compatibility"
        Assert-SelfTest ($incompatible.Issues.Count -ge 2) "baseline compatibility issue count"
        $compatWarnings = [System.Collections.Generic.List[string]]::new()
        $compatFailures = [System.Collections.Generic.List[string]]::new()
        Add-BaselineCompatibilityMessages $compatWarnings $compatFailures $incompatible $false
        Assert-SelfTest ($compatWarnings.Count -eq 1 -and $compatFailures.Count -eq 0) "baseline compatibility warning path"
        $compatWarnings.Clear()
        Add-BaselineCompatibilityMessages $compatWarnings $compatFailures $incompatible $true
        Assert-SelfTest ($compatWarnings.Count -eq 0 -and $compatFailures.Count -ge 2) "baseline compatibility failure path"

        $metric = New-DeltaMetric "WarmGenerateMedianMs" 110.0 100.0
        Assert-SelfTest ([Math]::Abs(([double]$metric.DeltaPercent) - 10.0) -lt 0.001) "delta percent"
        $failures = [System.Collections.Generic.List[string]]::new()
        Add-RegressionFailure $failures $metric 5.0
        Assert-SelfTest ($failures.Count -eq 1) "regression threshold failure"

        Write-Host "Benchmark parity smoke self-test passed."
    } finally {
        Remove-Item -Recurse -Force -LiteralPath $tempDir -ErrorAction SilentlyContinue
    }
}

if ($SelfTest) {
    Invoke-SelfTest
    exit 0
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cliExeResolved = Resolve-CliExe $CliExe
$modelDirResolved = Resolve-RepoPath $ModelDir
$speakerEmbeddingResolved = Resolve-RepoPath $SpeakerEmbedding
$outputDirResolved = Resolve-RepoPath $OutputDir
$baselineSummaryResolved = Resolve-RepoPath $BaselineSummary
$talkerKvCacheF32Enabled = -not [bool]$TalkerKvCacheF16
if ($TalkerKvCacheF32 -and $TalkerKvCacheF16) {
    throw "-TalkerKvCacheF32 and -TalkerKvCacheF16 cannot be combined."
}

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
if ($talkerKvCacheF32Enabled) {
    Write-Host "  Talker KV cache: F32 (default)"
} else {
    Write-Host "  Talker KV cache: F16 (QWEN3_TTS_TALKER_KV_F16=1)"
}
if ($null -ne $baselineSummaryObj) {
    Write-Host "  Baseline:       $baselineSummaryResolved"
}

$debugSnapshot = Save-DebugDumpEnv
Disable-DebugDumpEnv
$talkerKvCacheSnapshot = Save-TalkerKvCacheEnv
$gpuBefore = Get-GpuSnapshot
$gpuUtilBefore = Get-ObjectNumber $gpuBefore "UtilizationPercent"
if ($MaxGpuUtilizationBeforePercent -ge 0.0 -and
    $WaitForGpuIdleSeconds -gt 0 -and
    $null -ne $gpuUtilBefore -and
    $gpuUtilBefore -gt $MaxGpuUtilizationBeforePercent) {
    $deadline = (Get-Date).AddSeconds($WaitForGpuIdleSeconds)
    Write-Host ("GPU utilization before benchmark is {0}%, above threshold {1}%; waiting up to {2}s." -f `
        ($gpuUtilBefore.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)),
        ($MaxGpuUtilizationBeforePercent.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)),
        $WaitForGpuIdleSeconds) -ForegroundColor Yellow

    while ((Get-Date) -lt $deadline -and
        $null -ne $gpuUtilBefore -and
        $gpuUtilBefore -gt $MaxGpuUtilizationBeforePercent) {
        $remainingSeconds = [Math]::Max(0, [int][Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds))
        $sleepSeconds = [Math]::Min($GpuPollIntervalSeconds, [Math]::Max(1, $remainingSeconds))
        Start-Sleep -Seconds $sleepSeconds
        $gpuBefore = Get-GpuSnapshot
        $gpuUtilBefore = Get-ObjectNumber $gpuBefore "UtilizationPercent"
    }

    if ($null -ne $gpuUtilBefore -and $gpuUtilBefore -le $MaxGpuUtilizationBeforePercent) {
        Write-Host ("GPU utilization settled at {0}%; running benchmark." -f `
            ($gpuUtilBefore.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)))
    }
}
if ($MaxGpuUtilizationBeforePercent -ge 0.0 -and
    $null -ne $gpuUtilBefore -and
    $gpuUtilBefore -gt $MaxGpuUtilizationBeforePercent) {
    $summary = [PSCustomObject]@{
        CliExe = $cliExeResolved
        ModelDir = $modelDirResolved
        SpeakerEmbedding = $speakerEmbeddingResolved
        Text = $Text
        Language = $Language
        MaxTokens = $MaxTokens
        Temperature = $Temperature
        TopK = $TopK
        TopP = $TopP
        Seed = $Seed
        TalkerKvCacheF32 = [bool]$talkerKvCacheF32Enabled
        Repeat = $Repeat
        Skipped = $true
        SkipReason = "GPU utilization before benchmark was $gpuUtilBefore%, above threshold $MaxGpuUtilizationBeforePercent%."
        GpuBefore = $gpuBefore
    }
    $summary | ConvertTo-Json -Depth 6 | Tee-Object -FilePath $summaryPath
    Restore-DebugDumpEnv $debugSnapshot
    Restore-TalkerKvCacheEnv $talkerKvCacheSnapshot
    Write-Host ""
    Write-Host "BENCHMARK SKIPPED: $($summary.SkipReason)" -ForegroundColor Yellow
    exit 2
}

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
        Remove-Item Env:QWEN3_TTS_TALKER_KV_F32 -ErrorAction SilentlyContinue
        if ($talkerKvCacheF32Enabled) {
            Remove-Item Env:QWEN3_TTS_TALKER_KV_F16 -ErrorAction SilentlyContinue
        } else {
            $env:QWEN3_TTS_TALKER_KV_F16 = "1"
        }
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
    Restore-TalkerKvCacheEnv $talkerKvCacheSnapshot
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
$warmMetricStats = [PSCustomObject]@{
    GenerateMs = New-WarmMetricStats "GenerateMs" ([double[]]@($warmRecords | ForEach-Object { $_.GenerateMs }))
    TalkerMs = New-WarmMetricStats "TalkerMs" ([double[]]@($warmRecords | ForEach-Object { $_.TalkerMs }))
    CodePredMs = New-WarmMetricStats "CodePredMs" ([double[]]@($warmRecords | ForEach-Object { $_.CodePredMs }))
    PipelineTotalMs = New-WarmMetricStats "PipelineTotalMs" ([double[]]@($warmRecords | ForEach-Object { $_.PipelineTotalMs }))
    RTF = New-WarmMetricStats "RTF" ([double[]]@($warmRecords | ForEach-Object { $_.RTF }))
}
$benchmarkWarnings = [System.Collections.Generic.List[string]]::new()
foreach ($warning in @(New-BenchmarkWarnings $warmRecords.Count $MinWarmRuns)) {
    $benchmarkWarnings.Add($warning)
}
$baselineComparison = $null
$regressionFailures = [System.Collections.Generic.List[string]]::new()
$stabilityFailures = [System.Collections.Generic.List[string]]::new()
$baselineCompatibility = $null
$compatibilityFailures = [System.Collections.Generic.List[string]]::new()
$currentWorkload = [PSCustomObject]@{
    ModelDir = $modelDirResolved
    SpeakerEmbedding = $speakerEmbeddingResolved
    Text = $Text
    Language = $Language
    MaxTokens = $MaxTokens
    Temperature = $Temperature
    TopK = $TopK
    TopP = $TopP
    Seed = $Seed
    TalkerKvCacheF32 = [bool]$talkerKvCacheF32Enabled
}

Add-WarmRangeFailure $stabilityFailures $warmMetricStats.GenerateMs $MaxWarmGenerateRangePercent
Add-WarmRangeFailure $stabilityFailures $warmMetricStats.CodePredMs $MaxWarmCodePredRangePercent
Add-WarmRangeFailure $stabilityFailures $warmMetricStats.PipelineTotalMs $MaxWarmPipelineRangePercent
Add-WarmRangeFailure $stabilityFailures $warmMetricStats.RTF $MaxWarmRtfRangePercent

if ($null -ne $baselineSummaryObj) {
    $compatFields = @("ModelDir", "SpeakerEmbedding", "Text", "Language", "MaxTokens", "Temperature", "TopK", "TopP", "Seed", "TalkerKvCacheF32")
    $baselineCompatibility = New-BaselineCompatibility $currentWorkload $baselineSummaryObj $compatFields $RequireComparableBaseline.IsPresent
    Add-BaselineCompatibilityMessages $benchmarkWarnings $compatibilityFailures $baselineCompatibility $RequireComparableBaseline.IsPresent

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
        Compatibility = $baselineCompatibility
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
    Language = $Language
    MaxTokens = $MaxTokens
    Temperature = $Temperature
    TopK = $TopK
    TopP = $TopP
    Seed = $Seed
    TalkerKvCacheF32 = [bool]$talkerKvCacheF32Enabled
    Repeat = $Repeat
    WarmRepeatStart = if ($records.Count -ge 2) { 2 } else { 1 }
    Runs = $records.Count
    WarmRuns = $warmRecords.Count
    WarmGenerateMedianMs = $warmGenerateMedianMs
    WarmTalkerMedianMs = $warmTalkerMedianMs
    WarmCodePredMedianMs = $warmCodePredMedianMs
    WarmPipelineTotalMedianMs = $warmPipelineTotalMedianMs
    WarmRtfMedian = $warmRtfMedian
    MinWarmRuns = $MinWarmRuns
    WarmMetricStats = $warmMetricStats
    BenchmarkWarnings = @($benchmarkWarnings)
    StabilityThresholds = [PSCustomObject]@{
        MaxWarmGenerateRangePercent = $MaxWarmGenerateRangePercent
        MaxWarmCodePredRangePercent = $MaxWarmCodePredRangePercent
        MaxWarmPipelineRangePercent = $MaxWarmPipelineRangePercent
        MaxWarmRtfRangePercent = $MaxWarmRtfRangePercent
    }
    StabilityFailures = @($stabilityFailures)
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
}
if ($stabilityFailures.Count -gt 0) {
    foreach ($failure in $stabilityFailures) {
        Write-Host "BENCHMARK UNSTABLE: $failure" -ForegroundColor Red
    }
}
if ($compatibilityFailures.Count -gt 0) {
    foreach ($failure in $compatibilityFailures) {
        Write-Host "BASELINE INCOMPARABLE: $failure" -ForegroundColor Red
    }
}
if ($regressionFailures.Count -gt 0 -or $stabilityFailures.Count -gt 0 -or $compatibilityFailures.Count -gt 0) {
    exit 1
}
