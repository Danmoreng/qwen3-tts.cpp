param(
    [int]$Repetitions = 3,
    [int]$MaxTokens = 320,
    [string]$ModelDir = "models",
    [string]$OutputDir = "benchmark_output\python_vs_cpp",
    [string]$PythonExe = "",
    [string]$CliExe = "",
    [string]$OriginalCliExe = "",
    [switch]$Deterministic = $true,
    [string]$Text = "Hello from qwen3-tts.cpp benchmark."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Requested)
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return (Resolve-Path $Requested).Path
    }

    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }

    return "python"
}

function Resolve-CliExe {
    param([string]$Requested)
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return (Resolve-Path $Requested).Path
    }

    $defaultCli = Join-Path $PSScriptRoot "..\build\qwen3-tts-cli.exe"
    if (Test-Path $defaultCli) {
        return (Resolve-Path $defaultCli).Path
    }

    throw "CLI executable not found. Use -CliExe to specify the path."
}

function Resolve-OriginalCliExe {
    param([string]$Requested)
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return (Resolve-Path $Requested).Path
    }

    $defaultOriginalCli = Join-Path $PSScriptRoot "..\third_party\qwen3-tts-original.cpp\build-cuda-ninja\qwen3-tts-cli.exe"
    if (Test-Path $defaultOriginalCli) {
        return (Resolve-Path $defaultOriginalCli).Path
    }

    return ""
}

function Convert-ToInvariantDouble {
    param([string]$Value)
    return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-WavDurationSec {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return 0.0
    }

    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $br = [System.IO.BinaryReader]::new($fs)
        try {
            $riff = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            [void]$br.ReadUInt32()
            $wave = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            if ($riff -ne "RIFF" -or $wave -ne "WAVE") {
                throw "Invalid WAV header in '$Path'"
            }

            $sampleRate = 0
            $bitsPerSample = 0
            $channels = 0
            $dataBytes = 0

            while ($br.BaseStream.Position -le ($br.BaseStream.Length - 8)) {
                $chunkId = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
                $chunkSize = $br.ReadUInt32()

                if ($chunkId -eq "fmt ") {
                    [void]$br.ReadUInt16()
                    $channels = $br.ReadUInt16()
                    $sampleRate = [int]$br.ReadUInt32()
                    [void]$br.ReadUInt32()
                    [void]$br.ReadUInt16()
                    $bitsPerSample = $br.ReadUInt16()

                    if ($chunkSize -gt 16) {
                        [void]$br.ReadBytes([int]($chunkSize - 16))
                    }
                } elseif ($chunkId -eq "data") {
                    $dataBytes = [int64]$chunkSize
                    [void]$br.BaseStream.Seek([int64]$chunkSize, [System.IO.SeekOrigin]::Current)
                } else {
                    [void]$br.BaseStream.Seek([int64]$chunkSize, [System.IO.SeekOrigin]::Current)
                }

                if (($chunkSize % 2) -ne 0 -and $br.BaseStream.Position -lt $br.BaseStream.Length) {
                    [void]$br.ReadByte()
                }
            }

            if ($sampleRate -le 0 -or $channels -le 0 -or $bitsPerSample -le 0 -or $dataBytes -le 0) {
                return 0.0
            }

            $bytesPerSample = [int]($bitsPerSample / 8)
            if ($bytesPerSample -le 0) {
                return 0.0
            }

            $numSamples = $dataBytes / ($channels * $bytesPerSample)
            return [double]$numSamples / [double]$sampleRate
        } finally {
            $br.Dispose()
        }
    } finally {
        $fs.Dispose()
    }
}

function Quote-CommandArgument {
    param([string]$Arg)

    if ([string]::IsNullOrEmpty($Arg)) {
        return '""'
    }

    if ($Arg -notmatch '[\s"]') {
        return $Arg
    }

    $escaped = $Arg -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Get-ProcessTreeWorkingSetBytes {
    param([int]$RootPid)

    $sum = 0L
    $childrenByParent = @{}

    try {
        $procRows = Get-CimInstance Win32_Process -ErrorAction Stop
        foreach ($row in $procRows) {
            $parentId = [int]$row.ParentProcessId
            if (-not $childrenByParent.ContainsKey($parentId)) {
                $childrenByParent[$parentId] = New-Object System.Collections.Generic.List[int]
            }
            $childrenByParent[$parentId].Add([int]$row.ProcessId)
        }
    } catch {
        try {
            $p = Get-Process -Id $RootPid -ErrorAction Stop
            return [int64]$p.WorkingSet64
        } catch {
            return 0L
        }
    }

    $queue = New-Object System.Collections.Generic.Queue[int]
    $seen = New-Object "System.Collections.Generic.HashSet[int]"
    [void]$queue.Enqueue($RootPid)
    [void]$seen.Add($RootPid)

    while ($queue.Count -gt 0) {
        $currentPid = $queue.Dequeue()

        try {
            $proc = Get-Process -Id $currentPid -ErrorAction Stop
            $sum += [int64]$proc.WorkingSet64
        } catch {
            # Process may have exited before sampling.
        }

        if ($childrenByParent.ContainsKey($currentPid)) {
            foreach ($childId in $childrenByParent[$currentPid]) {
                if ($seen.Add($childId)) {
                    [void]$queue.Enqueue($childId)
                }
            }
        }
    }

    return $sum
}

function Invoke-BenchmarkProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $stdoutPath = Join-Path $env:TEMP ("qwen3tts_bench_out_{0}.log" -f ([guid]::NewGuid().ToString("N")))
    $stderrPath = Join-Path $env:TEMP ("qwen3tts_bench_err_{0}.log" -f ([guid]::NewGuid().ToString("N")))

    $argumentLine = ($Arguments | ForEach-Object { Quote-CommandArgument $_ }) -join " "

    $proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $argumentLine `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $peakWorkingSet = 0L

    while (-not $proc.HasExited) {
        $sampleWs = Get-ProcessTreeWorkingSetBytes -RootPid $proc.Id
        if ($sampleWs -gt $peakWorkingSet) {
            $peakWorkingSet = $sampleWs
        }
        Start-Sleep -Milliseconds 100
    }
    $proc.WaitForExit()
    $sw.Stop()

    $finalWs = Get-ProcessTreeWorkingSetBytes -RootPid $proc.Id
    if ($finalWs -gt $peakWorkingSet) {
        $peakWorkingSet = $finalWs
    }

    $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Raw } else { "" }
    $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { "" }

    if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
    if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }

    [pscustomobject]@{
        ExitCode       = $proc.ExitCode
        WallMs         = [int]$sw.ElapsedMilliseconds
        PeakWorkingSet = [int64]$peakWorkingSet
        Stdout         = $stdout
        Stderr         = $stderr
    }
}

function Parse-CppAudioDuration {
    param([string]$Stdout)

    $m = [regex]::Match($Stdout, "Audio duration:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds")
    if ($m.Success) {
        return Convert-ToInvariantDouble $m.Groups[1].Value
    }
    return 0.0
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Resolve-PythonExe -Requested $PythonExe
$cliPath = Resolve-CliExe -Requested $CliExe
$originalCliPath = Resolve-OriginalCliExe -Requested $OriginalCliExe
$outRoot = Join-Path $repoRoot $OutputDir
$logDir = Join-Path $outRoot "logs"
$modelDirPath = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }

if (-not (Test-Path $modelDirPath)) {
    throw "Missing C++ model directory: $modelDirPath"
}

$models = @(
    [pscustomobject]@{
        Name = "0.6B"
        HfModelDir = Join-Path $repoRoot "models\Qwen3-TTS-12Hz-0.6B-Base"
        CppModelName = "qwen3-tts-0.6b-f16.gguf"
    },
    [pscustomobject]@{
        Name = "1.7B"
        HfModelDir = Join-Path $repoRoot "models\Qwen3-TTS-12Hz-1.7B-Base"
        CppModelName = "qwen3-tts-1.7b-base-f16.gguf"
    }
)

foreach ($m in $models) {
    if (-not (Test-Path $m.HfModelDir)) {
        throw "Missing HF model directory for $($m.Name): $($m.HfModelDir)"
    }
    $cppModelPath = Join-Path $modelDirPath $m.CppModelName
    if (-not (Test-Path $cppModelPath)) {
        throw "Missing C++ model file for $($m.Name): $cppModelPath"
    }
}

New-Item -ItemType Directory -Path $outRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "Running Python vs C++ benchmark (basic TTS only)"
Write-Host "  Repo:        $repoRoot"
Write-Host "  Python:      $pythonPath"
Write-Host "  Fork CLI:    $cliPath"
if ([string]::IsNullOrWhiteSpace($originalCliPath)) {
    Write-Host "  Orig CLI:    not found (original benchmark disabled)"
} else {
    Write-Host "  Orig CLI:    $originalCliPath"
}
Write-Host "  ModelDir:    $modelDirPath"
Write-Host "  Repetitions: $Repetitions"
Write-Host "  Deterministic: $Deterministic"
Write-Host "  Prompt:      $Text"
Write-Host ""

$rows = New-Object System.Collections.Generic.List[object]

foreach ($model in $models) {
    Write-Host ("=== Model {0} ===" -f $model.Name)
    for ($rep = 1; $rep -le $Repetitions; $rep++) {
        Write-Host ("--- basic ({0}) run {1}/{2} ---" -f $model.Name, $rep, $Repetitions)

        $pyOut = Join-Path $outRoot ("python_qwen_tts_basic_{0}_run{1}.wav" -f $model.Name, $rep)
        $pyArgs = @(
            "scripts/benchmark_pytorch_vs_cpp.py",
            "--worker",
            "--backend", "qwen_tts",
            "--mode", "basic",
            "--hf-model-dir", $model.HfModelDir,
            "--text", $Text,
            "--output", $pyOut,
            "--max-tokens", $MaxTokens.ToString()
        )
        if ($Deterministic) {
            $pyArgs += "--deterministic"
        }

        $pyRun = Invoke-BenchmarkProcess -FilePath $pythonPath -Arguments $pyArgs -WorkingDirectory $repoRoot
        $pyAudioSec = Get-WavDurationSec -Path $pyOut
        $pyRtf = if ($pyAudioSec -gt 0) { ($pyRun.WallMs / 1000.0) / $pyAudioSec } else { 0.0 }
        $pyStatus = if ($pyRun.ExitCode -eq 0) { "PASS" } else { "FAIL" }

        $pyLog = Join-Path $logDir ("basic_{0}_python_qwen_tts_run{1}.log" -f $model.Name, $rep)
        @(
            "Command: $pythonPath $(($pyArgs | ForEach-Object { Quote-CommandArgument $_ }) -join ' ')",
            "ExitCode: $($pyRun.ExitCode)",
            "",
            "[stdout]",
            $pyRun.Stdout,
            "",
            "[stderr]",
            $pyRun.Stderr
        ) | Set-Content -Path $pyLog

        $rows.Add([pscustomobject]@{
            Model     = $model.Name
            Scenario  = "basic"
            Pipeline  = "python_qwen_tts"
            Run       = $rep
            Status    = $pyStatus
            ExitCode  = $pyRun.ExitCode
            WallMs    = $pyRun.WallMs
            PeakRssMB = [math]::Round($pyRun.PeakWorkingSet / 1MB, 2)
            AudioSec  = [math]::Round($pyAudioSec, 3)
            RTF       = [math]::Round($pyRtf, 3)
            Note      = ""
            LogPath   = $pyLog
            OutputWav = $pyOut
        })

        $cppOut = Join-Path $outRoot ("cpp_fork_basic_{0}_run{1}.wav" -f $model.Name, $rep)
        $cppArgs = @(
            "-m", $modelDirPath,
            "--model-name", $model.CppModelName,
            "-t", $Text,
            "-o", $cppOut,
            "--max-tokens", $MaxTokens.ToString(),
            "--temperature", "0",
            "--top-k", "0",
            "--top-p", "1.0"
        )

        $cppRun = Invoke-BenchmarkProcess -FilePath $cliPath -Arguments $cppArgs -WorkingDirectory $repoRoot
        $cppAudioSec = Parse-CppAudioDuration -Stdout $cppRun.Stdout
        if ($cppAudioSec -le 0.0) {
            $cppAudioSec = Get-WavDurationSec -Path $cppOut
        }
        $cppRtf = if ($cppAudioSec -gt 0) { ($cppRun.WallMs / 1000.0) / $cppAudioSec } else { 0.0 }
        $cppStatus = if ($cppRun.ExitCode -eq 0) { "PASS" } else { "FAIL" }

        $cppLog = Join-Path $logDir ("basic_{0}_cpp_run{1}.log" -f $model.Name, $rep)
        @(
            "Command: $cliPath $(($cppArgs | ForEach-Object { Quote-CommandArgument $_ }) -join ' ')",
            "ExitCode: $($cppRun.ExitCode)",
            "",
            "[stdout]",
            $cppRun.Stdout,
            "",
            "[stderr]",
            $cppRun.Stderr
        ) | Set-Content -Path $cppLog

        $rows.Add([pscustomobject]@{
            Model     = $model.Name
            Scenario  = "basic"
            Pipeline  = "cpp_fork"
            Run       = $rep
            Status    = $cppStatus
            ExitCode  = $cppRun.ExitCode
            WallMs    = $cppRun.WallMs
            PeakRssMB = [math]::Round($cppRun.PeakWorkingSet / 1MB, 2)
            AudioSec  = [math]::Round($cppAudioSec, 3)
            RTF       = [math]::Round($cppRtf, 3)
            Note      = ""
            LogPath   = $cppLog
            OutputWav = $cppOut
        })

        if ([string]::IsNullOrWhiteSpace($originalCliPath)) {
            $rows.Add([pscustomobject]@{
                Model     = $model.Name
                Scenario  = "basic"
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = "SKIP"
                ExitCode  = -1
                WallMs    = 0
                PeakRssMB = 0
                AudioSec  = 0
                RTF       = 0
                Note      = "original CLI not found"
                LogPath   = ""
                OutputWav = ""
            })
        } elseif ($model.Name -ne "0.6B") {
            $rows.Add([pscustomobject]@{
                Model     = $model.Name
                Scenario  = "basic"
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = "SKIP"
                ExitCode  = -1
                WallMs    = 0
                PeakRssMB = 0
                AudioSec  = 0
                RTF       = 0
                Note      = "original CLI has fixed 0.6B model (no --model-name)"
                LogPath   = ""
                OutputWav = ""
            })
        } else {
            $origOut = Join-Path $outRoot ("cpp_original_basic_{0}_run{1}.wav" -f $model.Name, $rep)
            $origArgs = @(
                "-m", $modelDirPath,
                "-t", $Text,
                "-o", $origOut,
                "--max-tokens", $MaxTokens.ToString(),
                "--temperature", "0",
                "--top-k", "0",
                "--top-p", "1.0"
            )

            $origRun = Invoke-BenchmarkProcess -FilePath $originalCliPath -Arguments $origArgs -WorkingDirectory $repoRoot
            $origAudioSec = Parse-CppAudioDuration -Stdout $origRun.Stdout
            if ($origAudioSec -le 0.0) {
                $origAudioSec = Get-WavDurationSec -Path $origOut
            }
            $origRtf = if ($origAudioSec -gt 0) { ($origRun.WallMs / 1000.0) / $origAudioSec } else { 0.0 }
            $origStatus = if ($origRun.ExitCode -eq 0) { "PASS" } else { "FAIL" }

            $origLog = Join-Path $logDir ("basic_{0}_cpp_original_run{1}.log" -f $model.Name, $rep)
            @(
                "Command: $originalCliPath $(($origArgs | ForEach-Object { Quote-CommandArgument $_ }) -join ' ')",
                "ExitCode: $($origRun.ExitCode)",
                "",
                "[stdout]",
                $origRun.Stdout,
                "",
                "[stderr]",
                $origRun.Stderr
            ) | Set-Content -Path $origLog

            $rows.Add([pscustomobject]@{
                Model     = $model.Name
                Scenario  = "basic"
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = $origStatus
                ExitCode  = $origRun.ExitCode
                WallMs    = $origRun.WallMs
                PeakRssMB = [math]::Round($origRun.PeakWorkingSet / 1MB, 2)
                AudioSec  = [math]::Round($origAudioSec, 3)
                RTF       = [math]::Round($origRtf, 3)
                Note      = ""
                LogPath   = $origLog
                OutputWav = $origOut
            })
        }
    }
}

$rawCsv = Join-Path $outRoot "benchmark_python_vs_cpp_raw.csv"
$rows | Sort-Object Model, Pipeline, Run | Export-Csv -Path $rawCsv -NoTypeInformation

$summary = New-Object System.Collections.Generic.List[object]
$pipelines = @($rows | Select-Object -ExpandProperty Pipeline -Unique | Sort-Object)

foreach ($model in $models) {
    $cppForkPass = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Pipeline -eq "cpp_fork" -and $_.Status -eq "PASS" })
    $cppForkWallAvg = if ($cppForkPass.Count -gt 0) { ($cppForkPass | Measure-Object WallMs -Average).Average } else { 0.0 }

    foreach ($pipeline in $pipelines) {
        $passRows = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Pipeline -eq $pipeline -and $_.Status -eq "PASS" })
        $wallAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object WallMs -Average).Average } else { 0.0 }
        $rtfAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object RTF -Average).Average } else { 0.0 }
        $memAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object PeakRssMB -Average).Average } else { 0.0 }
        $skipRows = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Pipeline -eq $pipeline -and $_.Status -eq "SKIP" })
        $skipReason = if ($skipRows.Count -gt 0) { ($skipRows[0].Note) } else { "" }
        $vsCppFork = if ($pipeline -eq "cpp_fork") { 1.0 } elseif ($cppForkWallAvg -gt 0) { $wallAvg / $cppForkWallAvg } else { 0.0 }

        $summary.Add([pscustomobject]@{
            Model      = $model.Name
            Scenario   = "basic"
            Pipeline   = $pipeline
            WallMs     = [math]::Round($wallAvg, 1)
            RTF        = [math]::Round($rtfAvg, 3)
            PeakRssMB  = [math]::Round($memAvg, 1)
            Pass       = ("{0}/{1}" -f $passRows.Count, $Repetitions)
            Skip       = $skipReason
            WallVsCppForkX = [math]::Round($vsCppFork, 2)
        })
    }
}

$pipelineOrder = @{
    "cpp_fork" = 0
    "cpp_original" = 1
    "python_qwen_tts" = 2
}

$summaryCsv = Join-Path $outRoot "benchmark_python_vs_cpp_summary.csv"
$summary |
    Sort-Object `
        @{Expression = "Model"; Ascending = $true }, `
        @{Expression = { if ($pipelineOrder.ContainsKey($_.Pipeline)) { $pipelineOrder[$_.Pipeline] } else { 99 } }; Ascending = $true } |
    Export-Csv -Path $summaryCsv -NoTypeInformation

Write-Host ""
Write-Host "Raw Results"
$rows | Sort-Object Model, Pipeline, Run | Format-Table Model, Scenario, Pipeline, Run, Status, ExitCode, WallMs, PeakRssMB, AudioSec, RTF, Note -AutoSize

Write-Host ""
Write-Host "Benchmark Summary (averages of PASS runs)"
$summary |
    Sort-Object `
        @{Expression = "Model"; Ascending = $true }, `
        @{Expression = { if ($pipelineOrder.ContainsKey($_.Pipeline)) { $pipelineOrder[$_.Pipeline] } else { 99 } }; Ascending = $true } |
    Format-Table Model, Scenario, Pipeline, WallMs, RTF, PeakRssMB, Pass, WallVsCppForkX, Skip -AutoSize

Write-Host ""
Write-Host "CSV outputs:"
Write-Host "  $rawCsv"
Write-Host "  $summaryCsv"
