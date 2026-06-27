param(
    [int]$Repetitions = 3,
    [int]$MaxTokens = 320,
    [string]$ModelDir = "models",
    [string]$OutputDir = "benchmark_output\python_vs_cpp",
    [string]$PythonExe = "",
    [string]$CliExe = "",
    [string]$OriginalCliExe = "",
    [switch]$Deterministic = $true,
    [string]$Text = "Hello from qwen3-tts.cpp benchmark.",
    [ValidateSet("qwen_tts", "faster_qwen3_tts")]
    [string]$PythonBackend = "qwen_tts",
    [string]$PythonDevice = "auto",
    [ValidateSet("auto", "float32", "float16", "bfloat16")]
    [string]$PythonDType = "auto",
    [string]$PythonAttnImplementation = "auto",
    [ValidateSet("basic", "voice_clone", "voice_clone_synth_only")]
    [string]$Scenario = "basic",
    [string]$ReferenceAudio = "examples\readme_clone_input.wav",
    [double]$MinAudioPeak = 1e-4,
    [double]$MinAudioRms = 1e-6,
    [int]$MinNonZeroSamples = 16,
    [switch]$AllowSilentAudio
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "wav_stats.ps1")

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

function Get-SlugComponent {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return "default"
    }

    $slug = $Value.ToLowerInvariant()
    $slug = $slug -replace '[^a-z0-9]+', '-'
    $slug = $slug.Trim('-')
    if ([string]::IsNullOrWhiteSpace($slug)) {
        return "default"
    }

    return $slug
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

function Parse-CppTimingMs {
    param(
        [string]$Text,
        [string]$Label
    )

    $m = [regex]::Match($Text, ("(?mi)^\s*{0}:\s*([0-9]+)\s*ms" -f [regex]::Escape($Label)))
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return $null
}

function Parse-BenchmarkMetric {
    param(
        [string]$Text,
        [string]$Name
    )

    $m = [regex]::Match($Text, ("(?mi)^\s*{0}=([0-9]+(?:\.[0-9]+)?)" -f [regex]::Escape($Name)))
    if ($m.Success) {
        return Convert-ToInvariantDouble $m.Groups[1].Value
    }
    return $null
}

function Get-BenchmarkAudioResult {
    param(
        [int]$ExitCode,
        [string]$OutputWav
    )

    $stats = if (Test-Path $OutputWav) { Get-WavAudioStats -Path $OutputWav } else { New-EmptyWavAudioStats -path $OutputWav -errorMessage "file not found" }
    $audioStatus = Get-WavAudioQualityStatus -Stats $stats -MinPeak $MinAudioPeak -MinRms $MinAudioRms -MinNonZeroSamples $MinNonZeroSamples
    $status = if ($ExitCode -ne 0 -or -not (Test-Path $OutputWav)) {
        "FAIL"
    } elseif ($audioStatus -eq "OK" -or $AllowSilentAudio) {
        "PASS"
    } else {
        $audioStatus
    }

    return [PSCustomObject]@{
        Status      = $status
        AudioStatus = $audioStatus
        Stats       = $stats
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Resolve-PythonExe -Requested $PythonExe
$cliPath = Resolve-CliExe -Requested $CliExe
$originalCliPath = Resolve-OriginalCliExe -Requested $OriginalCliExe
$outRoot = Join-Path $repoRoot $OutputDir
$logDir = Join-Path $outRoot "logs"
$modelDirPath = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
$pythonPipelineSlug = "python_{0}_{1}_{2}" -f (Get-SlugComponent $PythonBackend), (Get-SlugComponent $PythonDevice), (Get-SlugComponent $PythonDType)
$referenceAudioPath = if ([System.IO.Path]::IsPathRooted($ReferenceAudio)) { $ReferenceAudio } else { Join-Path $repoRoot $ReferenceAudio }

if ($PythonBackend -eq "faster_qwen3_tts" -and $Scenario -ne "voice_clone") {
    if ($Scenario -ne "voice_clone_synth_only") {
        throw "faster_qwen3_tts benchmark is only supported for -Scenario voice_clone or voice_clone_synth_only."
    }
}

if (-not (Test-Path $modelDirPath)) {
    throw "Missing C++ model directory: $modelDirPath"
}

if ($Scenario -like "voice_clone*" -and -not (Test-Path $referenceAudioPath)) {
    throw "Missing reference audio: $referenceAudioPath"
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

Write-Host ("Running Python vs C++ benchmark ({0})" -f $Scenario)
Write-Host "  Repo:        $repoRoot"
Write-Host "  Python:      $pythonPath"
Write-Host "  Fork CLI:    $cliPath"
if ([string]::IsNullOrWhiteSpace($originalCliPath)) {
    Write-Host "  Orig CLI:    not found (original benchmark disabled)"
} else {
    Write-Host "  Orig CLI:    $originalCliPath"
}
Write-Host "  Python backend: $PythonBackend"
Write-Host "  Python device:  $PythonDevice"
Write-Host "  Python dtype:   $PythonDType"
Write-Host "  Python attn:    $PythonAttnImplementation"
Write-Host "  ModelDir:    $modelDirPath"
if ($Scenario -like "voice_clone*") {
    Write-Host "  Ref audio:   $referenceAudioPath"
}
Write-Host "  Repetitions: $Repetitions"
Write-Host "  Deterministic: $Deterministic"
Write-Host "  Prompt:      $Text"
Write-Host ""

$rows = New-Object System.Collections.Generic.List[object]

foreach ($model in $models) {
    Write-Host ("=== Model {0} ===" -f $model.Name)
    for ($rep = 1; $rep -le $Repetitions; $rep++) {
        Write-Host ("--- {0} ({1}) run {2}/{3} ---" -f $Scenario, $model.Name, $rep, $Repetitions)

        $cppSpeakerEmbedding = ""
        $pythonSpeakerEmbedding = ""
        if ($Scenario -eq "voice_clone_synth_only") {
            $cppSpeakerEmbedding = Join-Path $outRoot ("cpp_speaker_embedding_{0}.json" -f $model.Name)
            if (-not (Test-Path $cppSpeakerEmbedding)) {
                $prepOut = Join-Path $outRoot ("prep_cpp_speaker_{0}.wav" -f $model.Name)
                $prepArgs = @(
                    "-m", $modelDirPath,
                    "--model-name", $model.CppModelName,
                    "-t", "Speaker embedding prep.",
                    "-o", $prepOut,
                    "-r", $referenceAudioPath,
                    "--dump-speaker-embedding", $cppSpeakerEmbedding,
                    "--max-tokens", "1",
                    "--temperature", "0",
                    "--top-k", "0",
                    "--top-p", "1.0"
                )
                $prepRun = Invoke-BenchmarkProcess -FilePath $cliPath -Arguments $prepArgs -WorkingDirectory $repoRoot
                if ($prepRun.ExitCode -ne 0 -or -not (Test-Path $cppSpeakerEmbedding)) {
                    throw "Failed to prepare C++ speaker embedding for $($model.Name): $($prepRun.Stderr)"
                }
            }

            $pythonSpeakerEmbedding = Join-Path $outRoot ("python_speaker_embedding_{0}.pt" -f $model.Name)
            if (-not (Test-Path $pythonSpeakerEmbedding)) {
                $extractDevice = if ($PythonDevice -eq "auto") { "cuda:0" } else { $PythonDevice }
                $pyPrepArgs = @(
                    "third_party/python_ref/faster-qwen3-tts/examples/extract_speaker.py",
                    "--ref_audio", $referenceAudioPath,
                    "--output", $pythonSpeakerEmbedding,
                    "--model_path", $model.HfModelDir,
                    "--device", $extractDevice
                )
                $pyPrepRun = Invoke-BenchmarkProcess -FilePath $pythonPath -Arguments $pyPrepArgs -WorkingDirectory $repoRoot
                if ($pyPrepRun.ExitCode -ne 0 -or -not (Test-Path $pythonSpeakerEmbedding)) {
                    throw "Failed to prepare Python speaker embedding for $($model.Name): $($pyPrepRun.Stderr)"
                }
            }
        }

        $pyOut = Join-Path $outRoot ("{0}_{1}_{2}_run{3}.wav" -f $pythonPipelineSlug, $Scenario, $model.Name, $rep)
        $pyArgs = @(
            "scripts/benchmark_pytorch_vs_cpp.py",
            "--worker",
            "--backend", $PythonBackend,
            "--mode", $Scenario,
            "--hf-model-dir", $model.HfModelDir,
            "--text", $Text,
            "--output", $pyOut,
            "--max-tokens", $MaxTokens.ToString(),
            "--device", $PythonDevice,
            "--dtype", $PythonDType,
            "--attn-implementation", $PythonAttnImplementation
        )
        if ($Scenario -like "voice_clone*") {
            $pyArgs += @("--ref-audio", $referenceAudioPath)
        }
        if ($Scenario -eq "voice_clone_synth_only") {
            $pyArgs += @("--speaker-embedding-file", $pythonSpeakerEmbedding)
        }
        if ($Deterministic) {
            $pyArgs += "--deterministic"
        }

        $pyRun = Invoke-BenchmarkProcess -FilePath $pythonPath -Arguments $pyArgs -WorkingDirectory $repoRoot
        $pyMetricText = $pyRun.Stdout + "`n" + $pyRun.Stderr
        $pySynthMs = if ($Scenario -eq "voice_clone_synth_only") { Parse-BenchmarkMetric -Text $pyMetricText -Name "BENCHMARK_SYNTH_MS" } else { $null }
        $pyEffectiveMs = if ($null -ne $pySynthMs) { $pySynthMs } else { [double]$pyRun.WallMs }
        $pyAudioSec = if ($Scenario -eq "voice_clone_synth_only") {
            $metricAudio = Parse-BenchmarkMetric -Text $pyMetricText -Name "BENCHMARK_AUDIO_SEC"
            if ($null -ne $metricAudio) { $metricAudio } else { Get-WavDurationSec -Path $pyOut }
        } else {
            Get-WavDurationSec -Path $pyOut
        }
        $pyRtf = if ($pyAudioSec -gt 0) { ($pyEffectiveMs / 1000.0) / $pyAudioSec } else { 0.0 }
        $pyAudioResult = Get-BenchmarkAudioResult -ExitCode $pyRun.ExitCode -OutputWav $pyOut
        $pyStatus = $pyAudioResult.Status
        if ($pyAudioSec -le 0.0 -and $pyAudioResult.Stats.Valid) {
            $pyAudioSec = $pyAudioResult.Stats.DurationSec
            $pyRtf = if ($pyAudioSec -gt 0) { ($pyEffectiveMs / 1000.0) / $pyAudioSec } else { 0.0 }
        }

        $pyLog = Join-Path $logDir ("{0}_{1}_{2}_run{3}.log" -f $Scenario, $model.Name, $pythonPipelineSlug, $rep)
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
            Scenario  = $Scenario
            Pipeline  = $pythonPipelineSlug
            Run       = $rep
            Status    = $pyStatus
            ExitCode  = $pyRun.ExitCode
            WallMs    = [math]::Round($pyEffectiveMs, 1)
            PeakRssMB = [math]::Round($pyRun.PeakWorkingSet / 1MB, 2)
            AudioSec  = [math]::Round($pyAudioSec, 3)
            RTF       = [math]::Round($pyRtf, 3)
            AudioStatus = $pyAudioResult.AudioStatus
            AudioPeak = if ($pyAudioResult.Stats.Valid) { [math]::Round($pyAudioResult.Stats.Peak, 8) } else { $null }
            AudioRms = if ($pyAudioResult.Stats.Valid) { [math]::Round($pyAudioResult.Stats.Rms, 8) } else { $null }
            AudioNonZeroSamples = if ($pyAudioResult.Stats.Valid) { $pyAudioResult.Stats.NonZeroSamples } else { $null }
            Note      = ""
            LogPath   = $pyLog
            OutputWav = $pyOut
        })

        $cppOut = Join-Path $outRoot ("cpp_fork_{0}_{1}_run{2}.wav" -f $Scenario, $model.Name, $rep)
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
        if ($Scenario -eq "voice_clone") {
            $cppArgs += @("-r", $referenceAudioPath)
        } elseif ($Scenario -eq "voice_clone_synth_only") {
            $cppArgs += @("--speaker-embedding", $cppSpeakerEmbedding)
        }

        $cppRun = Invoke-BenchmarkProcess -FilePath $cliPath -Arguments $cppArgs -WorkingDirectory $repoRoot
        $cppMetricText = $cppRun.Stdout + "`n" + $cppRun.Stderr
        $cppSynthMs = if ($Scenario -eq "voice_clone_synth_only") { Parse-CppTimingMs -Text $cppMetricText -Label "Total" } else { $null }
        $cppEffectiveMs = if ($null -ne $cppSynthMs) { [double]$cppSynthMs } else { [double]$cppRun.WallMs }
        $cppAudioSec = Parse-CppAudioDuration -Stdout $cppMetricText
        if ($cppAudioSec -le 0.0) {
            $cppAudioSec = Get-WavDurationSec -Path $cppOut
        }
        $cppRtf = if ($cppAudioSec -gt 0) { ($cppEffectiveMs / 1000.0) / $cppAudioSec } else { 0.0 }
        $cppAudioResult = Get-BenchmarkAudioResult -ExitCode $cppRun.ExitCode -OutputWav $cppOut
        $cppStatus = $cppAudioResult.Status
        if ($cppAudioSec -le 0.0 -and $cppAudioResult.Stats.Valid) {
            $cppAudioSec = $cppAudioResult.Stats.DurationSec
            $cppRtf = if ($cppAudioSec -gt 0) { ($cppEffectiveMs / 1000.0) / $cppAudioSec } else { 0.0 }
        }

        $cppLog = Join-Path $logDir ("{0}_{1}_cpp_run{2}.log" -f $Scenario, $model.Name, $rep)
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
            Scenario  = $Scenario
            Pipeline  = "cpp_fork"
            Run       = $rep
            Status    = $cppStatus
            ExitCode  = $cppRun.ExitCode
            WallMs    = [math]::Round($cppEffectiveMs, 1)
            PeakRssMB = [math]::Round($cppRun.PeakWorkingSet / 1MB, 2)
            AudioSec  = [math]::Round($cppAudioSec, 3)
            RTF       = [math]::Round($cppRtf, 3)
            AudioStatus = $cppAudioResult.AudioStatus
            AudioPeak = if ($cppAudioResult.Stats.Valid) { [math]::Round($cppAudioResult.Stats.Peak, 8) } else { $null }
            AudioRms = if ($cppAudioResult.Stats.Valid) { [math]::Round($cppAudioResult.Stats.Rms, 8) } else { $null }
            AudioNonZeroSamples = if ($cppAudioResult.Stats.Valid) { $cppAudioResult.Stats.NonZeroSamples } else { $null }
            Note      = ""
            LogPath   = $cppLog
            OutputWav = $cppOut
        })

        if ([string]::IsNullOrWhiteSpace($originalCliPath)) {
            $rows.Add([pscustomobject]@{
                Model     = $model.Name
                Scenario  = $Scenario
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = "SKIP"
                ExitCode  = -1
                WallMs    = 0
                PeakRssMB = 0
                AudioSec  = 0
                RTF       = 0
                AudioStatus = ""
                AudioPeak = $null
                AudioRms = $null
                AudioNonZeroSamples = $null
                Note      = "original CLI not found"
                LogPath   = ""
                OutputWav = ""
            })
        } elseif ($model.Name -ne "0.6B") {
            $rows.Add([pscustomobject]@{
                Model     = $model.Name
                Scenario  = $Scenario
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = "SKIP"
                ExitCode  = -1
                WallMs    = 0
                PeakRssMB = 0
                AudioSec  = 0
                RTF       = 0
                AudioStatus = ""
                AudioPeak = $null
                AudioRms = $null
                AudioNonZeroSamples = $null
                Note      = "original CLI has fixed 0.6B model (no --model-name)"
                LogPath   = ""
                OutputWav = ""
            })
        } else {
            $origOut = Join-Path $outRoot ("cpp_original_{0}_{1}_run{2}.wav" -f $Scenario, $model.Name, $rep)
            $origArgs = @(
                "-m", $modelDirPath,
                "-t", $Text,
                "-o", $origOut,
                "--max-tokens", $MaxTokens.ToString(),
                "--temperature", "0",
                "--top-k", "0",
                "--top-p", "1.0"
            )
            if ($Scenario -eq "voice_clone") {
                $origArgs += @("-r", $referenceAudioPath)
            } elseif ($Scenario -eq "voice_clone_synth_only") {
                $origArgs += @("--speaker-embedding", $cppSpeakerEmbedding)
            }

            $origRun = Invoke-BenchmarkProcess -FilePath $originalCliPath -Arguments $origArgs -WorkingDirectory $repoRoot
            $origMetricText = $origRun.Stdout + "`n" + $origRun.Stderr
            $origSynthMs = if ($Scenario -eq "voice_clone_synth_only") { Parse-CppTimingMs -Text $origMetricText -Label "Total" } else { $null }
            $origEffectiveMs = if ($null -ne $origSynthMs) { [double]$origSynthMs } else { [double]$origRun.WallMs }
            $origAudioSec = Parse-CppAudioDuration -Stdout $origMetricText
            if ($origAudioSec -le 0.0) {
                $origAudioSec = Get-WavDurationSec -Path $origOut
            }
            $origRtf = if ($origAudioSec -gt 0) { ($origEffectiveMs / 1000.0) / $origAudioSec } else { 0.0 }
            $origAudioResult = Get-BenchmarkAudioResult -ExitCode $origRun.ExitCode -OutputWav $origOut
            $origStatus = $origAudioResult.Status
            if ($origAudioSec -le 0.0 -and $origAudioResult.Stats.Valid) {
                $origAudioSec = $origAudioResult.Stats.DurationSec
                $origRtf = if ($origAudioSec -gt 0) { ($origEffectiveMs / 1000.0) / $origAudioSec } else { 0.0 }
            }

            $origLog = Join-Path $logDir ("{0}_{1}_cpp_original_run{2}.log" -f $Scenario, $model.Name, $rep)
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
                Scenario  = $Scenario
                Pipeline  = "cpp_original"
                Run       = $rep
                Status    = $origStatus
                ExitCode  = $origRun.ExitCode
                WallMs    = [math]::Round($origEffectiveMs, 1)
                PeakRssMB = [math]::Round($origRun.PeakWorkingSet / 1MB, 2)
                AudioSec  = [math]::Round($origAudioSec, 3)
                RTF       = [math]::Round($origRtf, 3)
                AudioStatus = $origAudioResult.AudioStatus
                AudioPeak = if ($origAudioResult.Stats.Valid) { [math]::Round($origAudioResult.Stats.Peak, 8) } else { $null }
                AudioRms = if ($origAudioResult.Stats.Valid) { [math]::Round($origAudioResult.Stats.Rms, 8) } else { $null }
                AudioNonZeroSamples = if ($origAudioResult.Stats.Valid) { $origAudioResult.Stats.NonZeroSamples } else { $null }
                Note      = ""
                LogPath   = $origLog
                OutputWav = $origOut
            })
        }
    }
}

$rawCsv = Join-Path $outRoot "benchmark_python_vs_cpp_raw.csv"
$rows | Sort-Object Model, Scenario, Pipeline, Run | Export-Csv -Path $rawCsv -NoTypeInformation

$summary = New-Object System.Collections.Generic.List[object]
$pipelines = @($rows | Select-Object -ExpandProperty Pipeline -Unique | Sort-Object)

foreach ($model in $models) {
    $cppForkPass = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Scenario -eq $Scenario -and $_.Pipeline -eq "cpp_fork" -and $_.Status -eq "PASS" })
    $cppForkWallAvg = if ($cppForkPass.Count -gt 0) { ($cppForkPass | Measure-Object WallMs -Average).Average } else { 0.0 }

    foreach ($pipeline in $pipelines) {
        $passRows = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Scenario -eq $Scenario -and $_.Pipeline -eq $pipeline -and $_.Status -eq "PASS" })
        $wallAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object WallMs -Average).Average } else { 0.0 }
        $rtfAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object RTF -Average).Average } else { 0.0 }
        $memAvg = if ($passRows.Count -gt 0) { ($passRows | Measure-Object PeakRssMB -Average).Average } else { 0.0 }
        $skipRows = @($rows | Where-Object { $_.Model -eq $model.Name -and $_.Scenario -eq $Scenario -and $_.Pipeline -eq $pipeline -and $_.Status -eq "SKIP" })
        $skipReason = if ($skipRows.Count -gt 0) { ($skipRows[0].Note) } else { "" }
        $vsCppFork = if ($pipeline -eq "cpp_fork") { 1.0 } elseif ($cppForkWallAvg -gt 0) { $wallAvg / $cppForkWallAvg } else { 0.0 }

        $summary.Add([pscustomobject]@{
            Model      = $model.Name
            Scenario   = $Scenario
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
}

$summaryCsv = Join-Path $outRoot "benchmark_python_vs_cpp_summary.csv"
$summary |
    Sort-Object `
        @{Expression = "Model"; Ascending = $true }, `
        @{Expression = { if ($pipelineOrder.ContainsKey($_.Pipeline)) { $pipelineOrder[$_.Pipeline] } else { 99 } }; Ascending = $true } |
    Export-Csv -Path $summaryCsv -NoTypeInformation

Write-Host ""
Write-Host "Raw Results"
$rows | Sort-Object Model, Scenario, Pipeline, Run | Format-Table Model, Scenario, Pipeline, Run, Status, ExitCode, WallMs, PeakRssMB, AudioSec, RTF, AudioStatus, AudioPeak, AudioRms, Note -AutoSize

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
