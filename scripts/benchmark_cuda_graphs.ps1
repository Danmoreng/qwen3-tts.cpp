param(
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$ModelDir = "models",
    [string]$OutputDir = "benchmark_output",
    [string]$Text = "Hello from benchmark.",
    [int]$MaxTokens = 128,
    [int]$TopK = 1,
    [double]$Temperature = 0.0,
    [string]$ModelF16 = "qwen3-tts-1.7b-base-f16.gguf",
    [string]$ModelQ80 = "qwen3-tts-1.7b-base-q8_0.gguf",
    [string]$ModelQ4K = "qwen3-tts-1.7b-base-q4_k.gguf"
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
    $gen = Parse-Int (Get-FirstMatch $text '(?mi)^\s*(?:Code generation|Generate):\s*([0-9]+)\s*ms')
    $dec = Parse-Int (Get-FirstMatch $text '(?mi)^\s*(?:Vocoder decode|Decode):\s*([0-9]+)\s*ms')
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repoRoot $BuildDir }
$resolvedModelDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $repoRoot $ModelDir }
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }

New-Item -Path $resolvedOutputDir -ItemType Directory -Force | Out-Null

$cliExe = Resolve-BinaryPath -name "qwen3-tts-cli" -buildDir $resolvedBuildDir -config $Configuration
if (-not $cliExe) {
    throw "Could not find qwen3-tts-cli.exe in '$resolvedBuildDir'. Build first with .\build.ps1."
}

$runs = @(
    [PSCustomObject]@{ Name = "f16_graphs_on";  ModelName = $ModelF16; GraphsOn = $true  },
    [PSCustomObject]@{ Name = "q8_0_graphs_on"; ModelName = $ModelQ80; GraphsOn = $true  },
    [PSCustomObject]@{ Name = "q4_k_graphs_on"; ModelName = $ModelQ4K; GraphsOn = $true  },
    [PSCustomObject]@{ Name = "q8_0_graphs_off"; ModelName = $ModelQ80; GraphsOn = $false },
    [PSCustomObject]@{ Name = "q4_k_graphs_off"; ModelName = $ModelQ4K; GraphsOn = $false }
)

$originalDisableGraphs = [Environment]::GetEnvironmentVariable("GGML_CUDA_DISABLE_GRAPHS", "Process")
$results = New-Object System.Collections.Generic.List[object]

try {
    foreach ($run in $runs) {
        $modelPath = Join-Path $resolvedModelDir $run.ModelName
        $wavPath = Join-Path $resolvedOutputDir ("{0}.wav" -f $run.Name)
        $logPath = Join-Path $resolvedOutputDir ("{0}.log" -f $run.Name)

        if (-not (Test-Path $modelPath)) {
            $results.Add([PSCustomObject]@{
                Run       = $run.Name
                Model     = $run.ModelName
                Graphs    = if ($run.GraphsOn) { "ON" } else { "OFF" }
                Status    = "MISSING_MODEL"
                ExitCode  = ""
                TotalMs   = ""
                GenMs     = ""
                DecodeMs  = ""
                AudioSec  = ""
                RTF       = ""
                XRealtime = ""
                LogPath   = $logPath
                WavPath   = $wavPath
            })
            continue
        }

        if ($run.GraphsOn) {
            Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
        } else {
            $env:GGML_CUDA_DISABLE_GRAPHS = "1"
        }

        $graphsLabel = if ($run.GraphsOn) { "graphs ON" } else { "graphs OFF" }
        Write-Host ""
        Write-Host ("--- {0} ({1}) ---" -f $run.Name, $graphsLabel) -ForegroundColor Cyan

        $args = @(
            "-m", $resolvedModelDir,
            "--model-name", $run.ModelName,
            "--temperature", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Temperature)),
            "--top-k", "$TopK",
            "--max-tokens", "$MaxTokens",
            "-t", $Text,
            "-o", $wavPath
        )

        $cmdLine = ('"{0}" {1}' -f $cliExe, (($args | ForEach-Object {
            if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
        }) -join ' '))

        $res = Invoke-CommandCapture -exe $cliExe -commandArgs $args
        $timing = Parse-Timing -text $res.Output

        $logBody = @(
            "Command: $cmdLine"
            "ExitCode: $($res.ExitCode)"
            ""
            $res.Output
        ) -join "`r`n"
        Set-Content -Path $logPath -Value $logBody -Encoding UTF8

        $status = if ($res.ExitCode -eq 0 -and (Test-Path $wavPath)) { "PASS" } else { "FAIL" }

        $results.Add([PSCustomObject]@{
            Run       = $run.Name
            Model     = $run.ModelName
            Graphs    = if ($run.GraphsOn) { "ON" } else { "OFF" }
            Status    = $status
            ExitCode  = $res.ExitCode
            TotalMs   = $timing.TotalMs
            GenMs     = $timing.GenerateMs
            DecodeMs  = $timing.DecodeMs
            AudioSec  = if ($null -ne $timing.AudioSec) { [Math]::Round($timing.AudioSec, 3) } else { $null }
            RTF       = if ($null -ne $timing.RTF) { [Math]::Round($timing.RTF, 3) } else { $null }
            XRealtime = if ($null -ne $timing.XRealtime) { [Math]::Round($timing.XRealtime, 3) } else { $null }
            LogPath   = $logPath
            WavPath   = $wavPath
        })
    }
}
finally {
    if ($null -eq $originalDisableGraphs) {
        Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    } else {
        $env:GGML_CUDA_DISABLE_GRAPHS = $originalDisableGraphs
    }
}

$csvPath = Join-Path $resolvedOutputDir "benchmark_summary.csv"
$results |
    Select-Object Run, Model, Graphs, Status, ExitCode, TotalMs, GenMs, DecodeMs, AudioSec, RTF, XRealtime, LogPath, WavPath |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csvPath

Write-Host ""
Write-Host "Benchmark Summary" -ForegroundColor Cyan
$results |
    Select-Object Run, Model, Graphs, Status, ExitCode, TotalMs, GenMs, DecodeMs, AudioSec, RTF, XRealtime |
    Format-Table -AutoSize

Write-Host ""
Write-Host "CSV: $csvPath" -ForegroundColor Green
Write-Host "Logs/WAVs: $resolvedOutputDir" -ForegroundColor Green
