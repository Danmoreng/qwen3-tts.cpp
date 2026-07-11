[CmdletBinding()]
param(
    [int]$Runs = 8,
    [int]$Warmup = 1,
    [int]$MaxTokens = 4096,
    [int]$Threads = 4,
    [string]$Text = "The strange thing about knowledge is that the more you gather, the more you realize how much remains unknown. Perhaps that is the whole point of the journey.",
    [string]$ReferenceAudio = "",
    [string]$ReferenceText = "",
    [string]$Language = "en",
    [int]$Seed = 42,
    [double]$Temperature = 0.9,
    [int]$TopK = 50,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.05,
    [string]$OutDir = "",
    [string]$WorkspaceRoot = "",
    [string]$QwenCppExe = "",
    [string]$QwenCppModels = "",
    [string]$QwenCppModelName = "qwen-talker-1.7b-base-Q8_0.gguf",
    [string]$QwenCppCodecName = "qwen-tokenizer-12hz-Q8_0.gguf",
    [string]$ServeurRepo = "",
    [string]$ServeurExe = "",
    [string]$ServeurCodecExe = "",
    [string]$ServeurServerExe = "",
    [string]$ServeurTalker = "",
    [string]$ServeurCodec = "",
    [switch]$SkipCold,
    [switch]$SkipServer,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

. (Join-Path $PSScriptRoot "wav_stats.ps1")

function Find-FirstExisting([string[]]$Paths) {
    foreach ($path in $Paths) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return ""
}

function Resolve-ExistingPath([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Find-StudioModelDir() {
    $candidate = Join-Path $env:USERPROFILE ".qwen-tts-studio\models"
    if ((Test-Path -LiteralPath (Join-Path $candidate "qwen-talker-1.7b-base-Q8_0.gguf")) -and
        (Test-Path -LiteralPath (Join-Path $candidate "qwen-tokenizer-12hz-Q8_0.gguf"))) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    return ""
}

function Convert-ToServeurLanguage([string]$Value) {
    $map = @{
        "en" = "English"; "zh" = "Chinese"; "de" = "German"; "fr" = "French"
        "es" = "Spanish"; "it" = "Italian"; "ja" = "Japanese"; "ko" = "Korean"
        "pt" = "Portuguese"; "ru" = "Russian"; "auto" = "auto"
    }
    if ($map.ContainsKey($Value)) { return $map[$Value] }
    return $Value
}

function Quote-Arg([string]$Arg) {
    if ([string]::IsNullOrEmpty($Arg)) { return '""' }
    if ($Arg -notmatch '[\s"]') { return $Arg }
    return '"' + ($Arg -replace '"', '\"') + '"'
}

function Format-CommandLine([string]$Exe, [string[]]$CommandArgs) {
    return (Quote-Arg $Exe) + " " + (($CommandArgs | ForEach-Object { Quote-Arg $_ }) -join " ")
}

function Invoke-BenchCommand([string]$Name, [string]$Exe, [string[]]$CommandArgs, [string]$Cwd, [string]$LogPath, [string]$StdinText = "") {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Exe
    $psi.Arguments = ($CommandArgs | ForEach-Object { Quote-Arg $_ }) -join " "
    $psi.WorkingDirectory = $Cwd
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = -not [string]::IsNullOrEmpty($StdinText)
    $psi.UseShellExecute = $false

    $p = [System.Diagnostics.Process]::new()
    $p.StartInfo = $psi
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$p.Start()
    if ($psi.RedirectStandardInput) {
        $p.StandardInput.Write($StdinText)
        $p.StandardInput.Close()
    }
    $stdoutTask = $p.StandardOutput.ReadToEndAsync()
    $stderrTask = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    $sw.Stop()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $text = $stdout + $stderr

    New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
    @(
        "Implementation: $Name",
        "Command: $(Format-CommandLine $Exe $CommandArgs)",
        "ExitCode: $($p.ExitCode)",
        "WallSeconds: $($sw.Elapsed.TotalSeconds)",
        "",
        "[stdout]",
        $stdout,
        "",
        "[stderr]",
        $stderr
    ) | Set-Content -LiteralPath $LogPath -Encoding UTF8

    [PSCustomObject]@{
        Name = $Name
        ExitCode = $p.ExitCode
        WallSeconds = $sw.Elapsed.TotalSeconds
        LogText = $text
        Stdout = $stdout
        Stderr = $stderr
        CommandLine = Format-CommandLine $Exe $CommandArgs
        LogPath = $LogPath
    }
}

function Get-RegexMetric([string]$Text, [string]$Pattern) {
    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($m.Success) {
        return [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return $null
}

function New-ResultRow([string]$Engine, [string]$Scope, [int]$Run, [string]$OutWav, [double]$WallSeconds, [int]$ExitCode, [string]$LogPath, [string]$Command, [Nullable[double]]$TtfaMs = $null, [object]$BenchJson = $null, [string]$LogText = "") {
    $stats = if (-not [string]::IsNullOrWhiteSpace($OutWav) -and (Test-Path -LiteralPath $OutWav)) {
        Get-WavAudioStats -Path $OutWav
    } else {
        New-EmptyWavAudioStats -path $(if ($OutWav) { $OutWav } else { "N/A" }) -errorMessage "file not found"
    }
    $audioSec = if ($stats.Valid) { [double]$stats.DurationSec } elseif ($BenchJson -and $BenchJson.audio_sec) { [double]$BenchJson.audio_sec } else { 0.0 }
    $rtf = if ($audioSec -gt 0.0) { $WallSeconds / $audioSec } else { $null }
    $xRealtime = if ($WallSeconds -gt 0.0) { $audioSec / $WallSeconds } else { $null }
    $internalTotalMs = $null
    $generateMs = $null
    $decodeMs = $null
    $parsedTtfaMs = $null
    if ($BenchJson) {
        $internalTotalMs = [double]$BenchJson.internal_total_ms
        $generateMs = [double]$BenchJson.generate_ms
        $decodeMs = [double]$BenchJson.decode_ms
        if ($BenchJson.ttfa_ms -ge 0) { $parsedTtfaMs = [double]$BenchJson.ttfa_ms }
    } elseif (-not [string]::IsNullOrWhiteSpace($LogText)) {
        if ($Engine -eq "qwen3-tts.cpp") {
            $internalTotalMs = Get-RegexMetric $LogText "^\s*Total:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            $generateMs = Get-RegexMetric $LogText "^\s*Generate:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            if ($null -eq $generateMs) {
                $generateMs = Get-RegexMetric $LogText "^\s*Code\+streaming:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            }
            $decodeMs = Get-RegexMetric $LogText "^\s*Decode:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            if ($null -eq $decodeMs) {
                $decodeMs = Get-RegexMetric $LogText "^\s*Streaming decode:\s*([0-9]+(?:\.[0-9]+)?)\s+ms"
            }
        } elseif ($Engine -eq "qwentts.cpp") {
            $internalTotalMs = Get-RegexMetric $LogText "\[Perf\]\s+Total\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            $talkerMs = Get-RegexMetric $LogText "\[Perf\]\s+TalkerDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            $codePredMs = Get-RegexMetric $LogText "\[Perf\]\s+CodePredictor\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            if ($null -ne $talkerMs -and $null -ne $codePredMs) {
                $generateMs = [double]$talkerMs + [double]$codePredMs
            } else {
                $generateMs = $talkerMs
            }
            $decodeMs = Get-RegexMetric $LogText "\[Perf\]\s+CodecDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
            $parsedTtfaMs = Get-RegexMetric $LogText "\[Perf\]\s+TTFA\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        }
    }
    [PSCustomObject]@{
        Engine = $Engine
        Scope = $Scope
        Run = $Run
        ExitCode = $ExitCode
        AudioStatus = if ($stats.Valid) { Get-WavAudioQualityStatus -Stats $stats } elseif ($audioSec -gt 0.0) { "RAW_PCM" } else { "N/A" }
        WallSeconds = [Math]::Round($WallSeconds, 3)
        AudioSeconds = [Math]::Round($audioSec, 3)
        RTF_WallPerAudio = if ($null -ne $rtf) { [Math]::Round($rtf, 4) } else { $null }
        XRealtime_AudioPerWall = if ($null -ne $xRealtime) { [Math]::Round($xRealtime, 3) } else { $null }
        MsPerAudioSecond = if ($null -ne $rtf) { [Math]::Round($rtf * 1000.0, 1) } else { $null }
        TTFA_Ms = if ($null -ne $TtfaMs) { [Math]::Round([double]$TtfaMs, 1) } elseif ($null -ne $parsedTtfaMs) { [Math]::Round([double]$parsedTtfaMs, 1) } else { $null }
        InternalTotalMs = if ($null -ne $internalTotalMs) { [Math]::Round([double]$internalTotalMs, 1) } else { $null }
        GenerateMs = if ($null -ne $generateMs) { [Math]::Round([double]$generateMs, 1) } else { $null }
        DecodeMs = if ($null -ne $decodeMs) { [Math]::Round([double]$decodeMs, 1) } else { $null }
        StreamChunks = if ($BenchJson) { [int]$BenchJson.stream_chunks } else { $null }
        Output = $OutWav
        LogPath = $LogPath
        Command = $Command
    }
}

function Convert-RawPcm16ToWav([string]$PcmPath, [string]$WavPath, [int]$SampleRate = 24000) {
    $pcm = [System.IO.File]::ReadAllBytes($PcmPath)
    $dataSize = $pcm.Length
    $riffSize = 36 + $dataSize
    $fs = [System.IO.File]::Create($WavPath)
    try {
        $bw = [System.IO.BinaryWriter]::new($fs)
        try {
            $bw.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
            $bw.Write([uint32]$riffSize)
            $bw.Write([System.Text.Encoding]::ASCII.GetBytes("WAVEfmt "))
            $bw.Write([uint32]16)
            $bw.Write([uint16]1)
            $bw.Write([uint16]1)
            $bw.Write([uint32]$SampleRate)
            $bw.Write([uint32]($SampleRate * 2))
            $bw.Write([uint16]2)
            $bw.Write([uint16]16)
            $bw.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
            $bw.Write([uint32]$dataSize)
            $bw.Write($pcm)
        } finally {
            $bw.Dispose()
        }
    } finally {
        $fs.Dispose()
    }
}

function Invoke-StreamingSpeechRequest([string]$Url, [object]$Payload, [string]$PcmPath) {
    $client = [System.Net.Http.HttpClient]::new()
    try {
        $json = $Payload | ConvertTo-Json -Depth 8 -Compress
        $content = [System.Net.Http.StringContent]::new($json, [System.Text.Encoding]::UTF8, "application/json")
        $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, $Url)
        $req.Content = $content
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        [void]$resp.EnsureSuccessStatusCode()
        $stream = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $fs = [System.IO.File]::Create($PcmPath)
        $buffer = New-Object byte[] 65536
        $ttfa = $null
        $bytes = 0L
        try {
            while (($n = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                if ($null -eq $ttfa) { $ttfa = $sw.Elapsed.TotalMilliseconds }
                $fs.Write($buffer, 0, $n)
                $bytes += $n
            }
        } finally {
            $fs.Dispose()
            $stream.Dispose()
            $resp.Dispose()
        }
        $sw.Stop()
        return [PSCustomObject]@{ WallSeconds = $sw.Elapsed.TotalSeconds; TtfaMs = $ttfa; Bytes = $bytes }
    } finally {
        $client.Dispose()
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = (Split-Path -Parent $repoRoot)
}
$WorkspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
if ([string]::IsNullOrWhiteSpace($ServeurRepo)) {
    $ServeurRepo = Join-Path $WorkspaceRoot "qwentts.cpp-serveurperso"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $WorkspaceRoot ("benchmark_output\qwentts_matrix\{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
}
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$logDir = Join-Path $OutDir "logs"
$artifactDir = Join-Path $OutDir "artifacts"
New-Item -ItemType Directory -Path $logDir, $artifactDir -Force | Out-Null

$studioModels = Find-StudioModelDir
if ([string]::IsNullOrWhiteSpace($QwenCppExe)) {
    $QwenCppExe = Find-FirstExisting @(
        (Join-Path $repoRoot "build\Release\qwen3-tts-cli.exe"),
        (Join-Path $repoRoot "build\qwen3-tts-cli.exe"),
        (Join-Path $repoRoot "build-timing\Release\qwen3-tts-cli.exe"),
        (Join-Path $repoRoot "build-timing\qwen3-tts-cli.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($QwenCppModels)) {
    $QwenCppModels = if ($studioModels) { $studioModels } else { Join-Path $repoRoot "models" }
}
if ([string]::IsNullOrWhiteSpace($ServeurExe)) {
    $ServeurExe = Find-FirstExisting @(
        (Join-Path $ServeurRepo "build-sm120-cuda133\Release\qwen-tts.exe"),
        (Join-Path $ServeurRepo "build\Release\qwen-tts.exe"),
        (Join-Path $ServeurRepo "build\qwen-tts.exe"),
        (Join-Path $ServeurRepo "build-sm120\Release\qwen-tts.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($ServeurCodecExe)) {
    $ServeurCodecExe = Find-FirstExisting @(
        (Join-Path $ServeurRepo "build-sm120-cuda133\Release\qwen-codec.exe"),
        (Join-Path $ServeurRepo "build\Release\qwen-codec.exe"),
        (Join-Path $ServeurRepo "build\qwen-codec.exe"),
        (Join-Path $ServeurRepo "build-sm120\Release\qwen-codec.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($ServeurServerExe)) {
    $ServeurServerExe = Find-FirstExisting @(
        (Join-Path $ServeurRepo "build-sm120-cuda133\Release\tts-server.exe"),
        (Join-Path $ServeurRepo "build\Release\tts-server.exe"),
        (Join-Path $ServeurRepo "build\tts-server.exe"),
        (Join-Path $ServeurRepo "build-sm120\Release\tts-server.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($ServeurTalker)) {
    $ServeurTalker = if ($studioModels) { Join-Path $studioModels $QwenCppModelName } else { Join-Path $ServeurRepo "models\$QwenCppModelName" }
}
if ([string]::IsNullOrWhiteSpace($ServeurCodec)) {
    $ServeurCodec = if ($studioModels) { Join-Path $studioModels "qwen-tokenizer-12hz-Q8_0.gguf" } else { Join-Path $ServeurRepo "models\qwen-tokenizer-12hz-Q8_0.gguf" }
}
if ([string]::IsNullOrWhiteSpace($ReferenceAudio)) {
    $ReferenceAudio = Join-Path $ServeurRepo "examples\freeman.wav"
}
if ([string]::IsNullOrWhiteSpace($ReferenceText)) {
    $sidecar = [System.IO.Path]::ChangeExtension($ReferenceAudio, ".txt")
    if (Test-Path -LiteralPath $sidecar) {
        $ReferenceText = (Get-Content -LiteralPath $sidecar -Raw).Trim()
    } else {
        $fallback = Join-Path $repoRoot "reference_text.txt"
        if (Test-Path -LiteralPath $fallback) {
            $ReferenceText = (Get-Content -LiteralPath $fallback -Raw).Trim()
        }
    }
}

$QwenCppExe = Resolve-ExistingPath $QwenCppExe "qwen3-tts.cpp CLI"
$QwenCppModels = Resolve-ExistingPath $QwenCppModels "qwen3-tts.cpp model directory"
[void](Resolve-ExistingPath (Join-Path $QwenCppModels $QwenCppModelName) "qwen3-tts.cpp talker GGUF")
$QwenCppCodecPath = Resolve-ExistingPath (Join-Path $QwenCppModels $QwenCppCodecName) "qwen3-tts.cpp codec GGUF"
$ServeurExe = Resolve-ExistingPath $ServeurExe "qwentts.cpp qwen-tts"
$ServeurCodecExe = Resolve-ExistingPath $ServeurCodecExe "qwentts.cpp qwen-codec"
$ServeurTalker = Resolve-ExistingPath $ServeurTalker "qwentts.cpp talker GGUF"
$ServeurCodec = Resolve-ExistingPath $ServeurCodec "qwentts.cpp codec GGUF"
$ReferenceAudio = Resolve-ExistingPath $ReferenceAudio "reference audio"
if ([string]::IsNullOrWhiteSpace($ReferenceText)) {
    throw "Reference text is required for full ICL voice-clone benchmarks."
}
if (-not $SkipServer) {
    $ServeurServerExe = Resolve-ExistingPath $ServeurServerExe "qwentts.cpp tts-server"
}

$referenceTextFile = Join-Path $artifactDir "reference.txt"
Set-Content -LiteralPath $referenceTextFile -Value $ReferenceText -Encoding UTF8

Write-Host "Benchmark matrix preflight" -ForegroundColor Cyan
Write-Host "  OutDir:        $OutDir"
Write-Host "  Runs/Warmup:   $Runs / $Warmup"
Write-Host "  qwen3 CLI:     $QwenCppExe"
Write-Host "  qwentts CLI:   $ServeurExe"
Write-Host "  qwentts server:$ServeurServerExe"
Write-Host "  Models:        $QwenCppModels"
Write-Host "  Reference:     $ReferenceAudio"
Write-Host ""

if ($ValidateOnly) {
    Write-Host "ValidateOnly completed. No benchmark commands were run." -ForegroundColor Green
    return
}

$rows = New-Object System.Collections.Generic.List[object]
$qwenIclPrompt = Join-Path $artifactDir "qwen3_icl_prompt.json"
$serveurRefCopy = Join-Path $artifactDir ([System.IO.Path]::GetFileName($ReferenceAudio))
Copy-Item -LiteralPath $ReferenceAudio -Destination $serveurRefCopy -Force
$serveurSpk = [System.IO.Path]::ChangeExtension($serveurRefCopy, ".spk")
$serveurRvq = [System.IO.Path]::ChangeExtension($serveurRefCopy, ".rvq")

Write-Host "[artifacts] qwen3 ICL prompt" -ForegroundColor Yellow
$cmd = Invoke-BenchCommand "qwen3-tts.cpp" $QwenCppExe @(
    "-m", $QwenCppModels, "--model-name", $QwenCppModelName,
    "--codec-model", $QwenCppCodecPath,
    "-r", $ReferenceAudio, "--reference-text-file", $referenceTextFile,
    "--extract-icl-prompt", $qwenIclPrompt, "-j", "$Threads"
) $repoRoot (Join-Path $logDir "qwen3_extract_icl.log")
if ($cmd.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $qwenIclPrompt)) {
    throw "qwen3 ICL prompt extraction failed. See $($cmd.LogPath)"
}

Write-Host "[artifacts] qwentts voice latents" -ForegroundColor Yellow
$cmd = Invoke-BenchCommand "qwentts.cpp" $ServeurCodecExe @(
    "--model", $ServeurCodec, "--talker", $ServeurTalker, "-i", $serveurRefCopy
) $ServeurRepo (Join-Path $logDir "qwentts_codec_extract.log")
if ($cmd.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $serveurSpk) -or -not (Test-Path -LiteralPath $serveurRvq)) {
    throw "qwentts voice latent extraction failed. See $($cmd.LogPath)"
}

$qwenBaseArgs = @("-m", $QwenCppModels, "--model-name", $QwenCppModelName, "--codec-model", $QwenCppCodecPath, "-t", $Text, "--max-tokens", "$MaxTokens", "--seed", "$Seed", "--temperature", "$Temperature", "--top-k", "$TopK", "--top-p", "$TopP", "--repetition-penalty", "$RepetitionPenalty", "-l", $Language, "-j", "$Threads")
$serveurBaseArgs = @("--model", $ServeurTalker, "--codec", $ServeurCodec, "--lang", (Convert-ToServeurLanguage $Language), "--max-new", "$MaxTokens", "--seed", "$Seed", "--temp", "$Temperature", "--top-k", "$TopK", "--top-p", "$TopP", "--rep-pen", "$RepetitionPenalty")

if (-not $SkipCold) {
    for ($run = 1; $run -le $Runs; ++$run) {
        Write-Host "[$run/$Runs] cold_e2e_ref_wav" -ForegroundColor Yellow
        $out = Join-Path $OutDir "qwen3_cold_ref_run$run.wav"
        $log = Join-Path $logDir "qwen3_cold_ref_run$run.log"
        $cmd = Invoke-BenchCommand "qwen3-tts.cpp" $QwenCppExe ($qwenBaseArgs + @("-o", $out, "-r", $ReferenceAudio, "--reference-text-file", $referenceTextFile)) $repoRoot $log
        $rows.Add((New-ResultRow "qwen3-tts.cpp" "cold_e2e_ref_wav" $run $out $cmd.WallSeconds $cmd.ExitCode $log $cmd.CommandLine -LogText $cmd.LogText)) | Out-Null

        $out = Join-Path $OutDir "qwentts_cold_ref_run$run.wav"
        $log = Join-Path $logDir "qwentts_cold_ref_run$run.log"
        $cmd = Invoke-BenchCommand "qwentts.cpp" $ServeurExe ($serveurBaseArgs + @("-o", $out, "--ref-wav", $ReferenceAudio, "--ref-text", $referenceTextFile)) $ServeurRepo $log $Text
        $rows.Add((New-ResultRow "qwentts.cpp" "cold_e2e_ref_wav" $run $out $cmd.WallSeconds $cmd.ExitCode $log $cmd.CommandLine -LogText $cmd.LogText)) | Out-Null

        Write-Host "[$run/$Runs] cold_e2e_preencoded" -ForegroundColor Yellow
        $out = Join-Path $OutDir "qwen3_cold_preencoded_run$run.wav"
        $log = Join-Path $logDir "qwen3_cold_preencoded_run$run.log"
        $cmd = Invoke-BenchCommand "qwen3-tts.cpp" $QwenCppExe ($qwenBaseArgs + @("-o", $out, "--icl-prompt", $qwenIclPrompt)) $repoRoot $log
        $rows.Add((New-ResultRow "qwen3-tts.cpp" "cold_e2e_preencoded" $run $out $cmd.WallSeconds $cmd.ExitCode $log $cmd.CommandLine -LogText $cmd.LogText)) | Out-Null

        $out = Join-Path $OutDir "qwentts_cold_preencoded_run$run.wav"
        $log = Join-Path $logDir "qwentts_cold_preencoded_run$run.log"
        $cmd = Invoke-BenchCommand "qwentts.cpp" $ServeurExe ($serveurBaseArgs + @("-o", $out, "--ref-spk", $serveurSpk, "--ref-rvq", $serveurRvq, "--ref-text", $referenceTextFile)) $ServeurRepo $log $Text
        $rows.Add((New-ResultRow "qwentts.cpp" "cold_e2e_preencoded" $run $out $cmd.WallSeconds $cmd.ExitCode $log $cmd.CommandLine -LogText $cmd.LogText)) | Out-Null
    }
}

Write-Host "[resident] qwen3 buffered" -ForegroundColor Yellow
$out = Join-Path $OutDir "qwen3_resident.wav"
$log = Join-Path $logDir "qwen3_resident.log"
$cmd = Invoke-BenchCommand "qwen3-tts.cpp" $QwenCppExe ($qwenBaseArgs + @("-o", $out, "--icl-prompt", $qwenIclPrompt, "--bench-server", "$Runs", "--bench-warmup", "$Warmup")) $repoRoot $log
foreach ($line in ($cmd.Stdout -split "`r?`n")) {
    if ($line -match '^BENCH_JSON\s+(\{.+\})') {
        $json = $matches[1] | ConvertFrom-Json
        if (-not $json.warmup) {
            $rows.Add((New-ResultRow "qwen3-tts.cpp" "resident_preencoded" ([int]$json.iteration) $json.output ([double]$json.wall_ms / 1000.0) ([int]$json.exit_code) $log $cmd.CommandLine $null $json)) | Out-Null
        }
    }
}

Write-Host "[resident] qwen3 streaming" -ForegroundColor Yellow
$out = Join-Path $OutDir "qwen3_resident_stream.wav"
$log = Join-Path $logDir "qwen3_resident_stream.log"
$cmd = Invoke-BenchCommand "qwen3-tts.cpp" $QwenCppExe ($qwenBaseArgs + @("-o", $out, "--icl-prompt", $qwenIclPrompt, "--stream", "--bench-server", "$Runs", "--bench-warmup", "$Warmup")) $repoRoot $log
foreach ($line in ($cmd.Stdout -split "`r?`n")) {
    if ($line -match '^BENCH_JSON\s+(\{.+\})') {
        $json = $matches[1] | ConvertFrom-Json
        if (-not $json.warmup) {
            $rows.Add((New-ResultRow "qwen3-tts.cpp" "resident_streaming_preencoded" ([int]$json.iteration) $json.output ([double]$json.wall_ms / 1000.0) ([int]$json.exit_code) $log $cmd.CommandLine $null $json)) | Out-Null
        }
    }
}

if (-not $SkipServer) {
    $port = Get-Random -Minimum 18080 -Maximum 25000
    $serverOut = Join-Path $logDir "qwentts_server_stdout.log"
    $serverErr = Join-Path $logDir "qwentts_server_stderr.log"
    $serverArgs = @("--model", $ServeurTalker, "--codec", $ServeurCodec, "--host", "127.0.0.1", "--port", "$port", "--lang", (Convert-ToServeurLanguage $Language))
    Write-Host "[server] starting qwentts on port $port" -ForegroundColor Yellow
    $server = Start-Process -FilePath $ServeurServerExe -ArgumentList $serverArgs -WorkingDirectory $ServeurRepo -NoNewWindow -PassThru -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
    try {
        $health = "http://127.0.0.1:$port/health"
        $ready = $false
        for ($i = 0; $i -lt 80; ++$i) {
            try {
                [void](Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 1)
                $ready = $true
                break
            } catch {
                Start-Sleep -Milliseconds 250
            }
        }
        if (-not $ready) { throw "qwentts server did not become ready. See $serverErr" }

        $voicePayload = [ordered]@{
            name = "bench_voice"
            ref_text = $ReferenceText
            spk_b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($serveurSpk))
            rvq_b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($serveurRvq))
        }
        Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/voices" -Method Post -ContentType "application/json" -Body ($voicePayload | ConvertTo-Json -Depth 8 -Compress) | Out-Null

        for ($run = 1; $run -le ($Warmup + $Runs); ++$run) {
            $warmupRun = $run -le $Warmup
            $iteration = if ($warmupRun) { $run } else { $run - $Warmup }
            $payload = [ordered]@{
                input = $Text
                voice = "bench_voice"
                response_format = "wav"
                seed = $Seed
                max_new_tokens = $MaxTokens
                temperature = $Temperature
                top_k = $TopK
                top_p = $TopP
                repetition_penalty = $RepetitionPenalty
            }
            $out = Join-Path $OutDir "qwentts_server_buffered_run$iteration.wav"
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/audio/speech" -Method Post -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 8 -Compress) -OutFile $out -UseBasicParsing | Out-Null
            $sw.Stop()
            if (-not $warmupRun) {
                $rows.Add((New-ResultRow "qwentts.cpp" "http_server_preencoded" $iteration $out $sw.Elapsed.TotalSeconds 0 $serverErr "POST /v1/audio/speech response_format=wav")) | Out-Null
            }
        }

        for ($run = 1; $run -le ($Warmup + $Runs); ++$run) {
            $warmupRun = $run -le $Warmup
            $iteration = if ($warmupRun) { $run } else { $run - $Warmup }
            $payload = [ordered]@{
                input = $Text
                voice = "bench_voice"
                response_format = "pcm"
                seed = $Seed
                max_new_tokens = $MaxTokens
                temperature = $Temperature
                top_k = $TopK
                top_p = $TopP
                repetition_penalty = $RepetitionPenalty
            }
            $pcm = Join-Path $OutDir "qwentts_server_stream_run$iteration.pcm"
            $wav = Join-Path $OutDir "qwentts_server_stream_run$iteration.wav"
            $res = Invoke-StreamingSpeechRequest "http://127.0.0.1:$port/v1/audio/speech" $payload $pcm
            Convert-RawPcm16ToWav $pcm $wav
            if (-not $warmupRun) {
                $rows.Add((New-ResultRow "qwentts.cpp" "http_server_streaming_preencoded" $iteration $wav $res.WallSeconds 0 $serverErr "POST /v1/audio/speech response_format=pcm" $res.TtfaMs)) | Out-Null
            }
        }
    } finally {
        if ($server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
            $server.WaitForExit()
        }
    }
}

$csv = Join-Path $OutDir "benchmark_matrix_results.csv"
$jsonPath = Join-Path $OutDir "benchmark_matrix_results.json"
$rows | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
$rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$summary = $rows |
    Where-Object { $_.ExitCode -eq 0 -and $null -ne $_.RTF_WallPerAudio } |
    Group-Object Engine, Scope |
    ForEach-Object {
        $items = @($_.Group)
        [PSCustomObject]@{
            Engine = $items[0].Engine
            Scope = $items[0].Scope
            Runs = $items.Count
            AvgWallSec = [Math]::Round((($items | Measure-Object WallSeconds -Average).Average), 3)
            AvgAudioSec = [Math]::Round((($items | Measure-Object AudioSeconds -Average).Average), 3)
            AvgRTF = [Math]::Round((($items | Measure-Object RTF_WallPerAudio -Average).Average), 4)
            AvgXRealtime = [Math]::Round((($items | Measure-Object XRealtime_AudioPerWall -Average).Average), 3)
            AvgTTFAms = $(
                $ttfaItems = @($items | Where-Object { $null -ne $_.TTFA_Ms })
                if ($ttfaItems.Count -gt 0) {
                    [Math]::Round((($ttfaItems | Measure-Object TTFA_Ms -Average).Average), 1)
                } else {
                    $null
                }
            )
        }
    } | Sort-Object Scope, Engine

$summaryPath = Join-Path $OutDir "benchmark_matrix_summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Summary (standard RTF is wall/audio, lower is better):" -ForegroundColor Cyan
$summary | Format-Table Engine, Scope, Runs, AvgWallSec, AvgAudioSec, AvgRTF, AvgXRealtime, AvgTTFAms -AutoSize
Write-Host "Results: $csv"
Write-Host "Summary: $summaryPath"
