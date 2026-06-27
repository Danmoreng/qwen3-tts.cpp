Set-StrictMode -Version Latest

function ConvertTo-DbFs([double]$value) {
    if ($value -le 0.0) {
        return [double]::NegativeInfinity
    }
    return 20.0 * [Math]::Log10($value)
}

function New-EmptyWavAudioStats([string]$path, [string]$errorMessage) {
    return [PSCustomObject]@{
        Path           = $path
        Exists         = (Test-Path $path)
        Valid          = $false
        Error          = $errorMessage
        FormatTag      = 0
        BitsPerSample  = 0
        Channels       = 0
        SampleRate     = 0
        Samples        = 0
        DurationSec    = 0.0
        Min            = 0.0
        Max            = 0.0
        Peak           = 0.0
        Rms            = 0.0
        PeakDbFS       = [double]::NegativeInfinity
        RmsDbFS        = [double]::NegativeInfinity
        NonZeroSamples = 0
    }
}

function Read-Int24LE([byte[]]$bytes, [int]$offset) {
    $value = [int]$bytes[$offset] -bor ([int]$bytes[$offset + 1] -shl 8) -bor ([int]$bytes[$offset + 2] -shl 16)
    if (($value -band 0x800000) -ne 0) {
        $value -= 0x1000000
    }
    return $value
}

function Get-WavAudioStats {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return New-EmptyWavAudioStats -path $Path -errorMessage "file not found"
    }

    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $br = [System.IO.BinaryReader]::new($fs)
        try {
            $riff = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            [void]$br.ReadUInt32()
            $wave = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            if ($riff -ne "RIFF" -or $wave -ne "WAVE") {
                return New-EmptyWavAudioStats -path $Path -errorMessage "invalid RIFF/WAVE header"
            }

            $formatTag = 0
            $channels = 0
            $sampleRate = 0
            $bitsPerSample = 0
            $dataOffset = 0L
            $dataBytes = 0L

            while ($br.BaseStream.Position -le ($br.BaseStream.Length - 8)) {
                $chunkId = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
                $chunkSize = [int64]$br.ReadUInt32()
                $chunkStart = $br.BaseStream.Position

                if ($chunkId -eq "fmt ") {
                    $fmt = $br.ReadBytes([int]$chunkSize)
                    if ($fmt.Length -lt 16) {
                        return New-EmptyWavAudioStats -path $Path -errorMessage "fmt chunk too small"
                    }
                    $formatTag = [BitConverter]::ToUInt16($fmt, 0)
                    $channels = [BitConverter]::ToUInt16($fmt, 2)
                    $sampleRate = [int][BitConverter]::ToUInt32($fmt, 4)
                    $bitsPerSample = [BitConverter]::ToUInt16($fmt, 14)

                    if ($formatTag -eq 0xfffe -and $fmt.Length -ge 26) {
                        $formatTag = [BitConverter]::ToUInt16($fmt, 24)
                    }
                } elseif ($chunkId -eq "data") {
                    $dataOffset = $chunkStart
                    $dataBytes = $chunkSize
                    [void]$br.BaseStream.Seek($chunkSize, [System.IO.SeekOrigin]::Current)
                } else {
                    [void]$br.BaseStream.Seek($chunkSize, [System.IO.SeekOrigin]::Current)
                }

                if (($chunkSize % 2) -ne 0 -and $br.BaseStream.Position -lt $br.BaseStream.Length) {
                    [void]$br.ReadByte()
                }
            }

            if ($channels -le 0 -or $sampleRate -le 0 -or $bitsPerSample -le 0 -or $dataBytes -le 0) {
                return New-EmptyWavAudioStats -path $Path -errorMessage "missing or empty audio data"
            }

            $bytesPerSample = [int]($bitsPerSample / 8)
            if ($bytesPerSample -le 0 -or ($dataBytes % $bytesPerSample) -ne 0) {
                return New-EmptyWavAudioStats -path $Path -errorMessage "unsupported sample width"
            }
            if (($formatTag -ne 1 -or ($bitsPerSample -ne 16 -and $bitsPerSample -ne 24)) -and
                ($formatTag -ne 3 -or $bitsPerSample -ne 32)) {
                return New-EmptyWavAudioStats -path $Path -errorMessage "unsupported WAV format tag=$formatTag bits=$bitsPerSample"
            }

            [void]$br.BaseStream.Seek($dataOffset, [System.IO.SeekOrigin]::Begin)
            $data = $br.ReadBytes([int]$dataBytes)
            $sampleCount = [int]($dataBytes / $bytesPerSample)
            if ($sampleCount -le 0) {
                return New-EmptyWavAudioStats -path $Path -errorMessage "no samples"
            }

            $min = [double]::PositiveInfinity
            $max = [double]::NegativeInfinity
            $peak = 0.0
            $sumSquares = 0.0
            $nonZero = 0

            for ($i = 0; $i -lt $sampleCount; $i++) {
                $offset = $i * $bytesPerSample
                if ($formatTag -eq 1 -and $bitsPerSample -eq 16) {
                    $v = [double]([BitConverter]::ToInt16($data, $offset)) / 32768.0
                } elseif ($formatTag -eq 1 -and $bitsPerSample -eq 24) {
                    $v = [double](Read-Int24LE -bytes $data -offset $offset) / 8388608.0
                } else {
                    $v = [double]([BitConverter]::ToSingle($data, $offset))
                }

                if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) {
                    $v = 0.0
                }
                if ($v -ne 0.0) {
                    $nonZero++
                }
                if ($v -lt $min) {
                    $min = $v
                }
                if ($v -gt $max) {
                    $max = $v
                }
                $abs = [Math]::Abs($v)
                if ($abs -gt $peak) {
                    $peak = $abs
                }
                $sumSquares += $v * $v
            }

            $rms = [Math]::Sqrt($sumSquares / [double]$sampleCount)
            $frames = [double]$sampleCount / [double]$channels

            return [PSCustomObject]@{
                Path           = $Path
                Exists         = $true
                Valid          = $true
                Error          = ""
                FormatTag      = $formatTag
                BitsPerSample  = $bitsPerSample
                Channels       = $channels
                SampleRate     = $sampleRate
                Samples        = $sampleCount
                DurationSec    = $frames / [double]$sampleRate
                Min            = $min
                Max            = $max
                Peak           = $peak
                Rms            = $rms
                PeakDbFS       = ConvertTo-DbFs $peak
                RmsDbFS        = ConvertTo-DbFs $rms
                NonZeroSamples = $nonZero
            }
        } finally {
            $br.Dispose()
        }
    } finally {
        $fs.Dispose()
    }
}

function Get-WavAudioQualityStatus {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Stats,
        [double]$MinPeak = 1e-4,
        [double]$MinRms = 1e-6,
        [int]$MinNonZeroSamples = 16
    )

    if (-not $Stats.Valid) {
        return "INVALID_AUDIO"
    }
    if ($Stats.Samples -le 0) {
        return "EMPTY_AUDIO"
    }
    if ($Stats.NonZeroSamples -lt $MinNonZeroSamples) {
        return "SILENT_AUDIO"
    }
    if ($Stats.Peak -lt $MinPeak) {
        return "LOW_PEAK_AUDIO"
    }
    if ($Stats.Rms -lt $MinRms) {
        return "LOW_RMS_AUDIO"
    }
    return "OK"
}
