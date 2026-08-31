param(
    [string]$OutputDirectory = "",
    [int]$Connections = 8
)

$ErrorActionPreference = "Stop"
$recordUrl = "https://zenodo.org/api/records/10202590/files/DiscoBAX_GeneDisco_datasets.zip/content"
$expectedSize = 246500391
$expectedMd5 = "9f2fb895e32c85377e4cf1b2d2658ed9"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\data"))
} else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$finalPath = Join-Path $OutputDirectory "DiscoBAX_GeneDisco_datasets.zip"
if (Test-Path -LiteralPath $finalPath) {
    $existing = Get-Item -LiteralPath $finalPath
    if ($existing.Length -eq $expectedSize) {
        $hash = (Get-FileHash -LiteralPath $finalPath -Algorithm MD5).Hash.ToLowerInvariant()
        if ($hash -eq $expectedMd5) {
            Write-Host "Verified existing archive: $finalPath"
            $archiveVerified = $true
        }
    }
    if (-not $archiveVerified) {
        throw "Existing archive is incomplete or has the wrong checksum: $finalPath"
    }
}

if (-not $archiveVerified) {
    $chunkSize = [math]::Ceiling($expectedSize / $Connections)
    $processes = @()
    $parts = @()
    for ($index = 0; $index -lt $Connections; $index++) {
    $start = $index * $chunkSize
    $end = [math]::Min($expectedSize - 1, (($index + 1) * $chunkSize) - 1)
    if ($start -gt $end) { break }
    $partPath = Join-Path $OutputDirectory ("zenodo.part.{0:D2}" -f $index)
    $existingLength = 0
    if (Test-Path -LiteralPath $partPath) {
        $existingLength = (Get-Item -LiteralPath $partPath).Length
    }
    $expectedPartSize = $end - $start + 1
    if ($existingLength -gt $expectedPartSize) {
        throw "Existing range is too large: $partPath"
    }
    $resumeStart = $start + $existingLength
    $resumePath = "$partPath.resume"
    $parts += [PSCustomObject]@{
        Path = $partPath
        ResumePath = $resumePath
        Start = $start
        End = $end
        ExistingLength = $existingLength
    }
    if ($resumeStart -gt $end) { continue }
    if (Test-Path -LiteralPath $resumePath) {
        throw "Stale resume fragment found: $resumePath"
    }
    $arguments = @(
        "-L", "--fail", "--retry", "5", "--range", "$resumeStart-$end",
        "--output", $resumePath, $recordUrl
    )
    $process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden
    $processes += [PSCustomObject]@{ Process = $process; Part = $parts[-1] }
    }

    $processes.Process | Wait-Process
    foreach ($item in $processes) {
    if ($item.Process.ExitCode -ne 0) {
        throw "A curl range request failed with exit code $($item.Process.ExitCode)."
    }
    $expectedResumeSize = $item.Part.End - $item.Part.Start + 1 - $item.Part.ExistingLength
    $actualResumeSize = (Get-Item -LiteralPath $item.Part.ResumePath).Length
    if ($actualResumeSize -ne $expectedResumeSize) {
        throw "Resume size mismatch for $($item.Part.ResumePath): expected $expectedResumeSize, got $actualResumeSize"
    }
    $destination = [IO.File]::Open($item.Part.Path, [IO.FileMode]::Append, [IO.FileAccess]::Write)
    try {
        $source = [IO.File]::OpenRead($item.Part.ResumePath)
        try { $source.CopyTo($destination) } finally { $source.Dispose() }
    } finally {
        $destination.Dispose()
    }
    Remove-Item -LiteralPath $item.Part.ResumePath
    }
    foreach ($part in $parts) {
    $expectedPartSize = $part.End - $part.Start + 1
    $actualPartSize = (Get-Item -LiteralPath $part.Path).Length
    if ($actualPartSize -ne $expectedPartSize) {
        throw "Range size mismatch for $($part.Path): expected $expectedPartSize, got $actualPartSize"
    }
    }

    $temporaryPath = "$finalPath.assembling"
    $destination = [IO.File]::Open($temporaryPath, [IO.FileMode]::Create, [IO.FileAccess]::Write)
    try {
        foreach ($part in $parts) {
            $source = [IO.File]::OpenRead($part.Path)
            try { $source.CopyTo($destination) } finally { $source.Dispose() }
        }
    } finally {
        $destination.Dispose()
    }

    $hash = (Get-FileHash -LiteralPath $temporaryPath -Algorithm MD5).Hash.ToLowerInvariant()
    if ($hash -ne $expectedMd5) {
        throw "MD5 mismatch after assembly: expected $expectedMd5, got $hash"
    }
    Move-Item -LiteralPath $temporaryPath -Destination $finalPath
    foreach ($part in $parts) { Remove-Item -LiteralPath $part.Path }
    Write-Host "Downloaded and verified: $finalPath"
}

$releaseRoot = Join-Path $OutputDirectory "release"
Expand-Archive -LiteralPath $finalPath -DestinationPath $releaseRoot -Force
$cachePath = Join-Path $releaseRoot "data"
$requiredFiles = @(
    "achilles.h5",
    "schmidt_2021_ifng.h5",
    "schmidt_2021_il2.h5",
    "zhuang_2019.h5",
    "sanchez_2021_neurons_tau.h5",
    "zhu_2021_sarscov2_host_factors.h5"
)
foreach ($name in $requiredFiles) {
    $path = Join-Path $cachePath $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected release file is missing after extraction: $path"
    }
}
Write-Host "Extracted exact cache: $cachePath"
