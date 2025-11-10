param(
    [string]$Timestamp = (Get-Date -Format 'yyyyMMdd_HHmmss')
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path $scriptDir -Parent
$backupDir = Join-Path $project 'backups'
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
$zipPath = Join-Path $backupDir ("ECG_Research_$Timestamp.zip")

$targets = @('paper', 'artifacts', 'mlflow.db')
$existingTargets = @()
foreach ($target in $targets) {
    $path = Join-Path $project $target
    if (Test-Path $path) {
        $existingTargets += $path
    }
}
if (-not $existingTargets) {
    Write-Host 'Nothing to archive.'
    exit 0
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ECG_Backup_" + [System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tempDir | Out-Null
try {
    foreach ($item in $existingTargets) {
        $destination = Join-Path $tempDir (Split-Path $item -Leaf)
        if (Test-Path $item -PathType Container) {
            Copy-Item $item -Destination $destination -Recurse -Force
        } else {
            Copy-Item $item -Destination $destination -Force
        }
    }
    Add-Type -AssemblyName 'System.IO.Compression.FileSystem'
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    [System.IO.Compression.ZipFile]::CreateFromDirectory($tempDir, $zipPath)
    Write-Host "Backup created at $zipPath"
}
finally {
    Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
