$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$python = 'E:\real\anaconda\envs\plant_gpu\python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $root 'freeze.json'))) {
    throw 'Prepare and verify the isolated dataset first.'
}
$logs = Join-Path $root 'logs'
if (Test-Path -LiteralPath (Join-Path $logs 'status.json')) {
    throw 'A supervisor has already started. Inspect the recorded process and heartbeat; do not start a duplicate.'
}
& $python (Join-Path $root 'launch_detached.py')
if ($LASTEXITCODE -ne 0) { throw 'Detached supervisor launch failed.' }
