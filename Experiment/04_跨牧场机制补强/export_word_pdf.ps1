param([string]$Folder = (Join-Path $PSScriptRoot '20260921_v1'))
$ErrorActionPreference = 'Stop'
$path = Join-Path $folder '奶牛热红外呼吸检测_中文初稿_20260921.docx'
$pdf = Join-Path $folder '奶牛热红外呼吸检测_中文初稿_20260921.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($path, $false, $true)
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdf, 17)
    Write-Output "Word rendered $pages pages: $pdf"
} finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word) { $word.Quit() }
}
