# RenamePics.ps1
$ErrorActionPreference = "Stop"

$ToNatural = { [regex]::Replace($_, '\d+.jpg', { $args[0].Value.PadLeft(20) }) }

$allImageFiles = Get-ChildItem -Path .\image*.jpg -Recurse -Force
$largestImageFile = $allImageFiles | Sort-Object $ToNatural -Descending | Select-Object -First 1

Write-Verbose "Largest Image File $largestImageFile" -Verbose

$matches = [regex]::Match($largestImageFile, '(\d+)(.jpg)')
$startingImageCount = [int]$matches.Groups[1].Value + 1

Write-Verbose "Starting Image Count $startingImageCount" -Verbose

$count = $startingImageCount

$allNewJpgFiles = Get-ChildItem -Path .\*.jpg -Exclude image*.jpg -Recurse -Force
$allNewJpgFiles | ForEach-Object -begin { $count } -process {
    Write-Verbose "Renaming $_ to image$count.jpg" -Verbose
    Rename-Item $_ -NewName "image$count.jpg"
    $count++
}