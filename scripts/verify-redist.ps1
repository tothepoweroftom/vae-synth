param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference = 'Stop'
$signature = Get-AuthenticodeSignature -LiteralPath $Path
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
    throw 'The VC++ redistributable must have a valid Microsoft signature.'
}
$info = (Get-Item -LiteralPath $Path).VersionInfo
if ($info.ProductName -notmatch 'Visual C\+\+.*\(x64\)') {
    throw 'The prerequisite must be the x64 Microsoft Visual C++ Redistributable.'
}
$version = [version]::new($info.FileMajorPart, $info.FileMinorPart, $info.FileBuildPart, $info.FilePrivatePart)
if ($version -lt [version]'14.44.35211.0') {
    throw 'The VC++ redistributable must be version 14.44.35211.0 or newer.'
}
Write-Output $version.ToString()
