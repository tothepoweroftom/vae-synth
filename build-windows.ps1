$ErrorActionPreference = 'Stop'
py -3 "$PSScriptRoot/scripts/release.py" @args
exit $LASTEXITCODE
