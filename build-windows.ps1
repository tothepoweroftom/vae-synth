$ErrorActionPreference = 'Stop'
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 "$PSScriptRoot/scripts/release.py" @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python "$PSScriptRoot/scripts/release.py" @args
} else {
    throw 'Python 3 is required. Install Python and put py or python on PATH.'
}
exit $LASTEXITCODE
