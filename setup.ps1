$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "=========================================="
Write-Host "  Aria Global Workspace Setup (Windows)"
Write-Host "=========================================="
Write-Host ""

function Write-Info { param([string]$Message); Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Err { param([string]$Message); Write-Host "[ERROR] $Message" -ForegroundColor Red }

# UNSET CONFLICTING ENV VARS TO AVOID 401 ERRORS
$env:HF_TOKEN = ""
$env:HUGGINGFACE_HUB_TOKEN = ""

Write-Host "Step 1: Checking Rust toolchain..."
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    Write-Err "Rust/Cargo not found. Please install Rust: https://rustup.rs"
    exit 1
}

Write-Host "`nStep 2: Installing Aria CLI globally..."
try {
    cargo install --path aria-cli --force
    Write-Info "Aria CLI installed successfully to PATH."
} catch {
    Write-Err "Failed to install Aria CLI."
    exit 1
}

Write-Host "`nStep 3: Installing AriaMem service globally..."
try {
    cargo install --path ariamem --force
    Write-Info "AriaMem installed successfully to PATH."
} catch {
    Write-Err "Failed to install AriaMem."
    exit 1
}

# REFRESH PATH FOR CURRENT SESSION
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ($env:PATH -notlike "*$CargoBin*") {
    $env:PATH = "$CargoBin;$env:PATH"
}

Write-Host "`nStep 4: Running global tests..."
# Clean config
$ConfigPath = Join-Path $env:APPDATA "aria-project\aria\config\aria.config.json"
if (Test-Path $ConfigPath) { Remove-Item $ConfigPath -Force }

Write-Info "Initializing Memory Engine via global CLI..."
aria mem init

Write-Info "Testing global storage..."
aria mem store -c "Global CLI setup successful" -m "experience" | Out-Null
$Result = aria mem search -q "setup" -l 1

if ($Result -match "setup successful") {
    Write-Info "Functionality test PASSED!"
} else {
    Write-Err "Functionality test FAILED!"
    exit 1
}

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Info "Starting AriaMem Server in the background..."

# Use the global aria command to start the memory module
aria start mem

Write-Host "`nNext steps:"
Write-Host "  1. Check running services: aria status"
Write-Host "  2. Search your memory: aria mem search -q `"query`""
Write-Host "  3. Stop services: aria stop mem"
Write-Host ""
