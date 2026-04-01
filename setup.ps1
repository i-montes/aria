# $ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "=========================================="
Write-Host "  Aria Global Workspace Setup (Windows)"
Write-Host "=========================================="
Write-Host ""

function Write-Info { param([string]$Message); Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Err { param([string]$Message); Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Step 0: LLM Configuration
$EnvFile = Join-Path $ScriptDir ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "`n--- ARIA LLM CONFIGURATION ---" -ForegroundColor Cyan
    Write-Host "Please choose your preferred LLM provider:"
    Write-Host "1) Google Gemini (Recommended)"
    Write-Host "2) Cloudflare Workers AI"
    Write-Host "3) DeepSeek"
    
    $Choice = Read-Host "Choice (1-3)"
    $EnvContent = ""

    match ($Choice) {
        "1" {
            $Key = Read-Host "Enter GOOGLE_AI_API_KEY"
            $EnvContent = "ARIA_LLM_CONNECTOR=google`nARIA_LLM_MODEL=gemini-1.5-flash`nGOOGLE_AI_API_KEY=$Key"
        }
        "2" {
            $AccId = Read-Host "Enter CLOUDFLARE_ACCOUNT_ID"
            $Token = Read-Host "Enter CLOUDFLARE_API_TOKEN"
            $EnvContent = "ARIA_LLM_CONNECTOR=cloudflare`nARIA_LLM_MODEL=@cf/meta/llama-3.1-70b-instruct`nCLOUDFLARE_ACCOUNT_ID=$AccId`nCLOUDFLARE_API_TOKEN=$Token"
        }
        "3" {
            $Key = Read-Host "Enter DEEPSEEK_API_KEY"
            $EnvContent = "ARIA_LLM_CONNECTOR=deepseek`nARIA_LLM_MODEL=deepseek-chat`nDEEPSEEK_API_KEY=$Key"
        }
        Default {
            Write-Err "Invalid choice. Skipping LLM configuration."
        }
    }

    if ($EnvContent -ne "") {
        $EnvContent | Out-File -FilePath $EnvFile -Encoding utf8
        Write-Info ".env file created successfully."
    }
}

# Step 0.5: Aria Configuration File
$ConfigFile = Join-Path $ScriptDir "aria.config.json"
if (-not (Test-Path $ConfigFile)) {
    Write-Host "`nStep 0.5: No aria.config.json found. Creating default configuration..." -ForegroundColor Yellow
    $DefaultConfig = @{
        workspace_root = $ScriptDir
        database_path = "aria_whiteboard.db"
        memory_port = 8080
        core_port = 3000
        llm_provider = "google"
        system = @{
            log_level = "info"
            data_dir = "data"
        }
        storage = @{
            storage_type = "sqlite"
            path = "ariamem.db"
            wal_mode = $true
        }
        embedder = @{
            provider = "model2vec"
            model2vec = @{
                model_name = "minishlab/potion-base-32M"
                model_path = "models/potion-base-32M"
            }
            http = @{
                url = "http://localhost:11434/api/embeddings"
                timeout_seconds = 30
            }
        }
        engine = @{
            default_search_limit = 10
            cache_size = 10000
            recency_lambda = 0.1
        }
    } | ConvertTo-Json -Depth 10
    $DefaultConfig | Out-File -FilePath $ConfigFile -Encoding utf8
    Write-Info "aria.config.json created successfully."
}

# UNSET CONFLICTING ENV VARS TO AVOID 401 ERRORS
$env:HF_TOKEN = ""
$env:HUGGINGFACE_HUB_TOKEN = ""

Write-Host "Step 1: Checking Rust toolchain..."
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    Write-Err "Rust/Cargo not found. Please install Rust: https://rustup.rs"
    exit 1
}

Write-Host "`nStep 1.5: Stopping existing Aria processes..."
taskkill /F /IM aria.exe /T 2>$null
taskkill /F /IM ariamem.exe /T 2>$null
taskkill /F /IM ariacore.exe /T 2>$null
Start-Sleep -Seconds 1

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

Write-Host "`nStep 3.5: Installing AriaCore service globally..."
try {
    cargo install --path aria-core --force
    Write-Info "AriaCore installed successfully to PATH."
} catch {
    Write-Err "Failed to install AriaCore."
    exit 1
}

# REFRESH PATH FOR CURRENT SESSION
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ($env:PATH -notlike "*$CargoBin*") {
    $env:PATH = "$CargoBin;$env:PATH"
}

Write-Host "`nStep 4: Running global tests..."
# Clean local project config for testing if requested, but ariamem now handles it
# If we want a fresh start, we can delete the local aria.config.json here, 
# but usually we want to keep it after Step 0.5.

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

Write-Info "Starting AriaCore Orchestrator in the background..."
aria start core

Write-Host "`nNext steps:"
Write-Host "  1. Check running services: aria status"
Write-Host "  2. Search your memory: aria mem search -q `"query`""
Write-Host "  3. REST API is running at: http://localhost:9090"
Write-Host "  4. Stop services: aria stop mem"
Write-Host ""
