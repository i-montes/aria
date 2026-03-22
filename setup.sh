#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  Aria Global Workspace Setup"
echo "=========================================="

echo -e "\nStep 1: Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    error "Rust/Cargo not found. Please install Rust: https://rustup.rs"
    exit 1
fi

echo -e "\nStep 2: Installing Aria CLI globally..."
cargo install --path aria-cli --force
info "Aria CLI installed successfully to PATH."

echo -e "\nStep 3: Installing AriaMem service globally..."
cargo install --path ariamem --force
info "AriaMem installed successfully to PATH."

echo -e "\nStep 4: Running global tests..."
rm -f ~/.local/share/aria-project/aria/config/aria.config.json

info "Initializing Memory Engine via global CLI..."
aria mem init

info "Testing global storage..."
aria mem store -c "Global CLI setup successful" -m "experience" > /dev/null 2>&1
RESULT=$(aria mem search -q "setup" -l 1 2>&1)

if echo "$RESULT" | grep -q "setup successful"; then
    info "Functionality test PASSED!"
else
    error "Functionality test FAILED!"
    exit 1
fi

echo -e "\n=========================================="
echo -e "  ${GREEN}SETUP COMPLETE!${NC}"
echo -e "==========================================\n"

info "Starting AriaMem Server in the background..."
aria start mem

echo -e "\nNext steps:"
echo "  1. Check running services: aria status"
echo "  2. Search your memory: aria mem search -q \"query\""
echo "  3. REST API is running at: http://localhost:9090"
echo "  4. Stop services: aria stop mem"
echo ""
