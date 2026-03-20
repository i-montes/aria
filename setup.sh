#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/ariamem"
MODEL_DIR="${HOME}/.local/share/ariamem/models/potion-base-32M"
MODEL_ID="minishlab/potion-base-32M"
MODEL_URL_BASE="https://huggingface.co/${MODEL_ID}/resolve/main"

echo "=========================================="
echo "  AriaMem Setup Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_DIR"

echo "Step 1: Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    error "Rust/Cargo not found. Please install Rust: https://rustup.rs"
    exit 1
fi
rustc --version
cargo --version
echo ""

echo "Step 2: Checking/Building project..."
if ! cargo build 2>&1; then
    error "Build failed!"
    exit 1
fi
echo ""

echo "Step 3: Downloading Model2Vec model..."
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/model.safetensors" ]; then
    info "Model already exists at $MODEL_DIR"
else
    info "Downloading model files from HuggingFace..."

    files=("config.json" "tokenizer_config.json" "tokenizer.json" "sentence_bert_config.json" "model.safetensors")

    for file in "${files[@]}"; do
        url="${MODEL_URL_BASE}/${file}"
        dest="$MODEL_DIR/$file"
        
        if [ -f "$dest" ]; then
            info "  $file already exists, skipping..."
        else
            info "  Downloading $file..."
            if ! curl -L "$url" -o "$dest" 2>&1 | tail -3; then
                error "Failed to download $file"
                rm -f "$dest"
                exit 1
            fi
        fi
    done
fi

echo ""
info "Model files:"
ls -lh "$MODEL_DIR"
echo ""

echo "Step 4: Running tests..."
if ! cargo test 2>&1; then
    error "Tests failed!"
    exit 1
fi
echo ""

echo "Step 5: Quick functionality test..."
TEST_DB=$(mktemp /tmp/ariamem_test_XXXXXX.db)
trap "rm -f $TEST_DB" EXIT

./target/debug/ariamem -d "$TEST_DB" init > /dev/null 2>&1
./target/debug/ariamem -d "$TEST_DB" store -c "Testing embeddings" > /dev/null 2>&1

RESULT=$(./target/debug/ariamem -d "$TEST_DB" search -q "embeddings" 2>&1)

if echo "$RESULT" | grep -q "Testing embeddings"; then
    info "Functionality test PASSED!"
    echo ""
    echo "Sample search result:"
    echo "$RESULT" | head -5
else
    error "Functionality test FAILED!"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "  ${GREEN}SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: ./target/debug/ariamem init"
echo "  2. Try: ./target/debug/ariamem store -c \"Your memory here\""
echo "  3. Search: ./target/debug/ariamem search -q \"query\""
echo ""
