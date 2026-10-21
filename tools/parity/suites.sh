#!/bin/sh
# Every offline suite (ARIA-SETUP item 35), one crate at a time (the engine's timing tests are sensitive to load).
#   sh tools/parity/suites.sh
set -e
cd "$(dirname "$0")/../.."
for crate in aria-engine aria-core aria aria-hyprland; do cargo test -q -p "$crate"; done
bun test ui
bun test extensions
bun test tools/parity
sh install/install.test.sh
sh install/install-mac.test.sh
sh tools/parity/scratch-home.test.sh
# The Mac build, type-checked from Linux (offline once the target and mise's zig are installed; a Mac builds it for real).
if [ "$(uname -s)" = Linux ]; then sh tools/check-macos.sh; fi
echo "all suites passed"
