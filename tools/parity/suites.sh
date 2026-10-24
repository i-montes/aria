#!/bin/sh
# Every offline suite (ARIA-SETUP item 35), one crate at a time (the engine's timing tests are sensitive to load).
#   sh tools/parity/suites.sh
set -e
cd "$(dirname "$0")/../.."
for crate in aria-engine aria-core aria aria-hyprland; do cargo test -q -p "$crate"; done
# Linux only (a Mac builds the dock). The service with no window (no `gui` feature): its tests and the acceptance of the hub run from a build of their own, so the
# default build's artifacts stay (hub_init.rs and the other tests of the crate ran above, with the dock).
if [ "$(uname -s)" = Linux ]; then
  CARGO_TARGET_DIR=target/headless cargo test -q -p aria --no-default-features
  sh tools/hub/no-gui-libs.test.sh
  CARGO_TARGET_DIR=target/headless cargo build -q -p aria --no-default-features
  sh tools/hub/no-gui-libs.sh target/headless/debug/aria
  sh tools/hub/accept.sh target/headless/debug/aria
fi
bun test ui
bun test extensions
bun test tools/parity
sh install/install.test.sh
sh install/install-mac.test.sh
sh tools/parity/scratch-home.test.sh
sh tools/release.test.sh
# The Mac build, type-checked from Linux (offline once the target and mise's zig are installed; a Mac builds it for real).
if [ "$(uname -s)" = Linux ]; then sh tools/check-macos.sh; fi
echo "all suites passed"
