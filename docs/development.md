# Development

How to build Aria, run your build without touching your own data, run every test, find your way around the repository, and make a change the project accepts. It is for people who work on the code; if you only want to use Aria, start with [getting started](guide/getting-started.md).

Contents:

- [Build](#build)
- [Run your build](#run-your-build)
- [The tests](#the-tests)
- [The repository](#the-repository)
- [The rules the project keeps](#the-rules-the-project-keeps)
- [Versions and releases](#versions-and-releases)
- [Contributing a provider or a platform](#contributing-a-provider-or-a-platform)
- [Design documents](#design-documents)
- [How a change goes in](#how-a-change-goes-in)

## Build

### What you need

| Tool | Why |
|---|---|
| Rust (the 2021 edition, with `cargo`) | the workspace. The repository sets no minimum Rust version and has no toolchain file. `rustup` must be on your `PATH` for `tools/check-macos.sh` and `tools/build-headless.sh`, which run `rustup target add` |
| `bun` | the UI, extensions and tools tests (`suites.sh` runs them as `bun test`) and the TypeScript tools such as `bun tools/ui-sandbox/serve.ts`. It must be on your `PATH`; there is no `package.json` and nothing to install |
| `cmake` and `clang` | dictation's built-in whisper.cpp (`clang` because bindgen reads its headers through libclang) |
| a C compiler | a dependency (`ring`) builds C even in the headless build |
| `python3` | the pseudo-terminal drivers in `tools/term` and `tools/hub`, and the compiler wrappers in the cross-build scripts |
| `mise` with `zig` 0.14.1 | only for `tools/check-macos.sh` and `tools/build-headless.sh`: each runs `mise x zig@0.14.1 -- which zig`, so `mise` must be on your `PATH`. The repository neither installs `mise` nor pins zig in a mise file |

On Omarchy (Arch) the build also needs these libraries: `gtk3`, `webkit2gtk-4.1`, `gtk-layer-shell`, `alsa-lib` (the microphone), and, for whisper.cpp's Vulkan backend, the Vulkan headers and a shader compiler (`vulkan-headers` and `shaderc` on Arch; no file in the repository checks them). Running Aria also uses `bubblewrap`, `iproute2` and `libsecret` (`secret-tool`); the installer's list is in `install/install.sh` (`PACKAGES`).

On macOS the AppKit, WebKit and CoreAudio frameworks come with the system and you install `cmake` yourself (Homebrew has it). A Mac has no build without a window.

### Build and install your build

```sh
cargo build --release
sh install/install.sh --local target/release/aria
```

`--local` installs the binary you built instead of downloading one. `sh install/install.sh --dry-run --local target/release/aria` prints what it would do and changes nothing.

### The `gui` feature and the headless build

The `aria` crate has one cargo feature, `gui`, on by default. It holds everything that needs a desktop: GTK, WebKitGTK, layer-shell, the Hyprland adapter, dictation (`whisper-rs`, `cpal`) and, on macOS, the dock. Without it none of those is compiled or linked, and `aria dock` runs `aria serve`, the service with no window ([the hub](guide/hub.md)):

```sh
CARGO_TARGET_DIR=target/headless cargo build -p aria --no-default-features
sh tools/hub/no-gui-libs.sh target/headless/debug/aria      # fails if the binary links gtk, webkit, vulkan, asound...
```

This is Linux only: on macOS the build stops with an error that says the `gui` feature is required. A separate `CARGO_TARGET_DIR` keeps the headless artifacts from replacing the default build's.

To build the headless binary for a server from your own computer, `tools/build-headless.sh` cross-compiles it with zig as the C compiler and linker, pinned to glibc 2.39 (Ubuntu 24.04):

```sh
sh tools/build-headless.sh                           # aarch64-unknown-linux-gnu, the default
sh tools/build-headless.sh x86_64-unknown-linux-gnu  # the other supported target
```

The last line of its output is the path of the binary, so `sh install/install.sh --headless --local "$(sh tools/build-headless.sh | tail -1)"` installs it. `aria hub init --binary <path>` ships one to a server.

## Run your build

In a terminal, plain `aria` is the terminal client, not the dock. To run the dock from a terminal use `aria dock`:

```sh
ARIA_DEBUG=1 target/release/aria dock
```

`ARIA_DEBUG=1` makes the dock log more to the terminal: on Omarchy each visibility transition and the page's console, on macOS the name of each page message (never its contents). `ARIA_DEBUG_PI=1` sends Pi's stderr to the dock's stderr instead of nowhere. On macOS `ARIA_FACE=<word>` forces the face beside the notch, to look at one.

To keep your own data out of it, give the run a scratch home:

```sh
ARIA_HOME=$(mktemp -d) target/release/aria dock
```

`ARIA_HOME` replaces `~/.aria` for everything Aria stores. A scratch home also gets its own keyring service (`aria:<path>`), so it never meets your real keys; the keyring entries it writes are not removed when you delete the folder.

On Linux that is not enough to run next to an installed dock. The control socket and the other sockets live in `$XDG_RUNTIME_DIR/aria/`, not under `ARIA_HOME`, so a second dock finds the first and exits with "the dock is already running", and a dock that did start would register the shortcut with your Hyprland. `tools/term/live-check.sh` shows what a separate run needs: its own `XDG_RUNTIME_DIR` (a `mktemp -d` folder, mode `700`), no `HYPRLAND_INSTANCE_SIGNATURE`, and an absolute `WAYLAND_DISPLAY`. On macOS the control socket is under `ARIA_HOME/run`, so the scratch home separates the instances.

To try the service without a window, build the headless binary and run `aria serve` the same way (own `ARIA_HOME`, own `XDG_RUNTIME_DIR`), then reach it with plain `aria` in a terminal that has the same environment variables; `tools/hub/accept.sh` does this with `aria connect` and a stand-in for ssh.

To see the dock's page without Pi or the Rust side, serve it with a stubbed bridge:

```sh
bun tools/ui-sandbox/serve.ts      # http://127.0.0.1:8770/ ; add ?lang=es for Spanish, ?first-run for the setup
```

It has the design's "simulate" buttons (permission and quota cards, questions, compaction, paged history, tool calls, the queue, tabs, subagents, background work), so each state of the chat can be seen without a model.

## The tests

### Everything at once

`sh tools/parity/suites.sh` runs the offline suites below, in this order, and stops at the first failure. `tools/release.sh` runs it before it makes a release. It needs the network only to fetch crates and, the first time, the macOS target and zig for step 6. A cold run builds the workspace several times (the default build, the headless build in its own target folder, the macOS type-check); the repository does not record how long that takes.

| Step | What it runs |
|---|---|
| 1 | `cargo test -q -p` each of `aria-engine`, `aria-core`, `aria` and `aria-hyprland`, one crate at a time (the engine's timing tests are sensitive to load) |
| 2 | Linux only, with `CARGO_TARGET_DIR=target/headless`: `cargo test -p aria --no-default-features`; `sh tools/hub/no-gui-libs.test.sh`; `cargo build -p aria --no-default-features`; `tools/hub/no-gui-libs.sh` and `tools/hub/accept.sh` on the binary that build made |
| 3 | `bun test ui`, `bun test extensions`, `bun test tools/parity` |
| 4 | `sh install/install.test.sh` and `sh install/install-mac.test.sh` |
| 5 | `sh tools/parity/scratch-home.test.sh`, `sh tools/release.test.sh` and `sh tools/release-assets.test.sh` |
| 6 | Linux only: `sh tools/check-macos.sh` |

It does not run `aria-mac`'s tests (`cargo test -p aria-mac`), `tools/hooks/pre-push.test.sh`, the UI sandbox check, `tools/term/live-check.sh` or anything marked `#[ignore]`; they are below. So before a change that touches a Mac-only crate or the pre-push hook, run `suites.sh` and those two commands.

Some things skip without failing the run, so "all suites passed" does not prove they ran: the tests in `bun test extensions` that need `bwrap` (they look for `/usr/bin/bwrap`) or `agent-browser` are skipped where it is missing, and bun counts them as skipped in its summary; `install/install.test.sh` prints `skip:` and succeeds off Linux; steps 2 and 6 run on Linux only. On a Mac the counterpart of step 6 is `cargo test --workspace`, which compiles the Mac code for real; step 2 has none, because a Mac has no build without a window. Say so when you did not run something.

### One suite at a time

| Command | What it covers |
|---|---|
| `cargo test --workspace` | the five crates: the unit tests beside the code and each crate's `tests/` |
| `cargo test -p aria-core` | settings, providers and their golden JSON, auth files, MCP files and import, the OAuth flows against a fake Notion-like server (feature `testing`), sessions, schedules, the keyring over its in-memory backend, the vault, the agent directory, the persona |
| `cargo test -p aria-engine` | Pi's RPC framing, protocol and the supervised child, against `tests/fake_pi.sh` |
| `cargo test -p aria-hyprland` | the hotkey's Lua, bind detection, the focused monitor and reload events, against fake sockets |
| `cargo test -p aria-mac` | the platform-free parts: Carbon's key codes and the order of registration, where the face's wing goes. The AppKit calls compile only on macOS |
| `cargo test -p aria` | the binary: sessions and the chat worker against fake Pis, the clients hub and the remote client's limits, background jobs, worktrees, checkpoints, the terminal client's state machine, the face, the dictation worker with stand-ins, and `tests/hub_init.rs` (`aria hub init` against a fake ssh) |
| `CARGO_TARGET_DIR=target/headless cargo test -p aria --no-default-features` | the build with no window, including `tests/serve.rs`, which runs `aria serve` as a process. Linux only |
| `bun test ui` | the page: each part of the chat, the state reducer, the onboarding, both languages' strings, and the golden files that tie the JS defaults to the Rust ones |
| `bun test extensions` | the Pi extensions that run inside Aria (router, sandbox policy, quota, connectors, background work, rotation, tasks and the rest). Tests that need `bwrap` or `agent-browser` skip where it is missing |
| `bun test tools/parity` | `matrix.test.ts` keeps [the parity matrix](parity.md) honest: every setup item has a row and every file or test a row names exists; `smoke.test.ts` tests the smoke test's judge |
| `sh install/install.test.sh` | the Omarchy installer branch and the headless one, without touching the system: the dry run's steps and their order, the download address (GitHub releases, `ARIA_VERSION`, `ARIA_REPO`, which variable wins) and what a failed or mismatching download says and leaves, the pinned Pi, Engram, agent-browser and Node (checksums, refusals), the units matching their embedded text, uninstall. Skips itself on macOS |
| `sh install/install-mac.test.sh` | the macOS branch: the LaunchAgent, the soft update, uninstall, the two Mac artifact names and the message for a missing Mac binary. Off a Mac it fakes `uname` and the system tools, so it runs anywhere |
| `sh tools/hub/no-gui-libs.test.sh` | the library check, against a fake `ldd` |
| `sh tools/hub/accept.sh <headless binary>` | the hub with no display, ssh or model provider: `aria serve` in scratch homes and `aria connect` through a pseudo-terminal, with ssh replaced by `tools/hub/ssh-shim.sh` (the list, a session with no folder, a killed client, `/schedules`, a refused protocol, a refused secret, an unknown session, `aria quit`). A run that survives a dropped client needs a provider and is checked by hand |
| `sh tools/parity/scratch-home.test.sh` | `scratch-home.sh` against a fake `~/.aria` |
| `sh tools/release.test.sh` | `tools/release.sh` in scratch repositories: the version it decides, the notes, the commit, the tag, its refusals |
| `sh tools/release-assets.test.sh` | `tools/release-assets.sh` in scratch repositories, with a fake binary instead of a build: the artifact names for each platform, the `.sha256` file (checked with the real installer's download step), the notes, the printed `gh release create` command (`gh` is never run), and its refusals: a dirty tree, a HEAD that is not at the release tag, a tag that is not the workspace version |
| `sh tools/hooks/pre-push.test.sh` | the pre-push hook in a scratch repository: a home path, a listed pattern and an unreleased `feat` are refused; a docs commit and a released main go through |
| `sh tools/check-macos.sh` | type-checks the macOS build from Linux: `clippy` for `aarch64-apple-darwin` over `aria`, `aria-mac`, `aria-core` and `aria-engine`, all targets, with zig (through mise) as the C compiler. Nothing links or runs. It stubs whisper.cpp's bindings, so dictation's C++ is not built |

### Checks that need more than the suites

- **The UI sandbox against headless Chrome.** `bun tools/ui-sandbox/serve.ts 8770 &`, then `bun tools/ui-sandbox/check.ts`. It drives the chat's controller (focus, the queue, `Esc` order, pickers, the question card) in headless Chrome, so it needs Chrome (`CHROME=<path>` names one; on Linux it looks for `google-chrome-stable`; on macOS it prefers Chrome for Testing, the one `agent-browser` installs). `URL=<address>` points it at another page.
- **The terminal client against a real dock.** `cargo build -p aria && sh tools/parity/scratch-home.sh sh tools/term/live-check.sh "$PWD/target/debug/aria"`. It needs a Wayland session (`WAYLAND_DISPLAY`) and an installed Aria to copy a scratch home from. The dock gets its own runtime folder, so it touches neither your binds nor your running dock. `drive.py` then starts `aria` in a pseudo-terminal and checks: the dock outlives the client, a new session in a folder, `!pwd`, `/context`, `Ctrl+C` leaving the client, attaching again, `/exit`.
- **Tests that talk to something real**, all `#[ignore]`, run by name:
  - `cargo test -p aria --test live_pi -- --ignored` and `--test live_rotation`: the real pinned Pi answering a prompt, and a session rotating (a few model calls). They need the setup done. `live_pi` also holds a test that downloads the Claude provider (about 300 MB); run one by name to leave it out.
  - the keyring's `linux_live_round_trip` and `mac_live_round_trip`: the real Secret Service or login keychain, under a service name of their own.
  - `records_the_microphone` and `transcribes_a_recording` (dictation): the first records from the microphone for a few seconds, so do not run it in a meeting; the second needs `ARIA_DICTATION_MODEL` and `ARIA_DICTATION_SAMPLE`.
  - `regenerate_golden` rewrites `crates/aria-core/tests/golden/` after an intended change to the settings defaults or a provider (`cargo test -p aria-core regenerate_golden -- --ignored`).
  - `render_every_face` (macOS only) writes every face as a raw RGBA file to `$ARIA_FACE_RENDER`, for a look by eye.
- **Evals with real models** live beside the extension they check and run through the scratch home: `extensions/jev-router/eval.ts`, `extensions/sandbox/review-eval.ts`, `extensions/sandbox/accept.sh` and `tools/parity/smoke.sh`. How to run each is in `tools/parity/README.md`. Jev is free; the others spend a few cents of the daily model.
- **Manual checks.** `cargo run -p aria-hyprland --example register -- 15 "SUPER + SHIFT + A"` binds a shortcut and rebinds it on every `hyprctl reload`. `cargo run -p aria-core --example prepare_agent` and `parity_env` prepare an agent directory under `ARIA_HOME`; `live_mcp` checks Notion and Tavily over the network.

## The repository

```
Cargo.toml                workspace: five crates, one version (`[workspace.package]`)
pi-version, pi-packages   the pinned Pi and the Pi packages Aria installs
engram-version            the pinned Engram, with one SHA-256 per platform
agent-browser-version     the pinned agent-browser
node-version              the pinned Node for servers, with one SHA-256 per CPU
RELEASES.md               what changed in each version (written by tools/release.sh)
CHANGELOG.md              what the installer and runtime change outside the repository
ARIA-SETUP.md             the inventory of the Pi setup Aria was built to match (docs/parity.md walks it)
rustfmt.toml, LICENSE     the formatting rules (maximum width 120) and the MIT license
CLAUDE.md                 the project's rules, written for coding agents; the summary is below
crates/
  aria-core/              no GTK, no Hyprland, testable without a display: paths, settings, providers,
                          credentials, keyring and vault, the agent directory, sessions, schedules,
                          MCP files and import, OAuth, and the messages between the dock and the page
    examples/             manual checks (prepare_agent, parity_env, live_mcp)
    tests/golden/         the JSON that ties the Rust defaults to the page's
  aria-engine/            Pi's RPC protocol and the supervised Pi child process
  aria-hyprland/          the hotkey, the focused monitor and reload events, through Hyprland's IPC
  aria-mac/               the hotkey (Carbon) and the screen on macOS
  aria/                   the `aria` binary
    src/                  main, the service (service.rs, serve.rs), the chat and its sessions (chat.rs),
                          the clients hub (clients.rs), connect, stdio_bridge and hub_admin (the hub),
                          jobs, jobs_socket and worktree (background work), checkpoints, schedule_store,
                          trace, face, control, handlers, keyring_cli, omarchy_bar
    src/dock/             the window: gtk.rs (Omarchy), mac.rs (macOS), the face's wing and layers (macOS)
    src/term/             the terminal client: ratatui, one module per screen part
    src/dictation/        recorder, audio, whisper and the model download
    src/jobs/             background subagents
    tests/                hub init against a fake ssh, aria serve as a process, live tests (ignored)
extensions/               Aria's Pi extensions, written into Aria's Pi directory (see its README)
agent-files/              the persona, the experts and Aria's own skills, written into the same place
ui/                       the dock's page: plain HTML, CSS and ES modules, embedded in the binary
  chat/parts/             one module per component of the design, with its CSS and tests
  onboarding/, i18n/      the first-run setup and settings screen; the English and Spanish strings
install/                  install.sh (Omarchy, macOS and headless), the systemd units, the Omarchy bar widget
tools/
  check-macos.sh          type-checks the macOS build from Linux
  build-headless.sh       cross-builds `aria serve` with zig
  release.sh              makes a release; release.test.sh tests it
  release-assets.sh       collects a release's binary, checksum, pins and installer; release-assets.test.sh tests it
  hooks/                  the pre-push hook and its test
  hub/                    the headless checks and the hub's acceptance script
  parity/                 the parity matrix checks, the scratch home, the offline suites (suites.sh)
  term/                   the terminal client's live check
  ui-sandbox/             the page with a stubbed bridge
design/                   the visual reference (Aria.html) and its components
docs/                     these documents; docs/superpowers/ holds the designs and plans
```

The architecture is described in [architecture](architecture.md); the extensions in [their README](../extensions/README.md).

## The rules the project keeps

They are in `CLAUDE.md`, which is written for coding agents; the substance is below. The first four are for everyone who sends a change; the last two are the maintainer's, who pushes `main`.

- **Everything is in English**: code, names, messages, docs and commit messages. (The UI has English and Spanish strings; those are data.)
- **Both platforms keep working.** A change made on one platform must not break the other. `sh tools/check-macos.sh` type-checks the macOS build from Linux; on a Mac, run the counterparts of the Linux-only suites or say they were not run. Behaviour that depends on the platform (keys, user agent, service manager, keyring tool) is tested for both, or the test says which one it covers.
- **Conventional commit subjects decide the version** (below).
- **The repository is public, so it carries no personal data**: no real home paths, employer or client names, project or task names. Test fixtures use neutral ones (`/home/ana`, `acme/checkout-widget`).
- **A push of `main` goes through a hook** (it checks only pushes of `main`, so a branch is not touched by it). Turn it on once per clone with `git config core.hooksPath tools/hooks`. `tools/hooks/pre-push` refuses a push of `main` when the outgoing commits add this machine's home path or a line matching one of your own patterns, and when the commits call for a release that has not been made. Your patterns go in `.git/info/personal-data`, one extended regular expression per line; that file is never committed. `git push --no-verify` skips the hook.
- **Before a push of `main`:** the hook is on; the scan is empty; run `sh tools/release.sh --next`. If it prints a version, run `sh tools/release.sh`, read the new section of `RELEASES.md`, and push the tag with `main` (`git push origin main vX.Y.Z`); if it prints nothing, push `main` alone. The project's instructions say to ask the maintainer before pushing.

Rust is formatted with the repository's `rustfmt.toml` (maximum width 120).

## Versions and releases

Versions follow [SemVer](https://semver.org) and are git tags (`vX.Y.Z`) on `main`; the version itself is `[workspace.package]` in `Cargo.toml`. `aria --version` prints it.

The commit subjects since the last tag decide the next version:

| Subject | Effect |
|---|---|
| `feat(scope): …` | the minor number goes up |
| `fix(scope): …` | the patch number goes up |
| `feat!: …`, `fix!: …`, or `BREAKING CHANGE:` in a body | a real break: the major number goes up, or the minor one while the major is 0 |
| `docs:`, `style:`, `test:`, `chore:` and any other type | no release |

`sh tools/release.sh --next` prints the version the commits call for, or nothing when there is nothing to release. `sh tools/release.sh` makes the release: it requires `main` and a clean tree, runs `tools/parity/suites.sh` (`--skip-checks` leaves that out), sets the version in `Cargo.toml` and the workspace crates' entries in `Cargo.lock`, writes a section in `RELEASES.md` from the `feat` and `fix` subjects (a section you wrote by hand for that version is kept), commits `release: X.Y.Z` and creates the annotated tag `vX.Y.Z` with the notes. It pushes nothing. If a step fails it puts the files back. `sh tools/release.sh X.Y.Z` makes a version you chose; it refuses a version before the current one, an existing tag and a malformed number.

Five files at the repository root pin what Aria installs. Each is read by the installer and, where the code needs it, by the binary:

| File | Pins |
|---|---|
| `pi-version` | the Pi (the engine) the installer puts under `~/.aria/pi/<version>/`; changing it is a deliberate upgrade |
| `pi-packages` | the Pi packages Aria installs, one per line, each with its version or commit |
| `engram-version` | Engram (memory): the version on the first line, then one SHA-256 per platform; the installer refuses a download that does not match |
| `agent-browser-version` | the `agent-browser` the agents use to check pages |
| `node-version` | the Node a server gets when its own is missing or older than Pi needs: the version, then one SHA-256 per CPU. The repository is the trust root |

A release must put these files next to the binary, because the piped `curl | sh` install reads them from there (run from a checkout, the installer reads the checkout's own).

### How a release's binaries are made and published

`tools/release.sh` makes the version and the tag. What the installer downloads is a GitHub release of `i-montes/aria`, with these assets:

| Asset | What it is |
|---|---|
| `aria-x86_64-linux`, `aria-aarch64-apple-darwin`, `aria-x86_64-apple-darwin` | the binary of one platform: `cargo build --release` with the default features (the dock). The Linux one is built on Ubuntu 24.04, so it needs glibc 2.39 or newer; Omarchy (Arch) has a newer one |
| `<artifact>.sha256` | the binary's digest as `sha256sum` prints it (digest, two spaces, name). The installer takes the first field. It comes from the same release as the binary, so it shows the download is whole, not that the release is authentic |
| the five pin files | copied from the repository root at the tag |
| `install.sh` | `install/install.sh`, which is what `curl -fsSL https://github.com/i-montes/aria/releases/latest/download/install.sh \| sh` runs |

Two routes make them, and they use the same script.

**The workflow.** Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`; so does "Run workflow" with the name of a tag that exists, to publish one pushed before the workflow was on GitHub or to run a failed run again. Three build jobs (Linux x86_64 on `ubuntu-24.04`, Apple silicon on `macos-14`, and Intel cross-compiled on `macos-14`) each run `sh tools/release-assets.sh dist` and upload their binary and checksum. Then `publish` adds the pin files and `install.sh` from the tagged commit, takes the notes from the tag's section of `RELEASES.md` and creates the release with `gh release create` (adding the files with `gh release upload` when the release already exists). Only `publish` can write to the repository. It publishes the Linux assets even when a macOS build failed. The workflow has never run ([pending](pending.md)); its header says what to look at first if the first tag run fails.

**By hand**, on a machine that should give its binary. Check out the tag with a clean tree (the script refuses otherwise), push the tag to GitHub (the command uses `--verify-tag`), then:

```sh
sh tools/release-assets.sh              # builds, and fills dist/ for this machine
```

It prints the exact `gh release create vX.Y.Z <files> --title … --notes-file dist/notes.md --verify-tag` command and does not run it; run it yourself. A Mac that wants to add its binary to a release that exists runs the script and then `gh release upload <tag> dist/<artifact> dist/<artifact>.sha256`. `ARIA_BINARY=<path>` skips the build (the tests use it) and `ARIA_TARGET=<triple>` cross-builds, which is how the workflow makes the Intel Mac's binary.

The server's headless build is not one of these: `aria hub init` uploads one you build with `tools/build-headless.sh`, and the release's Linux binary links the desktop libraries.

## Contributing a provider or a platform

### A provider

A provider is one module in `crates/aria-core/src/providers.rs`, but the setup page, both model pickers, the `/connect` card and the quota reader also name providers, so a new one touches several files. [Adding a provider](adding-a-provider.md) is the one list of those places and the steps, and it says how a provider reaches Pi and which existing module to copy. Only OpenCode Go has a free-model fallback today, and the fallback is tied to that provider; the same page says what that means for a new one.

### A platform

Aria runs on Omarchy (Hyprland) and macOS. Another platform (another Linux compositor, Windows) needs the three things `aria-hyprland` and `aria-mac` do today:

- **A global hotkey.** The `shortcut` setting is parsed in `aria_core::shortcut`, which is platform-free; it also decides the order in which candidates are tried when one is taken (the chosen one, the default, then fallbacks). The adapter registers what that module picks, registers again when the setting changes, and says which one is in use (`DockMessage::Platform`).
- **The focused monitor** (or screen), so the dock opens where you are looking. `aria-hyprland` also reports config reloads, because Hyprland drops runtime binds on a reload.
- **A floating panel that takes the keyboard when it opens**, hosting the page. It lives in `crates/aria/src/dock/`: `gtk.rs` on Omarchy, `mac.rs` on macOS, both exposing the same `Dock` API (`Dock::new(visible, on_message)`, `send`, `show`, `feed`, `show_notice`, `set_problem`, `clear_problem`). `ui.rs` hides the event loop, and `visibility::step` is the platform-free state machine for opening and closing.

The page has no platform fork: the panel serves `ui/` as `aria://app/…` and passes the page's messages through `window.webkit.messageHandlers.aria`. The platform also picks a few values the core owns, such as the shortcut's default and modifier names (`aria_core::shortcut::Platform`) and who animates the open and close (`aria_core::bridge::animation_of`).

The installer branches on `uname -s`: Linux gets the Arch packages and the systemd user unit `aria.service`, and `--headless` (Linux only) gets a server's packages (`bubblewrap`, `iproute2`) and the unit `aria-serve.service`; Darwin gets the LaunchAgent `com.aria.dock.plist` and no system packages. Its two tests, `install/install.test.sh` and `install/install-mac.test.sh`, are the place to extend when a branch is added. Keys use the platform's keyring tool (`security` on macOS, `secret-tool` on Linux): `aria-core/src/keyring.rs`.

## Design documents

Most features have a design and a plan under `docs/superpowers/`:

- `specs/` holds the designs, one file per feature or milestone, named `YYYY-MM-DD-<topic>-design.md`: what was decided, what the behaviour is, and the rulings behind it. The code's comments and the [architecture](architecture.md) link to them for the reasons.
- `plans/` holds the implementation plans, named `YYYY-MM-DD-<topic>.md`: a goal, the architecture and the interfaces, then the tasks in order, each with the files it touches, the tests to write first and the commit message. Some plans (the secrets one, the chat's design fidelity) have no separate design.

The base design, [2026-10-18-aria-base-design.md](superpowers/specs/2026-10-18-aria-base-design.md), has the roadmap and the maintainer's early decisions. [Ideas](ideas.md) lists what may come next (each one gets its own design before anything is built), [pending](pending.md) lists what the chat shows but nothing feeds yet and what each part still lacks, and [parity](parity.md) walks the maintainer's Pi setup against Aria.

## How a change goes in

**If you are sending a change**, the repository does not say how: it has no `CONTRIBUTING` file, no CI that tests a change (its one workflow publishes a release) and no defined review process. The usual way on a public repository, which this page assumes, is a branch and a pull request. The release and the push of `main` are the maintainer's, per `CLAUDE.md` (see [the rules](#the-rules-the-project-keeps)), so leave `tools/release.sh` and the push of `main` to them. Nothing checks your change for you, so run the checks yourself: `sh tools/parity/suites.sh`, `cargo test -p aria-mac` and `sh tools/hooks/pre-push.test.sh`, and say what you could not run (on a Mac, the Linux-only steps; the tests that skip without `bwrap` or `agent-browser`). Write commit subjects in the conventional form (`feat(scope): …`, `fix(scope): …`), because they decide the version, and keep everything in English and free of personal data.

The repository's own files describe how the project builds its features, and no more:

1. **A design.** A feature starts as a spec in `docs/superpowers/specs/` ([ideas](ideas.md) says each idea gets its own design before anything is built).
2. **A plan.** The plan in `docs/superpowers/plans/` breaks it into tasks. Each task lists the files and interfaces it touches, starts with failing tests, then the implementation, then the run of those tests (and `sh tools/check-macos.sh` where a platform is touched), then one commit with a conventional subject. Most plans open with their global constraints and a "Review Focus" listing the inputs the tasks' happy paths do not exercise; the hub's plan also ends with a self-review against its design. The plans are written to be executed task by task.
3. **Tests with the change.** The suites above are the project's checks; `sh tools/parity/suites.sh` is the offline set, with the exceptions listed there.
4. **Docs with the change.** A plan's last task is usually the acceptance check and the docs; the documents under `docs/` describe the code as it is.
5. **Commit, release, push**, the maintainer's part, as in [the rules](#the-rules-the-project-keeps) and [versions](#versions-and-releases).

Steps 1 and 2 are how the maintainer works; the repository does not say they are required of an outside change. Nor does it define a review process beyond the plans' review focus.

## See also

- [Architecture](architecture.md): how the crates, the page, the service and the hub fit together.
- [Adding a provider](adding-a-provider.md).
- [Settings](settings.md): every setting and its default.
- [Pending](pending.md): what is not there yet.
