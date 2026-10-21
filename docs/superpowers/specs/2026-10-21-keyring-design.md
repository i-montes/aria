# Credentials in the system keyring — design

Idea #4 of `docs/ideas.md`. Owner's goal, in their words: **nothing in plain text** — no key in a file under `~/.aria`
that a backup, a dotfiles sync or a `cat`/`grep` would show. Protecting the keys from commands Aria itself runs is a
non-goal of this step (the Linux sandbox already hides the files from boxed commands; the Mac has no box yet).

Decisions taken with the owner (2026-10-21): approach A below; OAuth tokens are sealed too, not left in files.

## What is a secret today

| File | What | Who writes it |
|---|---|---|
| `~/.aria/auth.json` | provider API keys (`opencode`, `opencode-go`, `anthropic`) and OAuth tokens (a Claude sign-in) | Aria's setup (`AuthStore`), Pi (`/login`, token refresh) |
| `~/.aria/mcp-auth.json` | MCP OAuth state (Notion): client registration, tokens, refresh token | Aria's setup (`mcp::store_oauth_record`), Pi (sign-in, refresh) |
| `~/.aria/mcp.json` | server list; Tavily's key inside `headers.Authorization`; imported servers may carry keys in `headers`/`env` | Aria (`mcp.rs`, `mcp_import.rs`), never Pi |

All three are mode 600, linked into `~/.aria/agent/` for Pi.

## What Pi offers

- An `auth.json` API-key entry whose `key` starts with `!` is a shell command; Pi runs it when the key is first needed
  and keeps the output in memory only (`docs/providers.md`).
- In `mcp.json`, a `headers` or `env` value that is a whole `!command` is run the same way (`docs/mcp.md`).
- OAuth credentials (provider sign-ins and MCP) have no such hook: Pi reads and rewrites the files themselves.

## Approach A (chosen)

1. **Fixed keys become commands.** Every API key in `auth.json`, Tavily's header and the secret-looking `headers`/`env`
   values of `mcp.json` move to the keyring; the file keeps `!'<aria>' secret --home '<root>' <account>`. These keys
   are never in plain text, not even while Aria runs.
2. **Files Pi rewrites are sealed.** `auth.json` (which then holds only commands and OAuth entries) and `mcp-auth.json`
   live in the keyring as whole files. While the dock runs, a *live copy* of each sits in a private folder outside
   `~/.aria` and outside backups; Pi reads and rewrites it through the links in `~/.aria/agent/`. The dock seals every
   change back into the keyring within a second, and removes the live copies when it quits.

Rejected: B, sealing all three files whole (every key in plain text while Aria runs); C, Pi asking the dock for
credentials over a socket (needs changes to Pi, another project).

## The keyring

`aria-core/src/keyring.rs`: one small interface, three implementations.

```rust
pub trait Backend: Send + Sync {
    fn get(&self, account: &str) -> Result<Option<String>, KeyringError>;
    fn set(&self, account: &str, value: &str) -> Result<(), KeyringError>;
    fn delete(&self, account: &str) -> Result<(), KeyringError>;
}
```

- **macOS — the `security` tool**, never the Security framework from `aria` itself. Aria is unsigned, so the keychain
  would ask "aria wants to use your confidential information" after every update (a new binary is a new app to it);
  items created and read by Apple's signed `/usr/bin/security` never ask. Writing goes through `security -i` with the
  command on **stdin**, so the secret is never in a process's arguments (`ps`):
  `add-generic-password -U -s <service> -a <account> -l <label> -w <base64>`, with no `-T`: on an item that already exists `-T` changes its access list, which macOS confirms with a password dialog on every update (found live on 2026-10-21); the creator, `security`, is trusted without it. Reading:
  `security find-generic-password -s <service> -a <account> -w`; exit 44 is "not there". Verified on the owner's Mac
  (2026-10-21): no dialog, values round-trip.
- **Linux — `secret-tool` (libsecret)**, talking to the Secret Service (gnome-keyring on Omarchy). `store` reads the
  secret from stdin; `lookup` prints it; `clear` deletes. The installer adds the `libsecret` package. When `secret-tool`
  is missing or no Secret Service answers, the keyring is *unavailable* (below).
- **A memory backend** for tests. No test ever touches the real keyring except one `#[ignore]` live test on macOS that
  uses its own service name and deletes what it made.

Values are stored **base64** (UTF-8 → base64), so JSON files and any byte survive `security -i`'s parser and
`secret-tool`'s stdin unchanged.

**Service and accounts.** Service `aria` for the default home; `aria:<root>` when `ARIA_HOME` points elsewhere, so a
development or test instance never reads or overwrites the real keys. Accounts: `key:<provider>` (an `auth.json` API
key), `mcp:<server>:header:<name>` and `mcp:<server>:env:<name>` (an `mcp.json` value), `file:auth.json` and
`file:mcp-auth.json` (sealed files).

## `aria secret`

`aria secret --home <root> <account>` prints the value (decoded) and exits 0; 1 with a message on stderr when it is
missing or the keyring fails. It is what the `!commands` in the files run, so it must be fast and print nothing else.
The command written into the files quotes the binary's own path (`std::env::current_exe`, without Linux's
` (deleted)` suffix) and the root, and is rewritten whenever the dock prepares Pi's folder, so a moved binary or root
never leaves a stale command behind.

## The vault: sealing, live copies, sync

`aria-core/src/vault.rs` owns the lifecycle; `AriaPaths` learns where credentials live.

- **Live folder.** Linux: `$XDG_RUNTIME_DIR/aria-secrets-<id>/` (tmpfs: never on disk; the sandbox already hides all of
  `$XDG_RUNTIME_DIR`). macOS: `<DARWIN_USER_TEMP_DIR>/aria-secrets-<id>/` (`confstr(_CS_DARWIN_USER_TEMP_DIR)`, the
  same for the LaunchAgent and a terminal; outside `~/.aria`, excluded from Time Machine). `<id>` is a short hash of
  the root. Mode 700, files 600.
- **`AriaPaths`** gets `sealed: Option<PathBuf>`. When set, `auth()` and `mcp_auth()` return the live copies;
  `mcp()` and everything else are unchanged. `AriaPaths::new` and `from_env` leave it unset: only the dock's vault sets
  it, so every existing test and the CLI subcommands keep today's paths.
- **Open (dock start, before anything reads credentials).** For each sealed file: a live copy left by a crash is newer
  than the keyring (the dock syncs every second) and is sealed first; otherwise the keyring's copy is written live.
  Then `~/.aria/agent/auth.json` and `mcp-auth.json` link to the live copies (the existing `agent::prepare` link step,
  with the new paths).
- **Sync (every second on the dock's tick).** A live copy whose content changed since it was last sealed is sealed
  again. A keyring error is logged once per change and retried next second; the live copy is kept.
- **Close (quit: signal, Cmd+Q twice, `aria quit`).** A last sync, then the live copies are deleted. A soft update's
  re-exec syncs and leaves them for the new process, which seals them on open.
- **Migration (first open with the keyring on and plain files present).** In order, each step verified before the next:
  1. Every literal API key in `~/.aria/auth.json` → `key:<provider>`, read back and compared, then replaced by its
     command.
  2. Tavily's header and every secret-looking `mcp.json` value → keyring, read back, replaced by its command.
     Secret-looking: every `headers` value of an HTTP server, and `env` values whose name contains `KEY`, `TOKEN`,
     `SECRET`, `PASSWORD` or `AUTH`. Values that are already `!commands` or `${VAR}` references are left alone.
  3. `auth.json` and `mcp-auth.json` are sealed (`file:*`), read back and compared, written as live copies, and only
     then removed from `~/.aria`.
  A failed step stops the migration and leaves that file and the later ones as they were; the dock runs on the files
  and logs why. Nothing is ever deleted that the keyring did not give back intact. No plain-text backup is written.
- **Writers.** `AuthStore::set` with an API key stores it in the keyring and writes the command; an OAuth credential is
  written into the live `auth.json`. `AuthStore::get` resolves Aria's own commands by reading the keyring directly
  (the quota card needs the real OpenCode Go key), never by running a shell. `mcp::set_tavily_key` and the MCP import
  store values in the keyring and write commands. `mcp::store_oauth_record` writes the live `mcp-auth.json`.
- **Reading what is connected** (`AuthStore::summary`, `mcp::connected`) keeps working unchanged: a command is a
  non-empty key, and the live `mcp-auth.json` holds the tokens.

## On and off

`settings.json` gets `credentials.keyring` (default `true`). Off, or with the keyring unavailable, Aria works exactly as
today (plain files, mode 600) and the dock's log says why. `aria keyring off` writes every secret back into
plain files in `~/.aria` (resolving the commands), sets the setting to `false` and deletes Aria's keyring items:
the way back if a keyring misbehaves. `aria keyring on` sets it to `true`; the next dock start migrates.
`aria keyring status` prints whether the keyring is on, available, and which accounts it holds (names only).

## Errors and edge cases

- Keyring locked (macOS asks to unlock the login keychain; Linux's Secret Service may prompt): `!commands` fail the
  same way a missing key does — Pi reports the provider as not signed in and the chat shows its usual problem.
- `aria secret` must never print to stdout on failure (Pi would take an error text as the key).
- Two docks (a development `ARIA_HOME`) use different services and live folders: no clash.
- The Linux sandbox: the live folder sits in `$XDG_RUNTIME_DIR`, already a secret directory for boxed commands; boxed
  commands could still run `aria secret` — the non-goal above, written in `docs/pending.md`.

## Testing

- `keyring.rs`: the memory backend; the macOS backend's command lines and its 44-is-missing rule with a fake `security`
  on `PATH`; one `#[ignore]` live round trip on a real Mac; the Linux backend's command lines with a fake `secret-tool`.
- `vault.rs` with the memory backend: open from the keyring, open after a crash (live copy wins), sync on change only,
  close removes the live copies, every migration step and its failure mode (a backend that fails or returns a different
  value leaves the files untouched).
- `AuthStore`, `mcp.rs` and `mcp_import.rs` with a vault: keys become commands, `get` resolves them, OAuth goes live.
- `aria secret` and `aria keyring` end to end against a fake `security` / `secret-tool`.
- A live check on the owner's Mac with a throwaway `ARIA_HOME`: Pi answers a prompt with a key read through `!aria
  secret`, and a Notion-style OAuth file is sealed, edited by hand, synced and restored after a restart.

## Out of scope

Protecting keys from commands Aria runs; moving Claude Code's own credentials (claude-bridge reads them from Claude
Code); a settings screen row to toggle the keyring (the setting, the CLI and the overview line are enough for now).
