# Running Aria on a server

A hub is Aria's service running on an always-on Linux server with no desktop. You use it from a terminal on any of your computers, over ssh, and its sessions, scheduled tasks and background work go on while your laptop is off. This guide is for the person who sets the server up and connects to it.

## What a hub is

Aria is split in two. The **service** owns the sessions: one Pi per session, the background jobs, the scheduled tasks, the checkpoints and the sandbox. A **client** shows them: the dock, or the terminal. On your desktop the dock is a window around the service. On a server there is no window, so Aria runs the service alone (`aria serve`), and a terminal on another computer attaches to it through ssh.

```text
your computer                       the server (the hub)
-------------                       -----------------------------------------------
aria connect aria-hub               sshd
  ssh --------------------------->    aria client-stdio    the only command the key may run
                                        clients.sock  (a unix socket, not a port)
                                          aria serve       the service
                                            sessions, one Pi each
                                            scheduled tasks
                                            background jobs
```

Closing the terminal, or losing the connection, stops nothing on the server. Open `aria connect` again, from the same computer or another one, and the sessions are there with their history. A run in progress belongs to the service, not to the terminal, so it goes on. The automated acceptance check covers a session and the service surviving a killed client. A run that outlives the client needs a model provider, which that check does not have, so it is checked by hand.

Nothing of Aria listens on a port that other computers can reach. The only way in is the ssh server (`sshd`) that the machine already has. The one exception is a sign-in, which listens briefly on the hub's own loopback address (see [The security model](#the-security-model)).

### What is built and what is planned

Built (milestone M1 of [the design](../superpowers/specs/2026-10-23-hub-and-machines-design.md)): `aria serve` (the service with no window), `aria connect` (the terminal on a hub), `aria client-stdio` (the bridge ssh runs on the hub), `aria hub init` (sets a server up from your computer), the restricted client key, and `/schedules` and `/signin` in the terminal.

Planned, not built: machines (other computers linked to a hub, so that a session can work in their folders), one memory shared across computers, a web client, the dock as a client of a hub, and notifications that find the device you are using. See [What is not there yet](#what-is-not-there-yet).

## What you need

On the server:

- Linux with systemd, `aarch64` or `x86_64`. `aria hub init` refuses any other architecture.
- A user for Aria, and ssh login as that user. `hub init` creates no accounts. A dedicated unprivileged user is the recommendation; pass it as `ana@hub.example`.
- Outbound internet. The installer downloads the Node that Pi needs (when the server's own is missing or older than 22.19), Engram, and Pi's npm packages.
- `sudo` for a few system packages, or an administrator who runs them: `bubblewrap` (the sandbox), `iproute2` (`ss`), `xz-utils` (only when Aria has to unpack its own Node), `dbus-user-session` on Ubuntu, and `loginctl enable-linger` for the user. `hub init` prints the command for each one it cannot run itself.

On your computer:

- A checkout of the Aria repository, and Aria already set up (`~/.aria/settings.json` exists): `hub init` sends your settings and credentials to the server, so you do not type a key twice.
- `ssh`. `aria connect` runs from Omarchy or macOS; the hub itself is Linux only.
- A way to build the headless binary for the server's architecture. Nothing is compiled on the server. `sh tools/build-headless.sh` cross-builds it with the Rust toolchain (`rustup`), zig (through `mise`) and `python3`, for `aarch64` by default or for `x86_64` with `sh tools/build-headless.sh x86_64-unknown-linux-gnu`. It is linked against glibc 2.39 (Ubuntu 24.04), so an older distribution may not run it.

There are no published releases or hosted installer yet, so a hub is installed from a checkout.

## Set up a hub

Build the binary, make a key that Aria alone uses, and run `hub init` from the root of the checkout:

```sh
sh tools/build-headless.sh
ssh-keygen -t ed25519 -f ~/.ssh/aria-hub -N ''
aria hub init ana@hub.example --client-key ~/.ssh/aria-hub.pub
```

Use a dedicated key, not the one you log in with. `hub init` logs in to the server the way you normally do, and adds the public key you give it to `~/.ssh/authorized_keys` with a restriction that lets it run only Aria's bridge (see [The security model](#the-security-model)). If that key is already in `authorized_keys` without the restriction, `hub init` reports it and leaves it alone.

`hub init` opens more than a dozen ssh connections. If your login key has a passphrase, use `ssh-agent` so it does not ask each time.

### What you see

`hub init` first asks the server some read-only questions: the machine architecture, the operating system, the home folder, which programs are there (`xz`, `bwrap`, `node`, `npm`, `tailscale`), the Node version, what is listening on which ports, whether `aria-serve.service` is already active, and which of `settings.json`, `auth.json`, `mcp.json` and `mcp-auth.json` already exist in the server's `~/.aria`. It also asks your own `ssh -G` which host name and port your ssh would use. Nothing on the server changes in this phase.

Then it prints the plan and asks `Go ahead? [y/N]`. Only a `y` or `yes` goes on. With no terminal on stdin and no `--yes`, the answer is no: it prints `nothing was changed` and exits 0.

### What it does, in order

1. Uploads the installer, the service unit and the version pin files (`pi-version`, `pi-packages`, `engram-version`, `agent-browser-version`, `node-version`) and your binary to `~/.aria/hub-src` on the server. The plan shows the binary's path and when it was built, because a stale cross-build is easy to ship.
2. Runs `install.sh --headless` there (below).
3. Sends your settings and credentials as one bundle to `aria hub-import` on the server, for the files the server does not have yet (below).
4. Restarts `aria-serve.service`, unless it was already active. An active service takes the new binary when it is idle and the settings within a second.
5. Runs `aria --version` on the server and prints the version it found.
6. With `--client-key`: tries the sandbox on the server first, and only then adds your key to `authorized_keys`.

After step 5 it reads whether the user lingers, tries the sandbox, and after step 6 checks whether sshd is reachable from the internet. Steps 1 to 5 stop the run when they fail, and the error says what is already on the server. Step 6 and the checks report and go on.

A successful run ends with `The hub is ready. Connect from this computer with: aria connect aria-hub`, or with the host you typed when no key was installed. Without linger it says the hub is installed but not ready, because the service stops when you log out of the server.

### What the installer does on the server

`install.sh --headless` runs as the user, on Linux only:

- Installs `bubblewrap` and `iproute2` with `apt-get` or `pacman` when `sudo` allows it. When it cannot, it prints the command for you to run and goes on without the sandbox.
- Keeps the server's Node when it is 22.19 or newer and has `npm`. Otherwise it installs the Node pinned in `node-version` into `~/.aria/node`, checked against the SHA-256 pinned there for the server's CPU, and never replaces the system's Node.
- Installs the pinned Pi, Pi's packages, and Engram, all under `~/.aria`.
- Does not install the browser that agents use to check pages (`agent-browser` and its Chrome are a large download). Run `sh install.sh --headless --with-browser` on the server to add it.
- Writes the user unit `~/.config/systemd/user/aria-serve.service`, enables and starts it, and runs `loginctl enable-linger` for the user.
- Sets `XDG_RUNTIME_DIR` and the user bus address when a non-login ssh left them unset, so that `systemctl --user` works.

### Flags

| Flag | What it does |
|---|---|
| `--binary <path>` | The headless binary to install. Default: `target/aarch64-unknown-linux-gnu/release/aria` in the checkout. Use it for an `x86_64` server. |
| `--projects <dir>` | The folder where the hub's sessions work. Default: `Projects` in the server's home. It is written into the `settings.json` that is sent, so it has an effect only when that file is sent. |
| `--client-key <pubkey file>` | The public key to allow, restricted to Aria's bridge. The file holds one bare public key line (`ssh-ed25519 AAAA... comment`), with no options and no quotes. |
| `--settings` | Replace the server's `settings.json` with this computer's. |
| `--credentials` | Replace the server's `auth.json`, `mcp.json` and `mcp-auth.json` with this computer's. |
| `--allow-no-sandbox` | Install the client key even when the sandbox does not work on the server. Read [The sandbox on a server](#the-sandbox-on-a-server) first. |
| `--yes`, `-y` | Do not ask. |

### What it refuses to do

- It stops before changing anything when the binary is not a Linux binary for the server's architecture, when the checkout is not where it runs, and when the installer would need `xz` that the server lacks (it prints `sudo apt-get install -y xz-utils` for you to run).
- It does not install the client key on a server whose sandbox does not work, unless you pass `--allow-no-sandbox`.
- It does not restrict a key that is already in `authorized_keys` without the restriction. Use a dedicated key, or edit that line yourself.
- It never edits `sshd_config`, the firewall or Tailscale, and never creates a user.
- It never runs the fixes it prints (a `sudo` command, an AppArmor change).

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The run finished, or you answered no (`nothing was changed`). |
| 1 | Something failed: ssh, the checkout, the binary, the settings export, a step 1 to 5, or step 6 (key skipped, key unrestricted, or the write failed). |
| 2 | The command line is wrong: a missing or extra host, an unknown option, a host that starts with `-`, or a key file that is not one bare public key. |

When step 6 fails, the earlier steps have still run, so the hub is installed and the exit code is 1.

### The ssh entry

When the key is installed, `hub init` prints an entry for `~/.ssh/config` on your computer:

```text
Host aria-hub
    HostName hub.example
    User ana
    IdentityFile /home/ana/.ssh/aria-hub
    IdentitiesOnly yes
```

Add it. It makes `aria connect aria-hub` use the dedicated key and no other. To reach the hub over a private network, put that network's address or name in `HostName`.

## Connect to the hub

```sh
aria connect aria-hub
```

`aria connect [user@]host` runs `ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 <host> ~/.local/bin/aria client-stdio` and gives the pipe to the terminal client. Everything after that is the terminal described in [the terminal guide](terminal.md): the same list, the same keys and commands. Set `ARIA_SSH` to use another program than `ssh`. A host that starts with `-` is refused.

What differs from a local terminal:

- The list's last row is `+ new on <host>`. A new session works in the hub's projects folder itself. Your current folder is on another computer, so the client never chooses a folder for a session on the hub. Two consequences: when that folder is not a git repository, a command in the sandbox can write the whole projects folder, every repository in it; and Aria keeps no code checkpoints there, so `/undo` can only rewind the conversation (it says `Code is not kept in the projects folder.`).
- The footer shows no git branch, because git runs on your computer and the session's folder is on the hub.
- `@` lists files on the hub.
- A dead connection is noticed by ssh after about 45 seconds. The terminal then tries once to reconnect, with the screen frozen for up to 60 seconds, and otherwise exits.
- Aria does not notify you when a run settles on a hub. Each session in the list shows `working`, `waiting for you` or `done` and how long ago.

A terminal on the hub itself (a login as the hub's user, then `aria`) is a local terminal, not a remote one. The limits below apply to `aria connect`.

## Sign in from a terminal

A browser sign-in cannot reach a hub: the redirect at the end goes to `localhost`, which on your computer is not the hub. The terminal handles it:

1. In a session on the hub, type `/signin anthropic` (a Claude plan) or `/signin notion`.
2. The panel shows an address. Open it in a browser on your computer and sign in.
3. The browser then fails to load a page at `localhost`. Copy that page's address from the address bar, or only what follows its `?`, paste it into the panel and press `Enter`.
4. The panel says `Signed in.` Esc closes it.

The panel shows how many characters you pasted, never the text, because the address holds a code. `That address was not accepted.` means to copy the address of the page that failed and paste it again. The hub gives up on a sign-in after five minutes.

API keys have no sign-in. They reach the hub with `hub init` (the first time, or `--credentials` to replace them), or by editing `~/.aria/auth.json` on the hub.

## Scheduled tasks on the hub

The hub runs scheduled tasks while your computers are off, in the same way the dock does ([automation](automation.md)). A task whose time passed while the service was down runs once when it is back, and only if that time was today.

`/schedules` in the terminal lists the tasks and lets you pause, resume and delete them. Creating a task stays the job of the model's `schedule` tool in a conversation.

Two limits matter on a hub:

- A task that can write (a connector it may change, at the `allow` access level, or extra writable folders) waits for your approval, and the terminal leaves that approval to the dock. It shows such a task as `waiting for approval in the dock`. A hub has no dock, so these tasks stay paused as far as the terminal goes. Read-only tasks run. The service itself does not tell the terminal's resume from anyone else's: a client that sends the resume message approves the task (see [what a stolen key can do](#the-security-model)).
- A scheduled task never asks questions. On a hub whose sandbox does not work, every command needs permission, so a scheduled task cannot run shell commands there at all.

## Settings and credentials on the hub

`hub init` writes `settings.json`, `auth.json`, `mcp.json` and `mcp-auth.json` into the hub's `~/.aria` as plain files, mode 600. The first copy of a file it replaces is kept once as `<name>.bak`. Keys that live in your computer's keyring are read out and written as their values, because the hub runs with the keyring off (`credentials.keyring` is false in the settings it sends). Secrets you saved by name (the ones a command reads as `ARIA_SECRET_<NAME>`) are not part of the bundle.

Running `hub init` again keeps what the hub already has. Its own Pi may have refreshed a token, and a copy from your computer would overwrite it. Add `--settings` or `--credentials` to replace them on purpose. If a sign-in stops working after you copy it to a second place, sign in again with `/signin` on the hub.

The hub reads `~/.aria/settings.json` and picks up a change within about a second. Edit it on the hub with your own ssh login ([settings](../settings.md)). A remote terminal has no screen for settings. `hub init` sends the whole file with only `sandbox.projectsDir` and `credentials.keyring` rewritten, so a path-like setting (`sandbox.extraWritable`, `integrations.packages.path`) still points at the path it has on your computer.

## Update a hub

Nothing updates a hub by itself. Pull the repository, build the binary again, and run `hub init` again with the same arguments:

```sh
sh tools/build-headless.sh
aria hub init ana@hub.example --client-key ~/.ssh/aria-hub.pub
```

Running it again is safe. It uploads the new tree and binary, runs the installer again, sends only the bundle files the hub lacks, leaves an active service running, and says `already allowed` for a key it has. An active service takes the new binary when it is idle: no run in progress and no background work that would be cut. If the unit file itself changed, the installer restarts the service at once. A client and a hub that speak different protocol numbers refuse each other, and the message says to update the older one, so update your computers and the hub together.

## The sandbox on a server

The sandbox runs the model's commands inside `bwrap` so they can write only to the project. It needs unprivileged user namespaces. Ubuntu 24.04 restricts them with AppArmor, so `bwrap` can be installed and still fail.

`hub init` runs a real probe on the server, `bwrap --unshare-user --ro-bind / / true`, and tells a missing `bwrap` from a blocked one. For a blocked one it prints the usual fix and never applies it:

```sh
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

(Put `kernel.apparmor_restrict_unprivileged_userns=0` in a file under `/etc/sysctl.d/` to keep it after a reboot, or give `bwrap` its own AppArmor profile.) After fixing it, restart the service so each session's Pi checks again, and run `hub init` again to install the key.

Where the sandbox does not work, every command asks first, as on a Mac. The question goes to whichever client is attached, so anyone holding the client key can approve it. That is shell access to the hub's user in practice, which is why `hub init` does not install the client key in that state unless you pass `--allow-no-sandbox`. A working sandbox narrows this less than it may seem: the model can also ask to run a command outside the box, and the same client answers that card. Treat the key as a login either way.

## What a remote terminal may and may not do

The hub's bridge marks every connection that comes through `aria connect` as remote and applies a narrower set than a local terminal gets. One test (`the_whole_surface_a_client_may_reach_is_pinned` in `crates/aria/src/clients.rs`) lists everything both kinds of client may reach.

| A remote terminal may | Notes |
|---|---|
| List, open and close sessions | A new session never gets a folder from the client. |
| In a session the hub has: send a message, stop a run, load the history | A made-up session id is refused. |
| Answer questions and approvals | Including the card that lets a command out of the sandbox or writes outside the project, an install, and a connector write. |
| List the slash commands, list models and change the session's model, list files for `@` | |
| Undo to an earlier message (`/undo`) | Sessions opened over `aria connect` work in the projects folder, where no code checkpoints are kept: only the conversation goes back. |
| `/context` | The only Pi command it may send is `get_session_stats`. |
| List, pause, resume and delete scheduled tasks | |
| Start a sign-in to Claude or Notion and paste its address | |

| A remote terminal may not | What it gets |
|---|---|
| Save a secret | `not-for-remote` |
| Run a `!command` | `not-for-remote` |
| Send any other Pi command (`export_html`, `compact`, and the rest) | `not-for-remote` |
| Talk to a session the hub does not have | `unknown-session` |
| Read or change settings or keys, import MCP servers, stop or dismiss background jobs, restart the engines, dictate, or anything else that changes the service's own state | `not-for-clients` |

The terminal shows the first two refusals as `not available on a connected hub`.

## The security model

**Credentials on the hub.** Your model keys and sign-ins sit in plain files, mode 600, in the hub user's `~/.aria`, with the keyring off. Whoever has the server's disk, root, or a shell as that user has them, and so does a command that runs outside the sandbox (inside it, these files are hidden). Disk encryption is the server host's business. Treat the server as holding every key you sent it.

**Transport.** ssh and nothing else. The service listens on unix sockets in a folder only its user can enter (`aria/` in the user's runtime directory, usually `/run/user/<uid>`). During a sign-in it also listens briefly on `127.0.0.1` of the hub, which other users of that server could reach, so a hub is for a server you do not share.

**The restricted key.** The line `hub init` adds is `restrict,command="/home/ana/.local/bin/aria client-stdio" ssh-ed25519 AAAA...`. The key gets no shell, no terminal, no port forwarding and no agent forwarding. Whatever command the client asks ssh to run, sshd runs only the bridge. The bridge writes its own `hello` marked remote and forwards only JSON lines, so the limits above hold for `aria connect` whichever key it uses. They protect nothing for an unrestricted key, which is a shell. To revoke the key, delete that line from `authorized_keys`: new connections with it are refused at once, and one already open stays open until it closes.

**What a stolen client key can do.** It gets no shell, but it is a terminal at the keyboard, and a terminal can answer every card. As far as the code shows, that is close to a shell as the hub's user, and the sandbox does not change it:

- It can chat with the model, which spends your model quota and reads what the sandbox lets it read (the rest of the disk, except credentials).
- The model can ask to run a command outside the sandbox, to write a file outside the project, or to install a skill, a Pi package or an MCP server (a local server is a command of its own, outside the box). Each is a card, and the key can answer yes, including "no sandbox for the rest of the session". Jev's review on the card is advice, not a gate. Without a working sandbox every command is such a card.
- It can ask the model to create a scheduled task that may write a folder it names, and then send the resume message that approves it. The task runs with that folder writable and no card. The terminal never sends that resume, but the service does not stop it. The folder cannot be one that holds credentials or Aria's own folder, and can be any other folder that exists.
- It can change a session's model, close sessions, pause or delete scheduled tasks, and start a sign-in, which points the hub's Claude or Notion account at an account of the attacker's. Commands typed in a session, such as `/autocompact` and `/effort`, still change their own setting in `settings.json`.
- It cannot save a secret, run a `!command`, read or change your keys or settings through the protocol, or choose a folder for a session.

Setting `sandbox.askToLeave` to `false` in the hub's `settings.json` makes a command that needs to leave the box refused instead of asked. It does not close the other cards or the scheduled-task route above. Nothing in Aria narrows a stolen key further today, so treat the key as a login: keep it in one place, and revoke it (see the restricted key above) if it may have leaked.

**The public ssh port.** Aria does not change how sshd is reachable. After the install, `hub init` checks whether sshd listens on a public address, meaning anything except loopback, a private range, link-local, or Tailscale's range. If it does, it says so once, tells you the port stays open until you close it, and reads `sshd_config` (without `sudo`) to say whether password logins look on or off. It warns and goes on. Until you close the port, the protection is key-only login and the restricted key.

The safe end state is a hub whose ssh is reachable only over a private network, such as Tailscale: every device you use joins the network, the server's firewall closes the public ssh port, and sshd listens on the private address. That is your administrator's work. Aria never touches sshd, the firewall or Tailscale. You can point `HostName` in the ssh entry at the private address today, so the traffic already goes through it.

## `aria serve` and the service's files

`aria serve` is the service with no window: no dock, no face, no hotkey, no notifications, no dictation. `aria dock` also runs it when the binary was built without a window. It takes the same control socket as the dock, so a dock and a serve on one computer exclude each other. It runs until `aria quit` or a stop signal, ends background work and Pis cleanly, and exits 1 when terminals have no way in.

Manage it with your own ssh login (the `aria-hub` entry's key is restricted and gives no shell):

```sh
ssh ana@hub.example
systemctl --user status aria-serve.service
journalctl --user -u aria-serve.service -f
systemctl --user restart aria-serve.service
```

The unit restarts the service five seconds after a failure. `aria quit` exits 0, so the unit stays stopped. On a hub, start it again with `systemctl --user start aria-serve.service`: `aria start` starts `aria.service`, the dock's unit, which a headless install does not have.

| Path on the hub | What it is |
|---|---|
| `~/.local/bin/aria` | The binary. The key's forced command and `aria connect` both call this path. |
| `~/.aria/` | Settings, credentials (mode 600), `sessions.json`, the `.bak` first copies, Pi under `pi/`, `node/` (the private Node), `bin/`, `agent/`, Engram's database. |
| `~/.aria/hub-src/` | What `hub init` uploaded. It is never cleaned up. |
| `~/.config/systemd/user/aria-serve.service` | The user unit. |
| `$XDG_RUNTIME_DIR/aria/` | `control.sock` and `clients.sock`. |
| `~/.ssh/authorized_keys` | The restricted line for the client key. |

## Troubleshooting

### Setting up (`aria hub init`)

| Message | What it means and what to do |
|---|---|
| `a host is needed`, `only one host`, `unknown option ...`, `--binary needs a value` (exit 2) | The command line is wrong. The usage line follows the message. |
| `... : not a host` (exit 2) | The host starts with `-`, which ssh would read as an option. |
| `<key file>: there is no key in it`, `it must be a bare public key (it starts with a key type such as ssh-ed25519), not ...`, `a key line holds no quote`, `the key is missing or is not base64` (exit 2) | `--client-key` needs the `.pub` file: one line, no options. A private key or an `authorized_keys` line with options is refused. |
| `the Aria checkout (install/install.sh and the version pins) is not here: run this from it` | Run `hub init` from the checkout. |
| ``there is no binary to install (... is not built): run `sh tools/build-headless.sh`, or pass --binary <path>`` | The default `aarch64` binary is missing. Build it, or pass `--binary`. |
| `... is built for aarch64 but the hub is x86_64: build one for it with ...` | The architectures differ. Build for the hub's and pass it with `--binary`. |
| `ssh could not reach or log in to <host>: ...` | ssh failed (exit status 255). Try `ssh <host>` by hand first. |
| `cannot build the settings to send: ...` | There is no `settings.json` on this computer, or a key could not be read from the keyring. Unlock the keyring and run it again. |
| `note: <file>: no copy found and the keyring could not be opened (...)` | A warning, for `auth.json` or `mcp-auth.json`. That file is left out. A computer that never had a keyring has nothing to send there. |
| `the installer would stop on this hub (see above); nothing was changed` (exit 1) | The server needs `xz-utils`. Run the `sudo apt-get install -y xz-utils` it printed, then run `hub init` again. |
| `nothing was changed` (exit 0) | You answered no, or there was no terminal and no `--yes`. |
| `step N (...) failed: ...` (exit 1) | The text after it says what is already on the server. Fix what it names, then run `hub init` again. For step 4 it also suggests `dbus-user-session` and linger. |
| `the hub refused the bundle (exit status 1)` | Step 3. `aria hub-import` on the hub rejected what was sent (a file that is not a JSON object, an unexpected file name, or a write that failed). Its own text is never shown because it could quote a value. |
| `warning: linger is not on for <user> ...` | The service stops when you log out of the server. Run `sudo loginctl enable-linger <user>` there. Exit 0, but `hub init` does not call the hub ready. |
| `warning: the sandbox cannot start on the hub ...` | See [The sandbox on a server](#the-sandbox-on-a-server). |
| `step 6 (key) skipped: the sandbox does not work on the hub ...` (exit 1) | Fix the sandbox and run it again, or pass `--allow-no-sandbox`. |
| `step 6 (key) failed: that key is already in authorized_keys WITHOUT the restriction ...` (exit 1) | Use a dedicated key, or edit that line to start with the `restrict,command=` text it shows. |
| `warning: sshd on <host> listens on port N on a public address ...` | See [The security model](#the-security-model). |
| `warning: could not check whether sshd is open to the internet ...` | `ss` is missing or failed on the hub. Check it yourself. |
| `sshd_config: password logins look on; set PasswordAuthentication no.` | Turn password logins off in sshd, as the administrator. |

### Connecting (`aria connect`)

| Message | What it means and what to do |
|---|---|
| `ssh is not installed here (or ARIA_SSH names a program that is not there).` (exit 1) | Install ssh, or fix `ARIA_SSH`. |
| `ssh did not finish connecting to <host> in 60 seconds. ...` (exit 1) | The first connection to a host asks you to accept its key. Run `ssh <host>` by hand once. A passphrase prompt or a slow login can cause this too. |
| ``Aria is not running on that computer: start it there with `aria start`.`` (exit 1) | The service is down. On a hub, log in and run `systemctl --user start aria-serve.service` (see above for why not `aria start`). |
| `Aria on that computer is older than this aria: update it there.` (exit 1) | The hub does not know this client's first line. Update the hub. |
| `This aria speaks protocol N and Aria on that computer speaks M: update the older one.` (exit 3) | Update whichever side has the lower number. |
| `Aria's dock could not start: the connection closed before the hub answered` | ssh ended before the bridge answered: the key was refused (look for `Permission denied` above it), or there is no `~/.local/bin/aria` on the hub. |
| ``Aria's dock stopped. Run `aria` again to start it.`` | The connection or the service went away. On a hub, run `aria connect` again, and check the service if it keeps happening. |
| `not available on a connected hub` | The hub refused a `!command`, a secret, or a session it does not have. |
| `There are already 8 sessions: close one first.` | Close a session in the list. |
| `Sign-in failed: ...`, `That address was not accepted. ...`, `No sign-in is waiting for that address. Start one with /signin.` | See [Sign in from a terminal](#sign-in-from-a-terminal). The last one means the hub has no sign-in pending: it timed out, or none was started. |

### On the hub

| Message | What it means and what to do |
|---|---|
| `aria: Aria is already running` (exit 0) | A dock or a serve already holds the control socket. |
| `terminals cannot connect: <socket>: <error>`, then `aria: not serving: terminals have no way in (see above)` (exit 1) | The service could not create `clients.sock`. Check that the runtime directory exists and is writable. The unit tries again after five seconds. |
| ``no settings yet: this is a first run, sign in from a terminal (`aria connect`)`` | The hub has no `settings.json`. Run `hub init` from a computer where Aria is set up. |
| ``no window here: `toggle` is ignored`` | `aria toggle` or `aria show` reached a service with no window. |
| `aria client-stdio` exits 1, 3 or 4 | The service is down (1), the protocols differ (3), or the client sent a line over 8 MiB (4). The terminal turns the first two into the messages above. |

## What is not there yet

- **Machines.** Linking other computers to a hub, so that a session works in a folder on your laptop and every session shows in one list, is designed and not built. So are the shared credentials and memory it needs. Today a hub's sessions work in the hub's own folders: chat, research, memory, schedules, and code in repositories you clone there over ssh with the server's own git credentials. Aria never handles git credentials.
- **A web client** and **the dock as a client of a hub**, and **notifications that find the device you are using**.
- **`aria update` for a hub**, releases and a hosted installer. A hub is updated by running `hub init` again.
- **A run that survives a dropped client** is checked by hand, not by the acceptance script.

The full list is in [pending](../pending.md), and the design is in [the hub and machines spec](../superpowers/specs/2026-10-23-hub-and-machines-design.md). How the pieces fit is in [the architecture](../architecture.md).

## See also

- [The terminal](terminal.md): sessions, keys and commands, which are the same over `aria connect`.
- [Safety](safety.md): the sandbox, access levels, secrets and undo.
- [Automation](automation.md): scheduled tasks and background work.
- [Commands](../reference/commands.md): every `aria` command.
- [Data and paths](../reference/data-and-paths.md): where everything lives.
