# One installer, two uses: this computer or a hub

Status: design, 2026-10-26. Builds on [the hub design](2026-10-23-hub-and-machines-design.md) (milestone M1).

## The three rules (the owner's)

1. **Nobody builds, edits or copies anything.** One command installs Aria, asks what it needs to know, and does the
   rest. What it cannot do without the owner's say-so (a `sudo` step, a system change) it asks for in one line, in
   plain words, with the safe answer as the default.
2. **A guided screen, easy to understand.** The installer is a terminal interface, not a stream of log lines: a title,
   numbered steps ("Step 3 of 7"), what each one is for in one sentence, a mark for done or failed, the noisy output
   (apt, npm, downloads) kept out of sight and shown only when a step fails. Questions say what each answer means, name
   the default, and take Enter for it. No jargon without a gloss ("a hub: a server that keeps Aria running day and
   night; you connect to it from any of your computers"). Colour only on a terminal (`NO_COLOR` and a dumb terminal
   turn it off), plain ASCII marks where the locale is not UTF-8, and the same sentences in a log or a pipe.
3. **When it ends there is nothing left to do but use it.** The last screen says what is ready and the one thing to type
   or press to use it, and that is not a setup step. A hub is ready when this computer is linked to it and a connection
   was tried and worked.

## The questions

`curl -fsSL https://github.com/i-montes/aria/releases/latest/download/install.sh | sh` asks, on Linux:

```
How will you use Aria?
  1) On this computer   the dock, with its window and shortcut                      (default with a desktop)
  2) As a hub           a server that keeps Aria running; you connect to it         (default with no desktop)
```

Choosing a hub asks one more thing, because a hub is only usable from a computer that is linked to it:

```
Which computer is the server?
  1) Another one    you give me its address (user@host); I set it up from here and link this computer to it
  2) This one       it is the server; it will wait for your computer to connect                (default with no desktop)
```

- **Another one** (the way that leaves nothing to do): the installer installs the dock here if this is a desktop, then runs
  `aria hub init <user@host> --yes` (below), which installs the hub on the server, links this computer and tries the
  connection. The last screen says: type `aria connect`.
- **This one** (a server installed on the server itself): the hub is installed and running, and the last screen gives the one
  line a computer needs to link to it. This is the only path that ends with a command for another machine, because a
  server cannot link a computer it has never seen; the screen says so in one sentence.
- Answered without asking by `--mode desktop|hub`, `--server user@host` (another one) or `--server here`, or the same in
  `ARIA_MODE` / `ARIA_SERVER`; `--headless` stays as the old spelling of `--mode hub --server here`. No question is
  asked when the answer is known: a flag or variable, an Aria already installed here (an update keeps its use), macOS (always
  the dock), a dry run or no terminal (the default, said in one line).
- Questions read `/dev/tty`, so they work under `curl | sh`.

## What "hub on this server" does, with no checkout

1. Downloads a **headless build** for the CPU (`aria-x86_64-linux-headless`, `aria-aarch64-linux-headless`) from the same
   release as the installer, with its `.sha256` and the pins. The dock's Linux build stays `aria-x86_64-linux`. The
   workflow builds both headless binaries with `tools/build-headless.sh` (zig, glibc 2.39).
2. Installs `bubblewrap` and `iproute2` with `sudo`, private Node, Pi, its packages and Engram, the user unit, linger.
3. **Makes the sandbox work.** It runs the real probe. If Ubuntu 24.04 blocks user namespaces
   (`kernel.apparmor_restrict_unprivileged_userns=1`), it offers an AppArmor profile for `bwrap` alone
   (`/etc/apparmor.d/aria-bwrap`, loaded with `apparmor_parser`), which leaves the restriction on for everything else. It
   never touches the sysctl, which is system-wide. Consent: the answer to the question, or `--yes`. With no terminal and no
   `--yes` it changes nothing and says what to run.

## What the computer does for "another one": `aria hub init <user@host>` with no checkout

- The hub is installed by running the installer **on the server** over ssh with a terminal allocated (`ssh -t`, so `sudo`
  can ask for its password), pinned to this Aria's version so the two match. `--binary` and the checkout stay as the
  developer's path.
- The dedicated client key is made if it does not exist (`~/.ssh/aria-hub`; `ssh-keygen` asks for a passphrase, and the screen
  says why one is a good idea) and authorised, restricted, on the server. `--client-key` stays to use another one.
- The computer's settings and sign-ins are sent so the hub can use the same models (asked once, default yes), and the screen
  says plainly when the only provider is a Claude plan, which a hub cannot use, and what to do (an API key).
- `aria connect` with no host connects to the remembered hub (`~/.aria/hubs.json`) and finds the key by itself, so there is no ssh
  entry to paste. `hub init` ends by trying that connection.

## Slices

1. **The installer** (`install/install.sh`, tests): the two questions, the guided screen, the headless download, the sandbox step,
   and the hand-off to `aria hub init`. Release pipeline: `tools/release-assets.sh` and `release.yml` carry the headless assets.
2. **`aria hub init` and `aria connect` without a checkout:** installs through the release installer, makes and authorises the
   key, sends settings and sign-ins, remembers the hub, tries the connection.
3. **Documentation:** the install flow in `docs/guide/getting-started.md` and `docs/guide/hub.md` (one path for each use,
   what is asked, what is changed on the machine), `docs/reference/commands.md`, `docs/pending.md`.

## Not in this design

Machines (M2), the web client, notifications. A release must be published (and the tag pushed) before the one-line install
can work; that is the owner's step.
