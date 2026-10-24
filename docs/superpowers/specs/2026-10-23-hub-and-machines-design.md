# Aria anywhere — a hub and its machines

Aria's work lives in one always-on place (a VPS, the **hub**) and every device is a window onto it. Other computers (a
laptop, a Mac, another server) can be linked as **machines**: sessions that need their files run there, and still show
up in the same list from anywhere.

Written 2026-10-23 from the owner's request: "have Aria in one place like a VPS and use it from anywhere, or have
workers anywhere and use them". Decided by me (the owner asked for a plan; every call is listed as a ruling at the
end). This is one design with four milestones; each milestone gets its own plan.

## Why

- A laptop that sleeps stops the schedules, the background jobs and every long run. A server does not.
- The owner works from several computers (Omarchy, a Mac) and wants the same sessions, the same memory and the same
  credentials on all of them, and later from a phone.
- The terminal work (2026-10-23) already split Aria into a **service** that owns every session and **clients** that
  show them. That split is the seam: this design only makes the client side reachable from another computer and lets
  one service reach another.

## Words

- **Service** — today's `aria dock` process minus its window: sessions (`sessions.json`), one Pi per session, jobs,
  schedules, checkpoints, the sandbox. Unchanged in what it does.
- **Hub** — a service on a VPS, headless, always on, that also keeps the list of linked machines and relays clients to
  them.
- **Machine** — any other service linked to the hub (a dock on a laptop, a headless box). Code, files, Pi, the sandbox,
  jobs and checkpoints stay on the machine whose files they touch. The code and the docs say "machine", never "slave".
- **Client** — what shows sessions: the terminal client (exists), later a web page, later a dock pointed at the hub.

## Goals

1. Aria runs 24/7 on a VPS: sessions, schedules and background jobs go on while every computer is off.
2. `aria connect <host>` opens the terminal client on that hub from any computer with ssh, identical to local `aria`.
3. A computer can be linked as a machine; the client then lists every session of the hub and of all machines (each
   tagged with its machine) and can start one on any of them.
4. One place for credentials and memory: set up once, used by every machine.
5. Both platforms: Linux and macOS clients and machines; the hub is Linux.

## Non-goals and anti-goals

- No Aria port of its own: the hub opens no listening port besides the sshd that is already there. The target posture is
  that sshd reachable **only through a private network** (an intranet the owner's devices join). The owner's VPS cannot
  close its public ssh port for now (2026-10-23), so M1 works with it open and treats closing it as a later step that
  needs nothing from Aria: key-only ssh and the restricted keys below carry the weight until then.
- Not multi-user: one owner, their devices. No sharing sessions with other people.
- Not remote tools: Pi does **not** run on the hub with its `bash`/`edit` proxied to machines (rejected, below).
- Remote clients cannot change settings, read keys or touch the service's own state: the clients protocol's whitelist
  (`clients.rs::allowed`) stays as it is. Settings are edited where the service runs (file, or the dock on that
  machine).
- Not Windows. Not a phone app (a web client in M3 covers a phone).

## Architecture

```
 terminal (any computer) ──ssh──▶ hub: `aria client-stdio` ──▶ clients.sock ─┐
                                                                              ▼
 machine (laptop/Mac/box) ──ssh──▶ hub: `aria hub-link machine <name>` ──▶ hub service
   its own service + sessions         (a stdio bridge to the hub's machines.sock)       │
   `aria link` relays the hub's lines to its own clients.sock ◀──────────────────────────┘
```

- **Transport: ssh over a private network (v1).** A client or a machine opens `ssh <hub> <aria command>` and speaks JSON
  lines over its stdio. No new listening port, no new authentication code, NAT is no problem because every connection is
  outbound, and a phone with an ssh app works as a terminal client the same day.
- **The intranet: Tailscale (owner's choice, 2026-10-23).** Two layers when it is fully in place, so one failing is not
  enough: the VPS's firewall closes public ssh and its sshd listens only on the Tailscale address; then the restricted
  keys below. Every device that uses the hub (laptop, Mac, phone, a headless machine) joins the tailnet. Aria never
  configures Tailscale or the firewall (changes outside the repo); it only needs `ssh <name>` to reach the hub, and the
  owner can point their ssh alias at the tailnet address today (the VPS is already on the tailnet) so the traffic goes
  through it even while the public port stays open.
- **Until the public port can be closed**, `aria hub init` says so once, plainly (`sshd answers on a public address:
  port N`), and checks what it can without sudo: that the hub user's `authorized_keys` lines for Aria are restricted
  and, best effort (`sudo -n sshd -T`), that password logins are off. It warns and goes on; it never blocks and never
  edits `sshd_config` or the firewall. `aria connect` says nothing about it.
- **Restricted keys.** The hub's user (a dedicated unprivileged one) gets two kinds of keys in `authorized_keys`, each
  `restrict,command="…"`: `client` keys run only `aria client-stdio`, `machine` keys run only
  `aria hub-link machine <name>` (and `hub-link engram`, M2). The owner's own admin key is separate.
- **What a stolen client key can do, said plainly.** It gets no shell, but it is a terminal at the keyboard: it can chat
  with a model that runs commands inside the sandbox and it can answer a sandbox-escape prompt. Remote clients get a
  narrower set than the local terminal: **no `secretSave`, no `chatBash`** (a `!command` runs outside the model's
  judgment and writing a key is the one thing a stolen key must not do). The approvals stay, or a phone is useless.
  Built in M1, the remote set is exactly this: chat, answers and the session tools on sessions that already exist
  (an unknown session id is refused), a new session never gets a folder from the client, the only Pi command is
  `get_session_stats`, schedules list/pause/resume/delete, and the sign-ins (`startOauth`, `startMcpOauth`,
  `oauthCode`, which can also cancel a pending sign-in of the same service or re-bind the hub's Claude or Notion
  account: accepted, because such a key already uses those accounts through the model). On a hub where no sandbox
  works, every command asks and anyone holding the client key can approve it: that key is then shell access to the hub
  user, and `hub init` does not install it without `--allow-no-sandbox`.
  Revoking a key (one line in `authorized_keys`) is immediate. The hub's `client-stdio` applies this set, the
  machine's `aria link` applies it again for relayed lines.
- **Versions.** The first line each side sends is a hello with a protocol number (and the binary's version). A client,
  hub or machine with a different protocol number is refused with a message saying which side is old and to run
  `aria update` there; messages inside one protocol number are additive only. A hub updates by that command (the dock's
  existing wait-for-jobs logic, which `aria serve` keeps), never by itself.
- **Machines are federated, not proxied.** `aria link` on a machine connects to the hub and to the machine's own
  `clients.sock`, and relays lines both ways wrapped in an envelope (`{ m: machine, c: client, line }`). A remote
  client is, for the machine, one more terminal client: the same `allowed()` whitelist, the same ids rewriting, the
  same fan-out. A hub (or a stolen machine key) can do what a terminal can do — chat, answer dialogs, open and close
  sessions, all inside the machine's sandbox — and not more.
- **The hub's list.** The hub keeps, per machine, the last `sessions` and `sessionStatus` it was sent (a machine pushes
  both on every change, and after hello). It answers a client's `sessionsLoad` at once from those caches, merged with
  its own and with `machine` added to every entry, and pushes later changes to clients as they arrive: no waiting for a
  slow machine. It routes a client's `session { id, … }` message to the machine that owns that id (ids are already
  unique), which rewrites the request ids as today. A machine that drops stays in the list as offline, from its cache;
  its sessions cannot be attached until it is back.
- **Which service a client talks to.** A client talks to exactly one: `aria connect` to the hub; plain `aria` on a
  computer that is linked to the hub, to the hub if it is reachable (and the hub routes "new here" back to that computer),
  otherwise to the computer's own service with a one-line notice that only its local sessions show. A computer's local
  sessions therefore never appear twice (a client sees the list through one path only). A client learns its own
  machine's name from the machine's config.
- **What stays local to a client's device:** the hotkey, the dock window, dictation (the microphone), the face widget,
  desktop notifications (a device notifies about what it shows; routing one to the device in use is M4).

### Credentials and settings

- The hub is the one place keys are set up. A machine gets **live copies** of `auth.json`, `mcp-auth.json` and the
  secrets `mcp.json` names, pushed over its link at connect and on every change, written where the keyring design
  already writes them: the private folder outside `~/.aria` that is removed when the service quits. They are not
  persisted on the machine. A machine that loses the link keeps running with what it has until it quits.
- **The hub is the only writer; credentials travel both ways.** Pi rewrites `auth.json` and `mcp-auth.json` when a token
  is refreshed, and a provider that rotates refresh tokens would sign every other copy out. The vault already notices a
  change to a live copy within a second and seals it; on a machine "sealing" means sending the new file to the hub over
  the link, and the hub (which takes the newest by its own clock) writes it and pushes it to every other machine. Whether
  Claude's and Notion's tokens really rotate is the first thing M2 checks with the real accounts; the design holds either
  way.
- Shared by the hub (an in-memory overlay on the machine, never written into its `settings.json`): provider and models,
  persona and experts, language, integrations. Local to each machine: shortcut, projects folder, sandbox folders,
  dictation, keyring, face. The exact split is derived from `docs/settings.md` by the builder (a key is local when it
  names something of the computer). On a linked machine the settings screen shows the shared keys read-only ("managed
  by <hub>"), and the first-run setup is skipped (a machine that was never set up gets everything from the hub). If the
  hub is down when the machine starts, it uses its own `settings.json` and credentials, as the standalone Aria it was.
- On the hub itself (no desktop keyring): the keyring is off and the existing mode-600 files hold the keys, as the
  vault already does with the keyring off. Disk encryption is the VPS's business.
- Memory (Engram): one database, on the hub. A machine's Pi reaches it through its link (`aria hub-link engram`, a
  stdio bridge under the same restricted key). Hub down means no memory in that session; the session goes on
  (verify in M2 that Pi tolerates a failing MCP server).

## Milestones

### M1 — Headless service and `aria connect`

- **A build without a window.** A cargo feature `gui` (on by default) holds GTK, WebKit, layer-shell, Hyprland and
  dictation on Linux; without it `aria serve` runs the service on a plain event loop: chat, handlers, jobs, schedules,
  `clients.sock`, the control socket. macOS keeps its panel under the same feature. `tools/check-macos.sh` and the Linux
  suites run both feature sets.
- **`aria serve`** is the service on a server; its systemd user unit (`loginctl enable-linger`) is `install/aria-serve.service`.
  No notifier, no face, no hotkey. A run that settles raises nothing; `sessionStatus` already tells every client.
- **`aria client-stdio`** bridges its stdio to `clients.sock` (one connection per call). **`aria connect [user@]host`** runs
  `ssh host aria client-stdio` and gives the pipe to the terminal client where `term/link.rs` now opens the socket; the
  client's code does not change otherwise. Over `connect`, `+ new in <current folder>` is replaced by a new session in
  the hub's projects folder (the current folder is on another computer); M2 brings "new here" back.
- **Getting Aria onto the VPS.** There are no published releases or hosted `install.sh` yet (`docs/pending.md`), so M1
  installs from a checkout: the headless binary is **cross-built on the owner's computer** for the hub's architecture
  (`tools/build-headless.sh <triple>`, zig as the C compiler as `tools/check-macos.sh` already does; the owner's VPS is
  `aarch64` Ubuntu 24.04 with glibc 2.39, no Rust toolchain, so nothing is compiled there). `aria hub init <user@host>`
  prints its plan and, on a yes, copies that binary, the pinned Pi, Engram and the extensions as `install.sh` installs
  them today, and a Node of the version Pi needs (the system's may be too old: the owner's VPS has Node 18) into
  `~/.aria/node/`, checked against a SHA-256 pinned in the repo's `node-version`, never replacing the system's. When releases exist the
  same step downloads instead.
- **The hub's user.** `hub init` installs as the user it logs in as and creates no accounts (a dedicated unprivileged
  user is the recommendation, and it works by passing that user). Installing a system package (`bubblewrap`) needs
  sudo; `hub init` prints the command and asks, and goes on without it, in which case the sandbox is unavailable and
  commands ask every time (see below).
- **First setup without a page:** the same command, run on a computer that already has Aria configured, then sends
  `settings.json` and the credentials over ssh into the hub's `~/.aria`, so no key is typed twice and none passes
  through a third place.
- **What the hub works on.** Its sessions work in its own projects folder: chat, research, memory, schedules, jobs, and
  code in whatever the owner clones there over ssh with their own git credentials on the VPS (Aria never handles git
  credentials). Code that lives on the owner's computers is worked on there, as a machine (M2).
- **Schedules from a remote client.** Listing, pausing and deleting schedules join the clients' allowed set (they are
  the owner's own tasks and a runaway one on a server must be stoppable from where it is seen), with a `/schedules`
  panel in the terminal. Creating one stays the `schedule` extension's tool, as today. This reverses the terminal
  design's "nothing about the schedules" for these three messages.
- **Sign-ins with no local browser** (Anthropic, Notion): the loopback callback cannot reach a hub from the owner's
  browser. The terminal client shows the URL and takes the pasted address of the failed redirect, which it sends to the
  service (a new client-allowed message, `oauthCode`, answered like the page's `oauthResult`).
- **Sandbox on the server:** bwrap needs unprivileged user namespaces, and Ubuntu 24.04 restricts them by default
  (AppArmor), so bwrap may be installed and still fail. `hub init` ends by running a real bwrap probe on the host
  (`bwrap --unshare-user --ro-bind / / true`) and says plainly if the sandbox cannot work, with the fix for that
  distribution printed (an AppArmor profile for bwrap, or the sysctl) and never applied by Aria; until it works, commands
  ask every time, as on a Mac without bwrap.
- **Done when:** a schedule created from the laptop's terminal runs while the laptop is off; a session started over
  `aria connect` from the Mac keeps working after the ssh window closes and shows up again from the Omarchy terminal.

### M2 — Machines

- **`aria machine add <user@hub> --name <name>`** on the machine: makes a dedicated ed25519 key, uses the owner's admin
  ssh once to add its line (restricted, as above) to the hub, and enables the link in the machine's service. `machine list`
  and `machine remove` on the hub (removing the line is the whole revocation, plus a credentials rotation the command
  offers).
- **`aria link`** (inside the machine's service, supervised): connects, says hello with the machine's name, OS and
  version, keeps the link alive (heartbeat, reconnect with backoff), pushes nothing of its own besides its session list.
- **The client's list** shows the machine of every session, offline ones greyed; `+ new here` runs on the machine the
  client is on if it is linked (a client on a linked laptop talks to the hub, and the hub routes the new session back to
  that laptop), `+ new on <machine>` anywhere else. A new session's folder is chosen on its machine (the client lists
  that machine's projects folder through `listFiles`).
- **Schedules belong to the machine they were created on** and run when it is up (the "once, when it is back" rule
  already handles a sleeping laptop). Attachments from a client's computer are inlined in the message up to a size cap.
- **Credentials push and the settings split** as above; `credentials: "own"` on a machine keeps its own keyring instead.
- **Done when:** from the VPS-attached terminal on the Mac, a session started "on laptop" edits the laptop's folder
  inside its sandbox; the laptop's lid closing shows it offline and opening it brings it back with the run still in
  its own Pi.

### M3 — A web client (sketch)

The page (`ui/`) already talks through one bridge and `tools/ui-sandbox/` stubs it. The hub serves the same page on
`127.0.0.1` with a WebSocket bridge to the clients protocol; reached through `ssh -L`, or through a private network
(Tailscale or WireGuard) with a token. The dock column's 470 px is a phone's width. Still no public port.

### M4 — The dock as a remote client, notifications that find you (sketch)

A dock on a laptop can point at the hub as well as at itself. A session "seen" is computed across devices, and a
notification goes to the device in use (a push service such as ntfy is the candidate for a phone).

## Rejected alternatives

- **Pi on the hub, tools proxied to machines.** Keys never leave the hub, which is attractive, but every `bash`, `read`,
  `write`, `edit`, the verify extension, background jobs, worktrees, checkpoints and the `@` picker all touch the
  filesystem and would each need a remote twin, and the sandbox policy would have to be enforced on the far side
  anyway. Federation reuses all of it as is. The price is credentials on the machines (live copies, memory-only).
- **A WebSocket + token server as the first transport.** New authentication, TLS and an open port for the same result
  ssh gives for free. It comes back in M3 only behind a private network or a tunnel.
- **A public hub protected only by ssh keys, as the end state.** Works, but it leaves the door in the street for every
  scanner and every future sshd bug; the owner asked for the private network (2026-10-23). It is accepted as the starting
  point only because their VPS cannot close the port yet.
- **Making one particular network a dependency of the code.** Aria does not know Tailscale or WireGuard; it needs a
  reachable private address, which keeps both open (a mesh VPN is easiest and works through NAT; WireGuard keeps
  everything under the owner's control but needs one UDP port open on the VPS and each peer configured).

## Both platforms

Hub: Linux only. Machines and clients: Linux and macOS. The link, the envelope and `client-stdio` are platform-free
Rust in `aria-core`/`aria`; the ssh invocation, the LaunchAgent for the link and the credentials folder differ and each
has a test for its platform or says which one it covers. `sh tools/check-macos.sh` must pass with and without `gui`.

## Left to the builder

The envelope's exact fields and framing limits, reconnect timings, the keys' file layout on the hub, how `aria hub init`
and `machine add` talk to the hub's admin ssh (run `ssh <host> <command>`), the `aria-serve` unit's hardening options,
the shared-versus-local settings list, and how the terminal shows a machine and an offline session.

## Rulings (mine, to be checked)

1. **Federation, not remote tools.** Cost: credentials live on each machine while it runs; a lost laptop means rotate.
2. **ssh is the only transport in v1.** Cost: the web client and push notifications wait for M3/M4; a client needs ssh.
3. **Restricted keys per role** instead of one login. Cost: setup writes `authorized_keys` lines.
4. **Keyring off on the VPS**, keys in mode-600 files. Cost: anyone with the VPS disk or root has the keys.
5. **Headless via a `gui` cargo feature**, not a second binary. Cost: every change builds both ways.
6. **Over `connect` there is no "current folder"** until M2 (it is on another computer).
7. **One shared memory on the hub**, reached through the link. Cost: no memory when the hub is unreachable.
8. **Settings stay local to where the service runs**; remote clients cannot change them.
9. **No provider or VPS size chosen**: any Linux with systemd and user namespaces, 2 vCPU / 2 GB to start.
10. **The word "machine"**, never the owner's "slave", in code and docs.

Added after the builder check (each is my call):

11. **The hub is the only credential writer**, machines send refreshed tokens back. Cost: a two-way link for secrets.
12. **Remote clients have a narrower set than the local terminal** (no `secretSave`, no `chatBash`); approvals stay.
13. **A protocol number in the first line; a mismatch is refused**, additive changes only inside one number.
14. **A client talks to one service**: the hub if reachable, else its own with a notice. No double lists.
15. **The machine's shared settings are an in-memory overlay**, read-only in its settings screen, first-run skipped.
16. **The hub keeps per-machine caches and pushes**, so the list never waits for a slow or sleeping machine.
17. **M1 installs from a checkout** (no releases exist); the hub's own work is non-code plus repos the owner clones.
18. **Schedule list, pause and delete are allowed to clients**, reversing the terminal design for those three messages.

From the owner, 2026-10-23 ("anchored to an intranet… I'm thinking about security"):

19. **Tailscale is the target intranet; the public ssh port stays open for now** (the owner cannot close it yet). Cost:
    until it is closed, every scanner can reach sshd, so the restricted keys, key-only login and a warning from
    `aria hub init` are the protection. When closed, every device (the phone included) must be on the tailnet. Aria
    never configures the network or the firewall.

Added during the build (the controller's rulings, each found by a review; costs in the branch's ledger):

20. **`oauthCode` is keyed by service**, not by request id (the hub remaps ids); a remote client may start a sign-in.
21. **A remote client never chooses a session's folder, only runs `get_session_stats` among Pi's commands, and only
    reaches sessions the hub already has.** Cost: a new terminal command that needs another Pi command must be added to
    the table in `clients.rs`.
22. **The sandbox check is a real bwrap probe**; with no working box every command asks, on Linux too. `hub init`
    installs the client key only after a working probe or with `--allow-no-sandbox`. Cost: on the owner's VPS (no bwrap,
    Ubuntu 24.04) the key waits for bubblewrap, an AppArmor profile, or the flag.
23. **Re-running `hub init` keeps what the hub already has** (settings, credentials) unless `--settings` /
    `--credentials`, restarts the service only the first time, and writes each `.bak` once.
24. **The headless install skips the browser download** unless `--with-browser`.
25. **The Node tarball is verified against a SHA-256 pinned in `node-version`**, not against the download host's own
    checksum file.
