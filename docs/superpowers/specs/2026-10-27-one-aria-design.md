# One Aria — the hub is the brain, this computer is a machine

Status: design, 2026-10-27. It builds on [the hub and its machines](2026-10-23-hub-and-machines-design.md) (M1 is shipped; this
is M2 and M4 as one piece) and changes three of its decisions, listed at the end.

## What the owner asked

After using `aria connect` the owner said it feels like **two environments**: the Aria on the computer, and the Aria on the
server behind a separate command. They want one:

1. **One `aria`.** No second command to remember. The computer's Aria and the server's are the same thing.
2. **The server replaces the computer's own Aria.** Everything one has on the computer (memory, keys, connectors, skills,
   settings) is on the server and is used from there, not from a local copy.
3. **Work happens where the person asks.** A project can be worked on **on this computer** (its files are here) or **on the
   server**, depending on what the person asks for.

## The model

```
            this computer (a machine)                         the server (the hub)
  ┌─────────────────────────────────┐            ┌────────────────────────────────────┐
  │ aria  (dock, terminal)          │            │ the brain                          │
  │   one list of sessions          │◀──ssh────▶│   keys, sign-ins, connectors        │
  │ its own service                 │            │   memory (one Engram)               │
  │   Pi + sandbox + the files here │            │   settings shared by every device   │
  │   no copy of the brain on disk  │            │   persona, experts, skills list     │
  └─────────────────────────────────┘            │ sessions that live on the server   │
                                                  │ schedules, background jobs         │
                                                  └────────────────────────────────────┘
```

- **The hub is the brain.** It is the only place the brain is stored: provider keys and sign-ins, connector secrets, the
  memory, the settings that are not about one device, the persona and the experts, the list of skills and packages.
- **A machine is the hands.** Each linked computer runs its own service for the sessions whose files are on it: Pi, the
  sandbox, checkpoints and background jobs stay where the files are (the rejected alternative of the earlier design stands:
  Pi on the hub with its tools sent to the machines would need a remote twin of every tool). What it needs to think it takes
  from the hub when it starts and keeps **in memory and in the service's private runtime folder only**, never as files in `~/.aria`.
- **A session has a place:** `server` or the name of a machine. The place is chosen when the session is created and never
  changes; a session is listed with where it lives.

## One command

- **`aria`** (the dock's shortcut, the terminal, and the `aria` typed in a shell) talks to the hub whenever this computer is linked
  to one. There is no mode to choose first. When the hub cannot be reached it says so in one line (see below).
- **`aria connect`** stays as an alias that does the same as `aria`, so nothing written before breaks, and `aria connect <host>`
  stays for reaching a hub this computer is not linked to. The docs stop teaching it.
- **The list is one list.** Every session of the server and of every machine, each tagged with where it lives; offline machines greyed.
- **Where does a new session go?** In this order: (1) what the person says (`+ new on the server`, `+ new here`, or a machine's
  name; the terminal accepts `--on server`); (2) if `aria` is started in a folder of this computer, or the dock is pointed at one,
  here; (3) otherwise the server, in its projects folder. The person is asked only when (2) and (3) disagree, once, with the
  default highlighted. A session on the server can still be given a project that exists on the server only.

## What a linked computer keeps

| Where | What |
|---|---|
| the hub only | provider keys and sign-ins, connector secrets and their sign-ins, the memory database, persona, experts, the skills and packages list, the settings that name no device (provider, models, language, integrations) |
| this computer only | what names the device: shortcut, projects folder, sandbox folders, dictation, face, window, notifications; the files of local projects, their sessions and checkpoints |
| in memory while a service runs | live copies of the sign-ins and secrets (the design of the earlier hub document: written to the private runtime folder that goes away when the service quits) |

So a stolen laptop holds no keys at rest.

## When the server cannot be reached

- A local session that is running keeps running with the live copies it has until the service quits.
- A new session cannot start without the brain. The dock and the terminal say: `The server is not reachable. Aria cannot start a
  session without it.` plus the one thing to try (the link, ssh, the key's passphrase in the agent).
- There is **no automatic fallback to old local keys**: the point is that there are none. A person who wants one can turn on
  `aria hub offline-copy on` (default off), which keeps an encrypted copy in the keyring, and `off` removes it.

## Moving what is local to the server

`aria hub init` already copies the computer's settings and sign-ins to a server that has none. This design adds, once, after a
successful link, an **adopt** step that asks nothing the first time it runs except one confirmation:

1. Copies what the brain needs that the hub lacks: the memory database (an Engram export and import, checked by a count of
   entries on both sides), the skills and packages the person added, the connectors.
2. Compares and says what moved: "Moved to the server: memory (212 notes), 3 skills, 4 connectors."
3. Offers to **remove the local copies** (keys in the keyring and in `~/.aria`, the memory database), default **keep**, never
   without a yes, and only for what step 2 showed on the server. `aria hub unlink` does the reverse: brings the brain back to this
   computer and removes the link.

## The pieces to build, in order (each one ships on its own)

- **S1 — versions.** `aria connect` shows the hub's version against its own, `aria hub update` brings the hub to this Aria's version
  with the same screens, and the installer updates the hubs this computer is linked to.
- **S2 — the machine link.** `aria hub init` makes the computer a machine (a restricted `machine` key, `aria link` supervised by its
  service): hello with name, OS and version; its session list cached by the hub; lines relayed both ways in an envelope; routing a
  session message to the machine that owns it. (M2 of the earlier design, unchanged.)
- **S3 — the brain from the hub.** Live copies of credentials pushed at connect and on every change; the shared-settings overlay;
  the memory bridge (`aria hub-link engram`) so a machine's Pi uses the hub's Engram; the skills and packages list pulled at link and
  on change. The first thing to check with the real accounts is whether Claude's and Notion's tokens rotate (the earlier design's
  check); and that Pi tolerates a memory server that fails.
- **S4 — one `aria`.** The dock and the terminal talk to the hub when linked; the one list with places; "new where" as above;
  `aria connect` becomes the alias; the hub-unreachable messages.
- **S5 — adopt and unlink.** The step and the commands above, with the counts.
- **S6 — the installer.** Installing the dock on a computer whose person already has a hub asks for the server's address once (or
  reads it from the account's other computer), and ends linked and adopted: nothing left to do but use it.

## Left to the builder

The exact envelope and message names of the link (the earlier document sketches them); how the dock draws the place of a session;
which settings keys are shared (derived from `docs/settings.md`, a key is local when it names something of the computer); the Engram
export format; what the macOS service does for the private runtime folder.

## Rulings (mine, to be checked)

1. **Pi stays on the machine whose files it touches.** The owner's "work where I ask" is met by choosing the session's place, not by
   proxying tools. This keeps every tool, the sandbox and the checkpoints as they are.
2. **No copy of the brain at rest on a linked computer**, except the opt-in encrypted offline copy. The owner said the local Aria is
   replaced; a copy would be the second environment again.
3. **A new session cannot start when the server is unreachable.** The alternative, falling back to local keys, is the second
   environment and a stolen-laptop risk.
4. **A session's place is fixed at creation.** Moving a running session between places would mean moving its Pi and its files.
5. **The default place follows where `aria` is started** and asks only when it cannot tell.
6. **`aria connect` stays as an alias**, so nothing breaks, and is dropped from the guides.
7. **Removing local copies is never automatic**: default keep, a yes per kind, only after the server was shown to hold them.
8. **Order:** versions first, then the link, then the brain, then the one command, because each step is useful alone and the one
   command is not safe until the brain is on the hub.

## What this changes in the earlier design

- "A hub is `aria connect <host>`" becomes "a hub is where `aria` talks when this computer is linked".
- "If the hub is down when the machine starts, it uses its own settings and credentials" becomes "it cannot start a session" (ruling 3).
- "A machine's live copies are not persisted" stays, and is now the only form a linked computer holds the brain in.
