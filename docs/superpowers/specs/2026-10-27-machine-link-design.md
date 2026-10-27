# The machine link (piece S2 of One Aria)

Status: design, 2026-10-27. This is piece S2 of [One Aria](2026-10-27-one-aria-design.md), and M2 of
[the hub design](2026-10-23-hub-and-machines-design.md): it links a computer to a hub **as a machine**, so the hub's clients see
the machine's sessions and can work on them. It does not move credentials or memory (S3) and does not change what `aria` opens
(S4).

## What a client sees when it is done

A terminal attached to the hub lists **every session of the hub and of every linked machine**, each tagged with where it lives;
a machine that is off shows greyed with its last known sessions; and a session of the machine can be opened, answered and closed
from the hub's terminal, running on the machine, inside its sandbox, on its files. `+ new on <machine>` starts one there.

## The picture

```
 a client ──ssh──▶ hub: aria client-stdio ──▶ clients.sock ┐
                                                            ▼
                                                     the hub's service
                                                            ▲ machines.sock (mode 600)
 machine's service ──ssh──▶ hub: aria hub-link machine <name>  (the forced command of the machine's key)
   `aria link` (a supervised thread)
   relays the hub's lines to its own clients.sock, one local connection per hub client
```

## The link

One ssh connection per machine, opened by the machine (outbound only: NAT is no problem), carrying JSON lines both ways.
`ssh -i <machine key> -o IdentitiesOnly=yes <hub> aria hub-link machine <name>`; on the hub the key's **forced command** runs
`aria hub-link machine <name>`, which connects the ssh stdio to the hub service's `machines.sock` (a bridge, like
`client-stdio`). The `<name>` is in the forced command the hub's `authorized_keys` holds, never taken from the machine's input:
a key is one machine.

Names: `^[a-z0-9][a-z0-9-]{0,30}$`. A name that is already linked is **replaced** by the newer connection (the old one is
almost always a dead socket the hub has not noticed), and the old one is closed.

### Lines (machine to hub, unless said)

| Line | Meaning |
|---|---|
| `{"type":"link-hello","protocol":1,"name":"laptop","os":"linux","version":"0.22.0"}` | The first line a machine sends. The hub answers `link-hello` with its own `protocol` and `version`, or `{"type":"error","code":"protocol-mismatch"}` and closes. The `name` must equal the one the forced command gave, else `name-mismatch` and close. |
| `{"type":"link-sessions","sessions":[...],"statuses":{...}}` | The machine's `sessions` list and `sessionStatus` map, exactly as its own clients receive them (the same JSON the dock sends the page). Sent after the hello and on every change. The hub keeps the last one as that machine's cache. |
| `{"type":"link-ping"}` / `{"type":"link-pong"}` | Heartbeat, either way, every 15 s; a side that hears nothing for 45 s closes the link. |
| `{"c":<n>,"line":{...}}` hub to machine | A line a hub client `n` sent for this machine (below). |
| `{"c":<n>,"line":{...}}` machine to hub | A line the machine's own clients hub produced for client `n`: the reply or event that client should get. |
| `{"c":<n>,"close":true}` either way | Client `n` is gone (hub to machine: the machine closes its local connection for `n`; machine to hub: that local connection died, the hub closes client `n`'s attachment to this machine). |

All of it is additive: a protocol number other than 1 is refused; unknown `type`s are ignored.

### What `aria link` does on the machine

The machine's service runs a supervised thread that keeps the link up (connect, hello, push the session list, heartbeat; on a
drop, reconnect with backoff 2 s, 4 s, 8 s ... up to 60 s; stop when the service quits). For every client number `n` the hub
names it opens **one local connection to its own `clients.sock`** with `hello {remote:true}` (so the machine applies exactly the
restrictions of a remote client: no `secretSave`, no `!command`, no folder chosen by the client, only sessions it has) and relays
lines both ways wrapped with `c: n`. When the hub says `close` for `n` it closes that connection; when the connection dies it says
`close` for `n`. The machine pushes `link-sessions` whenever its session list or statuses change (it already computes them for
its own clients).

Nothing else of the machine is exposed: the hub reaches the machine only through what a remote client may do.

## The hub's side

- **`aria hub-link machine <name>`**: connects to `machines.sock`, and relays stdin/stdout. Refuses (exit 2, nothing started)
  any other first argument and any `<name>` that does not match the pattern.
- **The registry**: per machine, `{name, os, version, connected_since, sessions, statuses, link queue}`; a machine that drops stays
  with its last cache and `offline: true` (the cache is in memory only: a restarted hub shows machines only once they link again).
- **The list**: when a client asks for `sessionsLoad` (and on every change), the hub answers with its own sessions plus each
  machine's cached ones, **each entry gaining `machine: "<name>"`** (the field is absent on the hub's own sessions, so a page or a
  terminal that ignores it keeps working), and machines' `statuses` merged. An offline machine's
  entries get `offline: true`.
- **Routing a client's line**: a line addressed to a session id the registry knows is a machine's is wrapped `{"c":<n>,"line":...}`
  and queued on that machine's link; a line for a session the hub has is handled as today. `attach`, `detach` and `focus` follow the
  session they name. A `sessionNew` with `"machine":"<name>"` goes to that machine (`machine-offline` error to the client when it
  is not linked); without it, to the hub. Lines from a machine for client `n` are delivered to that client unchanged (the machine
  rewrites ids as its own hub does, so the client gets its own ids back).
- **Remote restrictions stay**: a line from a remote client is checked by the hub's `for_remote` first (as today), and then again
  by the machine's own check. Lines machines send are treated as data shown to the owner's own clients.
- **A machine is not a client**: it cannot send a line that the hub handles as a client's; the hub reads only the types in the
  table from it.

## Keys and `hub init`

- `aria hub init` (and `aria hub link`, below) makes the machine key `~/.ssh/aria-machine-<name>` the way it makes the client key
  (never overwriting; passphrase asked; the key must NOT need a passphrase for the link to start unattended, so the link uses
  `ssh-agent` or an unencrypted machine key: the plan says which and why, and defaults to asking the person once; see rulings) and
  adds `restrict,command="~/.local/bin/aria hub-link machine <name>" <key>` to the hub's `authorized_keys`.
- The machine's name defaults to its hostname lowercased and cleaned to the pattern; `--machine-name <name>` overrides it.
- `aria hub machines` (on the hub, or with the person's ssh login from anywhere) lists linked machines with their state;
  `aria hub unlink-machine <name>` removes the key line (revocation is the line) and drops the link.
- The machine's own config records it: `~/.aria/link.json` `{"hub":"user@host","name":"laptop","key":"..."}` (mode 600). The service
  starts `aria link` when that file exists.

## Pieces to build

1. **Hub side (S2a)**: `machines.sock`, `aria hub-link machine <name>`, the registry, the merged list, routing, `sessionNew` on a
   machine, the heartbeat, replacement by name; tests with in-process fake machines over socket pairs.
2. **Machine side and keys (S2b)**: `aria link` in the service (supervised thread, relay, reconnect), `link.json`, `hub init` making the
   machine key and line, `aria hub machines` and `unlink-machine`, the screens in `hub_ui`, docs.
3. **Clients (S2c)**: the terminal's list shows the machine tag, greyed offline entries, and `+ new on <machine>`; the dock's page
   ignores the extra fields until S4.

## Rulings (mine, to be checked)

1. **One ssh connection per machine, opened by the machine.** It is outbound and survives NAT, and needs no port.
2. **A hub client reaches a machine through a local connection on the machine** (a remote client for the machine), so all the
   restrictions and the id rewriting already exist and are not written twice.
3. **Names are fixed by the forced command**, so a stolen machine key can only be that machine; replacing a live link of the same
   name is allowed because a dead socket is the common case.
4. **The hub's cache of a machine's sessions is memory only**, because it is only a view; the machine is the owner of its sessions.
5. **The machine key may be unencrypted** if the person chooses it (the link must start with nobody there); the default is to use
   the person's ssh-agent, and the plan says plainly that an unencrypted key lets anyone who copies the file link as this
   machine, which a restricted command limits to "being a machine that answers client lines".
6. **A machine's lines are data**: the hub passes only the fixed message types, and the owner's clients show what a machine says
   as they show what the dock says, so a stolen machine key is as harmful as a stolen computer, not more.
