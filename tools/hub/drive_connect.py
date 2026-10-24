# Drives `aria connect hub.example` in a pseudo-terminal against a hub started by accept.sh (ssh is the shim, which runs
# the hub's `client-stdio` on this computer): checks (a)-(d) of accept.sh. Prints `ok: ...`; the first check that fails
# prints `FAIL: ...` and ends the script with status 1.
#   python3 -I drive_connect.py <aria> <client folder> <hub's ARIA_HOME> <hub's clients.sock> <the shim's pid file>
import json, os, signal, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "term"))  # -I leaves it out
from pty_helpers import Screen, ok, screen_until, socket_alive, spawn

aria, folder, home, socket_path, pidfile = sys.argv[1:6]
SEEDED = {"seed-1": "Fix the checkout widget", "seed-2": "Review the release notes"}

def fail(what):
    raise SystemExit(f"FAIL: {what}")

def sessions():
    return json.load(open(os.path.join(home, "sessions.json")))["sessions"]

def connect():
    try:
        os.remove(pidfile)
    except FileNotFoundError:
        pass
    pid, fd = spawn([aria, "connect", "hub.example"], folder)
    return pid, fd, Screen()

def drop(pid, fd):
    """The connection dies the way a dropped network does: the client and the bridge ssh would have run are killed."""
    bridge = None
    if os.path.exists(pidfile):
        bridge = int(open(pidfile).read().strip())
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    if bridge:
        try:
            os.kill(bridge, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.close(fd)
    time.sleep(1)

# (a) the list shows the hub's own sessions, and offers a new one on the host (not in a folder of this computer)
pid, fd, screen = connect()
screen_until(fd, screen, [*SEEDED.values(), "+ new on hub.example"])
ok("the list over connect shows the hub's sessions")

# (b) a session opened over connect has no folder: the client's folder is not sent
for _ in range(10):
    os.write(fd, b"\x1b[B")  # Down to the last row: the new one
time.sleep(0.3)
os.write(fd, b"\r")
time.sleep(2)
new = [s for s in sessions() if s["id"] not in SEEDED]
if len(new) != 1:
    fail(f"expected one new session, got {new}")
if "workdir" in new[0] or folder in open(os.path.join(home, "sessions.json")).read():
    fail(f"the session opened over connect has a folder: {new[0]}")
ok("a session opened over connect has no folder of the client's")

# (c) the client dies: the hub keeps running and the session stays; a second connect lists it again
drop(pid, fd)
if not socket_alive(socket_path):
    fail("the hub stopped when the client was killed")
if not any(s["id"] == new[0]["id"] for s in sessions()):
    fail("the session went away when the client was killed")
pid, fd, screen = connect()
screen_until(fd, screen, ["untitled", *SEEDED.values()])
ok("a killed client leaves the hub and the session; a second connect lists it again")

# (d) /schedules over connect: nothing is scheduled on this hub
os.write(fd, b"\r")  # the first row
time.sleep(2)
os.write(fd, b"/schedules\r")
screen_until(fd, screen, ["No scheduled tasks"], 30)
ok("/schedules over connect says there are no scheduled tasks")
drop(pid, fd)
