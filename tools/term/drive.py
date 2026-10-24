# Drives the real `aria` client in a pseudo-terminal: a new session in a folder, !pwd there, Ctrl+C, back again, /exit.
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script's folder out
from pty_helpers import ok, read, socket_alive, spawn, until

aria, folder, home, socket_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def start():
    return spawn([aria], folder)

def sessions():
    return json.load(open(os.path.join(home, "sessions.json")))["sessions"]

def dock_alive():
    return socket_alive(socket_path)

# The client starts the dock; closing the client's terminal (a hang-up, no Ctrl+C) leaves the dock running.
assert not dock_alive(), "FAIL: a dock was already running"
pid, fd = start()
until(fd, folder)
assert dock_alive(), "FAIL: the client did not start the dock"
os.close(fd)
time.sleep(2)
os.waitpid(pid, 0)
assert dock_alive(), "FAIL: the dock died with the terminal that started it"
ok("the dock the client started outlives the client's terminal")

pid, fd = start()
until(fd, folder)  # the list offers a new session in this folder
ok("the list offers a new session in the current folder")
before = len(sessions()) if os.path.exists(os.path.join(home, "sessions.json")) else 0
for _ in range(10):
    os.write(fd, b"\x1b[B")  # Down to the last row: the new one
time.sleep(0.3)
os.write(fd, b"\r")
time.sleep(1.5)
new = [s for s in sessions() if s.get("workdir") == folder]
assert new, f"FAIL: no session with workdir {folder}: {sessions()}"
ok("a new session was saved with the folder it was opened in")
os.write(fd, b"!pwd\r")
until(fd, "!pwd", 30)
screen = until(fd, folder.split("/")[-1], 30)
ok("!pwd ran in the session")
os.write(fd, b"/context\r")
until(fd, "\u25a1", 30)  # the grid's free cells
ok("/context draws what fills the context")
os.write(fd, b"\x1b")
time.sleep(0.5)
os.write(fd, b"\x03")  # Ctrl+C
time.sleep(1)
done, status = os.waitpid(pid, os.WNOHANG)
assert done == pid, "FAIL: Ctrl+C did not leave the client"
ok("Ctrl+C left the client")
assert any(s["id"] == new[0]["id"] for s in sessions()), "FAIL: the session went away with the client"
ok("the session is still there after leaving")

pid, fd = start()
until(fd, folder)
# The list shows the latest activity first: the session just used is on top.
os.write(fd, b"\x1b[A" * 10)
time.sleep(0.2)
os.write(fd, b"\r")
until(fd, "!pwd", 30)
ok("attaching again shows the conversation (the !pwd is there)")
os.write(fd, b"/exit\r")
time.sleep(1.5)
done, status = os.waitpid(pid, os.WNOHANG)
assert done == pid, "FAIL: /exit did not leave the client"
assert not any(s["id"] == new[0]["id"] for s in sessions()), "FAIL: /exit did not close the session"
assert any(s.get("schedule") is None for s in sessions()), "FAIL: no session left"
ok("/exit closed the session, and there is still one")
print("live check passed")
