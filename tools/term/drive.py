# Drives the real `aria` client in a pseudo-terminal: a new session in a folder, !pwd there, Ctrl+C, back again, /exit.
import json, os, pty, re, select, sys, time

aria, folder, home, socket_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]|\x1b[=>]")

def start():
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(folder)
        os.environ["COLUMNS"], os.environ["LINES"] = "100", "30"
        os.execv(aria, [aria])
    import fcntl, struct, termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    return pid, fd

def read(fd, seconds):
    out = b""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                out += os.read(fd, 65536)
            except OSError:
                break
    return ANSI.sub(b"", out).decode("utf-8", "replace")

def until(fd, needle, seconds=20):
    seen = ""
    end = time.time() + seconds
    while time.time() < end:
        seen += read(fd, 0.5)
        if needle in seen:
            return seen
    raise SystemExit(f"FAIL: never saw {needle!r}; last screen text: {seen[-600:]!r}")

def ok(what):
    print(f"ok: {what}")

def sessions():
    return json.load(open(os.path.join(home, "sessions.json")))["sessions"]

import socket

def dock_alive():
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(socket_path)
        return True
    except OSError:
        return False
    finally:
        s.close()

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
