# What the scripts that drive the terminal client in a pseudo-terminal share (drive.py, ../hub/drive_connect.py).
import fcntl, os, pty, re, select, socket, struct, sys, termios, time

ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]|\x1b[=>]")

def spawn(argv, cwd):
    """Runs argv in a 100x30 pseudo-terminal started in cwd; returns (pid, fd)."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.environ["COLUMNS"], os.environ["LINES"] = "100", "30"
        os.execv(argv[0], argv)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    return pid, fd

def read_raw(fd, seconds):
    out = b""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                out += os.read(fd, 65536)
            except OSError:
                break
    return out

def text(raw):
    return ANSI.sub(b"", raw).decode("utf-8", "replace")

def read(fd, seconds):
    return text(read_raw(fd, seconds))

def until(fd, needle, seconds=20):
    raw = b""
    end = time.time() + seconds
    while time.time() < end:
        raw += read_raw(fd, 0.5)
        seen = text(raw)  # stripped as a whole: an escape sequence can be split between two reads
        if needle in seen:
            return seen
    raise SystemExit(f"FAIL: never saw {needle!r}; last screen text: {seen[-600:]!r}")

def ok(what):
    print(f"ok: {what}")

def socket_alive(path):
    """Whether something accepts connections on the unix socket at path."""
    s = socket.socket(socket.AF_UNIX)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()

class Screen:
    """What the terminal shows: the client draws the changed cells only, at absolute positions, so text of its own can be
    spread over several writes. Understands what the client sends: cursor positions and moves, erasing, and printable text."""
    CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])")

    def __init__(self, rows=30, cols=100):
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.row = self.col = 0
        self.pending = ""
        import codecs
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, raw):
        data = self.pending + self.decoder.decode(raw)
        self.pending = ""
        i = 0
        while i < len(data):
            c = data[i]
            if c == "\x1b":
                m = self.CSI.match(data, i)
                if m:
                    self.csi(m.group(1), m.group(2))
                    i = m.end()
                    continue
                if i + 1 >= len(data) or data[i + 1] == "[" or (data[i + 1] == "]" and "\x07" not in data[i:]):
                    self.pending = data[i:]  # cut in two by the read
                    return
                if data[i + 1] == "]":
                    i = data.index("\x07", i) + 1
                else:
                    i += 3 if data[i + 1] in "()" else 2
                continue
            if c == "\r":
                self.col = 0
            elif c == "\n":
                self.row = min(self.row + 1, self.rows - 1)
            elif c >= " ":
                if self.col < self.cols:
                    self.grid[self.row][self.col] = c
                    self.col += 1
            i += 1

    def csi(self, params, final):
        n = [int(p) if p.isdigit() else 0 for p in params.lstrip("?").split(";")]
        first = n[0] or 1
        if final in "Hf":
            self.row = min((n[0] or 1) - 1, self.rows - 1)
            self.col = min(((n[1] if len(n) > 1 else 0) or 1) - 1, self.cols - 1)
        elif final == "A": self.row = max(self.row - first, 0)
        elif final == "B": self.row = min(self.row + first, self.rows - 1)
        elif final == "C": self.col = min(self.col + first, self.cols - 1)
        elif final == "D": self.col = max(self.col - first, 0)
        elif final == "G": self.col = min(first - 1, self.cols - 1)
        elif final == "J" and n[0] in (2, 3):
            self.grid = [[" "] * self.cols for _ in range(self.rows)]
        elif final == "K":
            if n[0] == 0: self.grid[self.row][self.col:] = [" "] * (self.cols - self.col)
            elif n[0] == 1: self.grid[self.row][: self.col + 1] = [" "] * (self.col + 1)
            else: self.grid[self.row] = [" "] * self.cols
        elif final == "X":
            end = min(self.col + first, self.cols)
            self.grid[self.row][self.col:end] = [" "] * (end - self.col)

    def text(self):
        return "\n".join("".join(row).rstrip() for row in self.grid).rstrip()

def screen_until(fd, screen, needles, seconds=20):
    """Reads into screen until every string in needles is on it; returns the screen's text."""
    end = time.time() + seconds
    while True:
        screen.feed(read_raw(fd, 0.3))
        shown = screen.text()
        if all(n in shown for n in needles):
            return shown
        if time.time() >= end:
            raise SystemExit(f"FAIL: never saw {needles!r}; the screen: {shown[-600:]!r}")
