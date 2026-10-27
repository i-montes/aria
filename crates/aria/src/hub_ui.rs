//! The screen of `aria hub init`, as the events of a run. `hub_admin.rs` decides what happens and calls this trait for
//! everything it shows on stdout (the error texts on stderr stay where they are); an implementation decides how it looks.
//! [`PlainUi`] is the text version, and the fallback for every case that has no full-screen interface: no terminal, `--yes`,
//! `NO_COLOR`, `TERM=dumb`, a pipe, a log, the tests. What the events say is data and words, never the layout: the text of
//! a line does not depend on the implementation, so the tests that read PlainUi's output hold for any of them.
//!
//! The order of a run: `title`, `looking`, `probe_ok`/`probe_note` while the hub is read, `details` (only with `--details`),
//! `plan`, `confirm` (or `declined`), then for each step `step_start` with `step_ok`/`step_info` under it (and `failed` when
//! it breaks), `heads_up` once, and `done`. A step that hands the terminal to another program (the installer on the hub,
//! `ssh-keygen`) is wrapped by `hand_terminal_over` and `take_terminal_back`.

use std::io::{self, BufRead, IsTerminal, Write};

/// The widest line the text version prints, so it fits an 80-column terminal (a remote login often is one).
pub const WIDTH: usize = 78;

/// One line of a [`Note`].
#[derive(Debug, Clone, PartialEq)]
pub enum NoteLine {
    /// A sentence.
    Text(String),
    /// Something to type: shown set apart, never wrapped.
    Command(String),
}

/// Something the owner should know, or do, that is not a step: a warning with the one action it calls for.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Note {
    pub lines: Vec<NoteLine>,
}

impl Note {
    pub fn text(text: impl Into<String>) -> Note {
        Note { lines: vec![NoteLine::Text(text.into())] }
    }

    pub fn then_text(mut self, text: impl Into<String>) -> Note {
        self.lines.push(NoteLine::Text(text.into()));
        self
    }

    pub fn then_command(mut self, command: impl Into<String>) -> Note {
        self.lines.push(NoteLine::Command(command.into()));
        self
    }
}

/// What the owner is shown before the first yes: the steps, in the order they run, and what to know about them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Plan {
    /// The host as the owner typed it.
    pub host: String,
    /// The numbered steps; the `[k/n]` headers of the run use the same numbers (k is the position, from 1).
    pub steps: Vec<String>,
    /// "Good to know": sentences, each about one thing.
    pub good_to_know: Vec<String>,
    /// What stops the run before it starts, with the command that fixes it.
    pub blocked: Vec<Note>,
}

/// How a run that did not fail ends.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Working: `connect` is what to type to use it (`aria connect`, with the host when `aria connect` alone is not it).
    Ready { connect: String },
    /// Installed, but not working yet; `why` says what to do.
    NotReady { why: String },
    /// `aria hub update` ended: every server it looked at and the version it runs now, as (host, version).
    Updated { runs: Vec<(String, String)> },
}

/// The screen of `aria hub init`. Every method is one thing that happens in a run.
pub trait HubUi {
    /// The first line: what is being done, e.g. "linking this computer to your server".
    fn title(&mut self, text: &str);
    /// Reading the hub starts: nothing on it changes until the yes.
    fn looking(&mut self, host: &str);
    /// One thing found out that is fine ("Ubuntu 24.04.4 LTS, aarch64").
    fn probe_ok(&mut self, line: &str);
    /// Something being read or done while the hub is looked at, that is not a result (a keyring that may ask).
    fn probe_info(&mut self, line: &str);
    /// One thing found out that needs a word before the plan.
    fn probe_note(&mut self, note: &Note);
    /// The technical plan, command by command (`--details`), one paragraph per entry.
    fn details(&mut self, paragraphs: &[String]);
    /// The plan in plain words.
    fn plan(&mut self, plan: &Plan);
    /// Asks whether to go on. False for a no, and when there is nobody to ask.
    fn confirm(&mut self) -> bool;
    /// The owner said no (or could not be asked): nothing was changed.
    fn declined(&mut self);
    /// Step `k` of `n` begins: its title and one sentence of what it is for (empty for none).
    fn step_start(&mut self, k: usize, n: usize, title: &str, why: &str);
    /// Something in the current step worked.
    fn step_ok(&mut self, line: &str);
    /// A word about the current step that is not a result.
    fn step_info(&mut self, line: &str);
    /// Something failed. The error text itself goes to stderr as it always did; this is the mark beside it.
    fn failed(&mut self, line: &str);
    /// The notes of the run, once, after the steps.
    fn heads_up(&mut self, notes: &[Note]);
    /// The end of a run that did not fail.
    fn done(&mut self, outcome: &Outcome);
    /// The next thing writes to the terminal itself (the installer's output, ssh-keygen asking for a passphrase): a
    /// full-screen interface leaves the screen to it. The text version has nothing to do.
    fn hand_terminal_over(&mut self) {}
    /// That thing ended.
    fn take_terminal_back(&mut self) {}
}

/// The screen for this run: the full-screen version (`hub_tui.rs`) where the terminal can hold it, the text one everywhere
/// else (`force_plain` is `--plain`).
pub fn ui_for_terminal(force_plain: bool) -> Box<dyn HubUi> {
    match crate::hub_tui::for_terminal(force_plain) {
        Some(ui) => Box::new(ui),
        None => Box::new(PlainUi::stdio()),
    }
}

// ---- the text version ------------------------------------------------------------------------------------------------

/// Words of `text` in lines of at most `width` columns; a word longer than that keeps a line of its own.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.chars().count() + 1 + word.chars().count() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Whether an answer to "Press Enter to start" goes on: Enter, `y` or `yes`. Anything else, `n` and `no` included, does not.
pub fn answer_goes_on(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes")
}

/// What the text screen looks like: numbered steps, one sentence of what each is for, check marks. Colour only on a terminal
/// that takes it (`NO_COLOR` and `TERM=dumb` turn it off) and ASCII marks unless the locale says UTF-8; in a pipe or a log the
/// same sentences come out plain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Screen {
    color: bool,
    utf8: bool,
}

impl Screen {
    pub fn detect() -> Screen {
        let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
            .iter()
            .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
            .unwrap_or_default();
        Screen::with(
            io::stdout().is_terminal(),
            std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()),
            &std::env::var("TERM").unwrap_or_default(),
            &locale,
        )
    }

    pub fn with(is_terminal: bool, no_color: bool, term: &str, locale: &str) -> Screen {
        let locale = locale.to_ascii_lowercase();
        Screen {
            color: is_terminal && !no_color && !term.is_empty() && term != "dumb",
            utf8: locale.contains("utf-8") || locale.contains("utf8"),
        }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color && !text.is_empty() {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    fn mark_ok(&self) -> &'static str {
        if self.utf8 {
            "✓"
        } else {
            "ok"
        }
    }

    fn mark_failed(&self) -> &'static str {
        if self.utf8 {
            "✗"
        } else {
            "x"
        }
    }

    fn dot(&self) -> &'static str {
        if self.utf8 {
            "·"
        } else {
            "-"
        }
    }

    fn rule(&self) -> String {
        format!("  {}", (if self.utf8 { "─" } else { "-" }).repeat(56))
    }

    /// `lead` (painted with `lead_code`) and then `text` (painted with `text_code`) wrapped to the width, the continuation
    /// lines indented under the text.
    fn hang(&self, lead: &str, lead_code: &str, text: &str, text_code: &str) -> String {
        let cols = lead.chars().count();
        let lines = wrap(text, WIDTH.saturating_sub(cols).max(20));
        if lines.is_empty() {
            return self.paint(lead_code, lead.trim_end());
        }
        let indent = " ".repeat(cols);
        lines
            .iter()
            .enumerate()
            .map(|(at, line)| {
                let head = if at == 0 { self.paint(lead_code, lead) } else { indent.clone() };
                format!("{head}{}", self.paint(text_code, line))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn title(&self, text: &str) -> String {
        format!(
            "\n  {} {}\n{}",
            self.paint("1;36", "Aria"),
            self.paint("2", &format!("{} {text}", self.dot())),
            self.rule()
        )
    }

    /// `[k/n] title`, then the sentence of what it is for.
    pub fn step(&self, k: usize, n: usize, title: &str, why: &str) -> String {
        let head = format!("{} {}", self.paint("1;36", &format!("[{k}/{n}]")), self.paint("1", title));
        if why.is_empty() {
            format!("\n{head}")
        } else {
            format!("\n{head}\n{}", self.hang("      ", "", why, "2"))
        }
    }

    pub fn ok(&self, text: &str) -> String {
        self.hang(&format!("  {} ", self.mark_ok()), "32", text, "")
    }

    pub fn failed(&self, text: &str) -> String {
        self.hang(&format!("  {} ", self.mark_failed()), "31", text, "")
    }

    pub fn info(&self, text: &str) -> String {
        self.hang("      ", "", text, "2")
    }

    /// `!` and the sentences of a note, its commands set apart.
    pub fn note(&self, note: &Note) -> String {
        let mut out = Vec::new();
        for (at, line) in note.lines.iter().enumerate() {
            match line {
                NoteLine::Text(text) => out.push(self.hang(if at == 0 { "  ! " } else { "    " }, "33", text, "")),
                NoteLine::Command(command) => out.push(format!("        {}", self.paint("1", command))),
            }
        }
        out.join("\n")
    }
}

/// The text version of the screen, on any writers: stdout and stderr in a run, buffers in a test.
pub struct PlainUi {
    screen: Screen,
    out: Box<dyn Write>,
    err: Box<dyn Write>,
    /// Where the answer to `confirm` is read from; None when there is nobody to ask.
    input: Option<Box<dyn BufRead>>,
}

impl PlainUi {
    /// The screen of a run in this terminal: stdout, stderr, and stdin when it is a terminal.
    pub fn stdio() -> PlainUi {
        let input: Option<Box<dyn BufRead>> =
            if io::stdin().is_terminal() { Some(Box::new(io::BufReader::new(io::stdin()))) } else { None };
        PlainUi::new(Screen::detect(), Box::new(io::stdout()), Box::new(io::stderr()), input)
    }

    pub fn new(screen: Screen, out: Box<dyn Write>, err: Box<dyn Write>, input: Option<Box<dyn BufRead>>) -> PlainUi {
        PlainUi { screen, out, err, input }
    }

    /// Written whole and flushed, so what a program the terminal is handed to prints comes after it.
    fn put(&mut self, text: &str) {
        let _ = writeln!(self.out, "{text}");
        let _ = self.out.flush();
    }
}

impl HubUi for PlainUi {
    fn title(&mut self, text: &str) {
        let line = self.screen.title(text);
        self.put(&line);
    }

    fn looking(&mut self, host: &str) {
        self.put(&format!("\n  Looking at {host}..."));
    }

    fn probe_ok(&mut self, line: &str) {
        let line = self.screen.ok(line);
        self.put(&line);
    }

    fn probe_info(&mut self, line: &str) {
        let text = self.screen.hang("  ", "", line, "2");
        self.put(&text);
    }

    fn probe_note(&mut self, note: &Note) {
        let text = self.screen.note(note);
        self.put(&text);
    }

    fn details(&mut self, paragraphs: &[String]) {
        for paragraph in paragraphs {
            let text = wrap(paragraph, WIDTH).join("\n");
            self.put(&text);
        }
    }

    fn plan(&mut self, plan: &Plan) {
        let screen = self.screen;
        let mut text = format!("\n  {}", screen.paint("1", &format!("What I will do on {}", plan.host)));
        for (at, step) in plan.steps.iter().enumerate() {
            text.push('\n');
            text.push_str(&screen.hang(&format!("    {}  ", at + 1), "1;36", step, ""));
        }
        if !plan.good_to_know.is_empty() {
            text.push_str(&format!("\n\n  {}", screen.paint("1", "Good to know")));
            for item in &plan.good_to_know {
                text.push('\n');
                text.push_str(&screen.hang(&format!("    {} ", screen.dot()), "", item, ""));
            }
            text.push('\n');
            text.push_str(&screen.hang("    ", "", "--details lists every command.", "2"));
        }
        for note in &plan.blocked {
            text.push_str("\n\n");
            text.push_str(&screen.note(note));
        }
        self.put(&text);
    }

    fn confirm(&mut self) -> bool {
        let Some(input) = self.input.as_mut() else {
            return false;
        };
        let _ = write!(self.out, "\n  Press Enter to start, or Ctrl+C to cancel. ");
        let _ = self.out.flush();
        let mut answer = String::new();
        // An input that ends (read of 0 bytes) is not an Enter.
        matches!(input.read_line(&mut answer), Ok(read) if read > 0) && answer_goes_on(&answer)
    }

    fn declined(&mut self) {
        self.put("nothing was changed");
    }

    fn step_start(&mut self, k: usize, n: usize, title: &str, why: &str) {
        let text = self.screen.step(k, n, title, why);
        self.put(&text);
    }

    fn step_ok(&mut self, line: &str) {
        let text = self.screen.ok(line);
        self.put(&text);
    }

    fn step_info(&mut self, line: &str) {
        let text = self.screen.info(line);
        self.put(&text);
    }

    fn failed(&mut self, line: &str) {
        let text = self.screen.failed(line);
        let _ = writeln!(self.err, "{text}");
        let _ = self.err.flush();
    }

    fn heads_up(&mut self, notes: &[Note]) {
        if notes.is_empty() {
            return;
        }
        let mut text = format!("\n  {}", self.screen.paint("1", "Heads up"));
        for note in notes {
            text.push('\n');
            text.push_str(&self.screen.note(note));
        }
        self.put(&text);
    }

    fn done(&mut self, outcome: &Outcome) {
        let screen = self.screen;
        let mut text = format!("\n{}", screen.rule());
        match outcome {
            Outcome::Ready { connect } => {
                text.push('\n');
                text.push_str(&screen.paint("1;32", &format!("  {} Your server is ready.", screen.mark_ok())));
                text.push_str(&format!("\n\n  To use it, type:  {}", screen.paint("1", connect)));
            }
            Outcome::NotReady { why } => {
                text.push('\n');
                text.push_str(&screen.note(&Note::text(format!("The server is installed, but not ready: {why}"))));
            }
            Outcome::Updated { runs } => {
                for (host, version) in runs {
                    text.push('\n');
                    let line = format!("  {} {host} runs Aria {version}", screen.mark_ok());
                    text.push_str(&screen.paint("1;32", &line));
                }
            }
        }
        self.put(&text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Shared {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn plain(input: Option<&str>) -> (PlainUi, Shared, Shared) {
        let (out, err) = (Shared::default(), Shared::default());
        let input: Option<Box<dyn BufRead>> =
            input.map(|text| Box::new(io::Cursor::new(text.as_bytes().to_vec())) as Box<dyn BufRead>);
        let ui = PlainUi::new(
            Screen::with(false, false, "xterm", "en_US.UTF-8"),
            Box::new(out.clone()),
            Box::new(err.clone()),
            input,
        );
        (ui, out, err)
    }

    #[test]
    fn wrap_keeps_lines_within_the_width_and_a_long_word_on_its_own_line() {
        let lines = wrap("one two three four five six seven", 10);
        assert!(lines.iter().all(|line| line.chars().count() <= 10), "{lines:?}");
        assert_eq!(lines.join(" "), "one two three four five six seven");
        assert_eq!(
            wrap("a /a/very/long/path/that/does/not/break b", 10),
            vec!["a", "/a/very/long/path/that/does/not/break", "b"]
        );
        assert!(wrap("   ", 10).is_empty());
    }

    #[test]
    fn enter_y_and_yes_go_on_and_nothing_else_does() {
        for yes in ["\n", "", "y\n", "Y", " yes \n", "YES"] {
            assert!(answer_goes_on(yes), "{yes:?}");
        }
        for no in ["n\n", "no", "nope", "yess", "x", "go"] {
            assert!(!answer_goes_on(no), "{no:?}");
        }
    }

    #[test]
    fn confirm_reads_one_line_and_is_no_without_a_terminal_or_at_the_end_of_the_input() {
        assert!(plain(Some("\n")).0.confirm());
        assert!(plain(Some("y\n")).0.confirm());
        assert!(!plain(Some("n\n")).0.confirm());
        assert!(!plain(Some("")).0.confirm(), "an input that ends is not an Enter");
        let (mut none, out, _) = plain(None);
        assert!(!none.confirm());
        assert_eq!(out.text(), "", "nobody is asked when there is nobody to ask");
        let (mut asked, out, _) = plain(Some("\n"));
        asked.confirm();
        assert!(out.text().contains("Press Enter to start, or Ctrl+C to cancel."), "{}", out.text());
    }

    #[test]
    fn a_whole_run_fits_78_columns_and_the_failure_goes_to_stderr() {
        let (mut ui, out, err) = plain(Some("\n"));
        ui.title("linking this computer to your server");
        ui.looking("vps");
        ui.probe_ok("Ubuntu 24.04.4 LTS, aarch64");
        ui.probe_note(&Note::text("This computer has no keyring, so there are no sign-ins to send and the server will have no model until one is added.").then_command("aria hub init vps --credentials"));
        ui.plan(&Plan {
            host: "vps".into(),
            steps: vec![
                "Install Aria 0.21.0 on the server".into(),
                "Copy your settings and sign-ins (the server has none yet), a sentence that is long enough to need a second line".into(),
            ],
            good_to_know: vec!["Your sign-ins are copied to the server, which keeps and uses them, and nothing else changes in this computer.".into()],
            blocked: vec![Note::text("The server cannot unpack Node: it needs xz.").then_command("sudo apt-get install -y xz-utils")],
        });
        assert!(ui.confirm());
        ui.step_start(
            1,
            2,
            "Installing Aria on the server",
            "The server downloads Aria and sets itself up. It may ask for the server's sudo password.",
        );
        ui.step_ok("The server runs Aria 0.21.0");
        ui.step_info("(the server's installer prints its own steps here)");
        ui.failed("It did not work: ssh did not answer in time: if the key has a passphrase, add it to ssh-agent and run aria connect");
        ui.heads_up(&[Note::text("Linger is off for ubuntu on the server: it stops when you log out.")
            .then_command("sudo loginctl enable-linger ubuntu")]);
        ui.done(&Outcome::Ready { connect: "aria connect".into() });
        for (name, text) in [("stdout", out.text()), ("stderr", err.text())] {
            for line in text.lines() {
                assert!(line.chars().count() <= WIDTH, "{name}: {line:?} is {} columns", line.chars().count());
            }
        }
        assert!(err.text().contains("✗ It did not work"), "{}", err.text());
        assert!(!out.text().contains("It did not work"), "a failure is not on stdout: {}", out.text());
        let said = out.text();
        for part in [
            "What I will do on vps",
            "    1  Install Aria 0.21.0 on the server",
            "Good to know",
            "--details lists every command.",
            "[1/2] Installing Aria on the server",
            "Heads up",
            "✓ Your server is ready.",
            "aria connect",
        ] {
            assert!(said.contains(part), "{part}: {said}");
        }
    }

    #[test]
    fn the_screen_is_plain_in_a_pipe_and_coloured_on_a_terminal() {
        let plain = Screen::with(false, false, "xterm-256color", "en_US.UTF-8");
        assert!(!plain.step(1, 4, "Title", "Why.").contains('\x1b'));
        assert!(plain.ok("done").contains('✓') && plain.failed("no").contains('✗'));
        let ascii = Screen::with(false, false, "xterm", "C");
        assert_eq!(ascii.ok("done"), "  ok done");
        assert_eq!(ascii.failed("no"), "  x no");
        let colour = Screen::with(true, false, "xterm", "en_US.UTF-8");
        assert!(colour.step(1, 4, "Title", "Why.").contains("\x1b[1;36m[1/4]\x1b[0m"));
        assert!(colour.ok("done").contains("\x1b[32m"));
        assert!(!Screen::with(true, true, "xterm", "").step(1, 1, "T", "W").contains('\x1b'), "NO_COLOR");
        assert!(!Screen::with(true, false, "dumb", "").step(1, 1, "T", "W").contains('\x1b'), "TERM=dumb");
        assert!(!Screen::with(true, false, "", "").step(1, 1, "T", "W").contains('\x1b'), "no TERM");
        assert!(ascii.note(&Note::text("x")).contains("  ! x"));
    }

    #[test]
    fn the_ascii_screen_uses_ascii_for_the_rule_and_the_bullets_too() {
        let shared = Shared::default();
        let mut ui = PlainUi::new(
            Screen::with(false, false, "xterm", "C"),
            Box::new(shared.clone()),
            Box::new(Shared::default()),
            None,
        );
        ui.title("t");
        ui.plan(&Plan { host: "h".into(), steps: vec!["a".into()], good_to_know: vec!["b".into()], blocked: vec![] });
        let text = shared.text();
        assert!(text.is_ascii(), "{text}");
        assert!(text.contains("    - b"), "{text}");
    }
}
