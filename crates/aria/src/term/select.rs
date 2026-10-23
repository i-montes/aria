//! Selecting with the mouse: press, drag, release, and what was selected is copied to the clipboard at once (OSC 52,
//! which most terminals understand, and wl-copy or pbcopy beside it). The client reads the mouse, so the terminal's own
//! selection would need Shift; this one does not. A press without a drag is a click.

use ratatui::buffer::Buffer;
use ratatui::style::Modifier;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selection {
    anchor: (u16, u16),
    head: (u16, u16),
}

impl Selection {
    /// Where the press was (column, row).
    pub fn new(x: u16, y: u16) -> Selection {
        Selection { anchor: (x, y), head: (x, y) }
    }

    pub fn extend(&mut self, x: u16, y: u16) {
        self.head = (x, y);
    }

    pub fn is_click(&self) -> bool {
        self.anchor == self.head
    }

    /// (start, end) in reading order, both included.
    fn ordered(&self) -> ((u16, u16), (u16, u16)) {
        let (a, b) = (self.anchor, self.head);
        if (a.1, a.0) <= (b.1, b.0) {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn covers(&self, x: u16, y: u16, width: u16) -> bool {
        let ((sx, sy), (ex, ey)) = self.ordered();
        if y < sy || y > ey {
            return false;
        }
        let from = if y == sy { sx } else { 0 };
        let to = if y == ey { ex } else { width.saturating_sub(1) };
        x >= from && x <= to
    }

    /// The selected text as the screen shows it: line by line, without the spaces at their ends.
    pub fn text(&self, buffer: &Buffer) -> String {
        let area = buffer.area;
        let ((_, sy), (_, ey)) = self.ordered();
        let mut lines = Vec::new();
        for y in sy..=ey.min(area.height.saturating_sub(1)) {
            let row: String =
                (0..area.width).filter(|x| self.covers(*x, y, area.width)).map(|x| buffer[(x, y)].symbol()).collect();
            lines.push(row.trim_end().to_string());
        }
        lines.join("\n")
    }

    /// Shows the selection (reversed colours).
    pub fn paint(&self, buffer: &mut Buffer) {
        let area = buffer.area;
        for y in 0..area.height {
            for x in 0..area.width {
                if self.covers(x, y, area.width) {
                    let cell = &mut buffer[(x, y)];
                    cell.modifier.insert(Modifier::REVERSED);
                }
            }
        }
    }
}

/// The escape sequence that puts `text` on the terminal's clipboard.
pub fn osc52(text: &str) -> String {
    use base64::Engine as _;
    format!("\x1b]52;c;{}\x07", base64::engine::general_purpose::STANDARD.encode(text))
}

/// Puts `text` on the clipboard: through the terminal (OSC 52), and through the system's own tool when there is one
/// (wl-copy on Wayland, pbcopy on macOS), so it works whether the terminal passes OSC 52 on or not.
pub fn copy(text: &str) {
    use std::io::Write;
    let _ = std::io::stdout().write_all(osc52(text).as_bytes());
    let _ = std::io::stdout().flush();
    #[cfg(target_os = "macos")]
    let tool = Some("pbcopy");
    #[cfg(not(target_os = "macos"))]
    let tool = std::env::var_os("WAYLAND_DISPLAY").map(|_| "wl-copy");
    if let Some(tool) = tool {
        let child = std::process::Command::new(tool)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if let Ok(mut child) = child {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Modifier;

    fn screen(rows: &[&str]) -> Buffer {
        let mut buffer = Buffer::empty(Rect::new(0, 0, 20, rows.len() as u16));
        for (y, row) in rows.iter().enumerate() {
            buffer.set_string(0, y as u16, row, ratatui::style::Style::default());
        }
        buffer
    }

    #[test]
    fn a_drag_selects_from_where_it_started_to_where_it_ends() {
        let b = screen(&["  › fix the coupon", "", "  Listo: un cupón", "  y el total"]);
        let mut s = Selection::new(4, 0);
        s.extend(9, 2);
        assert_eq!(s.text(&b), "fix the coupon\n\n  Listo: u", "the cell under the release is included");
        let mut back = Selection::new(9, 2);
        back.extend(4, 0);
        assert_eq!(back.text(&b), s.text(&b), "dragging up selects the same");
    }

    #[test]
    fn a_press_without_a_drag_is_a_click() {
        let mut s = Selection::new(3, 4);
        assert!(s.is_click());
        s.extend(3, 4);
        assert!(s.is_click());
        s.extend(5, 4);
        assert!(!s.is_click());
    }

    #[test]
    fn the_selection_is_painted() {
        let mut b = screen(&["abcdef"]);
        let mut s = Selection::new(1, 0);
        s.extend(3, 0);
        s.paint(&mut b);
        assert!(b[(2, 0)].modifier.contains(Modifier::REVERSED));
        assert!(!b[(4, 0)].modifier.contains(Modifier::REVERSED));
    }

    #[test]
    fn the_clipboard_gets_it_by_osc_52() {
        assert_eq!(osc52("hola"), "\x1b]52;c;aG9sYQ==\x07");
    }
}
