//! The assistant's markdown as terminal lines: paragraphs wrapped at the width, headings and emphasis as bold and
//! italic, lists with `•` or their numbers, code blocks indented in grey and never wrapped, links as their text.

use crate::term::theme::Palette;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

struct Writer<'p> {
    width: usize,
    palette: &'p Palette,
    out: Vec<Line<'static>>,
    /// The block being written: (text, style) pieces.
    pieces: Vec<(String, Style)>,
    bold: u32,
    italic: u32,
    /// Open lists: the next number of an ordered one, None for bullets.
    lists: Vec<Option<u64>>,
    /// The marker of the item being written (`• `, `1. `), put on its first line.
    marker: Option<String>,
    in_code: bool,
}

impl Writer<'_> {
    fn style(&self) -> Style {
        let mut style = Style::default().fg(self.palette.text);
        if self.bold > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.italic > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        style
    }

    fn gap(&mut self) {
        if !self.out.is_empty() && self.out.last().is_some_and(|l| !l.spans.is_empty()) {
            self.out.push(Line::default());
        }
    }

    /// Wraps the block's pieces into lines: the item's marker on the first, its width as indent on the rest.
    fn flush(&mut self) {
        if self.pieces.iter().all(|(t, _)| t.trim().is_empty()) {
            self.pieces.clear();
            return;
        }
        let depth = self.lists.len().saturating_sub(1) * 2;
        let marker = self.marker.take().unwrap_or_default();
        let first = format!("{}{marker}", " ".repeat(depth));
        let rest = " ".repeat(first.width());
        let width = self.width.max(first.width() + 1);
        // Words with their styles, and whether each one is glued to the one before (`**two**.` has no space).
        let mut words: Vec<(String, Style, bool)> = Vec::new();
        let mut glued = false;
        for (text, style) in std::mem::take(&mut self.pieces) {
            for (i, word) in text.split(' ').enumerate() {
                if i > 0 {
                    glued = false;
                }
                if !word.is_empty() {
                    words.push((word.to_string(), style, glued && !words.is_empty()));
                    glued = true;
                }
            }
        }
        let mut line: Vec<Span<'static>> = vec![Span::raw(first.clone())];
        let mut used = first.width();
        let mut empty = true;
        for (word, style, joined) in words {
            let need = word.width() + usize::from(!empty && !joined);
            if !empty && !joined && used + need > width {
                self.out.push(Line::from(std::mem::take(&mut line)));
                line.push(Span::raw(rest.clone()));
                used = rest.width();
                empty = true;
            }
            if !empty && !joined {
                line.push(Span::raw(" "));
                used += 1;
            }
            used += word.width();
            line.push(Span::styled(word, style));
            empty = false;
        }
        let spans: Vec<Span<'static>> = line.into_iter().filter(|s| !s.content.is_empty()).collect();
        self.out.push(Line::from(spans));
    }
}

pub fn lines(text: &str, width: u16, palette: &Palette) -> Vec<Line<'static>> {
    let mut w = Writer {
        width: usize::from(width.max(1)),
        palette,
        out: Vec::new(),
        pieces: Vec::new(),
        bold: 0,
        italic: 0,
        lists: Vec::new(),
        marker: None,
        in_code: false,
    };
    for event in Parser::new_ext(text, Options::ENABLE_STRIKETHROUGH) {
        match event {
            Event::Start(Tag::Paragraph) if w.lists.is_empty() => w.gap(),
            Event::Start(Tag::Heading { .. }) => {
                w.gap();
                w.bold += 1;
            }
            Event::End(TagEnd::Heading(_)) => {
                w.flush();
                w.bold -= 1;
            }
            Event::End(TagEnd::Paragraph) => w.flush(),
            Event::Start(Tag::List(start)) => {
                if w.lists.is_empty() {
                    w.gap();
                } else {
                    w.flush();
                }
                w.lists.push(start);
            }
            Event::End(TagEnd::List(_)) => {
                w.lists.pop();
            }
            Event::Start(Tag::Item) => {
                w.flush();
                let marker = match w.lists.last_mut() {
                    Some(Some(n)) => {
                        *n += 1;
                        format!("{}. ", *n - 1)
                    }
                    _ => "• ".to_string(),
                };
                w.marker = Some(marker);
            }
            Event::End(TagEnd::Item) => w.flush(),
            Event::Start(Tag::Emphasis) => w.italic += 1,
            Event::End(TagEnd::Emphasis) => w.italic -= 1,
            Event::Start(Tag::Strong) => w.bold += 1,
            Event::End(TagEnd::Strong) => w.bold -= 1,
            Event::Start(Tag::CodeBlock(_)) => {
                w.flush();
                w.gap();
                w.in_code = true;
            }
            Event::End(TagEnd::CodeBlock) => w.in_code = false,
            Event::Text(text) if w.in_code => {
                for line in text.trim_end_matches('\n').split('\n') {
                    w.out.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(line.to_string(), Style::default().fg(palette.grey)),
                    ]));
                }
            }
            Event::Text(text) => {
                let style = w.style();
                w.pieces.push((text.to_string(), style));
            }
            Event::Code(code) => {
                let style = w.style();
                w.pieces.push((code.to_string(), style));
            }
            Event::SoftBreak => w.pieces.push((" ".into(), Style::default())),
            Event::HardBreak => w.flush(),
            Event::Rule => {
                w.gap();
                w.out.push(Line::styled("─".repeat(usize::from(width.min(40))), Style::default().fg(palette.rule)));
            }
            _ => {}
        }
    }
    w.flush();
    w.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::theme::Palette;

    fn plain(lines: &[ratatui::text::Line]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect()).collect()
    }

    #[test]
    fn markdown_renders_lists_and_code() {
        let text =
            "Done: **two** things.\n\n- first\n- second\n\n1. one\n\n```rust\nlet x = 1;\n```\n\nUse `cargo test`.";
        let got = plain(&lines(text, 40, &Palette::truecolor()));
        assert_eq!(
            got,
            ["Done: two things.", "", "• first", "• second", "", "1. one", "", "  let x = 1;", "", "Use cargo test."]
        );
    }

    #[test]
    fn long_paragraphs_wrap_at_the_width() {
        let got = plain(&lines("one two three four five six", 10, &Palette::truecolor()));
        assert_eq!(got, ["one two", "three four", "five six"]);
    }
}
