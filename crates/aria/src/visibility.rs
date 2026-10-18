//! When the dock is shown, closing, or hidden, as a pure state machine. `dock.rs` executes the
//! returned command; keeping the decisions here is what makes "press SUPER + A twice fast" testable.
//!
//! The surface stays mapped while the page animates closed, and is unmapped only when the close
//! transition is over (`CloseFinished`). Opening again mid-close goes straight back to `Opening`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Hidden,
    Opening,
    Open,
    Closing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// `aria toggle` (SUPER + A).
    Toggle,
    Escape,
    /// The page reports that its open transition finished.
    Opened,
    /// The close transition's time is up.
    CloseFinished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Map the surface on the focused monitor, take the keyboard, tell the page to open.
    Show,
    /// Release the keyboard and tell the page to close.
    StartClose,
    /// Unmap the surface.
    Unmap,
    None,
}

pub fn step(state: State, event: Event) -> (State, Command) {
    use Command::*;
    use Event::*;
    use State::*;
    match (state, event) {
        (Hidden, Toggle) | (Closing, Toggle) => (Opening, Show),
        (Opening, Toggle) | (Open, Toggle) | (Opening, Escape) | (Open, Escape) => (Closing, StartClose),
        (Opening, Opened) => (Open, Command::None),
        (Closing, CloseFinished) => (Hidden, Unmap),
        (state, _) => (state, Command::None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_from_hidden_shows() {
        assert_eq!(step(State::Hidden, Event::Toggle), (State::Opening, Command::Show));
    }

    #[test]
    fn toggle_while_opening_closes() {
        assert_eq!(step(State::Opening, Event::Toggle), (State::Closing, Command::StartClose));
    }

    #[test]
    fn toggle_while_open_closes() {
        assert_eq!(step(State::Open, Event::Toggle), (State::Closing, Command::StartClose));
    }

    #[test]
    fn toggle_while_closing_reopens() {
        assert_eq!(step(State::Closing, Event::Toggle), (State::Opening, Command::Show));
    }

    #[test]
    fn escape_closes_when_shown_and_is_a_noop_otherwise() {
        assert_eq!(step(State::Open, Event::Escape), (State::Closing, Command::StartClose));
        assert_eq!(step(State::Opening, Event::Escape), (State::Closing, Command::StartClose));
        assert_eq!(step(State::Hidden, Event::Escape), (State::Hidden, Command::None));
        assert_eq!(step(State::Closing, Event::Escape), (State::Closing, Command::None));
    }

    #[test]
    fn opened_settles_an_opening_dock() {
        assert_eq!(step(State::Opening, Event::Opened), (State::Open, Command::None));
        assert_eq!(step(State::Closing, Event::Opened), (State::Closing, Command::None));
        assert_eq!(step(State::Hidden, Event::Opened), (State::Hidden, Command::None));
    }

    #[test]
    fn close_finished_unmaps_only_a_closing_dock() {
        assert_eq!(step(State::Closing, Event::CloseFinished), (State::Hidden, Command::Unmap));
    }

    #[test]
    fn stale_close_finished_is_ignored() {
        assert_eq!(step(State::Opening, Event::CloseFinished), (State::Opening, Command::None));
        assert_eq!(step(State::Open, Event::CloseFinished), (State::Open, Command::None));
        assert_eq!(step(State::Hidden, Event::CloseFinished), (State::Hidden, Command::None));
    }
}
