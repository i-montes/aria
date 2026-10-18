//! OAuth sign-ins. Only Anthropic (Claude Pro/Max) in phase 1.

pub mod anthropic;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OAuthError {
    #[error("the state in the callback does not match")]
    StateMismatch,
    #[error("sign-in was denied: {0}")]
    Denied(String),
    #[error("the callback had no authorization code")]
    NoCode,
    #[error("sign-in timed out")]
    Timeout,
    #[error("unexpected token response: {0}")]
    BadToken(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("could not listen for the browser callback: {0}")]
    Io(String),
}
