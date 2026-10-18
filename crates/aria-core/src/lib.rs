//! Platform-independent core of Aria: paths, settings, credentials, providers.

pub mod json_file;
pub mod mcp;
pub mod paths;
pub mod auth;
pub mod providers;
pub mod settings;
pub mod validate;
pub mod oauth;
pub mod bridge;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
