//! Platform-independent core of Aria: paths, settings, credentials, providers.

pub mod agent;
pub mod agent_config;
pub mod auth;
pub mod bridge;
pub mod files;
pub mod json_file;
pub mod mcp;
pub mod mcp_import;
pub mod oauth;
pub mod paths;
pub mod persona;
pub mod pi;
pub mod providers;
pub mod quota;
pub mod sessions;
pub mod settings;
pub mod shortcut;
pub mod validate;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
