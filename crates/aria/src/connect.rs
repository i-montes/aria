//! `aria connect [user@]host`: the terminal on another computer's service. The transport is ssh and nothing else:
//! `ssh <host> ~/.local/bin/aria client-stdio` is the bridge (`stdio_bridge`) on that computer, and its stdin and stdout
//! are the link. Nothing of Aria listens on a port.

use crate::term::link::Link;
use std::ffi::OsString;
use std::io;
use std::process::Command;

/// Where the bridge lives on the other computer (a login shell expands the `~`).
const REMOTE_ARIA: &str = "~/.local/bin/aria";

#[derive(Debug)]
pub enum ConnectError {
    /// A host that starts with `-` would be an option of ssh (`-oProxyCommand=...`).
    OptionLikeHost,
    /// There is no ssh to run.
    NoSsh,
    /// ssh ran, but the hub did not open the link (see `LinkError`), or the link broke.
    Failed(io::Error),
}

/// The program that runs ssh commands: the one `ARIA_SSH` names, or `ssh`.
pub(crate) fn ssh_program() -> OsString {
    ssh_program_from(std::env::var_os("ARIA_SSH"))
}

/// `ssh_program` for a given value of `ARIA_SSH` (an empty one is none).
fn ssh_program_from(env: Option<OsString>) -> OsString {
    env.filter(|program| !program.is_empty()).unwrap_or_else(|| "ssh".into())
}

/// The command that runs `aria client-stdio` on `host` through `ssh`.
fn command_for(ssh: OsString, host: &str) -> Result<Command, ConnectError> {
    if host.starts_with('-') {
        return Err(ConnectError::OptionLikeHost);
    }
    let mut command = Command::new(ssh);
    command
        .args(["-T", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"])
        .arg(host)
        .args([REMOTE_ARIA, "client-stdio"]);
    Ok(command)
}

/// `ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 <host> ~/.local/bin/aria client-stdio`.
pub fn ssh_command(host: &str) -> Result<Command, ConnectError> {
    command_for(ssh_program(), host)
}

/// A link to the hub of `host`: ssh runs, the hellos are exchanged. ssh's own errors go to this terminal's stderr.
pub fn open(host: &str) -> Result<Link, ConnectError> {
    let mut command = ssh_command(host)?;
    Link::spawn(&mut command).map_err(|error| match error.kind() {
        io::ErrorKind::NotFound => ConnectError::NoSsh,
        _ => ConnectError::Failed(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn a_host_that_looks_like_an_option_is_refused() {
        assert!(matches!(ssh_command("-oProxyCommand=x"), Err(ConnectError::OptionLikeHost)));
    }

    #[test]
    fn the_command_is_ssh_with_keepalive_and_the_remote_binary() {
        let command = command_for(OsString::from("ssh"), "ana@hub.example").unwrap();
        assert_eq!(command.get_program(), "ssh");
        let args: Vec<_> = command.get_args().collect();
        assert_eq!(
            args,
            [
                "-T",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "ana@hub.example",
                "~/.local/bin/aria",
                "client-stdio"
            ]
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn ARIA_SSH_names_another_program() {
        assert_eq!(ssh_program_from(None), OsString::from("ssh"));
        assert_eq!(ssh_program_from(Some(OsString::new())), OsString::from("ssh"));
        assert_eq!(ssh_program_from(Some(OsString::from("/opt/bin/fake-ssh"))), OsString::from("/opt/bin/fake-ssh"));
        let command = command_for(ssh_program_from(Some(OsString::from("fake-ssh"))), "hub.example").unwrap();
        assert_eq!(command.get_program(), "fake-ssh");
    }
}
