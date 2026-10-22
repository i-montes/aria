//! Where Aria keeps its data: `~/.aria/` (or `$ARIA_HOME`). The repo never holds user data.

use std::io;
use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AriaPaths {
    pub root: PathBuf,
    /// The keyring and the live copies of the sealed credential files, once the dock has opened them
    /// (`crate::vault`); None everywhere else, where the credentials are the plain files in `root`.
    pub vault: Option<crate::vault::Vault>,
}

impl AriaPaths {
    pub fn new(root: PathBuf) -> AriaPaths {
        AriaPaths { root, vault: None }
    }

    /// `$ARIA_HOME` when set, otherwise `$HOME/.aria`.
    pub fn from_env() -> AriaPaths {
        let root = match std::env::var_os("ARIA_HOME") {
            Some(dir) => PathBuf::from(dir),
            None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".aria"),
        };
        AriaPaths::new(root)
    }

    pub fn settings(&self) -> PathBuf {
        self.root.join("settings.json")
    }

    /// Pi's `auth.json`: the live copy while the dock holds the keyring, else the plain file.
    pub fn auth(&self) -> PathBuf {
        self.credential("auth.json")
    }

    fn credential(&self, name: &str) -> PathBuf {
        match &self.vault {
            Some(vault) => vault.live.join(name),
            None => self.root.join(name),
        }
    }

    /// MCP servers, in Pi's `mcp.json` format (it holds Tavily's key, so mode 600).
    pub fn mcp(&self) -> PathBuf {
        self.root.join("mcp.json")
    }

    /// MCP OAuth state (registered client, tokens), in Pi's `mcp-auth.json` format.
    pub fn mcp_auth(&self) -> PathBuf {
        self.credential("mcp-auth.json")
    }

    /// The open sessions (the dock's tabs): ids, titles and which one is active.
    pub fn sessions(&self) -> PathBuf {
        self.root.join("sessions.json")
    }

    /// The scheduled tasks (`crate::schedules`).
    pub fn schedules(&self) -> PathBuf {
        self.root.join("schedules.json")
    }

    /// The background jobs the dock ran: what each one was and how it ended, kept across restarts.
    pub fn jobs(&self) -> PathBuf {
        self.root.join("jobs.json")
    }

    /// The files attached to chat messages, `<session>/<yyyy-mm-dd>/<n>-<safe name>` inside (folders 0700, files 0600).
    /// Pi reads them there; nothing in the box may write there.
    pub fn attachments(&self) -> PathBuf {
        self.root.join("attachments")
    }

    /// Pi's agent directory (`PI_CODING_AGENT_DIR`): its own settings, links to our credentials, the
    /// persona, extensions and sessions. Aria owns and regenerates it.
    pub fn agent(&self) -> PathBuf {
        self.root.join("agent")
    }

    /// First on the `PATH` of every Pi process: the `pi` shim and the `node` link.
    pub fn bin(&self) -> PathBuf {
        self.root.join("bin")
    }

    /// Where pinned Pi releases are installed, one directory per version.
    pub fn pi_root(&self) -> PathBuf {
        self.root.join("pi")
    }

    pub fn pi_install(&self, version: &str) -> PathBuf {
        self.pi_root().join(version)
    }

    /// Creates the root directory (mode 0700) when it is missing.
    pub fn ensure_root(&self) -> io::Result<()> {
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Environment variables are process-wide: serialize the tests that set them.
    static ENV: Mutex<()> = Mutex::new(());

    #[test]
    fn paths_use_aria_home() {
        let _guard = ENV.lock().unwrap();
        std::env::set_var("ARIA_HOME", "/tmp/x");
        let paths = AriaPaths::from_env();
        assert_eq!(paths.settings(), PathBuf::from("/tmp/x/settings.json"));
        assert_eq!(paths.auth(), PathBuf::from("/tmp/x/auth.json"));
        std::env::remove_var("ARIA_HOME");
    }

    #[test]
    fn agent_and_pi_paths() {
        let paths = AriaPaths::new(PathBuf::from("/r"));
        assert_eq!(paths.agent(), PathBuf::from("/r/agent"));
        assert_eq!(paths.jobs(), PathBuf::from("/r/jobs.json"));
        assert_eq!(paths.attachments(), PathBuf::from("/r/attachments"));
        assert_eq!(paths.bin(), PathBuf::from("/r/bin"));
        assert_eq!(paths.pi_root(), PathBuf::from("/r/pi"));
        assert_eq!(paths.pi_install("1.3.0"), PathBuf::from("/r/pi/1.3.0"));
    }

    #[test]
    fn ensure_root_creates_private_dir() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().join(".aria"));
        paths.ensure_root().unwrap();
        let mode = std::fs::metadata(&paths.root).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }
}
