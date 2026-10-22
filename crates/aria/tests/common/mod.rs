//! What the live tests share.
use aria_core::keyring;
use aria_core::paths::AriaPaths;
use std::path::PathBuf;

/// Puts the owner's real `auth.json` into the scratch home `scratch`: the file of `real` when it is there, else the
/// copy the keyring keeps (`file:auth.json`) once the keys moved into it.
pub fn borrow_auth(real: &AriaPaths, scratch: &AriaPaths) {
    if real.auth().is_file() {
        std::fs::copy(real.auth(), scratch.auth()).expect("copy auth.json");
        return;
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let auth = keyring::system(&keyring::service_for(&real.root, &home))
        .and_then(|keyring| keyring.get("file:auth.json"))
        .expect("the system keyring")
        .expect("run the Aria setup first (auth.json in the keyring)");
    std::fs::write(scratch.auth(), auth).expect("write auth.json");
    let private = <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o600);
    std::fs::set_permissions(scratch.auth(), private).expect("auth.json readable only by its owner");
}
