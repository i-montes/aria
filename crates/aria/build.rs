// The page (`ui/`) is embedded with `include_dir!`, which does not tell Cargo to rebuild when it changes.
//
// macOS: Aria is a bare binary, not an app bundle, so the reasons macOS shows when it asks for a permission (the
// microphone, for dictation) come from an `Info.plist` linked into the binary's `__TEXT,__info_plist` section.
fn main() {
    println!("cargo:rerun-if-changed=../../ui");
    println!("cargo:rerun-if-changed=Info.plist");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let plist = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("Info.plist");
        println!("cargo:rustc-link-arg-bins=-Wl,-sectcreate,__TEXT,__info_plist,{}", plist.display());
    }
}
