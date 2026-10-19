// The persona and the extensions are embedded with `include_dir!`, which does not tell Cargo to rebuild when they change.
fn main() {
    println!("cargo:rerun-if-changed=../../agent-files");
    println!("cargo:rerun-if-changed=../../extensions");
    println!("cargo:rerun-if-changed=../../pi-version");
}
