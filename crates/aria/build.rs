// The page (`ui/`) is embedded with `include_dir!`, which does not tell Cargo to rebuild when it changes.
fn main() {
    println!("cargo:rerun-if-changed=../../ui");
}
