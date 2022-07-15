use std::path::PathBuf;
use std::process::Command;

mod utils;

#[test]
fn check_python() {
    let tox = which::which("tox").expect("could not find tox");

    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut tox = Command::new(tox);
    tox.arg("--");
    if cfg!(debug_assertions) {
        // assume that debug assertion means that we are building the code in
        // debug mode, even if that could be not true in some cases
        tox.env("EQUISTORE_BUILD_TYPE", "debug");
    } else {
        tox.env("EQUISTORE_BUILD_TYPE", "release");
    }
    tox.current_dir(&root);
    let status = tox.status().expect("failed to run tox");
    assert!(status.success());
}
