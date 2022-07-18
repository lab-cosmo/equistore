#![allow(dead_code)]
#![allow(clippy::needless_return)]

use std::path::Path;
use std::process::Command;

pub fn cmake_config(source_dir: &Path, build_dir: &Path, build_type: &str) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_config = Command::new(cmake);
    cmake_config.current_dir(&build_dir);
    cmake_config.arg(&source_dir);

    // the cargo executable currently running
    let cargo_exe = std::env::var("CARGO").expect("CARGO env var is not set");
    cmake_config.arg(format!("-DCARGO_EXE={}", cargo_exe));


    cmake_config.arg(format!("-DCMAKE_BUILD_TYPE={}", build_type));

    let mut shared_lib = "ON";
    if let Ok(value) = std::env::var("RASCALINE_TEST_WITH_STATIC_LIB") {
        if value != "0" {
            shared_lib = "OFF";
        }
    }
    cmake_config.arg(format!("-DBUILD_SHARED_LIBS={}", shared_lib));

    return cmake_config;
}

pub fn cmake_build(build_dir: &Path, build_type: &str) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_build = Command::new(cmake);
    cmake_build.current_dir(&build_dir);
    cmake_build.arg("--build");
    cmake_build.arg(".");
    cmake_build.arg("--config");
    cmake_build.arg(build_type);

    return cmake_build;
}
