use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.metal");
    println!("cargo:rerun-if-changed=src/render.metal");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_air = out_dir.join("lib.air");
    let render_air = out_dir.join("render.air");
    let metallib = out_dir.join("shaders.metallib");

    // Compile lib.metal with optimizations
    // FIXME: this is where lib.metal is compiled
    let status = Command::new("xcrun")
        .args(&[
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-O3",
            // "-ffast-math",
            "-fno-fast-math",
            "-flto",
            "src/lib.metal",
            "-o",
        ])
        .arg(&lib_air)
        .status()
        .expect("Failed to run metal compiler for lib.metal");
    assert!(status.success(), "Failed to compile lib.metal");

    // Compile render.metal with optimizations
    let status = Command::new("xcrun")
        .args(&[
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-O3",
            "-ffast-math",
            "-flto",
            "src/render.metal",
            "-o",
        ])
        .arg(&render_air)
        .status()
        .expect("Failed to run metal compiler for render.metal");
    assert!(status.success(), "Failed to compile render.metal");

    // Link into final metallib
    let status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&lib_air)
        .arg(&render_air)
        .arg("-o")
        .arg(&metallib)
        .status()
        .expect("Failed to run metallib linker");
    assert!(status.success(), "Failed to link metallib");
}
