use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=AOCL_STATIC_RUNTIME");
    println!("cargo:rerun-if-changed=csrc/aocl_checked_shim.c");
    println!("cargo:rerun-if-changed=csrc/aocl_checked_shim.h");
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
    {
        return;
    }
    let root = PathBuf::from(env::var_os("AOCL_ROOT").expect(
        "AOCL_ROOT must name an installed AOCL-DLP 5.3.2 prefix containing include/ and lib/ (or lib64/)",
    ));
    let include = root.join("include");
    let header = include.join("aocl_dlp.h");
    let config = include.join("aocl_dlp_config.h");
    assert!(header.is_file(), "missing {}", header.display());
    assert!(config.is_file(), "missing {}", config.display());
    let config_text = fs::read_to_string(&config).expect("read installed AOCL config header");
    assert!(
        config_text.contains("#define AOCL_DLP_VERSION_STRING    \"5.3.2\""),
        "AOCL_ROOT must contain the pinned AOCL-DLP 5.3.2 headers"
    );
    assert!(
        config_text
            .lines()
            .any(|line| line.trim() == "#define DLP_ENABLE_OPENMP"),
        "AOCL-DLP must be built with DLP_THREADING_MODEL=openmp"
    );
    let lib = [root.join("lib"), root.join("lib64")]
        .into_iter()
        .find(|dir| dir.join("libaocl-dlp.a").is_file())
        .expect("AOCL_ROOT must contain lib/libaocl-dlp.a or lib64/libaocl-dlp.a");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", config.display());
    println!(
        "cargo:rerun-if-changed={}",
        lib.join("libaocl-dlp.a").display()
    );

    cc::Build::new()
        .file("csrc/aocl_checked_shim.c")
        .include("csrc")
        .flag("-isystem")
        .flag(include.to_str().expect("AOCL_ROOT must be UTF-8"))
        .flag_if_supported("-std=c11")
        .flag_if_supported("-fPIC")
        .warnings(true)
        .compile("runtime_aocl_checked_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    // Preserve JIT registration static constructors in upstream archive objects.
    println!("cargo:rustc-link-lib=static:+whole-archive=aocl-dlp");
    let runtime_kind = if env::var("AOCL_STATIC_RUNTIME").as_deref() == Ok("1") {
        for name in ["libgomp.a", "libstdc++.a", "libatomic.a"] {
            assert!(
                lib.join(name).is_file(),
                "AOCL_STATIC_RUNTIME requires {name} in AOCL_ROOT/lib"
            );
        }
        "static"
    } else {
        "dylib"
    };
    println!("cargo:rustc-link-lib={runtime_kind}=gomp");
    println!("cargo:rustc-link-lib={runtime_kind}=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib={runtime_kind}=atomic");
    println!("cargo:rustc-link-lib=dylib=pthread");
}
