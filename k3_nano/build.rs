// build.rs — Sovereign MiMo-4B Rust FFI Bridge
// Compiles k3_nano_harness.cu via nvcc and links CUDA runtime

fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cuda(true)
        .flag("-arch=sm_86")
        .flag("-O3")
        .flag("-use_fast_math")
        .file("k3_nano_harness.cu")
        .compile("k3_engine");
}
