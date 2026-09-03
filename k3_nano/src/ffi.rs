// src/ffi.rs — C FFI bindings to k3_nano_harness.cu

use std::os::raw::c_char;

pub enum K3EngineOpaque {}

extern "C" {
    /// Initialize the CUDA inference engine with a model binary.
    /// Returns an opaque pointer to the K3Model.
    pub fn k3_engine_init(path: *const c_char) -> *mut K3EngineOpaque;

    /// Run a single forward pass.
    /// token: vocabulary index
    /// pos: sequence position
    /// out_logits: caller-allocated buffer of size vocab_size
    /// Returns 0 on success.
    pub fn k3_engine_forward(
        engine: *mut K3EngineOpaque,
        token: i32,
        pos: i32,
        out_logits: *mut f32,
    ) -> i32;

    /// Free the engine and all GPU memory.
    pub fn k3_engine_free(engine: *mut K3EngineOpaque);
}

/// Safe wrapper around the raw FFI handle.
pub struct K3Engine {
    ptr: *mut K3EngineOpaque,
    vocab_size: usize,
}

impl K3Engine {
    /// Initialize a new engine from a model binary path.
    pub fn new(model_path: &str, vocab_size: usize) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(model_path)
            .map_err(|e| format!("Invalid path: {}", e))?;
        let ptr = unsafe { k3_engine_init(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err("Failed to initialize K3 engine".to_string());
        }
        Ok(Self { ptr, vocab_size })
    }

    /// Run a forward pass and return logits.
    pub fn forward(&self, token: i32, pos: i32) -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0f32; self.vocab_size];
        let ret = unsafe {
            k3_engine_forward(self.ptr, token, pos, logits.as_mut_ptr())
        };
        if ret != 0 {
            return Err(format!("Forward pass failed with code {}", ret));
        }
        Ok(logits)
    }

    /// Sample next token from logits (greedy).
    pub fn sample_greedy(&self, logits: &[f32]) -> i32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }
}

impl Drop for K3Engine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { k3_engine_free(self.ptr) };
        }
    }
}

// Safety: The CUDA engine is single-threaded (pinned to one OS thread).
// The daemon wrapper ensures no concurrent access.
unsafe impl Send for K3Engine {}
