// src/daemon.rs — Tokio Channel Actor for CUDA Inference
//
// Pins the CUDA context to a dedicated OS thread, exposes async Rust interface.
// Prevents thread-migration errors that cause CUDA context loss.

use tokio::sync::{mpsc, oneshot};
use std::ffi::CString;

pub struct InferenceRequest {
    pub token: i32,
    pub pos: i32,
    pub respond_to: oneshot::Sender<Vec<f32>>,
}

#[derive(Clone)]
pub struct K3DaemonHandle {
    sender: mpsc::Sender<InferenceRequest>,
    vocab_size: usize,
}

impl K3DaemonHandle {
    /// Spawn the CUDA inference daemon on a dedicated OS thread.
    ///
    /// All GPU work runs on the spawned thread. The async handle sends
    /// requests over an mpsc channel without blocking the Tokio reactor.
    pub fn spawn(model_path: &str, vocab_size: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<InferenceRequest>(64);
        let path_c = CString::new(model_path).expect("Invalid CString path");
        let vs = vocab_size;

        std::thread::spawn(move || {
            let engine = unsafe {
                crate::ffi::k3_engine_init(path_c.as_ptr())
            };
            if engine.is_null() {
                eprintln!("[DAEMON] Failed to init CUDA engine");
                return;
            }
            let mut logits_buf = vec![0.0f32; vs];

            while let Some(req) = rx.blocking_recv() {
                unsafe {
                    crate::ffi::k3_engine_forward(
                        engine,
                        req.token,
                        req.pos,
                        logits_buf.as_mut_ptr(),
                    );
                }
                let _ = req.respond_to.send(logits_buf.clone());
            }

            unsafe { crate::ffi::k3_engine_free(engine) };
            eprintln!("[DAEMON] CUDA engine freed");
        });

        Self { sender: tx, vocab_size: vs }
    }

    /// Send a forward pass request and await the logits.
    pub async fn predict(&self, token: i32, pos: i32) -> Result<Vec<f32>, String> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest { token, pos, respond_to: tx })
            .await
            .map_err(|e| format!("Channel send failed: {}", e))?;
        rx.await.map_err(|e| format!("Channel recv failed: {}", e))
    }

    /// Get vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
