//! Speaker-embedding loadable plugin built against **upstream**
//! `pyannote-rs` (crates.io 0.3.x), without forking the crate to
//! match the host workspace's `ort = "=2.0.0-rc.12"` pin.
//!
//! The plugin pins `ort = "=2.0.0-rc.10"` in its own Cargo.toml — the
//! only release where upstream pyannote compiles. This works because
//! the plugin's Cargo workspace is independent of the main workspace
//! and the plugin does **not** link `remotemedia-core` (which would
//! unify the tree onto rc.12).
//!
//! ## What changed in Phase A8
//!
//! Originally this plugin shipped a hand-rolled `impl FfiNode` with a
//! private `WireIn`/`WireOut` mirror so it could deserialize
//! `RuntimeData` without linking `remotemedia-core`. That mirror is
//! gone — the plugin now depends on `remotemedia-plugin-sdk`, which
//! re-exports the skinny `remotemedia-types::RuntimeData` and the
//! `StreamingNodeFfiAdapter` that handles `rmp-serde` encode/decode
//! at the FFI boundary. The plugin only writes business logic against
//! `AsyncStreamingNode` and registers via `plugin_export!()`.
//!
//! ## Node types exported
//!
//!   PyannoteEmbedding   — Audio (16 kHz mono) → Json{embedding,dim,…}
//!   PyannoteSegmenter   — Audio (16 kHz mono) → Json{segments:[…]}
//!
//! ## Params
//!
//!   { "embedding_model_path":   "/path/to/wespeaker_en_voxceleb_CAM++.onnx" }
//!   { "segmentation_model_path": "/path/to/segmentation-3.0.onnx" }

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{RErr, ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

fn audio_parts(data: RuntimeData) -> Result<(Vec<f32>, u32, u32), Error> {
    match data {
        RuntimeData::Audio {
            samples,
            sample_rate,
            channels,
            ..
        } => Ok((samples.into_vec(), sample_rate, channels)),
        other => Err(Error::InvalidData(format!(
            "expected RuntimeData::Audio, got {}",
            other.data_type()
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyannoteEmbedding
// ---------------------------------------------------------------------------

pub struct PyannoteEmbedding {
    // Arc<Mutex<…>> so the future returned from `process()` doesn't
    // borrow `&self` — the FfiFuture the adapter wraps requires
    // `'static`, and pyannote's extractor isn't `Sync`.
    extractor: Arc<Mutex<pyannote_rs::EmbeddingExtractor>>,
}

impl PyannoteEmbedding {
    pub fn new(model_path: &str) -> Result<Self, Error> {
        let extractor = pyannote_rs::EmbeddingExtractor::new(model_path).map_err(|e| {
            Error::Execution(format!("EmbeddingExtractor::new({model_path}): {e}"))
        })?;
        Ok(Self {
            extractor: Arc::new(Mutex::new(extractor)),
        })
    }
}

#[async_trait]
impl AsyncStreamingNode for PyannoteEmbedding {
    fn node_type(&self) -> &str {
        "PyannoteEmbedding"
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let (samples, sample_rate, channels) = audio_parts(data)?;
        if channels != 1 {
            return Err(Error::InvalidData(format!(
                "PyannoteEmbedding expects mono audio, got channels={channels}"
            )));
        }
        if sample_rate != 16_000 {
            return Err(Error::InvalidData(format!(
                "PyannoteEmbedding expects 16 kHz audio, got {sample_rate}"
            )));
        }

        let i16: Vec<i16> = f32_to_i16(&samples);
        let embedding: Vec<f32> = {
            let mut ex = match self.extractor.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            ex.compute(&i16)
                .map(|it| it.collect::<Vec<f32>>())
                .map_err(|e| Error::Execution(format!("EmbeddingExtractor::compute: {e}")))?
        };

        let dim = embedding.len();
        Ok(RuntimeData::Json(serde_json::json!({
            "embedding": embedding,
            "dim": dim,
            "input_samples": i16.len(),
        })))
    }
}

#[derive(Default)]
pub struct PyannoteEmbeddingFactory;

impl FfiNodeFactory for PyannoteEmbeddingFactory {
    fn node_type(&self) -> RString {
        RString::from("PyannoteEmbedding")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let params: serde_json::Value =
            serde_json::from_str(params.as_str()).unwrap_or(serde_json::json!({}));
        let model_path = match params.get("embedding_model_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                return RErr(RString::from(
                    "PyannoteEmbedding params missing 'embedding_model_path'",
                ))
            }
        };

        match PyannoteEmbedding::new(&model_path) {
            Ok(node) => ROk(FfiNode_TO::from_value(
                StreamingNodeFfiAdapter::new(node),
                TD_Opaque,
            )),
            Err(e) => RErr(RString::from(format!("{e}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// PyannoteSegmenter
// ---------------------------------------------------------------------------

pub struct PyannoteSegmenter {
    segmentation_model_path: String,
}

impl PyannoteSegmenter {
    pub fn new(model_path: String) -> Self {
        Self {
            segmentation_model_path: model_path,
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for PyannoteSegmenter {
    fn node_type(&self) -> &str {
        "PyannoteSegmenter"
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let (samples, sample_rate, channels) = audio_parts(data)?;
        if channels != 1 {
            return Err(Error::InvalidData(format!(
                "PyannoteSegmenter expects mono audio, got channels={channels}"
            )));
        }

        let i16: Vec<i16> = f32_to_i16(&samples);
        let segments_iter =
            pyannote_rs::get_segments(&i16, sample_rate, &self.segmentation_model_path)
                .map_err(|e| Error::Execution(format!("get_segments: {e}")))?;

        let mut out_segs = Vec::new();
        for seg in segments_iter {
            match seg {
                Ok(s) => out_segs.push(serde_json::json!({
                    "start": s.start,
                    "end": s.end,
                    "samples": s.samples.len(),
                })),
                Err(e) => return Err(Error::Execution(format!("segment: {e}"))),
            }
        }

        Ok(RuntimeData::Json(
            serde_json::json!({ "segments": out_segs }),
        ))
    }
}

#[derive(Default)]
pub struct PyannoteSegmenterFactory;

impl FfiNodeFactory for PyannoteSegmenterFactory {
    fn node_type(&self) -> RString {
        RString::from("PyannoteSegmenter")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let params: serde_json::Value =
            serde_json::from_str(params.as_str()).unwrap_or(serde_json::json!({}));
        let path = match params
            .get("segmentation_model_path")
            .and_then(|v| v.as_str())
        {
            Some(p) => p.to_string(),
            None => {
                return RErr(RString::from(
                    "PyannoteSegmenter params missing 'segmentation_model_path'",
                ))
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(PyannoteSegmenter::new(path)),
            TD_Opaque,
        ))
    }
}

// ---------------------------------------------------------------------------
// plugin registration
// ---------------------------------------------------------------------------

remotemedia_plugin_sdk::plugin_export!(PyannoteEmbeddingFactory, PyannoteSegmenterFactory);
