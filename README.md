# pyannote-rs — pyannote.audio as a dependency-isolated RemoteMedia SDK plugin

Hand-rolled `AsyncStreamingNode` plugin (Path 3 in the
[RemoteMedia SDK plugin guide](https://github.com/matbeedotcom/remotemedia-sdk/blob/main/docs/CUSTOM_NODE_REGISTRATION.md))
that registers two node types — `PyannoteEmbedding` and
`PyannoteSegmenter` — by wrapping
[upstream `pyannote-rs` 0.3.x](https://crates.io/crates/pyannote-rs).

## Why hand-rolled (Path 3)?

Dependency isolation. The dual-emit path (Path 2) would force the
plugin's `ort` version to match the host's workspace pin:

| Layer | `ort` version | Reason |
|---|---|---|
| Host (`remotemedia-core` etc.) | `=2.0.0-rc.12` | Latest, needed for other nodes |
| `pyannote-rs 0.3.x`            | `^2.0.0-rc.10` | Hard-pinned; rc.11 jumped ndarray 0.16→0.17, rc.12 added !Sync operators |

Path 2 would resolve these to a single version → broken build. Path 3
sits in its own workspace, depends ONLY on `remotemedia-plugin-sdk`
(NOT `remotemedia-core`), and pins `ort=rc.10` itself. The host links
`remotemedia-core` (and therefore `ort=rc.12`) and never sees the
plugin's `ort=rc.10` — they cross the dlopen boundary as opaque
`abi_stable` types.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["pyannote-rs@v0.1.0"],
  "nodes": [
    {
      "id": "embed",
      "node_type": "PyannoteEmbedding",
      "params": { "embedding_model_path": "./wespeaker_en_voxceleb_CAM++.onnx" }
    },
    {
      "id": "seg",
      "node_type": "PyannoteSegmenter",
      "params": { "segmentation_model_path": "./segmentation-3.0.onnx" }
    }
  ]
}
```

The SDK resolver expands `pyannote-rs@v0.1.0` to
`github.com/RemoteMedia-SDK/pyannote-rs`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

> **Status:** plugin.toml + source published. **Prebuilt release
> binaries are not yet uploaded** — the matrix-build CI workflow is
> pending. Until then, consumers should either build the cdylib
> themselves (see below) or use a local-path plugin entry.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/pyannote-rs
cd pyannote-rs
cargo build --release
# → target/release/libpyannote_conflict_plugin.so
```

Then reference it from your manifest:

```json
{ "plugins": ["./path/to/libpyannote_conflict_plugin.so"] }
```

## What it exports

| Node type            | Input                       | Output                                |
|----------------------|-----------------------------|---------------------------------------|
| `PyannoteEmbedding`  | Audio (16 kHz mono, ~10 s)  | JSON `{ embedding: [f32], dim: int }` |
| `PyannoteSegmenter`  | Audio (16 kHz mono)         | JSON `{ segments: [{start, end}] }`   |

Both nodes require an ONNX model file passed via params (`embedding_model_path` /
`segmentation_model_path`). See the
[pyannote-rs README](https://github.com/Sleepyhead01/pyannote-rs#models)
for the model URLs.

## License

See `LICENSE.md`. This plugin reuses RemoteMedia SDK source and is
governed by the same RemoteMedia SDK Community License 1.0. The
upstream `pyannote-rs` crate is MIT.
