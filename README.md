# KatVisualizer

An attempt at making music visualizer which better matches what you hear.

The current processing chain consists of the following:
- ERB-scale VQT utilizing ERB bandwidths
- NC method windowing & spectral reassignment
- ISO 226:2023 equal loudness contour
- Extraction of panning information

## Building

requires the Rust programming language

VST3 and CLAP plugins can be built with the following command:

```bash
cargo xtask bundle katvisualizer --release
```

alternatively, standalone binaries can be built using the following command:

```bash
cargo build --release --features $channel_config
```

where `$channel_config` is one of the following: `force-mono, force-mono-to-stereo, force-stereo`. due to a limitation of nih-plug, channel configurations cannot be changed in standalone mode at runtime.

## Usage

plugins: can be used like any other DAW metering plugin.

standalone binary: run it with the `--input-device` and `--output-device` options to select an input and output device (input is disabled by default). run it with `--help` for additional information.

## TODOs

- [ ] Finish adding usage information
- [ ] Configuration persistence
- [ ] Adjustable color coding (by channel, frequency, or amplitude)
- [ ] Horizontal mode