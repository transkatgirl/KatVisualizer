# KatVisualizer

A realtime music visualizer designed to better match human hearing.

The current processing chain consists of the following:
- ERB-scale VQT with ERB bandwidths
- NC method windowing & spectral reassignment
- ISO 226:2023 equal loudness contour
- Extraction of panning information

During rendering, color information is processed in the OkLCH color space.

## Building

Compiling this program requires the [Rust Programming Language](https://rust-lang.org/tools/install/).

In order to build this program as a VST3 or CLAP plugin, run the following command:

```bash
cargo xtask bundle katvisualizer --release
```

Alternatively, this program can run in a standalone mode which processes audio from the microphone. In order to build a standalone mode binary, run the following command:

```bash
cargo build --release --features $channel_config
```

(`$channel_config` must be set to one of the following: `force-mono, force-mono-to-stereo, force-stereo`. Due to a limitation of nih-plug, channel configurations cannot be changed in standalone mode at runtime.)

## Usage

The compiled VST3 or CLAP plugin can be in a DAW like any other metering plugin.

Usage information for the standalone binary can be found by running it with the `--help` command (keep in mind that not all available CLI flags are relevant to this program). You will likely want to use the `--input-device` and `--output-device` CLI flags.

## TODOs

- [ ] Finish adding usage information
- [ ] Configuration persistence
- [ ] Adjustable color coding (by channel, frequency, or amplitude)
- [ ] Horizontal mode