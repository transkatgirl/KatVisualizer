# KatVisualizer

my attempt at making a music visualizer which doesn't suck

a work in progress.

~~i know ~nothing about DSP, why am i even trying to do this~~

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

## Planned Features

- [ ] Processing Chain
	- [x] ERB-scale VQT
	- [x] ISO 226:2023 equal loudness contour
	- [ ] Spectral reassignment
	- [x] Stereo channel separation
	- [ ] Automatic gain control?
	- [ ] Data interpolation
- [ ] UI
	- [x] Bargraph
	- [x] Spectrogram
	- [x] Performance counters
	- [x] Settings
	- [ ] Frequency & amplitude display

Other important improvements: Support saving and loading configuration