# Kat Visualizer

my attempt at making a music visualizer which doesn't suck

a work in progress.

~~i know ~nothing about DSP, why am i even trying to do this~~

## Building

requires the Rust programming language

binaries can be built with the following command:

```bash
cargo xtask bundle katvisualizer --release
```

### Usage

once built, can be used like any other VST3 plugin.

audacity is useful for testing. once you have a track loaded into audacity, click on the "Effects" button on the track and add the visualizer to your project's master effects

## Planned Features

- [ ] Processing Chain
	- [x] ERB-scale VQT
	- [x] ISO 226:2023 equal loudness contour
	- [ ] Spectral reassignment
	- [x] Stereo channel separation
	- [ ] Automatic gain control
	- [ ] Data interpolation?
- [ ] UI
	- [x] Bargraph
	- [x] Spectrogram
	- [x] Performance counters
	- [ ] Settings
		- [ ] Processing chain settings
		- [ ] Visualization settings
	- [ ] Frequency & amplitude display
