# KatVisualizer

A realtime music visualizer designed to better match human hearing.

The current processing chain consists of the following:
- ERB-scale VQT with ERB based bandwidths
- NC method windowing & spectral reassignment
- Calcuation of simultaneous masking thresholds
- ISO 226:2023 equal loudness contour
- Extraction of panning information

During rendering, color information is processed in the OkLCH color space.

## Installation

This program is a VST3 & CLAP plugin, and can be downloaded from the [releases page](https://github.com/transkatgirl/KatVisualizer/releases). Platform-specific installation instructions can be found in the included README.txt file.

If you don't already have a host for the plugin, [Element](https://github.com/Kushview/Element) seems to be the least-bad open source option.

### Building from Source

Building from source code allows hardware-specific optimizations to be applied and allows you to change the default settings.

Compiling this program requires the [Rust Programming Language](https://rust-lang.org/tools/install/) to be installed.

In order to build this program as a VST3 plugin, run the following command:

```bash
RUSTFLAGS="-C target-cpu=native" cargo xtask bundle katvisualizer --release
```

Alternatively, this program can run in a standalone mode which processes audio from the microphone. In order to build a standalone mode binary, run the following command:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin "katvisualizer" --features $channel_config,mute-output
```

(`$channel_config` must be set to one of the following: `force-mono, force-mono-to-stereo, force-stereo`. Due to a limitation of nih-plug, channel configurations cannot be changed in standalone mode at runtime.)

Usage information for the standalone binary can be found by running it with the `--help` command (keep in mind that not all available CLI flags are relevant to this program). You will likely want to use the `--input-device`, `--output-device`, and `--period-size` CLI flags.

#### Updating defaults

Default values can be found in the following locations:

- [/src/chain/mod.rs](/src/chain/mod.rs), line 47
- [/src/editor.rs](/src/editor.rs), line 678
- [/src/lib.rs](/src/lib.rs), line 397

Most of these values *must* fit within certain constraints for the application to work properly; It is strongly recommended that you only set the defaults to values which are obtainable using the application's settings UI.

#### Building for Web

This program supports building as a WASM-based web application, similar to the standalone mode binary. However, not all of the application's features are supported when building for web, and performance will be significantly worse than a native binary.

Compiling for web requires [trunk](https://github.com/trunk-rs/trunk) and [binaryen](https://github.com/WebAssembly/binaryen).

In order to build this program as a web application, run the following commands:

```bash
trunk build --release
wasm-opt -O4 -all dist/katvisualizer_wasm_bg.wasm -o dist/katvisualizer_wasm_bg.opt.wasm
rm dist/katvisualizer_wasm_bg.wasm
mv dist/katvisualizer_wasm_bg.opt.wasm dist/katvisualizer_wasm_bg.wasm
```

The web application uses an audio file as input, but can be [easily modified](./assets/main.js) to take in any stream of samples.

## Usage

The compiled plugin can be loaded into a DAW like any other metering plugin. It's recommended that you use a buffer size under 10ms long and avoid sample rates below 40kHz.

Once the program is running, the window will display a graphical representation of the input audio, along with additional information in the top corners. The parameters used to render this graphical representation can be adjusted in the dragable settings window.

If you'd like to start improving the visualization's readability further, the settings with the largest impact are (in order of importance):
- Render Options -> Use signal-to-mask ratio when calculating spectrogram shading
	- Enabling this makes timbre more readable at the expense of amplitude differences; Disabling this makes amplitude differences more readable at the expense of timbre
		- This trade-off becomes more apparent as the spectrogram's dynamic range is reduced
- Render Options -> Bargraph averaging (ideal value is 1s / {DISPLAY_REFRESH_RATE})
- Render Options -> Range above masking mean
- Render Options -> Range below masking mean
- Analysis Options -> ERB bandwidth divisor (ideal value depends on what you're trying to analyze; if in doubt, the default value of 1.5 is usually a good middle ground)
- Analysis Options -> Frequency range (ideal value depends on what you're trying to analyze)
- Analysis Options -> Resolution (increasing it may hurt performance)
- Analysis Options -> Update rate (increasing it may hurt performance)

If you experience performance issues out of the box, the settings with the largest impact are:
- Analysis Options -> Approximate spreading function (enabling it will hurt psychoacoustic accuracy)
- Analysis Options -> Update rate (decreasing it may hurt readability)
- Analysis Options -> Resolution (decreasing it may hurt readability)
- Analysis Options -> Color lookup table size multiplier (decreasing it may hurt readability)
- Analysis Options -> Perform simultaneous masking (disabling may hurt readability)
- Render Options -> Spectrogram duration

At the moment, settings are not saved between sessions, but this functionality is planned to be added in a future release.

### Performance Details

The visualizer uses different threads for different tasks. The audio thread\* performs DSP on audio samples provided by the plugin host, and the render thread uses the resulting data to render a visualization.

<!--
Only one thread can access the shared data at a time: When the render thread is generating a spectrogram, the audio thread cannot continue processing, and vice versa. However, the different threads try to do as much of their work as possible before locking the shared data and try to lock it for the minimum amount of time necessary.
-->

\* When processing stereo inputs, some components of the audio processing chain use one thread per channel.

#### Interpreting performance counters

If you're having performance or latency issues, enabling performance counters can help you troubleshoot the issue.

- processing = Proportion of the available time budget spent processing audio.
	- Affected by buffering time
	- Affected by the following settings:
		- Update rate
			- If internal buffering is disabled, the update rate is determined by the buffer size set in the plugin's host
		- Resolution
		- Perform simultaneous masking
		- Use NC method
- rasterize = Proportion of the frame budget taken up by rasterization.
	- Affected by the following settings:
		- Spectrogram duration
		- Bargraph averaging
		- Bargraph height (setting this to 1 disables the spectrogram, setting this to 0 disables the bargraph)
		- Automatic amplitude ranging
		- Amplitude ranging duration
		- Highlight simultaneous masking thresholds
		- Update rate
		- Resolution
	- Affected by processing time
- buffering = Time spent waiting for the render thread to get data from the audio thread.
	- Affected by plugin buffer size (set by the host, or the `--period-size` flag in standalone mode)
	- Affected by processing time
- frame = Time between each frame.
	<!-- - This is rarely the issue. Generally, the appearance of dropped frames is caused by high buffering time, not a variance in frame times. -->
	- Affected by buffering time & rasterize time
		- These only increase the frame time when they exceed what can be compensated for by the renderer.