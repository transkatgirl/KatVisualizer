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

Compiling this program requires the [Rust Programming Language](https://rust-lang.org/tools/install/).

In order to build this program as a VST3 plugin, run the following command:

```bash
cargo xtask bundle katvisualizer --release
```

Alternatively, this program can run in a standalone mode which processes audio from the microphone. In order to build a standalone mode binary, run the following command:

```bash
cargo build --release --features $channel_config,mute-output
```

(`$channel_config` must be set to one of the following: `force-mono, force-mono-to-stereo, force-stereo`. Due to a limitation of nih-plug, channel configurations cannot be changed in standalone mode at runtime.)

Usage information for the standalone binary can be found by running it with the `--help` command (keep in mind that not all available CLI flags are relevant to this program). You will likely want to use the `--input-device`, `--output-device`, and `--period-size` CLI flags.

## Usage

The compiled plugin can be loaded into a DAW like any other metering plugin. It's recommended that you use a buffer size under 10ms long (unless you are using the plugin to generate outputs for an external program) and avoid sample rates below 40kHz.

Once the program is running, the window will display a graphical representation of the input audio, along with additional information in the top corners. The parameters used to render this graphical representation can be adjusted in the dragable settings window.

Keep in mind that this plugin is very CPU intensive, and some systems may struggle to run the default settings. If you experience performance issues, the first thing you should do is open the analysis settings and lower the update rate and resolution.

### Performance Details

The visualizer uses different threads for different tasks. The audio thread\* performs DSP on audio samples provided by the plugin host, and the render thread uses the resulting data to render a visualization. If OSC output is enabled, a separate thread is used to send OSC packets over the network.

Only one thread can access the shared data at a time: When the render thread is generating a spectrogram, the audio thread cannot continue processing, and vice versa. However, the different threads try to do as much of their work as possible before locking the shared data and try to lock it for the minimum amount of time necessary.

\* When processing stereo inputs, some aspects of the audio processing chain use one thread per channel.

#### Interpreting performance counters

If you're having performance or latency issues, enabling performance counters can help you troubleshoot the issue.

- processing = Proportion of the available time budget spent processing audio.
	- Affected by rasterize time
	- Affected by the following settings:
		- Update rate
			- If internal buffering is disabled, the update rate is determined by the buffer size set in the plugin's host
		- Resolution
		- Perform simultaneous masking
		- Use NC method
		- Output analysis via MIDI
		- Output analysis via OSC
- rasterize = Proportion of the processing time taken up by the render thread.
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
	- This is rarely the issue. Generally, the appearance of dropped frames is caused by high buffering time, not a variance in frame times.
	- Affected by buffering time & rasterize time
		- These only increase the frame time when they exceed what can be compensated for by the renderer.
