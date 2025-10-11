# KatVisualizer

A realtime music visualizer designed to better match human hearing.

The current processing chain consists of the following:
- ERB-scale VQT with ERB based bandwidths
- NC method windowing & spectral reassignment
- ISO 226:2023 equal loudness contour
- Extraction of panning information

During rendering, color information is processed in the OkLCH color space.

## Building

Compiling this program requires the [Rust Programming Language](https://rust-lang.org/tools/install/).

In order to build this program as a VST3 plugin, run the following command:

```bash
cargo xtask bundle katvisualizer --release
```

Alternatively, this program can run in a standalone mode which processes audio from the microphone. In order to build a standalone mode binary, run the following command:

```bash
cargo build --release --features $channel_config
```

(`$channel_config` must be set to one of the following: `force-mono, force-mono-to-stereo, force-stereo`. Due to a limitation of nih-plug, channel configurations cannot be changed in standalone mode at runtime.)

## Usage

The compiled plugin can be loaded into a DAW like any other metering plugin. It's recommended that you use a buffer size under 10ms long and avoid sample rates below 40kHz.

Usage information for the standalone binary can be found by running it with the `--help` command (keep in mind that not all available CLI flags are relevant to this program). You will likely want to use the `--input-device`, `--output-device`, and `--period-size` CLI flags.

<details>
<summary>MacOS specific</summary>

If you'd like to run the program on the system audio, standalone mode likely won't work. Instead, load the plugin into [Element](https://github.com/Kushview/Element) and use the [BlackHole](https://github.com/ExistentialAudio/BlackHole) loopback device (along with creating a multi-output device in the built-in "Audio MIDI Setup" app) to pass audio to it.

</details>

Once the program is running, the window will display a graphical representation of the input audio, along with additional information in the top corners. The parameters used to render this graphical representation can be adjusted in the dragable settings window.

### Performance

The visualizer uses different threads for different tasks. The audio thread performs DSP on audio samples provided by the plugin host, and the render thread uses the resulting data to render a visualization.

Only one thread can access the shared data at a time: When the render thread is generating a spectrogram, the audio thread cannot continue processing, and vice versa. However, the different threads try to do as much of their work as possible before locking the shared data and try to lock it for the minimum amount of time necessary.

#### Interpreting performance counters

If you're having performance or latency issues, enabling performance counters can help you troubleshoot the issue.

- processing = Proportion of the available time budget spent processing audio.
	- Affected by rasterize time
	- Affected by the following settings:
		- Update rate
		- Resolution
		- Use NC method
- rasterize = Proportion of the processing time taken up by the render thread.
	- Affected by the following settings:
		- Spectrogram duration
		- Bargraph averaging
		- Bargraph height (setting this to 1 disables the spectrogram, setting this to 0 disables the bargraph)
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

## TODOs

- [ ] Horizontal mode
- [ ] Adjustable color coding (by channel, frequency, or amplitude)
- [ ] Add documentation for render settings
- [ ] Configuration persistence
