# music visualizer plans

## pipeline
- load audio samples into RAM before rendering
	- process future CQTs ahead of time on separate threads
		- handle frame rendering on separate thread from CQT processing
	- handle audio playback on a separate thread
	- handle UX on a separate thread
- allow visualizing arbitrary audio devices

## processing
- switch window function based on chosen dynamic range
- window function raw dynamic range
	- 1, 0.49 = 65dB
	- 1, 0.493 = 70dB
	- 1, 0.495 = 75dB
	- 1, 0.497 = 80dB
	- 1, 0.4982 = 85dB
	- 1, 0.499 = 95dB
	- 1, 0.5 = 100dB
	- subtract ~15dB to account for weighting function
	- see also: https://web.archive.org/web/20210122032400/http://hyperphysics.phy-astr.gsu.edu/hbase/sound/mask.html#c3, https://pmc.ncbi.nlm.nih.gov/articles/PMC4753356/#sec6-2331216516630549
- use ISO 226:2023 equal loudness curve
- use VQT transform
	- take into account https://en.wikipedia.org/wiki/Just-noticeable_difference and https://en.wikipedia.org/wiki/Auditory_masking when turning parameters
		- minimum temporal resolution: somewhere between 50-200 ms (see: https://wiki.hydrogenaudio.org/index.php?title=Masking#Temporal_masking, https://ccrma.stanford.edu/~jos/bosse/Human_Audio_Perception_Masking.html)
- use loudness normalization
- overlay channel CQTs, use color coding (similar to ffmpeg showcqt)

### look into
- ac-3 window (https://www.atsc.org/wp-content/uploads/2015/03/A52-201212-17.pdf, https://ieeexplore.ieee.org/document/842996)
- non-linear color mapping, similar to ffmpeg showcqt color mapping and https://github.com/hqrrr/PerceptoMap
- time-frequency reassignment, like https://github.com/hqrrr/PerceptoMap

## rendering
- take advantage of WCG & HDR
- take advantage of high refresh rate and high resolution
- use window zero-padding to interpolate spectral data, like https://www.dsprelated.com/freebooks/sasp/Spectral_Interpolation.html
	- unsure how viable this is from a compute standpoint; this needs to run in real time

## synchronization
- use cpal::OutputStreamTimestamp.playback to get audio playback offset
- use pixels::Pixels.enable_vsync.set_present_mode(wgpu_types::PresentMode::Mailbox) to reduce frame offset to ~1
- use pixels::Pixels.queue().on_submitted_work_done() to keep track of frame offset

## UX
- allow tuning all key parameters (within the limits of human hearing):
	- erb scale (default), bark scale, mel scale, or logarithmic (octave) scale
	- time-domain / frequency-domain resolution tradeoff (default of 75ms - 200ms)
	- dynamic range / filter side lobe tradeoff (default of 60dB dynamic range)
	- minimum & maximum frequencies (default of 20Hz - 20kHz)
	- scroll speed
	- weighting curve listening volume (default of 85dB)
	- color mapping
	- latency offset
- allow changing settings in real time
- allow file loading
- show frequency and amplitude on hover
	- show musical notes from frequency
- allow showing frequency / musical note overlay
- (after implementing core features) add options for basic feature extraction:
	- chroma (see https://librosa.org/doc/latest/generated/librosa.feature.chroma_cens.html)
	- spectral centroid
	- bpm
	- cepstrum
	- autocorrelation tempogram
	- spectral flatness / contrast
	- waveform view

## may be useful as reference

- https://github.com/BrunoWallner/audiovis/tree/main
- https://github.com/waltonseymour/visualizer
- https://github.com/willianjusten/awesome-audio-visualization?tab=readme-ov-file
- https://github.com/mfcc64/showcqt-js?tab=readme-ov-file
- https://codepen.io/TF3RDL/pen/poQJwRW
- https://codepen.io/TF3RDL/pen/MWLzPoO
	- transform = variable q
	- window = hann or hamming
	- time res = min 75ms, max 200ms
	- target smoothing time constant = -450 db / second
		- 75 = -300 db / second @ 120 fps
		- 65 = -450 db / second @ 120 fps
		- 56 = -600 db / second @ 120 fps, -300 db / second @ 60 fps
		- 42 = -450 db / second @ 60 fps
		- 32 = -600 db / second @ 60 fps
	- bands per octave = 30
	- amplitude = 0dB - 60dB, 1, 0.495 custom window function
	- frequency scale = ERB or log (octaves)
