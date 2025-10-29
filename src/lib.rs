#[cfg(not(debug_assertions))]
use mimalloc::MiMalloc;

use nih_plug::{
    midi::control_change::{ALL_NOTES_OFF, POLY_MODE_ON, RESET_ALL_CONTROLLERS},
    prelude::*,
    util::StftHelper,
    util::{db_to_gain, freq_to_midi_note},
};
use nih_plug_egui::EguiState;
use parking_lot::{FairMutex, Mutex, RwLock};
use std::{
    num::NonZero,
    sync::Arc,
    time::{Duration, Instant},
};
use threadpool::ThreadPool;

#[cfg(not(debug_assertions))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use crate::analyzer::{
    BetterAnalyzer, BetterAnalyzerConfiguration, BetterSpectrogram, map_value_f32,
};

pub mod analyzer;
mod editor;

#[derive(Clone, Copy)]
pub(crate) struct AnalysisMetrics {
    processing: Duration,
    finished: Instant,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PluginStateInfo {
    audio_io_layout: AudioIOLayout,
    buffer_config: BufferConfig,
}

pub struct MyPlugin {
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    latency_samples: u32,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    midi_poly_on: bool,
    midi_notes_on: bool,
    midi_notes: [bool; 128],
    midi_volume_changed: bool,
    midi_needs_reset: bool,
    analysis_midi_output: Vec<AnalysisBufferMidi>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    state_info: Arc<RwLock<Option<PluginStateInfo>>>,
}

#[derive(Params)]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

const MAX_FREQUENCY_BINS: usize = 2048;
const SPECTROGRAM_SLICES: usize = 8192;

impl Default for MyPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(PluginParams::default()),
            analysis_chain: Arc::new(Mutex::new(None)),
            latency_samples: 0,
            analysis_output: Arc::new(FairMutex::new((
                BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
                AnalysisMetrics {
                    processing: Duration::ZERO,
                    finished: Instant::now(),
                },
            ))),
            midi_poly_on: false,
            midi_notes_on: false,
            midi_notes: [false; 128],
            midi_volume_changed: false,
            midi_needs_reset: false,
            analysis_midi_output: Vec::with_capacity(SPECTROGRAM_SLICES * MAX_FREQUENCY_BINS * 2),
            analysis_frequencies: Arc::new(RwLock::new(Vec::with_capacity(MAX_FREQUENCY_BINS))),
            state_info: Arc::new(RwLock::new(None)),
        }
    }
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(1200, 900),
        }
    }
}

impl Plugin for MyPlugin {
    const NAME: &'static str = "KatVisualizer";
    const VENDOR: &'static str = "transkatgirl";
    const URL: &'static str = "https://github.com/transkatgirl/katvisualizer";
    const EMAIL: &'static str = "08detour_dial@icloud.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    #[cfg(not(any(
        feature = "force-mono",
        feature = "force-mono-to-stereo",
        feature = "force-stereo"
    )))]
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(0),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(0),
            ..AudioIOLayout::const_default()
        },
    ];

    #[cfg(feature = "force-mono")]
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        ..AudioIOLayout::const_default()
    }];

    #[cfg(feature = "force-stereo")]
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    #[cfg(feature = "force-mono-to-stereo")]
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    const HARD_REALTIME_ONLY: bool = true;
    const MIDI_INPUT: MidiConfig = MidiConfig::None;

    #[cfg(feature = "midi")]
    const MIDI_OUTPUT: MidiConfig = MidiConfig::MidiCCs;

    #[cfg(not(feature = "midi"))]
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.analysis_chain.clone(),
            self.analysis_output.clone(),
            self.analysis_frequencies.clone(),
            self.state_info.clone(),
            async_executor,
        )
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let mut analysis_chain = self.analysis_chain.lock();

        let analysis_config = match &*analysis_chain {
            Some(old_chain) => old_chain.config(),
            None => AnalysisChainConfig::default(),
        };

        let new_chain = AnalysisChain::new(
            &analysis_config,
            buffer_config.sample_rate,
            audio_io_layout,
            self.analysis_frequencies.clone(),
        );
        context.set_latency_samples(new_chain.latency_samples);
        self.latency_samples = new_chain.latency_samples;

        *analysis_chain = Some(new_chain);

        *self.analysis_output.lock() = (
            BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
            AnalysisMetrics {
                processing: Duration::ZERO,
                finished: Instant::now(),
            },
        );

        *self.state_info.write() = Some(PluginStateInfo {
            audio_io_layout: *audio_io_layout,
            buffer_config: *buffer_config,
        });

        #[cfg(feature = "midi")]
        {
            self.midi_needs_reset = true;
        }

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if let Some(mut lock) = self.analysis_chain.try_lock() {
            let analysis_chain = lock.as_mut().unwrap();

            if analysis_chain.latency_samples != self.latency_samples {
                context.set_latency_samples(analysis_chain.latency_samples);
                self.latency_samples = analysis_chain.latency_samples;
            }

            analysis_chain.analyze(
                buffer,
                &self.analysis_output,
                &mut self.analysis_midi_output,
            );
        }

        #[cfg(feature = "midi")]
        if self.midi_needs_reset {
            context.send_event(NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc: RESET_ALL_CONTROLLERS,
                value: 0.0,
            });
            self.midi_poly_on = false;
            self.midi_volume_changed = false;
            self.midi_needs_reset = false;
        }

        #[cfg(feature = "midi")]
        if !self.midi_poly_on {
            context.send_event(NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc: POLY_MODE_ON,
                value: 0.0,
            });
            self.midi_notes_on = false;
            self.midi_notes = [false; 128];
            self.midi_poly_on = true;
        }

        #[cfg(feature = "midi")]
        for buffer in &self.analysis_midi_output {
            self.midi_notes_on = true;

            if let Some(pressures) = buffer.pressures {
                if self.midi_volume_changed {
                    for note in 0..127 {
                        context.send_event(NoteEvent::PolyVolume {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            gain: 1.0,
                        });
                    }
                    self.midi_volume_changed = false;
                }

                for (note, (pan, volume)) in buffer.notes.iter().enumerate() {
                    if *volume > buffer.min_value {
                        if !self.midi_notes[note] {
                            context.send_event(NoteEvent::NoteOn {
                                timing: buffer.timing,
                                voice_id: None,
                                channel: 0,
                                note: note as u8,
                                velocity: pressures[note],
                            });
                            self.midi_notes[note] = true;
                        } else if buffer.use_aftertouch {
                            context.send_event(NoteEvent::PolyPressure {
                                timing: buffer.timing,
                                voice_id: None,
                                channel: 0,
                                note: note as u8,
                                pressure: pressures[note],
                            });
                        } else {
                            context.send_event(NoteEvent::NoteOff {
                                timing: buffer.timing,
                                voice_id: None,
                                channel: 0,
                                note: note as u8,
                                velocity: pressures[note],
                            });
                            context.send_event(NoteEvent::NoteOn {
                                timing: buffer.timing,
                                voice_id: None,
                                channel: 0,
                                note: note as u8,
                                velocity: pressures[note],
                            });
                        }

                        context.send_event(NoteEvent::PolyPan {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            pan: *pan,
                        });
                    } else if self.midi_notes[note] {
                        context.send_event(NoteEvent::NoteOff {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            velocity: 0.0,
                        });
                        self.midi_notes[note] = false;
                    }
                }
            } else {
                self.midi_volume_changed = true;

                for (note, (pan, volume)) in buffer.notes.iter().enumerate() {
                    if *volume > buffer.min_value {
                        if !self.midi_notes[note] {
                            context.send_event(NoteEvent::NoteOn {
                                timing: buffer.timing,
                                voice_id: None,
                                channel: 0,
                                note: note as u8,
                                velocity: 1.0,
                            });
                            self.midi_notes[note] = true;
                        }
                        context.send_event(NoteEvent::PolyVolume {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            gain: db_to_gain(*volume) + 1.0,
                        });

                        context.send_event(NoteEvent::PolyPan {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            pan: *pan,
                        });
                    } else if self.midi_notes[note] {
                        context.send_event(NoteEvent::NoteOff {
                            timing: buffer.timing,
                            voice_id: None,
                            channel: 0,
                            note: note as u8,
                            velocity: 0.0,
                        });
                        self.midi_notes[note] = false;
                    }
                }
            }
        }

        #[cfg(feature = "midi")]
        if self.midi_notes_on && self.analysis_midi_output.is_empty() {
            context.send_event(NoteEvent::MidiCC {
                timing: 0,
                channel: 0,
                cc: ALL_NOTES_OFF,
                value: 0.0,
            });
            self.midi_notes_on = false;
            self.midi_notes = [false; 128];
        }

        self.analysis_midi_output.clear();

        #[cfg(feature = "mute-output")]
        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                *sample = 0.0;
            }
        }

        ProcessStatus::Normal
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AnalysisChainConfig {
    gain: f64,
    listening_volume: f64,
    normalize_amplitude: bool,
    internal_buffering: bool,
    update_rate_hz: f64,
    latency_offset: Duration,

    output_midi: bool,
    midi_max_simultaneous: u8,
    midi_amplitude_threshold: f32,
    midi_use_aftertouch: bool,
    midi_use_volume: bool,
    midi_pressure_min_amplitude: f32,
    midi_pressure_max_amplitude: f32,

    resolution: usize,
    start_frequency: f64,
    end_frequency: f64,
    erb_frequency_scale: bool,
    erb_time_resolution: bool,
    erb_bandwidth_divisor: f64,
    time_resolution_clamp: (f64, f64),
    q_time_resolution: f64,
    nc_method: bool,
}

impl Default for AnalysisChainConfig {
    fn default() -> Self {
        Self {
            gain: 0.0,
            listening_volume: 86.0,
            normalize_amplitude: true,
            internal_buffering: true,
            update_rate_hz: 1024.0,
            resolution: 512,
            latency_offset: Duration::ZERO,

            output_midi: false,
            midi_max_simultaneous: 32,
            midi_amplitude_threshold: 30.0 - 86.0,
            midi_use_aftertouch: true,
            midi_use_volume: false,
            midi_pressure_min_amplitude: 30.0 - 86.0,
            midi_pressure_max_amplitude: 70.0 - 86.0,

            start_frequency: BetterAnalyzerConfiguration::default().start_frequency,
            end_frequency: BetterAnalyzerConfiguration::default().end_frequency,
            erb_frequency_scale: BetterAnalyzerConfiguration::default().erb_frequency_scale,
            erb_time_resolution: BetterAnalyzerConfiguration::default().erb_time_resolution,
            erb_bandwidth_divisor: BetterAnalyzerConfiguration::default().erb_bandwidth_divisor,
            time_resolution_clamp: BetterAnalyzerConfiguration::default().time_resolution_clamp,
            q_time_resolution: BetterAnalyzerConfiguration::default().q_time_resolution,
            nc_method: BetterAnalyzerConfiguration::default().nc_method,
        }
    }
}

pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    left_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    right_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    gain: f64,
    internal_buffering: bool,
    output_midi: bool,
    midi_max_simultaneous: u8,
    midi_amplitude_threshold: f32,
    midi_use_aftertouch: bool,
    midi_use_volume: bool,
    midi_pressure_min_amplitude: f32,
    midi_pressure_max_amplitude: f32,
    update_rate: f64,
    listening_volume: Option<f64>,
    pub(crate) latency_samples: u32,
    additional_latency: Duration,
    sample_rate: f32,
    chunk_size: usize,
    chunk_duration: Duration,
    single_input: bool,
    pool: ThreadPool,
    pub(crate) frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
}

// See: https://nih-plug.robbertvanderhelm.nl/nih_plug/util/fn.midi_note_to_freq.html

struct AnalysisBufferMidi {
    timing: u32,
    min_value: f32,
    use_aftertouch: bool,
    notes: [(f32, f32); 128],
    pressures: Option<[f32; 128]>,
}

impl AnalysisChain {
    fn new(
        config: &AnalysisChainConfig,
        sample_rate: f32,
        layout: &AudioIOLayout,
        frequency_list_container: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    ) -> Self {
        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution: config.resolution,
            start_frequency: config.start_frequency,
            end_frequency: config.end_frequency,
            erb_frequency_scale: config.erb_frequency_scale,
            sample_rate,
            erb_time_resolution: config.erb_time_resolution,
            erb_bandwidth_divisor: config.erb_bandwidth_divisor,
            time_resolution_clamp: config.time_resolution_clamp,
            q_time_resolution: config.q_time_resolution,
            nc_method: config.nc_method,
        });

        let mut chunker = StftHelper::new(2, sample_rate.ceil() as usize, 0);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        chunker.set_block_size(chunk_size);

        {
            let mut frequencies = frequency_list_container.write();
            frequencies.clear();
            frequencies.extend(
                analyzer
                    .frequencies()
                    .iter()
                    .map(|(a, b, c)| (*a as f32, *b as f32, *c as f32)),
            );
        }

        Self {
            sample_rate,
            internal_buffering: config.internal_buffering,
            output_midi: config.output_midi,
            midi_max_simultaneous: config.midi_max_simultaneous,
            midi_amplitude_threshold: config.midi_amplitude_threshold,
            midi_use_aftertouch: config.midi_use_aftertouch,
            midi_use_volume: config.midi_use_volume,
            midi_pressure_min_amplitude: config.midi_pressure_min_amplitude,
            midi_pressure_max_amplitude: config.midi_pressure_max_amplitude,
            latency_samples: if config.internal_buffering {
                chunker.latency_samples()
            } else {
                0
            } + (config.latency_offset.as_secs_f64() * sample_rate as f64) as u32,
            additional_latency: config.latency_offset,
            chunker,
            frequencies: frequency_list_container,
            left_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], analyzer.clone()))),
            right_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], analyzer))),
            gain: config.gain,
            update_rate: config.update_rate_hz,
            listening_volume: if config.normalize_amplitude {
                Some(config.listening_volume)
            } else {
                None
            },
            chunk_size,
            chunk_duration: Duration::from_secs_f64(chunk_size as f64 / sample_rate as f64),
            single_input: layout.main_input_channels == NonZero::new(1),
            pool: ThreadPool::new(2),
        }
    }
    fn analyze(
        &mut self,
        buffer: &mut Buffer,
        output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
        midi_output: &mut Vec<AnalysisBufferMidi>,
    ) {
        let mut finished = Instant::now();

        let mut midi_timing = 0;

        if self.internal_buffering {
            self.chunker
                .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                    if channel_idx == 1 && self.single_input {
                        return;
                    }

                    if self.single_input {
                        let mut lock = self.left_analyzer.lock();
                        let (ref _buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64));
                    } else {
                        let analyzer = if channel_idx == 0 {
                            self.left_analyzer.clone()
                        } else {
                            self.right_analyzer.clone()
                        };

                        analyzer.lock().0.copy_from_slice(buffer);

                        self.pool.execute(move || {
                            let mut lock = analyzer.lock();
                            let (ref mut buffer, ref mut analyzer) = *lock;

                            analyzer.analyze(buffer.iter().map(|s| *s as f64));
                        });
                    }

                    if channel_idx == 1 || (channel_idx == 0 && self.single_input) {
                        let (ref mut spectrogram, ref mut metrics) = *output.lock();

                        self.pool.join();
                        let left_lock = self.left_analyzer.lock();
                        let right_lock = self.right_analyzer.lock();
                        let left_analyzer = &left_lock.1;
                        let right_analyzer = &right_lock.1;

                        spectrogram.update_fn(|analysis_output| {
                            if self.single_input {
                                analysis_output.update_mono(
                                    left_analyzer,
                                    self.gain,
                                    self.listening_volume,
                                    self.chunk_duration,
                                );
                            } else {
                                analysis_output.update_stereo(
                                    left_analyzer,
                                    right_analyzer,
                                    self.gain,
                                    self.listening_volume,
                                    self.chunk_duration,
                                );
                            }
                        });

                        let now = Instant::now();
                        metrics.processing = now.duration_since(finished);
                        metrics.finished = now;

                        finished = now;
                    }
                });
        } else {
            if self.single_input {
                let mut lock = self.left_analyzer.lock();
                let (ref _buffer, ref mut analyzer) = *lock;

                analyzer.analyze(buffer.as_slice()[0].iter().map(|s| *s as f64));
            } else {
                for (channel_idx, buffer) in buffer.as_slice().iter().enumerate() {
                    let analyzer = if channel_idx == 0 {
                        self.left_analyzer.clone()
                    } else {
                        self.right_analyzer.clone()
                    };

                    {
                        let mut analyzer = analyzer.lock();
                        if buffer.len() == analyzer.0.len() {
                            analyzer.0.copy_from_slice(buffer);
                        } else {
                            analyzer.0.clear();
                            analyzer.0.extend_from_slice(buffer);
                        }
                    }

                    self.pool.execute(move || {
                        let mut lock = analyzer.lock();
                        let (ref mut buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64));
                    });
                }
            }

            let chunk_duration =
                Duration::from_secs_f64(buffer.samples() as f64 / self.sample_rate as f64);

            let (ref mut spectrogram, ref mut metrics) = *output.lock();

            self.pool.join();
            let left_lock = self.left_analyzer.lock();
            let right_lock = self.right_analyzer.lock();
            let left_analyzer = &left_lock.1;
            let right_analyzer = &right_lock.1;

            spectrogram.update_fn(|analysis_output| {
                if self.single_input {
                    analysis_output.update_mono(
                        left_analyzer,
                        self.gain,
                        self.listening_volume,
                        chunk_duration,
                    );
                } else {
                    analysis_output.update_stereo(
                        left_analyzer,
                        right_analyzer,
                        self.gain,
                        self.listening_volume,
                        chunk_duration,
                    );
                }
            });

            #[cfg(feature = "midi")]
            if self.output_midi {
                let frequencies = self.frequencies.read();
                let mut note_scratchpad: [(f32, f32, f32); 128] = [(0.0, 0.0, 0.0); 128];
                for ((lower, _, upper), (pan, volume)) in spectrogram.data[0]
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (frequencies[i], d))
                {
                    let lower_note = freq_to_midi_note(lower).round().max(15.0) as usize;
                    let upper_note = freq_to_midi_note(upper).round().max(15.0) as usize;

                    if lower_note > 127 {
                        break;
                    }

                    #[allow(clippy::needless_range_loop)]
                    for note in lower_note..=upper_note {
                        if note > 127 {
                            break;
                        }

                        if !volume.is_finite() {
                            break;
                        }

                        note_scratchpad[note].0 += 1.0;
                        note_scratchpad[note].1 += pan;
                        note_scratchpad[note].2 += volume;
                    }
                }

                let mut analysis_midi = if !self.midi_use_volume {
                    let mut notes: [(f32, f32); 128] = [(0.0, 0.0); 128];
                    let mut pressures: [f32; 128] = [0.0; 128];

                    for (i, (items, pan_sum, volume_sum)) in note_scratchpad.into_iter().enumerate()
                    {
                        if items == 0.0 {
                            notes[i] = (0.0, f32::NEG_INFINITY);
                            pressures[i] = 0.0;
                        } else {
                            let volume = volume_sum / items;

                            notes[i] = (pan_sum / items, volume);
                            pressures[i] = map_value_f32(
                                volume,
                                self.midi_pressure_min_amplitude,
                                self.midi_pressure_max_amplitude,
                                0.0,
                                1.0,
                            )
                            .clamp(0.0, 1.0);
                        }
                    }

                    AnalysisBufferMidi {
                        timing: midi_timing,
                        min_value: self.midi_amplitude_threshold,
                        use_aftertouch: self.midi_use_aftertouch,
                        notes,
                        pressures: Some(pressures),
                    }
                } else {
                    let mut notes: [(f32, f32); 128] = [(0.0, 0.0); 128];

                    for (i, (items, pan_sum, volume_sum)) in note_scratchpad.into_iter().enumerate()
                    {
                        if items == 0.0 {
                            notes[i] = (0.0, f32::NEG_INFINITY);
                        } else {
                            notes[i] = (pan_sum / items, volume_sum / items);
                        }
                    }

                    AnalysisBufferMidi {
                        timing: midi_timing,
                        min_value: self.midi_amplitude_threshold,
                        use_aftertouch: self.midi_use_aftertouch,
                        notes,
                        pressures: None,
                    }
                };

                let mut sorted_notes: [(f32, usize); 128] = [(0.0, 0); 128];

                for (note, (_, volume)) in analysis_midi.notes.iter().enumerate() {
                    sorted_notes[note] = (*volume, note);
                }
                sorted_notes.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

                sorted_notes
                    .into_iter()
                    .rev()
                    .skip(self.midi_max_simultaneous as usize)
                    .for_each(|(_, note)| {
                        analysis_midi.notes[note].1 = f32::NEG_INFINITY;
                    });

                // TODO: Allow performing ISO 226 weighting when limiting simultaneous notes if amplitude normalization is off

                // TODO: Use models of auditory masking to determine which notes to remove?

                midi_output.push(analysis_midi);

                #[allow(unused_assignments)]
                {
                    midi_timing += buffer.samples() as u32;
                }
            }

            let now = Instant::now();
            metrics.processing = now.duration_since(finished);
            metrics.finished = now;
        }
    }
    pub(crate) fn config(&self) -> AnalysisChainConfig {
        let analyzer = self.left_analyzer.lock();
        let analyzer_config = analyzer.1.config();

        AnalysisChainConfig {
            gain: self.gain,
            listening_volume: self
                .listening_volume
                .unwrap_or(AnalysisChainConfig::default().listening_volume),
            normalize_amplitude: self.listening_volume.is_some(),
            internal_buffering: self.internal_buffering,
            output_midi: self.output_midi,
            midi_max_simultaneous: self.midi_max_simultaneous,
            midi_amplitude_threshold: self.midi_amplitude_threshold,
            midi_use_aftertouch: self.midi_use_aftertouch,
            midi_use_volume: self.midi_use_volume,
            midi_pressure_min_amplitude: self.midi_pressure_min_amplitude,
            midi_pressure_max_amplitude: self.midi_pressure_max_amplitude,
            update_rate_hz: self.update_rate,
            latency_offset: self.additional_latency,
            resolution: analyzer_config.resolution,
            start_frequency: analyzer_config.start_frequency,
            end_frequency: analyzer_config.end_frequency,
            erb_frequency_scale: analyzer_config.erb_frequency_scale,
            erb_time_resolution: analyzer_config.erb_time_resolution,
            erb_bandwidth_divisor: analyzer_config.erb_bandwidth_divisor,
            time_resolution_clamp: analyzer_config.time_resolution_clamp,
            q_time_resolution: analyzer_config.q_time_resolution,
            nc_method: analyzer_config.nc_method,
        }
    }
    pub(crate) fn update_config(&mut self, config: &AnalysisChainConfig) {
        self.gain = config.gain;
        self.listening_volume = if config.normalize_amplitude {
            Some(config.listening_volume)
        } else {
            None
        };

        let old_left_analyzer = self.left_analyzer.lock();
        let old_analyzer_config = old_left_analyzer.1.config();

        self.output_midi = config.output_midi;
        self.midi_max_simultaneous = config.midi_max_simultaneous;
        self.midi_amplitude_threshold = config.midi_amplitude_threshold;
        self.midi_use_aftertouch = config.midi_use_aftertouch;
        self.midi_use_volume = config.midi_use_volume;
        self.midi_pressure_min_amplitude = config.midi_pressure_min_amplitude;
        self.midi_pressure_max_amplitude = config.midi_pressure_max_amplitude;
        if self.internal_buffering {
            self.output_midi = false;
        }

        if self.update_rate != config.update_rate_hz {
            self.chunk_size = (self.sample_rate as f64 / config.update_rate_hz).round() as usize;
            self.chunker.set_block_size(self.chunk_size);
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
            self.chunk_duration =
                Duration::from_secs_f64(self.chunk_size as f64 / self.sample_rate as f64);
        } else if self.additional_latency != config.latency_offset {
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
        }

        if old_analyzer_config.resolution != config.resolution
            || old_analyzer_config.start_frequency != config.start_frequency
            || old_analyzer_config.end_frequency != config.end_frequency
            || old_analyzer_config.erb_frequency_scale != config.erb_frequency_scale
            || old_analyzer_config.erb_time_resolution != config.erb_time_resolution
            || old_analyzer_config.time_resolution_clamp != config.time_resolution_clamp
            || old_analyzer_config.erb_bandwidth_divisor != config.erb_bandwidth_divisor
            || old_analyzer_config.q_time_resolution != config.q_time_resolution
            || old_analyzer_config.nc_method != config.nc_method
        {
            let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
                resolution: config.resolution,
                start_frequency: config.start_frequency,
                end_frequency: config.end_frequency,
                erb_frequency_scale: config.erb_frequency_scale,
                sample_rate: self.sample_rate,
                erb_time_resolution: config.erb_time_resolution,
                erb_bandwidth_divisor: config.erb_bandwidth_divisor,
                time_resolution_clamp: config.time_resolution_clamp,
                q_time_resolution: config.q_time_resolution,
                nc_method: config.nc_method,
            });
            drop(old_left_analyzer);

            let mut frequencies = self.frequencies.write();
            frequencies.clear();
            frequencies.extend(
                analyzer
                    .frequencies()
                    .iter()
                    .map(|(a, b, c)| (*a as f32, *b as f32, *c as f32)),
            );

            self.left_analyzer =
                Arc::new(Mutex::new((vec![0.0; self.chunk_size], analyzer.clone())));
            self.right_analyzer = Arc::new(Mutex::new((vec![0.0; self.chunk_size], analyzer)));
        } else if self.update_rate != config.update_rate_hz
            || self.internal_buffering != config.internal_buffering
        {
            drop(old_left_analyzer);

            self.left_analyzer.lock().0 = vec![0.0; self.chunk_size];
            self.right_analyzer.lock().0 = vec![0.0; self.chunk_size];
        }

        self.internal_buffering = config.internal_buffering;
        self.update_rate = config.update_rate_hz;
    }
}

impl ClapPlugin for MyPlugin {
    const CLAP_ID: &'static str = "com.transkatgirl.katvisualizer";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Analyzer,
        ClapFeature::Mono,
        ClapFeature::Stereo,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for MyPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"transkatgirlVizu";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Analyzer];
}

nih_export_clap!(MyPlugin);
nih_export_vst3!(MyPlugin);
