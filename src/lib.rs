#[cfg(not(debug_assertions))]
use mimalloc::MiMalloc;

use nih_plug::{
    midi::control_change::{ALL_NOTES_OFF, POLY_MODE_ON},
    prelude::*,
    util::StftHelper,
    util::freq_to_midi_note,
};
use nih_plug_egui::EguiState;
use parking_lot::{FairMutex, Mutex, RwLock};
use rosc::{OscArray, OscBundle, OscMessage, OscPacket, OscTime, OscType, encoder};
use std::{
    net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, UdpSocket},
    num::NonZero,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
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
    midi_on: bool,
    midi_notes: [bool; 128],
    midi_output: [f32; 128],
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
const MAX_PEAK_OUTPUTS: usize = 512;

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
            midi_on: false,
            midi_notes: [false; 128],
            midi_output: [0.0; 128],
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

            analysis_chain.analyze(buffer, &self.analysis_output, &mut self.midi_output);

            drop(lock);

            #[cfg(feature = "midi")]
            {
                let mut midi_on = false;

                for (note, pressure) in self.midi_output.iter().enumerate() {
                    if *pressure > 0.0 {
                        if !self.midi_on {
                            context.send_event(NoteEvent::MidiCC {
                                timing: 0,
                                channel: 0,
                                cc: POLY_MODE_ON,
                                value: 0.0,
                            });
                            self.midi_notes = [false; 128];
                            self.midi_on = true;
                        }
                        midi_on = true;

                        if !self.midi_notes[note] {
                            context.send_event(NoteEvent::NoteOn {
                                timing: 0,
                                voice_id: Some(note as i32),
                                channel: 0,
                                note: note as u8,
                                velocity: *pressure,
                            });
                            self.midi_notes[note] = true;
                        } else {
                            context.send_event(NoteEvent::PolyPressure {
                                timing: 0,
                                voice_id: Some(note as i32),
                                channel: 0,
                                note: note as u8,
                                pressure: *pressure,
                            });
                        }
                    } else if self.midi_notes[note] {
                        context.send_event(NoteEvent::NoteOff {
                            timing: 0,
                            voice_id: Some(note as i32),
                            channel: 0,
                            note: note as u8,
                            velocity: 1.0 - *pressure,
                        });
                        self.midi_notes[note] = false;
                    }
                }

                if self.midi_on {
                    self.midi_output.iter_mut().for_each(|m| *m = 0.0);
                }

                if self.midi_on && !midi_on {
                    context.send_event(NoteEvent::MidiCC {
                        timing: 0,
                        channel: 0,
                        cc: ALL_NOTES_OFF,
                        value: 0.0,
                    });
                    self.midi_on = false;
                }
            }
        }

        #[cfg(feature = "mute-output")]
        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                *sample = 0.0;
            }
        }

        ProcessStatus::Normal
    }
}

#[derive(Clone)]
pub(crate) struct AnalysisChainConfig {
    gain: f64,
    listening_volume: f64,
    normalize_amplitude: bool,
    masking: bool,
    internal_buffering: bool,
    update_rate_hz: f64,
    latency_offset: Duration,

    output_tone_amplitude_threshold: f32,
    output_max_simultaneous_tones: usize,
    output_osc: bool,
    osc_socket_address: String,
    osc_resource_address_tones: String,
    osc_resource_address_stats: String,
    output_midi: bool,
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
            masking: true,
            internal_buffering: true,
            update_rate_hz: 1024.0,
            resolution: 512,
            latency_offset: Duration::ZERO,

            output_tone_amplitude_threshold: 30.0 - 86.0,
            output_max_simultaneous_tones: 24,
            output_osc: false,
            osc_socket_address: "127.0.0.1:8000".to_string(),
            osc_resource_address_tones: format!(
                "/katvisualizer/v{}/tones",
                env!("CARGO_PKG_VERSION")
            ),
            osc_resource_address_stats: format!(
                "/katvisualizer/v{}/stats",
                env!("CARGO_PKG_VERSION")
            ),
            output_midi: false,
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

#[allow(clippy::type_complexity)]
pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    left_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    right_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    gain: f64,
    internal_buffering: bool,
    output_tone_amplitude_threshold: f32,
    output_max_simultaneous_tones: usize,
    output_osc: bool,
    osc_socket_address: String,
    osc_resource_address_tones: Arc<String>,
    osc_resource_address_stats: Arc<String>,
    output_midi: bool,
    midi_pressure_min_amplitude: f32,
    midi_pressure_max_amplitude: f32,
    update_rate: f64,
    listening_volume: Option<f64>,
    masking: bool,
    pub(crate) latency_samples: u32,
    additional_latency: Duration,
    sample_rate: f32,
    chunk_size: usize,
    chunk_duration: Duration,
    single_input: bool,
    analyzer_pool: ThreadPool,
    osc_pool: ThreadPool,
    pub(crate) frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    osc_socket: Arc<Mutex<Option<UdpSocket>>>,
    osc_output: Arc<Mutex<Vec<(f32, f32, f32, f32, f32)>>>,
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
            masking: config.masking,
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
            output_tone_amplitude_threshold: config.output_tone_amplitude_threshold,
            output_max_simultaneous_tones: config.output_max_simultaneous_tones,
            output_osc: config.output_osc,
            osc_socket_address: config.osc_socket_address.clone(),
            osc_resource_address_tones: Arc::new(config.osc_resource_address_tones.clone()),
            osc_resource_address_stats: Arc::new(config.osc_resource_address_stats.clone()),
            output_midi: config.output_midi,
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
            masking: config.masking,
            chunk_size,
            chunk_duration: Duration::from_secs_f64(chunk_size as f64 / sample_rate as f64),
            single_input: layout.main_input_channels == NonZero::new(1),
            analyzer_pool: ThreadPool::new(2),
            osc_pool: ThreadPool::new(1),
            osc_output: Arc::new(Mutex::new(Vec::with_capacity(MAX_PEAK_OUTPUTS))),
            osc_socket: Arc::new(Mutex::new(None)),
        }
    }
    fn analyze(
        &mut self,
        buffer: &mut Buffer,
        output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
        midi_output: &mut [f32; 128],
    ) {
        let mut finished = Instant::now();

        if self.internal_buffering {
            self.chunker
                .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                    if channel_idx == 1 && self.single_input {
                        return;
                    }

                    if self.single_input {
                        let mut lock = self.left_analyzer.lock();
                        let (ref _buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64), self.listening_volume);
                    } else {
                        let analyzer = if channel_idx == 0 {
                            self.left_analyzer.clone()
                        } else {
                            self.right_analyzer.clone()
                        };
                        let listening_volume = self.listening_volume;

                        analyzer.lock().0.copy_from_slice(buffer);

                        self.analyzer_pool.execute(move || {
                            let mut lock = analyzer.lock();
                            let (ref mut buffer, ref mut analyzer) = *lock;

                            analyzer.analyze(buffer.iter().map(|s| *s as f64), listening_volume);
                        });
                    }

                    if channel_idx == 1 || (channel_idx == 0 && self.single_input) {
                        let (ref mut spectrogram, ref mut metrics) = *output.lock();

                        self.analyzer_pool.join();
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

                analyzer.analyze(
                    buffer.as_slice()[0].iter().map(|s| *s as f64),
                    self.listening_volume,
                );
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

                    let listening_volume = self.listening_volume;

                    self.analyzer_pool.execute(move || {
                        let mut lock = analyzer.lock();
                        let (ref mut buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64), listening_volume);
                    });
                }
            }

            let chunk_duration =
                Duration::from_secs_f64(buffer.samples() as f64 / self.sample_rate as f64);

            let (ref mut spectrogram, ref mut metrics) = *output.lock();

            let (left_ref, right_ref) = (self.left_analyzer.clone(), self.right_analyzer.clone());

            self.analyzer_pool.join();
            let mut left_lock = left_ref.lock();
            let mut right_lock = right_ref.lock();
            let left_analyzer = &mut left_lock.1;
            let right_analyzer = &mut right_lock.1;

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

            let osc_timestamp = if self.output_osc {
                let since_start = finished.elapsed();

                if chunk_duration > since_start {
                    (SystemTime::now() - since_start) + chunk_duration
                } else {
                    SystemTime::now()
                }
            } else {
                SystemTime::UNIX_EPOCH
            };

            self.generate_external_output(
                midi_output,
                spectrogram,
                left_analyzer,
                right_analyzer,
                osc_timestamp,
                chunk_duration,
            );

            let now = Instant::now();
            metrics.processing = now.duration_since(finished);
            metrics.finished = now;
        }
    }
    fn generate_external_output(
        &mut self,
        midi_output: &mut [f32; 128],
        spectrogram: &mut BetterSpectrogram,
        left_analyzer: &mut BetterAnalyzer,
        right_analyzer: &mut BetterAnalyzer,
        osc_timestamp: SystemTime,
        chunk_duration: Duration,
    ) {
        if !self.output_osc && !self.output_midi {
            return;
        }

        if self.masking {
            left_analyzer.remove_masked_components();
            right_analyzer.remove_masked_components();
        }

        let peaks = spectrogram.data[0].peaks(
            self.output_tone_amplitude_threshold,
            self.output_max_simultaneous_tones,
            left_analyzer,
        );

        let mut osc_output = self.osc_output.lock();
        osc_output.clear();

        #[cfg(feature = "midi")]
        if self.output_midi {
            for (frequency, width, pan, volume, signal_to_mask) in peaks {
                osc_output.push((frequency, width, pan, volume, signal_to_mask));

                let note = freq_to_midi_note(frequency).clamp(0.0, 128.0).round() as usize;
                if note != 128 {
                    midi_output[note] = map_value_f32(
                        volume,
                        self.midi_pressure_min_amplitude,
                        self.midi_pressure_max_amplitude,
                        0.0,
                        1.0,
                    )
                    .clamp(0.0, 1.0);
                }
            }
        } else {
            for (frequency, width, pan, volume, signal_to_mask) in peaks {
                osc_output.push((frequency, width, pan, volume, signal_to_mask));
            }
        }

        #[cfg(not(feature = "midi"))]
        for (frequency, width, pan, volume, signal_to_mask) in peaks {
            osc_output.push((frequency, width, pan, volume, signal_to_mask));
        }

        drop(osc_output);

        if self.output_osc
            && let Ok(socket_address) = self.osc_socket_address.parse()
        {
            let osc_resource_address_tones = self.osc_resource_address_tones.clone();
            let osc_resource_address_stats = self.osc_resource_address_stats.clone();
            let osc_socket = self.osc_socket.clone();
            let osc_output = self.osc_output.clone();
            let listening_volume = self.listening_volume.map(|l| l as f32);

            let osc_spectrum_metadata = if let Some(listening_volume) = listening_volume {
                (
                    spectrogram.data[0].masking_mean + listening_volume,
                    spectrogram.data[0].mean + listening_volume,
                    spectrogram.data[0].max + listening_volume,
                    chunk_duration.as_secs_f32(),
                )
            } else {
                (
                    spectrogram.data[0].masking_mean,
                    spectrogram.data[0].mean,
                    spectrogram.data[0].max,
                    chunk_duration.as_secs_f32(),
                )
            };

            self.osc_pool.execute(move || {
                let mut socket = osc_socket.lock();

                let new_socket = || {
                    UdpSocket::bind(match socket_address {
                        SocketAddr::V4(addr) => SocketAddr::V4(SocketAddrV4::new(
                            if addr.ip().is_loopback() {
                                *addr.ip()
                            } else {
                                Ipv4Addr::UNSPECIFIED
                            },
                            0,
                        )),
                        SocketAddr::V6(addr) => SocketAddr::V6(SocketAddrV6::new(
                            if addr.ip().is_loopback() {
                                Ipv6Addr::LOCALHOST
                            } else {
                                Ipv6Addr::UNSPECIFIED
                            },
                            0,
                            0,
                            0,
                        )),
                    })
                    .ok()
                };

                if let Some(active_socket) = &mut *socket {
                    if let Ok(active_address) = active_socket.local_addr() {
                        if socket_address.ip().is_loopback() != active_address.ip().is_loopback() {
                            *socket = new_socket();
                        } else {
                            match active_address {
                                SocketAddr::V4(addr) => {
                                    if !socket_address.is_ipv4()
                                        || (addr.ip().is_loopback()
                                            && socket_address.ip() != *addr.ip())
                                    {
                                        *socket = new_socket();
                                    }
                                }
                                SocketAddr::V6(_) => {
                                    if !socket_address.is_ipv6() {
                                        *socket = new_socket();
                                    }
                                }
                            }
                        }
                    } else {
                        *socket = new_socket();
                    }
                } else {
                    *socket = new_socket();
                }

                if let Some(socket) = &mut *socket {
                    let data = osc_output.lock();
                    let message_data = if let Some(listening_volume) = listening_volume {
                        data.iter()
                            .map(|(f, w, p, v, stm)| {
                                OscType::Array(OscArray {
                                    content: vec![
                                        OscType::Float(*f),
                                        OscType::Float(*w),
                                        OscType::Float(*p),
                                        OscType::Float(*v + listening_volume),
                                        OscType::Float(*stm),
                                    ],
                                })
                            })
                            .collect()
                    } else {
                        data.iter()
                            .map(|(f, w, p, v, stm)| {
                                OscType::Array(OscArray {
                                    content: vec![
                                        OscType::Float(*f),
                                        OscType::Float(*w),
                                        OscType::Float(*p),
                                        OscType::Float(*v),
                                        OscType::Float(*stm),
                                    ],
                                })
                            })
                            .collect()
                    };
                    let packet = OscPacket::Bundle(OscBundle {
                        timetag: OscTime::try_from(osc_timestamp).unwrap_or(OscTime {
                            seconds: 0,
                            fractional: 0,
                        }),
                        content: vec![
                            OscPacket::Message(OscMessage {
                                addr: osc_resource_address_stats.to_string(),
                                args: vec![
                                    OscType::Float(osc_spectrum_metadata.0),
                                    OscType::Float(osc_spectrum_metadata.1),
                                    OscType::Float(osc_spectrum_metadata.2),
                                    OscType::Float(osc_spectrum_metadata.3),
                                ],
                            }),
                            OscPacket::Message(OscMessage {
                                addr: osc_resource_address_tones.to_string(),
                                args: vec![OscType::Array(OscArray {
                                    content: message_data,
                                })],
                            }),
                        ],
                    });
                    let buf = encoder::encode(&packet).unwrap();
                    let _ = socket.send_to(&buf, socket_address);
                }
            });
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
            masking: self.masking,
            internal_buffering: self.internal_buffering,
            output_tone_amplitude_threshold: self.output_tone_amplitude_threshold,
            output_max_simultaneous_tones: self.output_max_simultaneous_tones,
            output_osc: self.output_osc,
            osc_socket_address: self.osc_socket_address.to_string(),
            osc_resource_address_tones: self.osc_resource_address_tones.to_string(),
            osc_resource_address_stats: self.osc_resource_address_stats.to_string(),
            output_midi: self.output_midi,
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
        self.masking = config.masking;

        let old_left_analyzer = self.left_analyzer.lock();
        let old_analyzer_config = old_left_analyzer.1.config();

        self.output_tone_amplitude_threshold = config.output_tone_amplitude_threshold;
        self.output_max_simultaneous_tones = config.output_max_simultaneous_tones;
        self.output_osc = config.output_osc;
        self.osc_socket_address = config.osc_socket_address.clone();
        self.osc_resource_address_tones = Arc::new(config.osc_resource_address_tones.clone());
        self.osc_resource_address_stats = Arc::new(config.osc_resource_address_stats.clone());
        self.output_midi = config.output_midi;
        self.midi_pressure_min_amplitude = config.midi_pressure_min_amplitude;
        self.midi_pressure_max_amplitude = config.midi_pressure_max_amplitude;
        if config.internal_buffering {
            self.output_osc = false;
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
            || old_analyzer_config.masking != config.masking
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
                masking: config.masking,
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

fn get_osc_packet_size(stats_address: &str, tones_address: &str, notes: usize) -> usize {
    let message_data = (0..=notes)
        .map(|_| {
            OscType::Array(OscArray {
                content: vec![
                    OscType::Float(0.0),
                    OscType::Float(0.0),
                    OscType::Float(0.0),
                    OscType::Float(0.0),
                ],
            })
        })
        .collect();
    let packet = OscPacket::Bundle(OscBundle {
        timetag: OscTime {
            seconds: 0,
            fractional: 0,
        },
        content: vec![
            OscPacket::Message(OscMessage {
                addr: stats_address.to_string(),
                args: vec![
                    OscType::Float(0.0),
                    OscType::Float(0.0),
                    OscType::Float(0.0),
                ],
            }),
            OscPacket::Message(OscMessage {
                addr: tones_address.to_string(),
                args: vec![OscType::Array(OscArray {
                    content: message_data,
                })],
            }),
        ],
    });
    encoder::encode(&packet).unwrap().len()
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
