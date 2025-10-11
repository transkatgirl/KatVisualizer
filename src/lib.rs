use mimalloc::MiMalloc;
use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::EguiState;
use parking_lot::{FairMutex, Mutex, RwLock};
use std::{
    num::NonZero,
    sync::Arc,
    time::{Duration, Instant},
};
use threadpool::ThreadPool;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use crate::analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration, BetterSpectrogram};

pub mod analyzer;
mod editor;

#[derive(Clone, Copy)]
pub(crate) struct AnalysisMetrics {
    processing: Duration,
    finished: Instant,
}

pub struct MyPlugin {
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    latency_samples: u32,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
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
            analysis_frequencies: Arc::new(RwLock::new(Vec::with_capacity(MAX_FREQUENCY_BINS))),
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
            buffer_config.sample_rate as usize,
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

            analysis_chain.analyze(buffer, &self.analysis_output);
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

    resolution: usize,
    start_frequency: f64,
    end_frequency: f64,
    erb_frequency_scale: bool,
    erb_time_resolution: bool,
    erb_time_resolution_clamp: (f64, f64),
    erb_bandwidth_divisor: f64,
    time_resolution: f64,
    nc_method: bool,
}

impl Default for AnalysisChainConfig {
    fn default() -> Self {
        Self {
            gain: 0.0,
            listening_volume: 92.0,
            normalize_amplitude: true,
            internal_buffering: true,
            update_rate_hz: 512.0,
            resolution: 512,
            latency_offset: Duration::ZERO,
            start_frequency: BetterAnalyzerConfiguration::default().start_frequency,
            end_frequency: BetterAnalyzerConfiguration::default().end_frequency,
            erb_frequency_scale: BetterAnalyzerConfiguration::default().erb_frequency_scale,
            erb_time_resolution: BetterAnalyzerConfiguration::default().erb_time_resolution,
            erb_time_resolution_clamp: BetterAnalyzerConfiguration::default()
                .erb_time_resolution_clamp,
            erb_bandwidth_divisor: BetterAnalyzerConfiguration::default().erb_bandwidth_divisor,
            time_resolution: BetterAnalyzerConfiguration::default().time_resolution,
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
    update_rate: f64,
    listening_volume: Option<f64>,
    pub(crate) latency_samples: u32,
    additional_latency: Duration,
    sample_rate: usize,
    chunk_size: usize,
    chunk_duration: Duration,
    single_input: bool,
    pool: ThreadPool,
    pub(crate) frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
}

impl AnalysisChain {
    fn new(
        config: &AnalysisChainConfig,
        sample_rate: usize,
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
            erb_time_resolution_clamp: config.erb_time_resolution_clamp,
            erb_bandwidth_divisor: config.erb_bandwidth_divisor,
            time_resolution: config.time_resolution,
            nc_method: config.nc_method,
        });

        let mut chunker = StftHelper::new(2, sample_rate, 0);
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

                        analyzer.analyze(
                            buffer.iter().map(|s| *s as f64),
                            self.gain,
                            self.listening_volume,
                        );
                    } else {
                        let analyzer = if channel_idx == 0 {
                            self.left_analyzer.clone()
                        } else {
                            self.right_analyzer.clone()
                        };

                        analyzer.lock().0.copy_from_slice(buffer);
                        let gain = self.gain;
                        let listening_volume = self.listening_volume;

                        self.pool.execute(move || {
                            let mut lock = analyzer.lock();
                            let (ref mut buffer, ref mut analyzer) = *lock;

                            analyzer.analyze(
                                buffer.iter().map(|s| *s as f64),
                                gain,
                                listening_volume,
                            );
                        });
                    }

                    if channel_idx == 1 || (channel_idx == 0 && self.single_input) {
                        let (ref mut spectrogram, ref mut metrics) = *output.lock();

                        self.pool.join();
                        let left_lock = self.left_analyzer.lock();
                        let right_lock = self.right_analyzer.lock();
                        let left_output = left_lock.1.last_analysis();
                        let right_output = right_lock.1.last_analysis();

                        spectrogram.update_fn(|analysis_output| {
                            if self.single_input {
                                analysis_output.update_mono(left_output, self.chunk_duration);
                            } else {
                                analysis_output.update_stereo(
                                    left_output,
                                    right_output,
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
                    self.gain,
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

                    let gain = self.gain;
                    let listening_volume = self.listening_volume;

                    self.pool.execute(move || {
                        let mut lock = analyzer.lock();
                        let (ref mut buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64), gain, listening_volume);
                    });
                }
            }

            let chunk_duration =
                Duration::from_secs_f64(buffer.samples() as f64 / self.sample_rate as f64);

            let (ref mut spectrogram, ref mut metrics) = *output.lock();

            self.pool.join();
            let left_lock = self.left_analyzer.lock();
            let right_lock = self.right_analyzer.lock();
            let left_output = left_lock.1.last_analysis();
            let right_output = right_lock.1.last_analysis();

            spectrogram.update_fn(|analysis_output| {
                if self.single_input {
                    analysis_output.update_mono(left_output, chunk_duration);
                } else {
                    analysis_output.update_stereo(left_output, right_output, chunk_duration);
                }
            });

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
            update_rate_hz: self.update_rate,
            latency_offset: self.additional_latency,
            resolution: analyzer_config.resolution,
            start_frequency: analyzer_config.start_frequency,
            end_frequency: analyzer_config.end_frequency,
            erb_frequency_scale: analyzer_config.erb_frequency_scale,
            erb_time_resolution: analyzer_config.erb_time_resolution,
            erb_time_resolution_clamp: analyzer_config.erb_time_resolution_clamp,
            erb_bandwidth_divisor: analyzer_config.erb_bandwidth_divisor,
            time_resolution: analyzer_config.time_resolution,
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
            || old_analyzer_config.erb_time_resolution_clamp != config.erb_time_resolution_clamp
            || old_analyzer_config.erb_bandwidth_divisor != config.erb_bandwidth_divisor
            || old_analyzer_config.time_resolution != config.time_resolution
            || old_analyzer_config.nc_method != config.nc_method
        {
            let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
                resolution: config.resolution,
                start_frequency: config.start_frequency,
                end_frequency: config.end_frequency,
                erb_frequency_scale: config.erb_frequency_scale,
                sample_rate: self.sample_rate,
                erb_time_resolution: config.erb_time_resolution,
                erb_time_resolution_clamp: config.erb_time_resolution_clamp,
                erb_bandwidth_divisor: config.erb_bandwidth_divisor,
                time_resolution: config.time_resolution,
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
