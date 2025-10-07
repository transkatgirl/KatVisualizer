use mimalloc::MiMalloc;
use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::EguiState;
use parking_lot::Mutex;
use std::{
    num::NonZero,
    sync::Arc,
    time::{Duration, Instant},
};
use threadpool::ThreadPool;
use triple_buffer::{Input, Output, triple_buffer};

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
    metrics: AnalysisMetrics,
    spectrogram: BetterSpectrogram,
    buffer_input: Input<(BetterSpectrogram, AnalysisMetrics)>,
    buffer_output: Arc<Mutex<Output<(BetterSpectrogram, AnalysisMetrics)>>>,
}

#[derive(Params)]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

const MAX_FREQUENCY_BINS: usize = 2048;
const SPECTROGRAM_SLICES: usize = 1024;

impl Default for MyPlugin {
    fn default() -> Self {
        let spectrogram = (
            BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
            AnalysisMetrics {
                processing: Duration::ZERO,
                finished: Instant::now(),
            },
        );
        let (buffer_input, buffer_output) = triple_buffer(&spectrogram);

        Self {
            params: Arc::new(PluginParams::default()),
            analysis_chain: Arc::new(Mutex::new(None)),
            latency_samples: 0,
            spectrogram: BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
            metrics: AnalysisMetrics {
                processing: Duration::ZERO,
                finished: Instant::now(),
            },
            buffer_input,
            buffer_output: Arc::new(Mutex::new(buffer_output)),
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
            self.buffer_output.clone(),
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
        );
        context.set_latency_samples(new_chain.latency_samples);
        self.latency_samples = new_chain.latency_samples;

        *analysis_chain = Some(new_chain);

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
                &mut self.spectrogram,
                &mut self.metrics,
                |spectrogram, metrics| {
                    let write_buffer = self.buffer_input.input_buffer_mut();
                    write_buffer.0.clone_from(spectrogram);
                    write_buffer.1 = *metrics;
                    self.buffer_input.publish();
                },
            );
        }

        ProcessStatus::Normal
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AnalysisChainConfig {
    gain: f64,
    listening_volume: f64,
    normalize_amplitude: bool,
    update_rate_hz: f64,
    buffer_update_rate_hz: f64,

    resolution: usize,
    start_frequency: f64,
    end_frequency: f64,
    log_frequency_scale: bool,
    time_resolution: (f64, f64),
    spectral_reassignment: bool,
}

impl Default for AnalysisChainConfig {
    fn default() -> Self {
        Self {
            gain: 0.0,
            listening_volume: 89.0,
            normalize_amplitude: true,
            update_rate_hz: 512.0,
            buffer_update_rate_hz: 256.0,
            resolution: 512,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            log_frequency_scale: false,
            time_resolution: (70.0, 120.0),
            spectral_reassignment: true,
        }
    }
}

pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    left_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    right_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    gain: f64,
    update_rate: f64,
    buffer_update_rate_hz: f64,
    listening_volume: Option<f64>,
    latency_samples: u32,
    chunk_size: usize,
    chunk_duration: Duration,
    single_input: bool,
    pool: ThreadPool,
    buffer_update_rate: usize,
    last_buffer_update: usize,
    duration_samples: usize,
}

impl AnalysisChain {
    fn new(config: &AnalysisChainConfig, sample_rate: usize, layout: &AudioIOLayout) -> Self {
        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution: config.resolution,
            start_frequency: config.start_frequency,
            end_frequency: config.end_frequency,
            log_frequency_scale: config.log_frequency_scale,
            sample_rate,
            time_resolution: config.time_resolution,
            spectral_reassignment: config.spectral_reassignment,
        });

        let mut chunker = StftHelper::new(2, sample_rate, 0);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        chunker.set_block_size(chunk_size);

        Self {
            latency_samples: chunker.latency_samples(),
            chunker,
            left_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], analyzer.clone()))),
            right_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], analyzer))),
            gain: config.gain,
            update_rate: config.update_rate_hz,
            buffer_update_rate_hz: config.buffer_update_rate_hz,
            listening_volume: if config.normalize_amplitude {
                Some(config.listening_volume)
            } else {
                None
            },
            chunk_size,
            chunk_duration: Duration::from_secs_f64(chunk_size as f64 / sample_rate as f64),
            single_input: layout.main_input_channels == NonZero::new(1),
            pool: ThreadPool::new(2),
            buffer_update_rate: (config.update_rate_hz / config.buffer_update_rate_hz)
                .floor()
                .max(1.0) as usize,
            last_buffer_update: usize::MAX - 1,
            duration_samples: 0,
        }
    }
    fn analyze<F>(
        &mut self,
        buffer: &mut Buffer,
        spectrogram: &mut BetterSpectrogram,
        metrics: &mut AnalysisMetrics,
        mut callback: F,
    ) where
        F: FnMut(&BetterSpectrogram, &AnalysisMetrics),
    {
        metrics.finished = Instant::now();
        self.duration_samples = 0;

        self.chunker
            .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                if channel_idx == 1 && self.single_input {
                    return;
                }
                let analyzer = if channel_idx == 0 {
                    self.left_analyzer.clone()
                } else {
                    self.right_analyzer.clone()
                };

                if self.single_input {
                    let mut lock = analyzer.lock();
                    let (ref _buffer, ref mut analyzer) = *lock;

                    analyzer.analyze(
                        buffer.iter().map(|s| *s as f64),
                        self.gain,
                        self.listening_volume,
                    );
                } else {
                    analyzer.lock().0.copy_from_slice(buffer);
                    let gain = self.gain;
                    let listening_volume = self.listening_volume;

                    self.pool.execute(move || {
                        let mut lock = analyzer.lock();
                        let (ref mut buffer, ref mut analyzer) = *lock;

                        analyzer.analyze(buffer.iter().map(|s| *s as f64), gain, listening_volume);
                    });
                }

                if channel_idx == 1 || (channel_idx == 0 && self.single_input) {
                    self.pool.join();
                    let left_lock = self.left_analyzer.lock();
                    let right_lock = self.right_analyzer.lock();
                    let left_output = left_lock.1.last_analysis();
                    let right_output = right_lock.1.last_analysis();

                    spectrogram.update_fn(|analysis_output| {
                        if self.single_input {
                            analysis_output.0.update_mono(left_output);
                        } else {
                            analysis_output.0.update_stereo(left_output, right_output);
                        }
                        analysis_output.1 = self.chunk_duration;
                    });

                    self.last_buffer_update += 1;
                    self.duration_samples += 1;

                    if self.last_buffer_update >= self.buffer_update_rate {
                        let finished = Instant::now();
                        metrics.processing = finished
                            .duration_since(metrics.finished)
                            .div_f64(self.duration_samples as f64);
                        metrics.finished = finished;

                        callback(spectrogram, metrics);

                        self.last_buffer_update = 0;
                        self.duration_samples = 0;
                    }
                }
            });
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
            update_rate_hz: self.update_rate,
            resolution: analyzer_config.resolution,
            buffer_update_rate_hz: self.buffer_update_rate_hz,
            start_frequency: analyzer_config.start_frequency,
            end_frequency: analyzer_config.end_frequency,
            log_frequency_scale: analyzer_config.log_frequency_scale,
            time_resolution: analyzer_config.time_resolution,
            spectral_reassignment: analyzer_config.spectral_reassignment,
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
        let sample_rate = old_analyzer_config.sample_rate;

        if self.update_rate != config.update_rate_hz {
            self.chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
            self.chunker.set_block_size(self.chunk_size);
            self.latency_samples = self.chunker.latency_samples();
            self.chunk_duration =
                Duration::from_secs_f64(self.chunk_size as f64 / sample_rate as f64);
        }

        if self.update_rate != config.update_rate_hz
            || self.buffer_update_rate_hz != config.buffer_update_rate_hz
        {
            self.buffer_update_rate_hz = config.buffer_update_rate_hz;
            self.buffer_update_rate = (config.update_rate_hz / self.buffer_update_rate_hz)
                .floor()
                .max(1.0) as usize;
        }

        if old_analyzer_config.resolution != config.resolution
            || old_analyzer_config.start_frequency != config.start_frequency
            || old_analyzer_config.end_frequency != config.end_frequency
            || old_analyzer_config.log_frequency_scale != config.log_frequency_scale
            || old_analyzer_config.time_resolution != config.time_resolution
            || old_analyzer_config.spectral_reassignment != config.spectral_reassignment
        {
            let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
                resolution: config.resolution,
                start_frequency: config.start_frequency,
                end_frequency: config.end_frequency,
                log_frequency_scale: config.log_frequency_scale,
                sample_rate,
                time_resolution: config.time_resolution,
                spectral_reassignment: config.spectral_reassignment,
            });
            drop(old_left_analyzer);

            self.left_analyzer =
                Arc::new(Mutex::new((vec![0.0; self.chunk_size], analyzer.clone())));
            self.right_analyzer = Arc::new(Mutex::new((vec![0.0; self.chunk_size], analyzer)));
        } else if self.update_rate != config.update_rate_hz {
            drop(old_left_analyzer);

            self.left_analyzer.lock().0 = vec![0.0; self.chunk_size];
            self.right_analyzer.lock().0 = vec![0.0; self.chunk_size];
        }

        self.update_rate = config.update_rate_hz;
        self.last_buffer_update = usize::MAX - 1;
        self.duration_samples = 0;
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
