use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::EguiState;
use std::{
    collections::VecDeque,
    mem,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use triple_buffer::{Input, Output, triple_buffer};

use crate::analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration};

pub mod analyzer;
mod editor;

type AnalyzerOutput = (Vec<f64>, Vec<f64>, Duration, Instant, Duration);
type Spectrogram = VecDeque<AnalyzerOutput>;

pub struct MyPlugin {
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    latency_samples: u32,
    buffer: AnalyzerOutput,
    spectrogram: Spectrogram,
    analyzer_input: Input<Spectrogram>,
    analyzer_output: Arc<Mutex<Output<Spectrogram>>>,
}

#[derive(Params)]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

const MAX_FREQUENCY_BINS: usize = 2048;
const SPECTROGRAM_SLICES: usize = 512;

impl Default for MyPlugin {
    fn default() -> Self {
        let spectrogram = VecDeque::from(vec![
            (
                Vec::with_capacity(MAX_FREQUENCY_BINS),
                Vec::with_capacity(MAX_FREQUENCY_BINS),
                Duration::ZERO,
                Instant::now(),
                Duration::from_secs_f64(1.0 / 512.0),
            );
            SPECTROGRAM_SLICES
        ]);

        let (analyzer_input, analyzer_output) = triple_buffer(&spectrogram);

        Self {
            params: Arc::new(PluginParams::default()),
            analysis_chain: Arc::new(Mutex::new(None)),
            latency_samples: 0,
            spectrogram,
            buffer: (
                Vec::with_capacity(MAX_FREQUENCY_BINS),
                Vec::with_capacity(MAX_FREQUENCY_BINS),
                Duration::ZERO,
                Instant::now(),
                Duration::from_secs_f64(1.0 / 512.0),
            ),
            analyzer_input,
            analyzer_output: Arc::new(Mutex::new(analyzer_output)),
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

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
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
            self.analyzer_output.clone(),
            async_executor,
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let analysis_chain = AnalysisChain::new(
            BetterAnalyzerConfiguration {
                resolution: 256,
                start_frequency: 20.0,
                end_frequency: 20000.0,
                log_frequency_scale: false,
                sample_rate: buffer_config.sample_rate as usize,
                time_resolution: (75.0, 200.0),
            },
            0.0,
            Some(85.0),
            512.0,
        );
        context.set_latency_samples(analysis_chain.latency_samples);
        self.latency_samples = analysis_chain.latency_samples;
        self.analysis_chain = Arc::new(Mutex::new(Some(analysis_chain)));

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if let Ok(mut lock) = self.analysis_chain.try_lock() {
            let analysis_chain = lock.as_mut().unwrap();

            if analysis_chain.latency_samples != self.latency_samples {
                context.set_latency_samples(analysis_chain.latency_samples);
                self.latency_samples = analysis_chain.latency_samples;
            }

            analysis_chain.analyze(buffer, &mut self.buffer, |output| {
                update_spectrogram(output, &mut self.spectrogram);
                publish_updated_spectrogram(&self.spectrogram, &mut self.analyzer_input);
            });
        }

        ProcessStatus::Normal
    }
}

fn update_spectrogram(buffer: &AnalyzerOutput, spectrogram: &mut Spectrogram) {
    let mut old = spectrogram.pop_back().unwrap();

    if old.0.len() == buffer.0.len() {
        old.0.copy_from_slice(&buffer.0);
    } else {
        old.0.clear();
        old.0.extend_from_slice(&buffer.0);
    }
    if old.1.len() == buffer.1.len() {
        old.1.copy_from_slice(&buffer.1);
    } else {
        old.1.clear();
        old.1.extend_from_slice(&buffer.1);
    }
    old.2 = buffer.2;
    old.3 = buffer.3;
    old.4 = buffer.4;

    spectrogram.push_front(old);
}

fn publish_updated_spectrogram(spectrogram: &Spectrogram, destination: &mut Input<Spectrogram>) {
    let write_buffer = destination.input_buffer_mut();
    let _ = mem::replace(write_buffer, spectrogram.clone());
    destination.publish();
}

pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    left_analyzer: BetterAnalyzer,
    right_analyzer: BetterAnalyzer,
    gain: f64,
    listening_volume: Option<f64>,
    latency_samples: u32,
    chunk_duration: Duration,
}

impl AnalysisChain {
    fn new(
        config: BetterAnalyzerConfiguration,
        gain: f64,
        listening_volume: Option<f64>,
        update_rate_hz: f64,
    ) -> Self {
        let sample_rate = config.sample_rate;

        let analyzer = BetterAnalyzer::new(config);

        let mut chunker = StftHelper::new(2, sample_rate, 0);
        chunker.set_block_size((sample_rate as f64 / update_rate_hz).round() as usize);

        Self {
            latency_samples: chunker.latency_samples(),
            chunker,
            left_analyzer: analyzer.clone(),
            right_analyzer: analyzer,
            gain,
            listening_volume,
            chunk_duration: Duration::from_secs_f64(1.0 / update_rate_hz),
        }
    }
    fn analyze<F>(
        &mut self,
        buffer: &mut Buffer,
        analysis_output: &mut AnalyzerOutput,
        mut callback: F,
    ) where
        F: FnMut(&mut AnalyzerOutput),
    {
        self.chunker
            .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                let analyzer = if channel_idx == 0 {
                    analysis_output.3 = Instant::now();
                    &mut self.left_analyzer
                } else {
                    &mut self.right_analyzer
                };

                let output = analyzer.analyze(
                    buffer.iter().map(|s| *s as f64),
                    self.gain,
                    self.listening_volume,
                );

                #[allow(clippy::collapsible_else_if)]
                if channel_idx == 0 {
                    if analysis_output.0.len() == output.len() {
                        analysis_output.0.copy_from_slice(output);
                    } else {
                        analysis_output.0.clear();
                        analysis_output.0.extend_from_slice(output);
                    }
                } else {
                    if analysis_output.1.len() == output.len() {
                        analysis_output.1.copy_from_slice(output);
                    } else {
                        analysis_output.1.clear();
                        analysis_output.1.extend_from_slice(output);
                    }

                    let finished = Instant::now();
                    analysis_output.2 = finished.duration_since(analysis_output.3);
                    analysis_output.3 = finished;
                    analysis_output.4 = self.chunk_duration;

                    callback(analysis_output);
                }
            });
    }
    pub(crate) fn update_analysis_config(&mut self, gain: f64, listening_volume: Option<f64>) {
        self.gain = gain;
        self.listening_volume = listening_volume;
    }
    pub(crate) fn update_chunking_config(&mut self, update_rate_hz: f64) {
        let sample_rate = self.left_analyzer.config().sample_rate;

        self.chunker
            .set_block_size((sample_rate as f64 / update_rate_hz).round() as usize);
        self.latency_samples = self.chunker.latency_samples();
        self.chunk_duration = Duration::from_secs_f64(1.0 / update_rate_hz);
    }
    pub(crate) fn update_analyzer_config(
        &mut self,
        resolution: usize,
        start_frequency: f64,
        end_frequency: f64,
        log_frequency_scale: bool,
        time_resolution: (f64, f64),
    ) {
        let sample_rate = self.left_analyzer.config().sample_rate;

        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution,
            start_frequency,
            end_frequency,
            log_frequency_scale,
            sample_rate,
            time_resolution,
        });

        self.left_analyzer = analyzer.clone();
        self.right_analyzer = analyzer;
    }
}

impl ClapPlugin for MyPlugin {
    const CLAP_ID: &'static str = "com.transkatgirl.katvisualizer";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Analyzer,
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
