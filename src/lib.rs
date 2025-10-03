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

type AnalyzerSet = Arc<Mutex<Option<(BetterAnalyzer, BetterAnalyzer)>>>;
type AnalyzerOutput = (Vec<f64>, Vec<f64>, Duration, Instant);
type Spectrogram = VecDeque<AnalyzerOutput>;

// TODO: Use f32 for spectrogram data

pub struct MyPlugin {
    params: Arc<PluginParams>,
    helper: util::StftHelper<0>,
    analyzers: AnalyzerSet,
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

impl Default for MyPlugin {
    fn default() -> Self {
        let spectrogram = VecDeque::from(vec![
            (
                Vec::with_capacity(4096),
                Vec::with_capacity(4096),
                Duration::ZERO,
                Instant::now(),
            );
            256
        ]);

        let (analyzer_input, analyzer_output) = triple_buffer(&spectrogram);

        Self {
            params: Arc::new(PluginParams::default()),
            helper: StftHelper::new(2, 96000, 0),
            analyzers: Arc::new(Mutex::new(None)),
            spectrogram,
            buffer: (
                Vec::with_capacity(4096),
                Vec::with_capacity(4096),
                Duration::ZERO,
                Instant::now(),
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
            AnalyzerSetWrapper::new(self.analyzers.clone()),
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
        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution: 270,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            log_frequency_scale: false,
            sample_rate: buffer_config.sample_rate as usize,
            time_resolution: (75.0, 200.0),
        });

        self.analyzers = Arc::new(Mutex::new(Some((analyzer.clone(), analyzer))));

        self.helper
            .set_block_size((buffer_config.sample_rate as f64 / 256.0).round() as usize);
        context.set_latency_samples(self.helper.latency_samples());

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if let Ok(mut lock) = self.analyzers.try_lock() {
            let analyzers = lock.as_mut().unwrap();

            assert!(buffer.channels() == 2);

            self.helper
                .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                    let analyzer = if channel_idx == 0 {
                        self.buffer.3 = Instant::now();
                        &mut analyzers.0
                    } else {
                        &mut analyzers.1
                    };

                    let output = analyzer.analyze(buffer.iter().map(|s| *s as f64), 0.0, 83.0);

                    #[allow(clippy::collapsible_else_if)]
                    if channel_idx == 0 {
                        if self.buffer.0.len() == output.len() {
                            self.buffer.0.copy_from_slice(output);
                        } else {
                            self.buffer.0.clear();
                            self.buffer.0.extend_from_slice(output);
                        }
                    } else {
                        if self.buffer.1.len() == output.len() {
                            self.buffer.1.copy_from_slice(output);
                        } else {
                            self.buffer.1.clear();
                            self.buffer.1.extend_from_slice(output);
                        }

                        let finished = Instant::now();
                        self.buffer.2 = finished.duration_since(self.buffer.3);
                        self.buffer.3 = finished;

                        update_spectrogram(&self.buffer, &mut self.spectrogram);
                        publish_updated_spectrogram(&self.spectrogram, &mut self.analyzer_input);
                    }
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

    spectrogram.push_front(old);
}

fn publish_updated_spectrogram(spectrogram: &Spectrogram, destination: &mut Input<Spectrogram>) {
    let write_buffer = destination.input_buffer_mut();
    let _ = mem::replace(write_buffer, spectrogram.clone());
    destination.publish();
}

#[derive(Clone)]
pub(crate) struct AnalyzerSetWrapper {
    analyzers: AnalyzerSet,
}

impl AnalyzerSetWrapper {
    fn new(analyzers: AnalyzerSet) -> Self {
        Self { analyzers }
    }

    pub(crate) fn update_config(
        &self,
        resolution: usize,
        start_frequency: f64,
        end_frequency: f64,
        log_frequency_scale: bool,
        time_resolution: (f64, f64),
    ) {
        let mut lock = self.analyzers.lock().unwrap();
        let analyzers = lock.as_mut().unwrap();
        let sample_rate = analyzers.0.config().sample_rate;

        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution,
            start_frequency,
            end_frequency,
            log_frequency_scale,
            sample_rate,
            time_resolution,
        });

        analyzers.0 = analyzer.clone();
        analyzers.1 = analyzer;
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
