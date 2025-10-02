use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::EguiState;
use std::sync::{Arc, Mutex};
use triple_buffer::{Input, Output, triple_buffer};

use crate::analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration};

pub mod analyzer;
mod editor;

type AnalyzerSet = Arc<Mutex<Option<(BetterAnalyzer, BetterAnalyzer)>>>;
type AnalyzerOutput = (Vec<f64>, Vec<f64>);

pub struct MyPlugin {
    params: Arc<PluginParams>,
    helper: util::StftHelper<0>,
    analyzers: AnalyzerSet,
    analyzer_input: Input<AnalyzerOutput>,
    analyzer_output: Arc<Mutex<Output<AnalyzerOutput>>>,
    block_size: usize,
}

#[derive(Params)]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for MyPlugin {
    fn default() -> Self {
        let (analyzer_input, analyzer_output) =
            triple_buffer(&(Vec::with_capacity(96000), Vec::with_capacity(96000)));

        Self {
            params: Arc::new(PluginParams::default()),
            helper: StftHelper::new(2, 96000, 0),
            analyzers: Arc::new(Mutex::new(None)),
            analyzer_input,
            analyzer_output: Arc::new(Mutex::new(analyzer_output)),
            block_size: 0,
        }
    }
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(900, 600),
        }
    }
}

impl Plugin for MyPlugin {
    const NAME: &'static str = "KatVisualizer";
    const VENDOR: &'static str = "transkatgirl";
    const URL: &'static str = "https://github.com/transkatgirl/katvisualizer";
    const EMAIL: &'static str = "08detour_dial@icloud.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

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
    ];

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
            resolution: 100,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            log_frequency_scale: false,
            sample_rate: buffer_config.sample_rate as usize,
            time_resolution: (75.0, 200.0),
        });
        self.block_size = analyzer.chunk_size();

        self.analyzers = Arc::new(Mutex::new(Some((analyzer.clone(), analyzer))));

        self.helper
            .set_block_size((buffer_config.sample_rate as f64 * 10.0 / 1000.0).round() as usize);
        context.set_latency_samples(self.helper.latency_samples());

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if let Ok(mut lock) = self.analyzers.try_lock() {
            let analyzers = lock.as_mut().unwrap();

            let block_size = analyzers.0.chunk_size();

            if block_size != self.block_size {
                self.block_size = block_size;
                self.helper.set_block_size(self.block_size);
                context.set_latency_samples(self.helper.latency_samples());
            }

            let mut write_buffer = self.analyzer_input.input_buffer_publisher();

            self.helper
                .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                    let analyzer = if channel_idx == 0 {
                        &mut analyzers.0
                    } else {
                        &mut analyzers.1
                    };

                    let output = analyzer.analyze(buffer.iter().map(|s| *s as f64), 80.0);

                    #[allow(clippy::collapsible_else_if)]
                    if channel_idx == 0 {
                        if write_buffer.0.len() == output.len() {
                            write_buffer.0.copy_from_slice(output);
                        } else {
                            write_buffer.0.clear();
                            write_buffer.0.extend_from_slice(output);
                        }
                    } else {
                        if write_buffer.1.len() == output.len() {
                            write_buffer.1.copy_from_slice(output);
                        } else {
                            write_buffer.1.clear();
                            write_buffer.1.extend_from_slice(output);
                        }
                    }
                });
        }

        ProcessStatus::Normal
    }
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
        ClapFeature::Mono,
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
