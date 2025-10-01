use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Vec2},
    resizable_window::ResizableWindow,
    widgets,
};
use std::{
    collections::HashMap,
    mem,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration};

mod analyzer;
mod editor;

type AnalyzerOutput = (Vec<f32>, Vec<f32>);

pub struct MyPlugin {
    params: Arc<PluginParams>,
    helper: util::StftHelper<0>,
    analyzers: Arc<Mutex<Option<(BetterAnalyzer, BetterAnalyzer)>>>,
    analyzer_scratchpad: AnalyzerOutput,
    analyzer_output: Arc<Mutex<AnalyzerOutput>>,
    block_size: usize,
}

#[derive(Params)]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
    /*#[id = "gain"]
    pub gain: FloatParam,

    #[id = "foobar"]
    pub some_int: IntParam,*/
}

impl Default for MyPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(PluginParams::default()),
            helper: StftHelper::new(2, 96000, 0),
            analyzers: Arc::new(Mutex::new(None)),
            analyzer_scratchpad: (Vec::with_capacity(96000), Vec::with_capacity(96000)),
            analyzer_output: Arc::new(Mutex::new((
                Vec::with_capacity(96000),
                Vec::with_capacity(96000),
            ))),
            block_size: 0,
        }
    }
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(300, 180),
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

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        /*let params = self.params.clone();
        let peak_meter = self.peak_meter.clone();
        let egui_state = params.editor_state.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                ResizableWindow::new("res-wind")
                    .min_size(Vec2::new(128.0, 128.0))
                    .show(egui_ctx, egui_state.as_ref(), |ui| {
                        // NOTE: See `plugins/diopser/src/editor.rs` for an example using the generic UI widget

                        // This is a fancy widget that can get all the information it needs to properly
                        // display and modify the parameter from the parametr itself
                        // It's not yet fully implemented, as the text is missing.
                        ui.label("Some random integer");
                        ui.add(widgets::ParamSlider::for_param(&params.some_int, setter));

                        ui.label("Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.gain, setter));

                        ui.label(
                        "Also gain, but with a lame widget. Can't even render the value correctly!",
                    );
                        // This is a simple naieve version of a parameter slider that's not aware of how
                        // the parameters work
                        ui.add(
                            egui::widgets::Slider::from_get_set(-30.0..=30.0, |new_value| {
                                match new_value {
                                    Some(new_value_db) => {
                                        let new_value = util::gain_to_db(new_value_db as f32);

                                        setter.begin_set_parameter(&params.gain);
                                        setter.set_parameter(&params.gain, new_value);
                                        setter.end_set_parameter(&params.gain);

                                        new_value_db
                                    }
                                    None => util::gain_to_db(params.gain.value()) as f64,
                                }
                            })
                            .suffix(" dB"),
                        );

                        // TODO: Add a proper custom widget instead of reusing a progress bar
                        let peak_meter =
                            util::gain_to_db(peak_meter.load(std::sync::atomic::Ordering::Relaxed));
                        let peak_meter_text = if peak_meter > util::MINUS_INFINITY_DB {
                            format!("{peak_meter:.1} dBFS")
                        } else {
                            String::from("-inf dBFS")
                        };

                        let peak_meter_normalized = (peak_meter + 60.0) / 60.0;
                        ui.allocate_space(egui::Vec2::splat(2.0));
                        ui.add(
                            egui::widgets::ProgressBar::new(peak_meter_normalized)
                                .text(peak_meter_text),
                        );
                    });
            },
        )*/

        None
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let analyzer = BetterAnalyzer::new(BetterAnalyzerConfiguration {
            resolution: 300,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            log_frequency_scale: false,
            sample_rate: buffer_config.sample_rate as usize,
            time_resolution: (75.0, 200.0),
        });
        self.block_size = analyzer.chunk_size();

        self.analyzers = Arc::new(Mutex::new(Some((analyzer.clone(), analyzer))));

        self.helper.set_block_size(self.block_size);
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

            self.helper
                .process_analyze_only(buffer, self.block_size / 4, |channel_idx, buffer| {
                    let analyzer = if channel_idx == 0 {
                        &mut analyzers.0
                    } else {
                        &mut analyzers.1
                    };

                    let output = analyzer.analyze(buffer.iter().cloned(), 80.0);

                    #[allow(clippy::collapsible_else_if)]
                    if channel_idx == 0 {
                        if self.analyzer_scratchpad.0.len() == output.len() {
                            self.analyzer_scratchpad.0.copy_from_slice(buffer);
                        } else {
                            self.analyzer_scratchpad.0.clear();
                            self.analyzer_scratchpad.0.extend_from_slice(buffer);
                        }
                    } else {
                        if self.analyzer_scratchpad.1.len() == output.len() {
                            self.analyzer_scratchpad.1.copy_from_slice(buffer);
                        } else {
                            self.analyzer_scratchpad.1.clear();
                            self.analyzer_scratchpad.1.extend_from_slice(buffer);
                        }
                    }
                });

            let mut output_lock = self.analyzer_output.lock().unwrap();

            mem::swap(&mut output_lock.0, &mut self.analyzer_scratchpad.0);
            mem::swap(&mut output_lock.1, &mut self.analyzer_scratchpad.1);
        }

        ProcessStatus::Normal
    }
}

impl MyPlugin {
    fn update_config(
        &self,
        resolution: usize,
        start_frequency: f32,
        end_frequency: f32,
        log_frequency_scale: bool,
        time_resolution: (f32, f32),
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
