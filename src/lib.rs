// TODO: Go through https://nnethercote.github.io/perf-book/title-page.html and apply applicable optimizations

#[cfg(all(not(debug_assertions), not(target_arch = "wasm32")))]
use mimalloc::MiMalloc;

use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use parking_lot::FairMutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZero;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
use keepawake::KeepAwake;
#[cfg(not(target_arch = "wasm32"))]
use nih_plug::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use nih_plug_egui::EguiState;

#[cfg(target_arch = "wasm32")]
use js_sys::Float32Array;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(all(not(debug_assertions), not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(target_arch = "wasm32")]
use crate::editor::{SharedState, build, render};

use crate::chain::{AnalysisChain, AnalysisChainConfig};
#[cfg(target_arch = "wasm32")]
use crate::output::DirectAnalysisSink;
#[cfg(not(target_arch = "wasm32"))]
use crate::output::{NativeAnalysisReceiver, NativeAnalysisSink, native_transport};

pub mod analyzer;
pub mod chain;
mod editor;
mod output;

#[derive(Clone, Copy, Default)]
pub(crate) struct AnalysisMetrics {
    processing: Duration,
}

#[derive(Clone, Debug)]
pub(crate) struct AudioState {
    pub(crate) buffer_size_range: (u32, u32),
    pub(crate) sample_rate: f32,
    pub(crate) process_mode_title: String,
    pub(crate) realtime: bool,
    pub(crate) input_channels: u32,
    pub(crate) output_channels: u32,
}

#[cfg(target_arch = "wasm32")]
impl Default for AudioState {
    fn default() -> Self {
        Self {
            buffer_size_range: (
                0,
                (SAMPLE_BUFFER_SIZE as u32).min((BUFFER_LIMIT_SECS * 48000.0).floor() as u32),
            ),
            sample_rate: 48000.0,
            process_mode_title: "Chunked".to_string(),
            realtime: false,
            input_channels: 2,
            output_channels: 0,
        }
    }
}

impl AudioState {
    #[cfg(not(target_arch = "wasm32"))]
    fn new(audio_io_layout: AudioIOLayout, buffer_config: BufferConfig) -> Self {
        Self {
            buffer_size_range: (
                buffer_config.min_buffer_size.unwrap_or(0),
                buffer_config.max_buffer_size,
            ),
            sample_rate: buffer_config.sample_rate,
            process_mode_title: format!("{:?}", buffer_config.process_mode),
            realtime: buffer_config.process_mode == ProcessMode::Realtime,
            input_channels: audio_io_layout
                .main_input_channels
                .map(u32::from)
                .unwrap_or(0),
            output_channels: audio_io_layout
                .main_output_channels
                .map(u32::from)
                .unwrap_or(0),
        }
    }
}

#[cfg(target_arch = "wasm32")]
const SAMPLE_BUFFER_SIZE: u16 = u16::MAX;

#[cfg(target_arch = "wasm32")]
const BUFFER_LIMIT_SECS: f32 = 0.1;

#[cfg(target_arch = "wasm32")]
struct SampleBuffers {
    position: u16,
    single_input: bool,
    rate: f32,
    latency: u16,
    left: Vec<f32>,
    right: Vec<f32>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
static SAMPLES: RefCell<SampleBuffers> = RefCell::new(SampleBuffers {
    position: 0,
    single_input: false,
    rate: 48000.0,
    latency: 0,
    left: vec![0.0; SAMPLE_BUFFER_SIZE as usize],
    right: vec![0.0; SAMPLE_BUFFER_SIZE as usize],
});
} // The WASM module and the sample passer MUST be on the same thread

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn left_sample_buffer() -> Float32Array {
    SAMPLES.with(|samples| unsafe { Float32Array::view(&samples.borrow().left) })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn right_sample_buffer() -> Float32Array {
    SAMPLES.with(|samples| unsafe { Float32Array::view(&samples.borrow().right) })
}

#[cfg(target_arch = "wasm32")]
pub fn drain_buffers(callback: impl FnOnce(bool, f32, [&mut [f32]; 2])) {
    SAMPLES.with(|samples| {
        let mut samples = samples.borrow_mut();

        let SampleBuffers {
            position,
            single_input,
            rate,
            latency,
            left: left_samples,
            right: right_samples,
        } = &mut *samples;

        if *latency >= *position {
            return;
        }

        let index = *position as usize;
        let latency_usize = *latency as usize;
        let compensated_index = index - latency_usize;

        let compensated_limited_index =
            compensated_index.min((*rate * BUFFER_LIMIT_SECS).floor() as usize);

        callback(
            *single_input,
            *rate,
            [
                unsafe { left_samples.get_unchecked_mut(0..compensated_limited_index) },
                unsafe { right_samples.get_unchecked_mut(0..compensated_limited_index) },
            ],
        );

        if compensated_index != index {
            left_samples.copy_within(compensated_index..index, 0);
            right_samples.copy_within(compensated_index..index, 0);
            *position = *latency;
        } else {
            *position = 0;
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_position() -> u16 {
    SAMPLES.with(|samples| samples.borrow().position)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_position(position: u16) {
    SAMPLES.with(|samples| samples.borrow_mut().position = position);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_rate_and_latency(rate: f32, latency: f32) {
    assert!(rate.is_normal() && rate > 0.0 && latency.is_finite() && latency >= 0.0);

    SAMPLES.with(|samples| {
        let mut samples = samples.borrow_mut();
        samples.rate = rate;
        samples.latency = if ((latency + BUFFER_LIMIT_SECS) * rate).ceil() as usize
            >= SAMPLE_BUFFER_SIZE as usize
        {
            0
        } else {
            (latency * rate).round() as u16
        };
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_mono() {
    SAMPLES.with(|samples| samples.borrow_mut().single_input = true);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_stereo() {
    SAMPLES.with(|samples| samples.borrow_mut().single_input = false);
}

#[cfg(target_arch = "wasm32")]
pub struct WasmApp {
    analysis_chain: AnalysisChain,
    state_info: AudioState,
    last_single_input: bool,
    last_sample_rate: f32,

    shared_state: SharedState,
}

#[cfg(target_arch = "wasm32")]
impl WasmApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut shared_state = SharedState::new();

        build(&cc.egui_ctx, &mut shared_state.spectrogram_texture);

        let analysis_chain = AnalysisChain::new(
            &AnalysisChainConfig::default(),
            AudioState::default().sample_rate,
            false,
        );

        Self {
            analysis_chain,
            state_info: AudioState::default(),
            last_single_input: AudioState::default().input_channels == 1,
            last_sample_rate: AudioState::default().sample_rate,
            shared_state,
        }
    }
    fn update_config(&mut self, single_input: bool, sample_rate: f32) {
        let analysis_config = self.analysis_chain.config();

        self.analysis_chain = AnalysisChain::new(&analysis_config, sample_rate, single_input);

        self.shared_state.invalidate_analysis_history();

        self.state_info = AudioState {
            input_channels: if single_input { 1 } else { 2 },
            sample_rate,
            buffer_size_range: (
                0,
                (SAMPLE_BUFFER_SIZE as u32).min((sample_rate * BUFFER_LIMIT_SECS).floor() as u32),
            ),
            ..Default::default()
        };
        self.last_single_input = single_input;
        self.last_sample_rate = sample_rate;
    }
}

#[cfg(target_arch = "wasm32")]
impl eframe::App for WasmApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        drain_buffers(|single_input, sample_rate, mut buffer| {
            if sample_rate != self.last_sample_rate || single_input != self.last_single_input {
                self.update_config(single_input, sample_rate);
            }

            let (spectrogram, metrics) = (
                &mut self.shared_state.spectrogram,
                &mut self.shared_state.metrics,
            );
            let mut sink = DirectAnalysisSink {
                spectrogram,
                metrics,
            };
            self.analysis_chain.analyze(&mut buffer, &mut sink);
        });

        render(
            ctx,
            &mut self.analysis_chain,
            &self.state_info,
            &mut self.shared_state,
            false,
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct MyPlugin {
    params: Arc<PluginParams>,
    analysis_chain: Arc<FairMutex<Option<AnalysisChain>>>,
    latency_samples: u32,
    analysis_sink: NativeAnalysisSink,
    analysis_receiver: Option<NativeAnalysisReceiver>,
    analysis_frequencies: Arc<FairMutex<Vec<(f32, f32, f32)>>>,
    state_info: Arc<FairMutex<Option<AudioState>>>,
    keepawake: Option<KeepAwake>,
}

#[derive(Params)]
#[cfg(not(target_arch = "wasm32"))]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_FREQUENCY_BINS: usize = 2048;

#[cfg(target_arch = "wasm32")]
const MAX_FREQUENCY_BINS: usize = 1024;

#[cfg(not(target_arch = "wasm32"))]
const SPECTROGRAM_SLICES: usize = 8192;

#[cfg(target_arch = "wasm32")]
const SPECTROGRAM_SLICES: usize = 2048;

#[cfg(not(target_arch = "wasm32"))]
impl Default for MyPlugin {
    fn default() -> Self {
        let (analysis_sink, analysis_receiver) = native_transport(MAX_FREQUENCY_BINS);

        Self {
            params: Arc::new(PluginParams::default()),
            analysis_chain: Arc::new(FairMutex::new(None)),
            latency_samples: 0,
            analysis_sink,
            analysis_receiver: Some(analysis_receiver),
            analysis_frequencies: Arc::new(FairMutex::new(Vec::with_capacity(MAX_FREQUENCY_BINS))),
            state_info: Arc::new(FairMutex::new(None)),
            keepawake: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for PluginParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(1400, 900),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let analysis_receiver = self.analysis_receiver.take()?;

        editor::create(
            self.params.clone(),
            self.analysis_chain.clone(),
            analysis_receiver,
            self.analysis_frequencies.clone(),
            self.state_info.clone(),
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
            audio_io_layout.main_input_channels == NonZero::new(1),
            self.analysis_frequencies.clone(),
        );
        context.set_latency_samples(new_chain.latency_samples);
        self.latency_samples = new_chain.latency_samples;

        *analysis_chain = Some(new_chain);

        self.analysis_sink.reset_stream();

        *self.state_info.lock() = Some(AudioState::new(*audio_io_layout, *buffer_config));

        self.keepawake = keepawake::Builder::default()
            .app_name("KatVisualizer")
            .app_reverse_domain("com.transkatgirl.katvisualizer")
            .reason("Video playback")
            .display(true)
            .idle(true)
            .create()
            .ok();

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

            analysis_chain.analyze(buffer.as_slice(), &mut self.analysis_sink);

            drop(lock);
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl Vst3Plugin for MyPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"transkatgirlVizu";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Analyzer];
}

#[cfg(not(target_arch = "wasm32"))]
nih_export_clap!(MyPlugin);

#[cfg(not(target_arch = "wasm32"))]
nih_export_vst3!(MyPlugin);
