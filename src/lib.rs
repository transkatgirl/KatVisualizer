#![feature(portable_simd)]
#![feature(float_algebraic)]

// TODO: Go through https://nnethercote.github.io/perf-book/title-page.html and apply applicable optimizations

#[cfg(all(not(debug_assertions), not(target_arch = "wasm32")))]
use mimalloc::MiMalloc;

use parking_lot::{FairMutex, Mutex, RwLock};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::{
    num::NonZero,
    time::{Duration, Instant},
};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use std::sync::LazyLock;

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

use crate::{
    analyzer::BetterSpectrogram,
    chain::{AnalysisChain, AnalysisChainConfig},
};

pub mod analyzer;
pub mod chain;
mod editor;

#[derive(Clone, Copy)]
pub(crate) struct AnalysisMetrics {
    processing: Duration,
    finished: Instant,
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
            buffer_size_range: (0, 9600),
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
static SAMPLES: LazyLock<Mutex<(u16, bool, f32, Vec<f32>, Vec<f32>)>> =
    LazyLock::new(|| Mutex::new((0, false, 48000.0, vec![0.0; 9600], vec![0.0; 9600]))); // The WASM module and the sample passer MUST be on the same thread

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn left_sample_buffer() -> Float32Array {
    let lock = LazyLock::force(&SAMPLES).lock();

    unsafe { Float32Array::view(&lock.3) }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn right_sample_buffer() -> Float32Array {
    let lock = LazyLock::force(&SAMPLES).lock();

    unsafe { Float32Array::view(&lock.4) }
}

#[cfg(target_arch = "wasm32")]
pub fn drain_buffers(callback: impl FnOnce(bool, f32, [&mut [f32]; 2])) {
    let mut lock = LazyLock::force(&SAMPLES).lock();

    let (ref mut position, ref single_input, ref rate, ref mut left_samples, ref mut right_samples) =
        *lock;

    let index = (*position).min(9599) as usize;

    callback(
        *single_input,
        *rate,
        [&mut left_samples[0..index], &mut right_samples[0..index]],
    );

    *position = 0;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_position() -> u16 {
    let lock = LazyLock::force(&SAMPLES).lock();

    lock.0
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_position(position: u16) {
    let mut lock = LazyLock::force(&SAMPLES).lock();

    lock.0 = position;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_rate(rate: f32) {
    let mut lock = LazyLock::force(&SAMPLES).lock();

    lock.2 = rate;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_mono(position: u16) {
    let mut lock = LazyLock::force(&SAMPLES).lock();

    lock.1 = true;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_stereo(position: u16) {
    let mut lock = LazyLock::force(&SAMPLES).lock();

    lock.1 = false;
}

#[cfg(target_arch = "wasm32")]
pub struct WasmApp {
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    state_info: Arc<RwLock<Option<AudioState>>>,
    last_single_input: bool,
    last_sample_rate: f32,

    shared_state: SharedState,
}

#[cfg(target_arch = "wasm32")]
impl WasmApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let shared_state = SharedState::new();

        let spectrogram_texture = shared_state.spectrogram_texture.clone();

        build(&cc.egui_ctx, &spectrogram_texture);

        let analysis_frequencies = Arc::new(RwLock::new(Vec::with_capacity(MAX_FREQUENCY_BINS)));

        let analysis_chain = AnalysisChain::new(
            &AnalysisChainConfig::default(),
            AudioState::default().sample_rate,
            false,
            analysis_frequencies.clone(),
        );

        Self {
            analysis_chain: Arc::new(Mutex::new(Some(analysis_chain))),
            analysis_output: Arc::new(FairMutex::new((
                BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
                AnalysisMetrics {
                    processing: Duration::ZERO,
                    finished: Instant::now(),
                },
            ))),
            analysis_frequencies,
            state_info: Arc::new(RwLock::new(Some(AudioState::default()))),
            last_single_input: AudioState::default().input_channels == 1,
            last_sample_rate: AudioState::default().sample_rate,
            shared_state,
        }
    }
    fn update_config(&mut self, single_input: bool, sample_rate: f32) {
        let mut analysis_chain = self.analysis_chain.lock();

        let analysis_config = match &*analysis_chain {
            Some(old_chain) => old_chain.config(),
            None => AnalysisChainConfig::default(),
        };

        *analysis_chain = Some(AnalysisChain::new(
            &analysis_config,
            sample_rate,
            single_input,
            self.analysis_frequencies.clone(),
        ));

        *self.analysis_output.lock() = (
            BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
            AnalysisMetrics {
                processing: Duration::ZERO,
                finished: Instant::now(),
            },
        );

        *self.state_info.write() = Some(AudioState {
            input_channels: if single_input { 1 } else { 2 },
            sample_rate,
            ..Default::default()
        });
        self.last_single_input = single_input;
        self.last_sample_rate = sample_rate;
    }
}

#[cfg(target_arch = "wasm32")]
impl eframe::App for WasmApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        drain_buffers(|single_input, sample_rate, mut buffer| {
            if sample_rate != self.last_sample_rate || single_input != self.last_single_input {
                self.update_config(false, sample_rate);
            }

            let mut lock = self.analysis_chain.lock();

            let mut analysis_chain = lock.as_mut().unwrap();

            analysis_chain.analyze(&mut buffer, &self.analysis_output);
        });

        render(
            ctx,
            &self.analysis_chain,
            &self.analysis_output,
            &self.analysis_frequencies,
            &self.state_info,
            &self.shared_state,
            false,
            |_| {},
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct MyPlugin {
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    latency_samples: u32,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    state_info: Arc<RwLock<Option<AudioState>>>,
    keepawake: Option<KeepAwake>,
}

#[derive(Params)]
#[cfg(not(target_arch = "wasm32"))]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

const MAX_FREQUENCY_BINS: usize = 2048;
const SPECTROGRAM_SLICES: usize = 8192;

#[cfg(not(target_arch = "wasm32"))]
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
            state_info: Arc::new(RwLock::new(None)),
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
        editor::create(
            self.params.clone(),
            self.analysis_chain.clone(),
            self.analysis_output.clone(),
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

        *self.analysis_output.lock() = (
            BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS),
            AnalysisMetrics {
                processing: Duration::ZERO,
                finished: Instant::now(),
            },
        );

        *self.state_info.write() = Some(AudioState::new(*audio_io_layout, *buffer_config));

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

            analysis_chain.analyze(buffer.as_slice(), &self.analysis_output);

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
