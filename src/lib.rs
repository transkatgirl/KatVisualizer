// TODO: Go through https://nnethercote.github.io/perf-book/title-page.html and apply applicable optimizations

#[cfg(all(not(debug_assertions), not(target_arch = "wasm32")))]
use mimalloc::MiMalloc;

use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use parking_lot::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use arc_swap::{ArcSwap, ArcSwapOption};
#[cfg(not(target_arch = "wasm32"))]
use rtrb::{Consumer, Producer, PushError, RingBuffer};

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

#[cfg(not(target_arch = "wasm32"))]
use crate::chain::AnalyzerPair;
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
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) generation: u64,
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
    fn new(audio_io_layout: AudioIOLayout, buffer_config: BufferConfig, generation: u64) -> Self {
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
            generation,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn use_realtime_analysis_worker(process_mode: ProcessMode) -> bool {
    process_mode == ProcessMode::Realtime
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct PreparedAnalyzers {
    generation: u64,
    config: AnalysisChainConfig,
    analyzers: Box<AnalyzerPair>,
    frequencies: Arc<Vec<(f32, f32, f32)>>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct FrequencySnapshot {
    generation: u64,
    frequencies: Vec<(f32, f32, f32)>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct AnalysisUpdate {
    config: AnalysisChainConfig,
    prepared: Option<PreparedAnalyzers>,
    clear_history: bool,
    requires_preparation: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ReclaimAcknowledgement {
    reclaimed: Box<AnalyzerPair>,
    generation: u64,
    frequencies: Arc<Vec<(f32, f32, f32)>>,
    publish_frequencies: bool,
    retry_clear_history: Option<bool>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct AnalysisUpdateSender {
    updates: Producer<AnalysisUpdate>,
    reclaims: Consumer<ReclaimAcknowledgement>,
    pending: Option<AnalysisUpdate>,
    last_queued: AnalysisChainConfig,
    desired_config: Arc<ArcSwap<AnalysisChainConfig>>,
    frequencies: Arc<ArcSwap<FrequencySnapshot>>,
    audio_state: Arc<ArcSwapOption<AudioState>>,
}

/// The render-side endpoints for communication with the analysis thread.
///
/// NIH-plug may destroy and recreate the thread running an editor. Keeping
/// these endpoints on the plugin and only lending access to editor instances
/// prevents a render-thread restart from permanently consuming either endpoint.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct RenderAnalysisBridge {
    analysis_receiver: NativeAnalysisReceiver,
    analysis_updates: AnalysisUpdateSender,
}

#[cfg(not(target_arch = "wasm32"))]
impl AnalysisUpdateSender {
    fn drain_reclaims(&mut self) {
        let mut retry_clear_history = None;
        while let Ok(acknowledgement) = self.reclaims.pop() {
            let ReclaimAcknowledgement {
                reclaimed,
                generation,
                frequencies,
                publish_frequencies,
                retry_clear_history: retry,
            } = acknowledgement;
            if publish_frequencies {
                let current = self.frequencies.load_full();
                if current.generation == generation {
                    let replacement = Arc::new(FrequencySnapshot {
                        generation,
                        frequencies: Arc::unwrap_or_clone(frequencies),
                    });
                    // Reinitialization publishes a newer generation directly.
                    // Only replace the exact snapshot observed above so an old
                    // acknowledgement cannot overwrite that newer metadata.
                    self.frequencies.compare_and_swap(&current, replacement);
                }
            }
            if let Some(clear_history) = retry {
                retry_clear_history = Some(retry_clear_history.unwrap_or(false) || clear_history);
            }
            drop(reclaimed);
        }

        if let Some(retry_clear_history) = retry_clear_history {
            let config = **self.desired_config.load();
            let (prepared, pending_clear_history) = match self.pending.take() {
                Some(mut update) => {
                    let prepared = update
                        .prepared
                        .take()
                        .filter(|prepared| prepared.config.structurally_eq(&config));
                    (prepared, update.clear_history)
                }
                None => (None, false),
            };
            self.pending = Some(AnalysisUpdate {
                config,
                prepared,
                clear_history: retry_clear_history || pending_clear_history,
                // The rejected analyzers were built for an obsolete audio
                // format. `last_queued` already contains this configuration, so
                // explicitly require a new preparation for the current format.
                requires_preparation: true,
            });
        }
    }

    fn prepare_pending(&mut self) {
        let Some(pending) = &mut self.pending else {
            return;
        };
        if !pending.requires_preparation && self.last_queued.structurally_eq(&pending.config) {
            pending.prepared = None;
            return;
        }

        let Some(audio_state) = self.audio_state.load_full() else {
            return;
        };
        let can_reuse = pending.prepared.as_ref().is_some_and(|prepared| {
            prepared.generation == audio_state.generation
                && prepared.config.structurally_eq(&pending.config)
        });
        if can_reuse {
            let prepared = pending.prepared.as_mut().unwrap();
            let chunk_size =
                (audio_state.sample_rate as f64 / pending.config.update_rate_hz).round() as usize;
            prepared.analyzers.resize_buffers(chunk_size);
            prepared.config = pending.config;
        } else {
            let analyzers = Box::new(AnalyzerPair::new(&pending.config, audio_state.sample_rate));
            let frequencies = Arc::new(analyzers.frequencies().to_vec());
            pending.prepared = Some(PreparedAnalyzers {
                generation: audio_state.generation,
                config: pending.config,
                analyzers,
                frequencies,
            });
        }
    }

    fn try_send_pending(&mut self) {
        // Analyzer construction can be expensive. Leave the latest update
        // coalesced until the audio thread has made room for it instead of
        // preparing a value that cannot be sent yet.
        if self.updates.slots() == 0 {
            return;
        }

        self.prepare_pending();
        let Some(update) = self.pending.take() else {
            return;
        };
        if (update.requires_preparation || !self.last_queued.structurally_eq(&update.config))
            && update.prepared.is_none()
        {
            self.pending = Some(update);
            return;
        }
        let config = update.config;
        match self.updates.push(update) {
            Ok(()) => self.last_queued = config,
            Err(PushError::Full(update)) => self.pending = Some(update),
        }
    }

    pub(crate) fn service(&mut self) {
        self.drain_reclaims();
        self.try_send_pending();
    }

    pub(crate) fn stage(&mut self, config: AnalysisChainConfig, clear_history: bool) {
        self.desired_config.store(Arc::new(config));
        self.drain_reclaims();

        if self.pending.is_none() && config == self.last_queued && !clear_history {
            return;
        }

        let (prepared, pending_clear_history, requires_preparation) = match self.pending.take() {
            Some(mut update) => {
                let prepared = update
                    .prepared
                    .take()
                    .filter(|prepared| prepared.config.structurally_eq(&config));
                (prepared, update.clear_history, update.requires_preparation)
            }
            None => (None, false, false),
        };
        let clear_history = clear_history || pending_clear_history;
        self.pending = Some(AnalysisUpdate {
            config,
            prepared,
            clear_history,
            requires_preparation,
        });
        self.try_send_pending();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_pending_analysis_update(
    updates: &mut Consumer<AnalysisUpdate>,
    reclaims: &mut Producer<ReclaimAcknowledgement>,
    audio_format_generation: u64,
    analysis_chain: &mut AnalysisChain,
    analysis_sink: &mut NativeAnalysisSink,
) -> bool {
    let Ok(update) = updates.peek() else {
        return false;
    };
    if update.prepared.is_some() && reclaims.slots() == 0 {
        return false;
    }

    let update = updates.pop().expect("peeked analysis update is available");
    if let Some(prepared) = update.prepared {
        if prepared.generation != audio_format_generation {
            reclaims
                .push(ReclaimAcknowledgement {
                    reclaimed: prepared.analyzers,
                    generation: prepared.generation,
                    frequencies: prepared.frequencies,
                    publish_frequencies: false,
                    retry_clear_history: Some(update.clear_history),
                })
                .unwrap_or_else(|_| unreachable!("reclaim capacity was checked"));
            return false;
        }

        let reclaimed = analysis_chain.replace_analyzers(prepared.analyzers);
        reclaims
            .push(ReclaimAcknowledgement {
                reclaimed,
                generation: prepared.generation,
                frequencies: prepared.frequencies,
                publish_frequencies: true,
                retry_clear_history: None,
            })
            .unwrap_or_else(|_| unreachable!("reclaim capacity was checked"));
    }

    analysis_chain.apply_runtime_config(&update.config);
    if update.clear_history {
        analysis_sink.reset_stream();
    }
    true
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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        let shared_state = SharedState::new();

        build(&cc.egui_ctx);
        let gl = cc
            .gl
            .as_deref()
            .ok_or_else(|| "WebGL2 renderer was not initialized".to_owned())?;
        shared_state.initialize_renderer(gl)?;

        let analysis_chain = AnalysisChain::new(
            &AnalysisChainConfig::default(),
            AudioState::default().sample_rate,
            false,
        );

        Ok(Self {
            analysis_chain,
            state_info: AudioState::default(),
            last_single_input: AudioState::default().input_channels == 1,
            last_sample_rate: AudioState::default().sample_rate,
            shared_state,
        })
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
    analysis_chain: Option<AnalysisChain>,
    analysis_updates: Consumer<AnalysisUpdate>,
    analysis_reclaims: Producer<ReclaimAcknowledgement>,
    render_analysis_bridge: Arc<Mutex<RenderAnalysisBridge>>,
    desired_config: Arc<ArcSwap<AnalysisChainConfig>>,
    audio_format_generation: u64,
    latency_samples: u32,
    analysis_sink: NativeAnalysisSink,
    analysis_frequencies: Arc<ArcSwap<FrequencySnapshot>>,
    state_info: Arc<ArcSwapOption<AudioState>>,
    keepawake: Option<KeepAwake>,
}

#[derive(Params)]
#[cfg(not(target_arch = "wasm32"))]
pub struct PluginParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_FREQUENCY_BINS: usize = 4096;

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
        let (updates, analysis_updates) = RingBuffer::new(1);
        let (analysis_reclaims, reclaims) = RingBuffer::new(1);
        let desired_config = Arc::new(ArcSwap::from_pointee(AnalysisChainConfig::default()));
        let analysis_frequencies = Arc::new(ArcSwap::from_pointee(FrequencySnapshot {
            generation: 0,
            frequencies: Vec::new(),
        }));
        let state_info = Arc::new(ArcSwapOption::empty());

        Self {
            params: Arc::new(PluginParams::default()),
            analysis_chain: None,
            analysis_updates,
            analysis_reclaims,
            render_analysis_bridge: Arc::new(Mutex::new(RenderAnalysisBridge {
                analysis_receiver,
                analysis_updates: AnalysisUpdateSender {
                    updates,
                    reclaims,
                    pending: None,
                    last_queued: AnalysisChainConfig::default(),
                    desired_config: desired_config.clone(),
                    frequencies: analysis_frequencies.clone(),
                    audio_state: state_info.clone(),
                },
            })),
            desired_config,
            audio_format_generation: 0,
            latency_samples: 0,
            analysis_sink,
            analysis_frequencies,
            state_info,
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
            Arc::clone(&self.render_analysis_bridge),
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
        self.audio_format_generation = self.audio_format_generation.wrapping_add(1);
        let analysis_config = **self.desired_config.load();
        let single_input = audio_io_layout.main_input_channels == NonZero::new(1);

        let new_chain = if use_realtime_analysis_worker(buffer_config.process_mode) {
            AnalysisChain::new_realtime(
                &analysis_config,
                buffer_config.sample_rate,
                single_input,
                buffer_config.max_buffer_size,
            )
        } else {
            AnalysisChain::new(&analysis_config, buffer_config.sample_rate, single_input)
        };
        context.set_latency_samples(new_chain.latency_samples);
        self.latency_samples = new_chain.latency_samples;
        self.analysis_frequencies.store(Arc::new(FrequencySnapshot {
            generation: self.audio_format_generation,
            frequencies: new_chain.frequencies().to_vec(),
        }));

        self.analysis_chain = Some(new_chain);

        self.analysis_sink.reset_stream();

        self.state_info.store(Some(Arc::new(AudioState::new(
            *audio_io_layout,
            *buffer_config,
            self.audio_format_generation,
        ))));

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
        if let Some(analysis_chain) = self.analysis_chain.as_mut() {
            apply_pending_analysis_update(
                &mut self.analysis_updates,
                &mut self.analysis_reclaims,
                self.audio_format_generation,
                analysis_chain,
                &mut self.analysis_sink,
            );
        }

        if let Some(analysis_chain) = self.analysis_chain.as_mut() {
            if analysis_chain.latency_samples != self.latency_samples {
                context.set_latency_samples(analysis_chain.latency_samples);
                self.latency_samples = analysis_chain.latency_samples;
            }

            analysis_chain.analyze(buffer.as_slice(), &mut self.analysis_sink);
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod update_tests {
    use super::*;

    #[test]
    fn only_realtime_process_mode_requests_realtime_worker() {
        assert!(use_realtime_analysis_worker(ProcessMode::Realtime));
        assert!(!use_realtime_analysis_worker(ProcessMode::Buffered));
        assert!(!use_realtime_analysis_worker(ProcessMode::Offline));
    }

    fn configured_plugin() -> (
        MyPlugin,
        Arc<Mutex<RenderAnalysisBridge>>,
        AnalysisChainConfig,
    ) {
        let mut plugin = MyPlugin::default();
        let bridge = Arc::clone(&plugin.render_analysis_bridge);
        let base = AnalysisChainConfig {
            resolution: 128,
            masking: false,
            normalize_amplitude: false,
            ..AnalysisChainConfig::default()
        };
        {
            let mut bridge = bridge.lock();
            bridge.analysis_updates.last_queued = base;
            bridge.analysis_updates.desired_config.store(Arc::new(base));
        }
        plugin.audio_format_generation = 1;
        plugin.analysis_chain = Some(AnalysisChain::new(&base, 48_000.0, false));
        plugin
            .analysis_frequencies
            .store(Arc::new(FrequencySnapshot {
                generation: 1,
                frequencies: plugin
                    .analysis_chain
                    .as_ref()
                    .unwrap()
                    .frequencies()
                    .to_vec(),
            }));
        plugin.state_info.store(Some(Arc::new(AudioState {
            buffer_size_range: (32, 512),
            sample_rate: 48_000.0,
            process_mode_title: "Realtime".to_owned(),
            realtime: true,
            input_channels: 2,
            output_channels: 2,
            generation: 1,
        })));
        (plugin, bridge, base)
    }

    fn apply(plugin: &mut MyPlugin) -> bool {
        apply_pending_analysis_update(
            &mut plugin.analysis_updates,
            &mut plugin.analysis_reclaims,
            plugin.audio_format_generation,
            plugin.analysis_chain.as_mut().unwrap(),
            &mut plugin.analysis_sink,
        )
    }

    #[test]
    fn render_bridge_survives_render_and_analysis_thread_replacement() {
        let (mut plugin, bridge, base) = configured_plugin();

        let first_render_bridge = Arc::clone(&bridge);
        std::thread::spawn(move || {
            first_render_bridge
                .lock()
                .analysis_updates
                .stage(AnalysisChainConfig { gain: 1.0, ..base }, false);
        })
        .join()
        .expect("first render thread did not panic");
        assert!(apply(&mut plugin));

        // Model NIH-plug replacing the processing thread and its analysis
        // chain while the editor-side bridge stays owned by the plugin.
        plugin.audio_format_generation = 2;
        plugin.analysis_chain = Some(AnalysisChain::new(&base, 96_000.0, false));
        plugin.state_info.store(Some(Arc::new(AudioState {
            buffer_size_range: (32, 512),
            sample_rate: 96_000.0,
            process_mode_title: "Realtime".to_owned(),
            realtime: true,
            input_channels: 2,
            output_channels: 2,
            generation: 2,
        })));

        let second_render_bridge = Arc::clone(&bridge);
        std::thread::spawn(move || {
            second_render_bridge
                .lock()
                .analysis_updates
                .stage(AnalysisChainConfig { gain: 2.0, ..base }, false);
        })
        .join()
        .expect("replacement render thread did not panic");
        assert!(apply(&mut plugin));
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config().gain, 2.0);
    }

    #[test]
    fn runtime_update_preserves_structural_analyzers() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        let frequencies = plugin
            .analysis_chain
            .as_ref()
            .unwrap()
            .frequencies()
            .as_ptr();
        let updated = AnalysisChainConfig { gain: 6.0, ..base };

        sender.stage(updated, false);
        assert!(apply(&mut plugin));

        let chain = plugin.analysis_chain.as_ref().unwrap();
        assert_eq!(chain.config(), updated);
        assert_eq!(chain.frequencies().as_ptr(), frequencies);
        assert_eq!(plugin.analysis_reclaims.slots(), 1);
    }

    #[test]
    fn structural_update_swaps_and_reclaims_on_ui_side() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        let updated = AnalysisChainConfig {
            resolution: 192,
            ..base
        };

        sender.stage(updated, false);
        assert!(apply(&mut plugin));
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config(), updated);
        assert_eq!(plugin.analysis_reclaims.slots(), 0);

        sender.service();
        assert_eq!(sender.frequencies.load().frequencies.len(), 192);
        assert_eq!(plugin.analysis_reclaims.slots(), 1);
    }

    #[test]
    fn reinitialization_wins_over_stale_frequency_acknowledgement() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..base
            },
            false,
        );
        assert!(apply(&mut plugin));

        let reinitialized = Arc::new(FrequencySnapshot {
            generation: 2,
            frequencies: vec![(1.0, 2.0, 3.0)],
        });
        plugin
            .analysis_frequencies
            .store(Arc::clone(&reinitialized));

        sender.service();

        let published = plugin.analysis_frequencies.load_full();
        assert!(Arc::ptr_eq(&published, &reinitialized));
        assert_eq!(published.generation, 2);
    }

    #[test]
    fn stale_structural_update_is_reclaimed_without_application() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..base
            },
            false,
        );
        plugin.audio_format_generation = 2;

        assert!(!apply(&mut plugin));
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config(), base);
        assert_eq!(plugin.analysis_reclaims.slots(), 0);
        sender.service();
        assert_eq!(plugin.analysis_reclaims.slots(), 1);
    }

    #[test]
    fn structural_update_racing_reinitialization_is_reprepared() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;

        // `initialize()` may have already snapshotted the old desired config
        // while the editor still sees the previous audio format generation.
        let initialization_config = **sender.desired_config.load();
        let updated = AnalysisChainConfig {
            resolution: 192,
            ..base
        };
        sender.stage(updated, false);

        plugin.audio_format_generation = 2;
        plugin.analysis_chain = Some(AnalysisChain::new(&initialization_config, 96_000.0, false));
        plugin.state_info.store(Some(Arc::new(AudioState {
            buffer_size_range: (32, 512),
            sample_rate: 96_000.0,
            process_mode_title: "Realtime".to_owned(),
            realtime: true,
            input_channels: 2,
            output_channels: 2,
            generation: 2,
        })));

        // The obsolete preparation is rejected, then the UI side rebuilds the
        // latest desired configuration for generation 2 and sends it again.
        assert!(!apply(&mut plugin));
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config(), base);
        sender.service();
        assert!(apply(&mut plugin));
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config(), updated);
    }

    #[test]
    fn full_update_queue_coalesces_latest_and_retains_history_reset() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        sender.stage(AnalysisChainConfig { gain: 1.0, ..base }, false);
        sender.stage(AnalysisChainConfig { gain: 2.0, ..base }, true);
        sender.stage(AnalysisChainConfig { gain: 3.0, ..base }, false);

        let first = plugin.analysis_updates.pop().unwrap();
        assert_eq!(first.config.gain, 1.0);
        assert!(!first.clear_history);
        sender.service();
        let coalesced = plugin.analysis_updates.pop().unwrap();
        assert_eq!(coalesced.config.gain, 3.0);
        assert!(coalesced.clear_history);
    }

    #[test]
    fn structural_preparation_waits_for_update_queue_capacity() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        sender.stage(AnalysisChainConfig { gain: 1.0, ..base }, false);
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..base
            },
            false,
        );

        assert!(
            sender
                .pending
                .as_ref()
                .is_some_and(|update| update.prepared.is_none())
        );

        let first = plugin.analysis_updates.pop().unwrap();
        assert_eq!(first.config.gain, 1.0);
        sender.service();

        let structural = plugin.analysis_updates.pop().unwrap();
        assert_eq!(structural.config.resolution, 192);
        assert!(structural.prepared.is_some());
    }

    #[test]
    fn retained_structural_preparation_tracks_latest_chunk_size() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        let runtime = AnalysisChainConfig {
            update_rate_hz: 1_000.0,
            ..base
        };
        sender.stage(runtime, false);
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..base
            },
            false,
        );
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..runtime
            },
            false,
        );

        assert!(apply(&mut plugin));
        sender.service();
        assert!(apply(&mut plugin));

        let mut left = vec![0.0; 48];
        let mut right = vec![0.0; 48];
        plugin
            .analysis_chain
            .as_mut()
            .unwrap()
            .analyze(&mut [&mut left, &mut right], &mut plugin.analysis_sink);
    }

    #[test]
    fn structural_update_waits_for_reclaim_capacity() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        sender.stage(
            AnalysisChainConfig {
                resolution: 192,
                ..base
            },
            false,
        );
        plugin
            .analysis_reclaims
            .push(ReclaimAcknowledgement {
                reclaimed: Box::new(AnalyzerPair::new(&base, 48_000.0)),
                generation: 1,
                frequencies: Arc::new(Vec::new()),
                publish_frequencies: false,
                retry_clear_history: None,
            })
            .unwrap_or_else(|_| unreachable!());

        assert!(!apply(&mut plugin));
        assert_eq!(plugin.analysis_updates.slots(), 1);
        assert_eq!(plugin.analysis_chain.as_ref().unwrap().config(), base);
    }

    #[test]
    fn history_reset_occurs_at_update_block_boundary() {
        let (mut plugin, bridge, base) = configured_plugin();
        let mut bridge = bridge.lock();
        let sender = &mut bridge.analysis_updates;
        let generation = plugin.analysis_sink.generation();
        sender.stage(AnalysisChainConfig { gain: 1.0, ..base }, true);
        assert_eq!(plugin.analysis_sink.generation(), generation);

        assert!(apply(&mut plugin));
        assert_eq!(plugin.analysis_sink.generation(), generation + 1);
    }
}
