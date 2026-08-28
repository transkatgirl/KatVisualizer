#[cfg(not(target_arch = "wasm32"))]
use std::{
    cell::UnsafeCell,
    hint,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use crossbeam_utils::{
    CachePadded,
    sync::{Parker, Unparker},
};

#[cfg(all(not(target_arch = "wasm32"), test))]
use std::sync::atomic::AtomicBool;

use crate::{
    AnalysisMetrics,
    analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration},
    chain::chunker::{StftHelper, StftInput},
    output::AnalysisSink,
};

mod chunker;

#[cfg(not(target_arch = "wasm32"))]
const WORKER_STARTING: u8 = 0;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_IDLE: u8 = 1;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_READY: u8 = 2;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_DONE: u8 = 3;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_FAILED: u8 = 4;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_STOP: u8 = 5;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_BATCH_IDLE: u8 = 6;
#[cfg(not(target_arch = "wasm32"))]
const WORKER_PARK_REQUEST: u8 = 7;

/// One stereo job transferred from the audio thread to the persistent analysis
/// worker. The pointed-to values stay alive and are not accessed by the audio
/// thread until the worker publishes completion.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct StereoJob {
    analyzer: *mut BetterAnalyzer,
    samples: *const f32,
    sample_count: usize,
    listening_volume: Option<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl StereoJob {
    const EMPTY: Self = Self {
        analyzer: ptr::null_mut(),
        samples: ptr::null(),
        sample_count: 0,
        listening_volume: None,
    };
}

#[cfg(not(target_arch = "wasm32"))]
struct StereoWorkerShared {
    state: CachePadded<AtomicU8>,
    job: UnsafeCell<StereoJob>,
    audio_unparker: Unparker,
    #[cfg(test)]
    force_failure: AtomicBool,
    #[cfg(test)]
    wake_count: std::sync::atomic::AtomicU64,
}

// SAFETY: `job` has exactly one producer and one consumer. The producer writes
// it only while `state` is idle or batch-idle, publishes it with a release
// store, and waits for an acquire load of done before reusing the slot or its
// pointed-to data.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for StereoWorkerShared {}
// SAFETY: Access to `job` follows the same release/acquire ownership protocol.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for StereoWorkerShared {}

/// A persistent right-channel worker. The audio thread wakes it, analyzes the
/// left channel itself, and waits for the right channel to finish.
#[cfg(not(target_arch = "wasm32"))]
struct StereoWorker {
    shared: Arc<StereoWorkerShared>,
    thread: Thread,
    audio_parker: Parker,
    join_handle: Option<JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl StereoWorker {
    fn new() -> Self {
        let audio_parker = Parker::new();
        let shared = Arc::new(StereoWorkerShared {
            state: CachePadded::new(AtomicU8::new(WORKER_STARTING)),
            job: UnsafeCell::new(StereoJob::EMPTY),
            audio_unparker: audio_parker.unparker().clone(),
            #[cfg(test)]
            force_failure: AtomicBool::new(false),
            #[cfg(test)]
            wake_count: std::sync::atomic::AtomicU64::new(0),
        });
        let worker_shared = Arc::clone(&shared);
        let join_handle = thread::Builder::new()
            .name("katvisualizer-analysis".to_owned())
            .spawn(move || Self::run(worker_shared))
            .expect("private analysis worker can be created");
        let thread = join_handle.thread().clone();

        // Warm the thread during construction so the first process block only
        // needs the steady-state wakeup path.
        while shared.state.load(Ordering::Acquire) == WORKER_STARTING {
            audio_parker.park();
        }
        assert_eq!(shared.state.load(Ordering::Relaxed), WORKER_IDLE);

        Self {
            shared,
            thread,
            audio_parker,
            join_handle: Some(join_handle),
        }
    }

    fn run(shared: Arc<StereoWorkerShared>) {
        shared.state.store(WORKER_IDLE, Ordering::Release);
        shared.audio_unparker.unpark();
        let mut should_park = true;

        loop {
            if should_park {
                // `unpark()` has token semantics, so a wakeup sent after idle is
                // published but just before this call is kept.
                thread::park();
                should_park = false;
            }

            match shared.state.load(Ordering::Acquire) {
                WORKER_READY => {
                    #[cfg(test)]
                    if shared.force_failure.swap(false, Ordering::Relaxed) {
                        let _ = shared.state.compare_exchange(
                            WORKER_READY,
                            WORKER_FAILED,
                            Ordering::Release,
                            Ordering::Acquire,
                        );
                        shared.audio_unparker.unpark();
                        return;
                    }

                    // SAFETY: the acquire load transfers the initialized job and
                    // exclusive access to its right analyzer to this thread. The
                    // sample slice remains immutable until completion.
                    let job = unsafe { *shared.job.get() };
                    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
                        let analyzer = &mut *job.analyzer;
                        let samples = std::slice::from_raw_parts(job.samples, job.sample_count);
                        analyzer.analyze(samples.iter().copied(), job.listening_volume);
                    }));
                    let completed_state = if result.is_ok() {
                        WORKER_DONE
                    } else {
                        WORKER_FAILED
                    };

                    // Drop may request shutdown while a job is in flight. Do not
                    // overwrite that request or park again in that case.
                    if shared
                        .state
                        .compare_exchange(
                            WORKER_READY,
                            completed_state,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_err()
                    {
                        debug_assert_eq!(shared.state.load(Ordering::Relaxed), WORKER_STOP);
                        return;
                    }

                    shared.audio_unparker.unpark();

                    if result.is_err() {
                        return;
                    }
                }
                WORKER_PARK_REQUEST => {
                    // Drop may request shutdown while the audio thread is ending
                    // a batch. Never overwrite that request with idle.
                    match shared.state.compare_exchange(
                        WORKER_PARK_REQUEST,
                        WORKER_IDLE,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            shared.audio_unparker.unpark();
                            should_park = true;
                        }
                        Err(WORKER_STOP) => return,
                        Err(state) => panic!("invalid analysis worker park state {state}"),
                    }
                }
                WORKER_STOP => return,
                WORKER_IDLE => should_park = true,
                WORKER_DONE | WORKER_BATCH_IDLE => hint::spin_loop(),
                state => panic!("invalid analysis worker state {state}"),
            }
        }
    }

    fn analyze(
        &self,
        right_analyzer: &mut BetterAnalyzer,
        right_samples: &[f32],
        listening_volume: Option<f32>,
        analyze_left: impl FnOnce(),
    ) {
        let previous_state = self.shared.state.load(Ordering::Acquire);
        assert!(
            previous_state == WORKER_IDLE || previous_state == WORKER_BATCH_IDLE,
            "analysis worker was not available before submission"
        );

        // SAFETY: idle and batch-idle grant the sole producer exclusive access
        // to the preallocated slot. The release store below publishes the job.
        unsafe {
            self.shared.job.get().write(StereoJob {
                analyzer: right_analyzer,
                samples: right_samples.as_ptr(),
                sample_count: right_samples.len(),
                listening_volume,
            });
        }
        self.shared.state.store(WORKER_READY, Ordering::Release);
        if previous_state == WORKER_IDLE {
            #[cfg(test)]
            self.shared.wake_count.fetch_add(1, Ordering::Relaxed);
            self.thread.unpark();
        }

        analyze_left();

        loop {
            match self.shared.state.load(Ordering::Acquire) {
                WORKER_DONE => {
                    self.shared
                        .state
                        .store(WORKER_BATCH_IDLE, Ordering::Release);
                    return;
                }
                WORKER_FAILED => panic!("right-channel analysis worker failed"),
                WORKER_READY => self.audio_parker.park(),
                state => panic!("invalid analysis worker completion state {state}"),
            }
        }
    }

    fn finish_batch(&self) {
        loop {
            match self.shared.state.load(Ordering::Acquire) {
                // A batch with no stereo jobs leaves the worker parked.
                WORKER_IDLE => return,
                WORKER_BATCH_IDLE => {
                    let _ = self.shared.state.compare_exchange(
                        WORKER_BATCH_IDLE,
                        WORKER_PARK_REQUEST,
                        Ordering::Release,
                        Ordering::Acquire,
                    );
                }
                WORKER_PARK_REQUEST => self.audio_parker.park(),
                WORKER_FAILED => panic!("right-channel analysis worker failed"),
                state => panic!("invalid analysis worker batch completion state {state}"),
            }
        }
    }

    fn is_idle(&self) -> bool {
        self.shared.state.load(Ordering::Acquire) == WORKER_IDLE
    }

    #[cfg(test)]
    fn thread_id(&self) -> thread::ThreadId {
        self.thread.id()
    }

    #[cfg(test)]
    fn force_failure(&self) {
        self.shared.force_failure.store(true, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn wake_count(&self) -> u64 {
        self.shared.wake_count.load(Ordering::Relaxed)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for StereoWorker {
    fn drop(&mut self) {
        self.shared.state.store(WORKER_STOP, Ordering::Release);
        self.thread.unpark();
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnalysisChainConfig {
    pub(crate) gain: f32,
    pub(crate) listening_volume: f32,
    pub(crate) normalize_amplitude: bool,
    pub(crate) masking: bool,
    pub(crate) approximate_masking: bool,
    pub(crate) internal_buffering: bool,
    pub(crate) strict_synchronization: bool,
    pub(crate) update_rate_hz: f64,
    pub(crate) latency_offset: Duration,

    pub(crate) resolution: usize,
    pub(crate) start_frequency: f32,
    pub(crate) end_frequency: f32,
    pub(crate) erb_frequency_scale: bool,
    pub(crate) erb_time_resolution: bool,
    pub(crate) erb_bandwidth_divisor: f32,
    pub(crate) time_resolution_clamp: (f32, f32),
    pub(crate) q_time_resolution: f32,
    pub(crate) nc_method: bool,
    pub(crate) strict_nc: bool,
}

impl AnalysisChainConfig {
    pub(crate) fn structurally_eq(&self, other: &Self) -> bool {
        self.resolution == other.resolution
            && self.start_frequency == other.start_frequency
            && self.end_frequency == other.end_frequency
            && self.erb_frequency_scale == other.erb_frequency_scale
            && self.erb_time_resolution == other.erb_time_resolution
            && self.erb_bandwidth_divisor == other.erb_bandwidth_divisor
            && self.time_resolution_clamp == other.time_resolution_clamp
            && self.q_time_resolution == other.q_time_resolution
            && self.nc_method == other.nc_method
            && self.strict_nc == other.strict_nc
            && self.masking == other.masking
            && self.approximate_masking == other.approximate_masking
    }

    fn analyzer_config(&self, sample_rate: f32) -> BetterAnalyzerConfiguration {
        BetterAnalyzerConfiguration {
            resolution: self.resolution,
            start_frequency: self.start_frequency,
            end_frequency: self.end_frequency,
            erb_frequency_scale: self.erb_frequency_scale,
            sample_rate,
            erb_time_resolution: self.erb_time_resolution,
            erb_bandwidth_divisor: self.erb_bandwidth_divisor,
            time_resolution_clamp: self.time_resolution_clamp,
            q_time_resolution: self.q_time_resolution,
            nc_method: self.nc_method,
            strict_nc: self.strict_nc,
            masking: self.masking,
            approximate_masking: self.approximate_masking,
        }
    }
}

impl Default for AnalysisChainConfig {
    fn default() -> Self {
        Self {
            gain: 0.0,
            listening_volume: 100.0,
            normalize_amplitude: true,
            masking: true,
            #[cfg(not(target_arch = "wasm32"))]
            approximate_masking: false,
            #[cfg(target_arch = "wasm32")]
            approximate_masking: true,
            internal_buffering: true,
            strict_synchronization: true,
            #[cfg(not(target_arch = "wasm32"))]
            update_rate_hz: 1536.0, // round((1408 * ((1400 / 900) * 0.75)) / 256) * 256
            #[cfg(target_arch = "wasm32")]
            update_rate_hz: 512.0,
            #[cfg(not(target_arch = "wasm32"))]
            resolution: 1408, // MUST be a multiple of 64
            // Total perceptible pitch steps in human hearing = ~1,400
            #[cfg(target_arch = "wasm32")]
            resolution: 448, // MUST be a multiple of 64
            latency_offset: Duration::ZERO,

            start_frequency: BetterAnalyzerConfiguration::default().start_frequency,
            end_frequency: BetterAnalyzerConfiguration::default().end_frequency,
            erb_frequency_scale: BetterAnalyzerConfiguration::default().erb_frequency_scale,
            erb_time_resolution: BetterAnalyzerConfiguration::default().erb_time_resolution,
            erb_bandwidth_divisor: BetterAnalyzerConfiguration::default().erb_bandwidth_divisor,
            time_resolution_clamp: BetterAnalyzerConfiguration::default().time_resolution_clamp,
            q_time_resolution: BetterAnalyzerConfiguration::default().q_time_resolution,
            nc_method: BetterAnalyzerConfiguration::default().nc_method,
            strict_nc: BetterAnalyzerConfiguration::default().strict_nc,
        }
    }
}

pub(crate) struct AnalyzerPair {
    left: BetterAnalyzer,
    right: BetterAnalyzer,
    left_buffer: Vec<f32>,
    right_buffer: Vec<f32>,
}

impl AnalyzerPair {
    pub(crate) fn new(config: &AnalysisChainConfig, sample_rate: f32) -> Self {
        let analyzer_config = config.analyzer_config(sample_rate);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        let scratch_capacity = sample_rate.ceil() as usize;
        let mut left_buffer = Vec::with_capacity(scratch_capacity);
        let mut right_buffer = Vec::with_capacity(scratch_capacity);
        left_buffer.resize(chunk_size, 0.0);
        right_buffer.resize(chunk_size, 0.0);

        Self {
            left: BetterAnalyzer::new(analyzer_config.clone()),
            right: BetterAnalyzer::new(analyzer_config),
            left_buffer,
            right_buffer,
        }
    }

    pub(crate) fn frequencies(&self) -> &[(f32, f32, f32)] {
        self.left.frequencies()
    }

    pub(crate) fn resize_buffers(&mut self, chunk_size: usize) {
        assert!(chunk_size <= self.left_buffer.capacity());
        assert!(chunk_size <= self.right_buffer.capacity());
        self.left_buffer.resize(chunk_size, 0.0);
        self.right_buffer.resize(chunk_size, 0.0);
    }
}

pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    // This field must be dropped before `analyzers`, whose right-hand member is
    // borrowed by jobs running on the worker.
    #[cfg(not(target_arch = "wasm32"))]
    stereo_worker: Option<StereoWorker>,
    analyzers: Box<AnalyzerPair>,
    gain: f32,
    internal_buffering: bool,
    strict_synchronization: bool,
    update_rate: f64,
    listening_volume: Option<f32>,
    masking: bool,
    pub(crate) latency_samples: u32,
    additional_latency: Duration,
    sample_rate: f32,
    chunk_size: usize,
    chunk_duration: Duration,
    single_input: bool,
}

impl AnalysisChain {
    pub(crate) fn new(config: &AnalysisChainConfig, sample_rate: f32, single_input: bool) -> Self {
        let mut chunker = StftHelper::new(2, sample_rate.ceil() as usize, 0);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        chunker.set_block_size(chunk_size);

        Self {
            sample_rate,
            internal_buffering: config.internal_buffering,
            strict_synchronization: config.strict_synchronization,
            latency_samples: if config.internal_buffering {
                chunker.latency_samples()
            } else {
                0
            } + (config.latency_offset.as_secs_f64() * sample_rate as f64) as u32,
            additional_latency: config.latency_offset,
            chunker,
            #[cfg(not(target_arch = "wasm32"))]
            stereo_worker: (!single_input).then(StereoWorker::new),
            analyzers: Box::new(AnalyzerPair::new(config, sample_rate)),
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
            single_input,
        }
    }

    pub(crate) fn frequencies(&self) -> &[(f32, f32, f32)] {
        self.analyzers.frequencies()
    }
    pub(crate) fn analyze<S: AnalysisSink>(&mut self, buffer: &mut [&mut [f32]], output: &mut S) {
        assert!(buffer.num_channels() == 1 || buffer.num_channels() == 2);

        output.begin_batch();
        if self.internal_buffering {
            self.analyze_buffered(buffer, output);
        } else {
            self.analyze_unbuffered(buffer, output);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(stereo_worker) = &self.stereo_worker {
            stereo_worker.finish_batch();
        }
        output.finish_batch();
    }
    fn analyze_buffered<S: AnalysisSink>(&mut self, buffer: &mut [&mut [f32]], output: &mut S) {
        let mut finished = Instant::now();
        let analyzers = &mut *self.analyzers;
        let single_input = self.single_input;
        let listening_volume = self.listening_volume;
        let gain = self.gain;
        let chunk_duration = self.chunk_duration;
        #[cfg(not(target_arch = "wasm32"))]
        let stereo_worker = self.stereo_worker.as_ref();
        let mut callback = |channel_idx, buffer: &[f32]| {
            if channel_idx == 1 && single_input {
                return;
            }

            if single_input {
                analyzers
                    .left
                    .analyze(buffer.iter().copied(), listening_volume);
            } else {
                if channel_idx == 0 {
                    analyzers.left_buffer.copy_from_slice(buffer);
                    return;
                }

                analyzers.right_buffer.copy_from_slice(buffer);
                let AnalyzerPair {
                    left,
                    right,
                    left_buffer,
                    right_buffer,
                } = analyzers;

                #[cfg(not(target_arch = "wasm32"))]
                stereo_worker
                    .expect("stereo chains have an analysis worker")
                    .analyze(right, right_buffer, listening_volume, || {
                        left.analyze(left_buffer.iter().copied(), listening_volume);
                    });
                #[cfg(target_arch = "wasm32")]
                {
                    left.analyze(left_buffer.iter().copied(), listening_volume);
                    right.analyze(right_buffer.iter().copied(), listening_volume);
                }
            }

            if channel_idx == 1 || (channel_idx == 0 && single_input) {
                output.submit(chunk_duration, |analysis_output| {
                    if single_input {
                        analysis_output.update_mono(
                            &analyzers.left,
                            gain,
                            listening_volume,
                            chunk_duration,
                        );
                    } else {
                        analysis_output.update_stereo(
                            &analyzers.left,
                            &analyzers.right,
                            gain,
                            listening_volume,
                            chunk_duration,
                        );
                    }

                    AnalysisMetrics {
                        processing: finished.elapsed(),
                    }
                });

                finished = Instant::now();
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        if self.strict_synchronization {
            self.chunker
                .process_overlap_add(buffer, 1, |channel_idx, buffer| {
                    callback(channel_idx, buffer);
                });
        } else {
            self.chunker
                .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                    callback(channel_idx, buffer);
                });
        }

        #[cfg(target_arch = "wasm32")]
        self.chunker
            .process_analyze_only(buffer, 1, |channel_idx, buffer| {
                callback(channel_idx, buffer);
            });
    }
    fn analyze_unbuffered<S: AnalysisSink>(&mut self, buffer: &mut [&mut [f32]], output: &mut S) {
        let finished = Instant::now();
        let analyzers = &mut *self.analyzers;

        if self.single_input {
            analyzers
                .left
                .analyze(buffer[0].iter().copied(), self.listening_volume);
        } else {
            let left_samples: &[f32] = buffer[0];
            let right_samples: &[f32] = buffer[1];
            let listening_volume = self.listening_volume;

            #[cfg(not(target_arch = "wasm32"))]
            self.stereo_worker
                .as_ref()
                .expect("stereo chains have an analysis worker")
                .analyze(
                    &mut analyzers.right,
                    right_samples,
                    listening_volume,
                    || {
                        analyzers
                            .left
                            .analyze(left_samples.iter().copied(), listening_volume);
                    },
                );
            #[cfg(target_arch = "wasm32")]
            {
                analyzers
                    .left
                    .analyze(left_samples.iter().copied(), listening_volume);
                analyzers
                    .right
                    .analyze(right_samples.iter().copied(), listening_volume);
            }
        }

        let chunk_duration =
            Duration::from_secs_f64(buffer.num_samples() as f64 / self.sample_rate as f64);

        output.submit(chunk_duration, |analysis_output| {
            if self.single_input {
                analysis_output.update_mono(
                    &analyzers.left,
                    self.gain,
                    self.listening_volume,
                    chunk_duration,
                );
            } else {
                analysis_output.update_stereo(
                    &analyzers.left,
                    &analyzers.right,
                    self.gain,
                    self.listening_volume,
                    chunk_duration,
                );
            }

            AnalysisMetrics {
                processing: finished.elapsed(),
            }
        });
    }
    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn config(&self) -> AnalysisChainConfig {
        let analyzer_config = self.analyzers.left.config();

        AnalysisChainConfig {
            gain: self.gain,
            listening_volume: self
                .listening_volume
                .unwrap_or(AnalysisChainConfig::default().listening_volume),
            normalize_amplitude: self.listening_volume.is_some(),
            masking: self.masking,
            approximate_masking: analyzer_config.approximate_masking,
            internal_buffering: self.internal_buffering,
            strict_synchronization: self.strict_synchronization,
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
            strict_nc: analyzer_config.strict_nc,
        }
    }

    pub(crate) fn apply_runtime_config(&mut self, config: &AnalysisChainConfig) {
        self.gain = config.gain;
        self.listening_volume = if config.normalize_amplitude {
            Some(config.listening_volume)
        } else {
            None
        };
        self.masking = config.masking;

        if self.update_rate != config.update_rate_hz {
            self.chunk_size = (self.sample_rate as f64 / config.update_rate_hz).round() as usize;
            self.chunker.set_block_size(self.chunk_size);
            self.analyzers.resize_buffers(self.chunk_size);
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
            self.chunk_duration =
                Duration::from_secs_f64(self.chunk_size as f64 / self.sample_rate as f64);
        } else if self.additional_latency != config.latency_offset
            || self.internal_buffering != config.internal_buffering
        {
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
        }

        self.internal_buffering = config.internal_buffering;
        self.strict_synchronization = config.strict_synchronization;
        self.update_rate = config.update_rate_hz;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn replace_analyzers(&mut self, analyzers: Box<AnalyzerPair>) -> Box<AnalyzerPair> {
        debug_assert!(
            self.stereo_worker
                .as_ref()
                .is_none_or(StereoWorker::is_idle),
            "analyzers cannot be replaced while the stereo worker is active"
        );
        std::mem::replace(&mut self.analyzers, analyzers)
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn update_config(&mut self, config: &AnalysisChainConfig) {
        if !self.config().structurally_eq(config) {
            self.analyzers = Box::new(AnalyzerPair::new(config, self.sample_rate));
        }
        self.apply_runtime_config(config);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{
        analyzer::{BetterAnalysis, Spectrogram},
        output::DirectAnalysisSink,
    };

    fn test_config(buffered: bool) -> AnalysisChainConfig {
        AnalysisChainConfig {
            resolution: 128,
            masking: false,
            normalize_amplitude: false,
            internal_buffering: buffered,
            strict_synchronization: false,
            update_rate_hz: 1_000.0,
            ..AnalysisChainConfig::default()
        }
    }

    fn samples(length: usize, multiplier: f32) -> Vec<f32> {
        (0..length)
            .map(|index| ((index as f32 * multiplier).sin() * 0.25) + 0.01)
            .collect()
    }

    fn assert_analysis_eq(actual: &BetterAnalysis, expected: &BetterAnalysis) {
        assert_eq!(actual.duration, expected.duration);
        assert_eq!(actual.data.len(), expected.data.len());
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        for (actual, expected) in actual.masking.iter().zip(&expected.masking) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
    }

    fn run_chain(
        config: &AnalysisChainConfig,
        single_input: bool,
        left: &mut [f32],
        right: Option<&mut [f32]>,
    ) -> BetterAnalysis {
        let mut chain = AnalysisChain::new(config, 48_000.0, single_input);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        match right {
            Some(right) => chain.analyze(&mut [left, right], &mut sink),
            None => chain.analyze(&mut [left], &mut sink),
        }
        let newest = spectrogram.newest();
        let mut result = BetterAnalysis::new(config.resolution);
        result.data.extend_from_slice(&newest.data);
        result.masking.extend_from_slice(&newest.masking);
        result.masking_mean = newest.masking_mean;
        result.duration = newest.duration;
        result
    }

    #[test]
    fn mono_analysis_matches_sequential_analyzer() {
        let config = test_config(false);
        let mut left = samples(256, 0.07);
        let actual = run_chain(&config, true, &mut left, None);
        let mut analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        analyzer.analyze(left.iter().copied(), None);
        let mut expected = BetterAnalysis::new(config.resolution);
        expected.update_mono(
            &analyzer,
            config.gain,
            None,
            Duration::from_secs_f64(left.len() as f64 / 48_000.0),
        );
        assert_analysis_eq(&actual, &expected);
    }

    #[test]
    fn unbuffered_stereo_matches_sequential_analyzers() {
        let config = test_config(false);
        let mut left = samples(256, 0.07);
        let mut right = samples(256, 0.11);
        let actual = run_chain(&config, false, &mut left, Some(&mut right));
        let mut left_analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        let mut right_analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        left_analyzer.analyze(left.iter().copied(), None);
        right_analyzer.analyze(right.iter().copied(), None);
        let mut expected = BetterAnalysis::new(config.resolution);
        expected.update_stereo(
            &left_analyzer,
            &right_analyzer,
            config.gain,
            None,
            Duration::from_secs_f64(left.len() as f64 / 48_000.0),
        );
        assert_analysis_eq(&actual, &expected);
    }

    #[test]
    fn buffered_stereo_matches_sequential_analyzers() {
        let config = test_config(true);
        let chunk_size = (48_000.0 / config.update_rate_hz).round() as usize;
        let mut left = samples(chunk_size, 0.07);
        let mut right = samples(chunk_size, 0.11);
        let actual = run_chain(&config, false, &mut left, Some(&mut right));
        let mut left_analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        let mut right_analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        left_analyzer.analyze(left.iter().copied(), None);
        right_analyzer.analyze(right.iter().copied(), None);
        let mut expected = BetterAnalysis::new(config.resolution);
        expected.update_stereo(
            &left_analyzer,
            &right_analyzer,
            config.gain,
            None,
            Duration::from_secs_f64(chunk_size as f64 / 48_000.0),
        );
        assert_analysis_eq(&actual, &expected);
    }

    #[test]
    fn worker_lifecycle_matches_channel_layout() {
        let config = test_config(false);
        let mono = AnalysisChain::new(&config, 48_000.0, true);
        assert!(mono.stereo_worker.is_none());

        let mut stereo = AnalysisChain::new(&config, 48_000.0, false);
        let worker = stereo
            .stereo_worker
            .as_ref()
            .expect("stereo chain has a worker");
        assert_ne!(worker.thread_id(), thread::current().id());
        assert!(worker.is_idle());

        let mut left = samples(64, 0.07);
        let mut right = samples(64, 0.11);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        stereo.analyze(&mut [&mut left, &mut right], &mut sink);

        assert!(
            stereo
                .stereo_worker
                .as_ref()
                .expect("stereo chain still has its worker")
                .is_idle()
        );
    }

    #[test]
    fn buffered_stereo_batch_wakes_worker_once() {
        let config = test_config(true);
        let chunk_size = (48_000.0 / config.update_rate_hz).round() as usize;
        let chunk_count = 4;
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        let mut left = samples(chunk_size * chunk_count, 0.07);
        let mut right = samples(chunk_size * chunk_count, 0.11);
        let mut spectrogram = Spectrogram::new(chunk_count + 1, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };

        chain.analyze(&mut [&mut left, &mut right], &mut sink);

        let worker = chain.stereo_worker.as_ref().unwrap();
        assert_eq!(worker.wake_count(), 1);
        assert!(worker.is_idle());
        assert!(
            spectrogram
                .newest_to_oldest()
                .take(chunk_count)
                .all(|analysis| analysis.duration == chain.chunk_duration)
        );
    }

    #[test]
    fn structural_swap_reuses_worker_across_repeated_jobs() {
        let config = test_config(false);
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        let worker_id = chain.stereo_worker.as_ref().unwrap().thread_id();
        let old = chain.replace_analyzers(Box::new(AnalyzerPair::new(&config, 48_000.0)));
        drop(old);

        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        let mut left = samples(16, 0.07);
        let mut right = samples(16, 0.11);
        for _ in 0..128 {
            chain.analyze(&mut [&mut left, &mut right], &mut sink);
        }

        let worker = chain.stereo_worker.as_ref().unwrap();
        assert_eq!(worker.thread_id(), worker_id);
        assert!(worker.is_idle());
    }

    #[test]
    fn worker_failure_is_reported_without_hanging() {
        let config = test_config(false);
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        chain
            .stereo_worker
            .as_ref()
            .expect("stereo chain has a worker")
            .force_failure();

        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        let mut left = samples(16, 0.07);
        let mut right = samples(16, 0.11);
        let result = catch_unwind(AssertUnwindSafe(|| {
            chain.analyze(&mut [&mut left, &mut right], &mut sink);
        }));

        assert!(result.is_err());
    }
}
