#[cfg(not(target_arch = "wasm32"))]
use std::{
    cell::{Cell, UnsafeCell},
    hint,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use crossbeam_utils::CachePadded;

use crate::{
    AnalysisMetrics,
    analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration},
    chain::chunker::{StftHelper, StftInput},
    output::AnalysisSink,
};

mod chunker;

/// One job transferred from the audio thread to the persistent analysis worker.
/// The pointed-to values stay alive and are not accessed by the audio thread
/// until the worker publishes completion.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct AnalysisJob {
    analyzer: *mut BetterAnalyzer,
    samples: *const f32,
    sample_count: usize,
    listening_volume: Option<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AnalysisJob {
    const EMPTY: Self = Self {
        analyzer: ptr::null_mut(),
        samples: ptr::null(),
        sample_count: 0,
        listening_volume: None,
    };
}

#[cfg(not(target_arch = "wasm32"))]
struct AnalysisWorkerShared {
    submission: CachePadded<AnalysisSubmission>,
    completion: CachePadded<AnalysisCompletion>,
    control: CachePadded<AnalysisControl>,
    #[cfg(test)]
    force_failure: AtomicBool,
    #[cfg(test)]
    wake_count: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    park_count: std::sync::atomic::AtomicU64,
}

#[cfg(not(target_arch = "wasm32"))]
struct AnalysisSubmission {
    sequence: AtomicU64,
    job: UnsafeCell<AnalysisJob>,
}

#[cfg(not(target_arch = "wasm32"))]
struct AnalysisCompletion {
    sequence: AtomicU64,
    failed: AtomicBool,
}

#[cfg(not(target_arch = "wasm32"))]
struct AnalysisControl {
    batch_active: AtomicBool,
    stop: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerScheduling {
    Normal,
    #[cfg(not(target_arch = "wasm32"))]
    Realtime {
        max_buffer_frames: u32,
        sample_rate_hz: u32,
        #[cfg(test)]
        force_failure: bool,
    },
}

impl WorkerScheduling {
    #[cfg(not(target_arch = "wasm32"))]
    fn realtime(max_buffer_frames: u32, sample_rate: f32) -> Self {
        Self::Realtime {
            max_buffer_frames,
            sample_rate_hz: sample_rate.round() as u32,
            #[cfg(test)]
            force_failure: false,
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn realtime_with_forced_failure(max_buffer_frames: u32, sample_rate: f32) -> Self {
        Self::Realtime {
            max_buffer_frames,
            sample_rate_hz: sample_rate.round() as u32,
            force_failure: true,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn requests_promotion(self) -> bool {
        matches!(self, Self::Realtime { .. })
    }
}

// SAFETY: the audio thread writes `job` only after observing completion of the
// previous sequence. Its release store transfers the initialized job and its
// pointed-to analyzer to the worker, which only reads it after acquiring that
// sequence. Completion transfers ownership back before the slot is reused.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for AnalysisWorkerShared {}
// SAFETY: access to the only interior-mutable non-atomic field follows the same
// single-producer/single-consumer release/acquire protocol.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for AnalysisWorkerShared {}

/// A persistent analysis worker. For stereo, the audio thread analyzes the left
/// channel concurrently. For mono, it waits while the worker analyzes the input.
#[cfg(not(target_arch = "wasm32"))]
struct AnalysisWorker {
    shared: Arc<AnalysisWorkerShared>,
    thread: Thread,
    submitted_sequence: Cell<u64>,
    batch_woken: Cell<bool>,
    join_handle: Option<JoinHandle<()>>,
    #[cfg(test)]
    promotion_requested: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl AnalysisWorker {
    #[cfg(test)]
    fn new() -> Self {
        Self::new_with_scheduling(0, WorkerScheduling::Normal)
    }

    #[cfg(test)]
    fn new_at_sequence(initial_sequence: u64) -> Self {
        Self::new_with_scheduling(initial_sequence, WorkerScheduling::Normal)
    }

    fn new_with_scheduling(initial_sequence: u64, scheduling: WorkerScheduling) -> Self {
        let shared = Arc::new(AnalysisWorkerShared {
            submission: CachePadded::new(AnalysisSubmission {
                sequence: AtomicU64::new(initial_sequence),
                job: UnsafeCell::new(AnalysisJob::EMPTY),
            }),
            completion: CachePadded::new(AnalysisCompletion {
                sequence: AtomicU64::new(initial_sequence),
                failed: AtomicBool::new(false),
            }),
            control: CachePadded::new(AnalysisControl {
                batch_active: AtomicBool::new(false),
                stop: AtomicBool::new(false),
            }),
            #[cfg(test)]
            force_failure: AtomicBool::new(false),
            #[cfg(test)]
            wake_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            park_count: std::sync::atomic::AtomicU64::new(0),
        });
        let worker_shared = Arc::clone(&shared);
        let (startup_sender, startup_receiver) = sync_channel(1);
        let join_handle = thread::Builder::new()
            .name("katvisualizer-analysis".to_owned())
            .spawn(move || Self::run(worker_shared, initial_sequence, scheduling, startup_sender))
            .expect("private analysis worker can be created");
        let thread = join_handle.thread().clone();
        if let Err(error) = startup_receiver
            .recv()
            .expect("private analysis worker reports startup status")
        {
            nih_plug::nih_warn!(
                "Could not promote the analysis worker to realtime priority: {error}"
            );
        }

        Self {
            shared,
            thread,
            submitted_sequence: Cell::new(initial_sequence),
            batch_woken: Cell::new(false),
            join_handle: Some(join_handle),
            #[cfg(test)]
            promotion_requested: scheduling.requests_promotion(),
        }
    }

    fn run(
        shared: Arc<AnalysisWorkerShared>,
        initial_sequence: u64,
        scheduling: WorkerScheduling,
        startup_sender: SyncSender<Result<(), String>>,
    ) {
        let priority_handle = match scheduling {
            WorkerScheduling::Normal => {
                let _ = startup_sender.send(Ok(()));
                None
            }
            WorkerScheduling::Realtime {
                max_buffer_frames,
                sample_rate_hz,
                #[cfg(test)]
                force_failure,
            } => {
                #[cfg(test)]
                let promotion = if force_failure {
                    Err("forced analysis worker priority promotion failure".to_owned())
                } else {
                    audio_thread_priority::promote_current_thread_to_real_time(
                        max_buffer_frames,
                        sample_rate_hz,
                    )
                    .map_err(|error| error.to_string())
                };
                #[cfg(not(test))]
                let promotion = audio_thread_priority::promote_current_thread_to_real_time(
                    max_buffer_frames,
                    sample_rate_hz,
                )
                .map_err(|error| error.to_string());

                match promotion {
                    Ok(handle) => {
                        let _ = startup_sender.send(Ok(()));
                        Some(handle)
                    }
                    Err(error) => {
                        let _ = startup_sender.send(Err(error));
                        None
                    }
                }
            }
        };

        Self::run_loop(shared, initial_sequence);

        if let Some(handle) = priority_handle
            && let Err(error) = audio_thread_priority::demote_current_thread_from_real_time(handle)
        {
            nih_plug::nih_warn!(
                "Could not demote the analysis worker from realtime priority: {error}"
            );
        }
    }

    fn run_loop(shared: Arc<AnalysisWorkerShared>, initial_sequence: u64) {
        let mut completed_sequence = initial_sequence;

        loop {
            let submitted_sequence = shared.submission.sequence.load(Ordering::Relaxed);
            if submitted_sequence != completed_sequence {
                // The relaxed probe keeps the hot polling loop cheap. Once it
                // observes a new job, acquire its publication before accessing
                // the slot and pointed-to analyzer.
                let submitted_sequence = shared.submission.sequence.load(Ordering::Acquire);
                if submitted_sequence == completed_sequence {
                    continue;
                }

                #[cfg(test)]
                let forced_failure = shared.force_failure.swap(false, Ordering::Relaxed);
                #[cfg(not(test))]
                let forced_failure = false;

                // SAFETY: acquiring a new submission sequence transfers the
                // initialized job and exclusive access to its analyzer.
                let job = unsafe { *shared.submission.job.get() };
                let result = if forced_failure {
                    Err(Box::new("forced analysis worker failure") as Box<dyn std::any::Any + Send>)
                } else {
                    catch_unwind(AssertUnwindSafe(|| unsafe {
                        let analyzer = &mut *job.analyzer;
                        let samples = std::slice::from_raw_parts(job.samples, job.sample_count);
                        analyzer.analyze(samples.iter().copied(), job.listening_volume);
                    }))
                };

                shared
                    .completion
                    .failed
                    .store(result.is_err(), Ordering::Relaxed);
                shared
                    .completion
                    .sequence
                    .store(submitted_sequence, Ordering::Release);
                completed_sequence = submitted_sequence;

                if result.is_err() {
                    return;
                }
                continue;
            }

            if shared.control.stop.load(Ordering::Relaxed) {
                return;
            }

            if shared.control.batch_active.load(Ordering::Acquire) {
                hint::spin_loop();
            } else {
                // `unpark()` has token semantics. A token sent while the worker
                // was still running may make this return immediately, so an
                // idle worker loops back and parks again instead of spinning.
                #[cfg(test)]
                shared.park_count.fetch_add(1, Ordering::Relaxed);
                thread::park();
            }
        }
    }

    fn begin_batch(&self) {
        debug_assert!(
            !self.batch_woken.get(),
            "analysis worker batch was already active"
        );
        self.shared
            .control
            .batch_active
            .store(true, Ordering::Release);
    }

    fn analyze(
        &self,
        analyzer: &mut BetterAnalyzer,
        samples: &[f32],
        listening_volume: Option<f32>,
        concurrent_analysis: impl FnOnce(),
    ) {
        let previous_sequence = self.submitted_sequence.get();
        assert_eq!(
            self.shared.completion.sequence.load(Ordering::Relaxed),
            previous_sequence,
            "analysis worker was not available before submission"
        );
        let submitted_sequence = previous_sequence.wrapping_add(1);

        // SAFETY: completion of the preceding sequence grants the audio thread
        // sole access to the preallocated slot. The release store publishes it.
        unsafe {
            self.shared.submission.job.get().write(AnalysisJob {
                analyzer,
                samples: samples.as_ptr(),
                sample_count: samples.len(),
                listening_volume,
            });
        }
        self.shared
            .submission
            .sequence
            .store(submitted_sequence, Ordering::Release);
        self.submitted_sequence.set(submitted_sequence);
        if !self.batch_woken.replace(true) {
            #[cfg(test)]
            self.shared.wake_count.fetch_add(1, Ordering::Relaxed);
            self.thread.unpark();
        }

        concurrent_analysis();

        loop {
            if self.shared.completion.sequence.load(Ordering::Relaxed) == submitted_sequence
                && self.shared.completion.sequence.load(Ordering::Acquire) == submitted_sequence
            {
                break;
            }
            hint::spin_loop();
        }
        if self.shared.completion.failed.load(Ordering::Relaxed) {
            panic!("analysis worker failed");
        }
    }

    fn finish_batch(&self) {
        self.batch_woken.set(false);
        self.shared
            .control
            .batch_active
            .store(false, Ordering::Release);
    }

    fn is_idle(&self) -> bool {
        self.shared.completion.sequence.load(Ordering::Acquire)
            == self.shared.submission.sequence.load(Ordering::Acquire)
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

    #[cfg(test)]
    fn park_count(&self) -> u64 {
        self.shared.park_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn promotion_requested(&self) -> bool {
        self.promotion_requested
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for AnalysisWorker {
    fn drop(&mut self) {
        self.shared.control.stop.store(true, Ordering::Relaxed);
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
            update_rate_hz: 2048.0, // roughly 0.5x JND for determining if two auditory events are simultaneous
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
    // This field must be dropped before `analyzers`, whose members can be
    // borrowed by jobs running on the worker.
    #[cfg(not(target_arch = "wasm32"))]
    analysis_worker: Option<AnalysisWorker>,
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
        Self::new_with_scheduling(config, sample_rate, single_input, WorkerScheduling::Normal)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new_realtime(
        config: &AnalysisChainConfig,
        sample_rate: f32,
        single_input: bool,
        max_buffer_frames: u32,
    ) -> Self {
        Self::new_with_scheduling(
            config,
            sample_rate,
            single_input,
            WorkerScheduling::realtime(max_buffer_frames, sample_rate),
        )
    }

    fn new_with_scheduling(
        config: &AnalysisChainConfig,
        sample_rate: f32,
        single_input: bool,
        worker_scheduling: WorkerScheduling,
    ) -> Self {
        let mut chunker = StftHelper::new(2, sample_rate.ceil() as usize, 0);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        chunker.set_block_size(chunk_size);
        #[cfg(target_arch = "wasm32")]
        let _ = worker_scheduling;

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
            analysis_worker: (!single_input || worker_scheduling.requests_promotion())
                .then(|| AnalysisWorker::new_with_scheduling(0, worker_scheduling)),
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
        assert!(
            (self.single_input && (buffer.num_channels() == 1 || buffer.num_channels() == 2))
                || (!self.single_input && buffer.num_channels() == 2)
        );

        output.begin_batch();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(analysis_worker) = &self.analysis_worker {
            analysis_worker.begin_batch();
        }
        if self.internal_buffering {
            self.analyze_buffered(buffer, output);
        } else {
            self.analyze_unbuffered(buffer, output);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(analysis_worker) = &self.analysis_worker {
            analysis_worker.finish_batch();
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
        let analysis_worker = self.analysis_worker.as_ref();
        let mut callback = |channel_idx, buffer: &[f32]| {
            if channel_idx == 1 && single_input {
                return;
            }

            if single_input {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(analysis_worker) = analysis_worker {
                    analysis_worker.analyze(&mut analyzers.left, buffer, listening_volume, || {});
                } else {
                    analyzers
                        .left
                        .analyze(buffer.iter().copied(), listening_volume);
                }
                #[cfg(target_arch = "wasm32")]
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
                analysis_worker
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
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(analysis_worker) = &self.analysis_worker {
                analysis_worker.analyze(
                    &mut analyzers.left,
                    buffer[0],
                    self.listening_volume,
                    || {},
                );
            } else {
                analyzers
                    .left
                    .analyze(buffer[0].iter().copied(), self.listening_volume);
            }
            #[cfg(target_arch = "wasm32")]
            analyzers
                .left
                .analyze(buffer[0].iter().copied(), self.listening_volume);
        } else {
            let left_samples: &[f32] = buffer[0];
            let right_samples: &[f32] = buffer[1];
            let listening_volume = self.listening_volume;

            #[cfg(not(target_arch = "wasm32"))]
            self.analysis_worker
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
            self.analysis_worker
                .as_ref()
                .is_none_or(AnalysisWorker::is_idle),
            "analyzers cannot be replaced while the analysis worker is active"
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
        assert!(mono.analysis_worker.is_none());

        let mut stereo = AnalysisChain::new(&config, 48_000.0, false);
        let worker = stereo
            .analysis_worker
            .as_ref()
            .expect("stereo chain has a worker");
        assert_ne!(worker.thread_id(), thread::current().id());
        assert!(worker.is_idle());
        assert!(!worker.promotion_requested());

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
                .analysis_worker
                .as_ref()
                .expect("stereo chain still has its worker")
                .is_idle()
        );
    }

    #[test]
    fn promotion_failure_keeps_stereo_worker_available() {
        let config = test_config(false);
        let scheduling = WorkerScheduling::realtime_with_forced_failure(512, 48_000.0);
        let mut chain = AnalysisChain::new_with_scheduling(&config, 48_000.0, false, scheduling);
        assert!(
            chain
                .analysis_worker
                .as_ref()
                .expect("stereo chain has a worker")
                .promotion_requested()
        );

        let mut left = samples(64, 0.07);
        let mut right = samples(64, 0.11);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        chain.analyze(&mut [&mut left, &mut right], &mut sink);

        assert_eq!(
            spectrogram.newest().duration,
            Duration::from_secs_f64(left.len() as f64 / 48_000.0)
        );
        assert!(
            chain
                .analysis_worker
                .as_ref()
                .expect("stereo chain still has its worker")
                .is_idle()
        );

        drop(chain);
    }

    #[test]
    fn promotion_failure_keeps_mono_worker_available() {
        let config = test_config(false);
        let scheduling = WorkerScheduling::realtime_with_forced_failure(512, 48_000.0);
        let mut chain = AnalysisChain::new_with_scheduling(&config, 48_000.0, true, scheduling);
        let worker = chain
            .analysis_worker
            .as_ref()
            .expect("realtime mono chain has a worker");
        assert!(worker.promotion_requested());
        assert!(worker.is_idle());

        let mut left = samples(64, 0.07);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        chain.analyze(&mut [&mut left], &mut sink);

        let mut analyzer = BetterAnalyzer::new(config.analyzer_config(48_000.0));
        analyzer.analyze(left.iter().copied(), None);
        let mut expected = BetterAnalysis::new(config.resolution);
        expected.update_mono(
            &analyzer,
            config.gain,
            None,
            Duration::from_secs_f64(left.len() as f64 / 48_000.0),
        );
        assert_analysis_eq(spectrogram.newest(), &expected);
        assert!(
            chain
                .analysis_worker
                .as_ref()
                .expect("realtime mono chain still has its worker")
                .is_idle()
        );

        drop(chain);
    }

    #[test]
    fn buffered_realtime_mono_batch_wakes_worker_once() {
        let config = test_config(true);
        let chunk_size = (48_000.0 / config.update_rate_hz).round() as usize;
        let chunk_count = 4;
        let scheduling = WorkerScheduling::realtime_with_forced_failure(512, 48_000.0);
        let mut chain = AnalysisChain::new_with_scheduling(&config, 48_000.0, true, scheduling);
        let mut left = samples(chunk_size * chunk_count, 0.07);
        let mut spectrogram = Spectrogram::new(chunk_count + 1, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };

        chain.analyze(&mut [&mut left], &mut sink);

        let worker = chain.analysis_worker.as_ref().unwrap();
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

        let worker = chain.analysis_worker.as_ref().unwrap();
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
        let worker_id = chain.analysis_worker.as_ref().unwrap().thread_id();
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

        let worker = chain.analysis_worker.as_ref().unwrap();
        assert_eq!(worker.thread_id(), worker_id);
        assert!(worker.is_idle());
    }

    #[test]
    fn worker_failure_is_reported_without_hanging() {
        let config = test_config(false);
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        chain
            .analysis_worker
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

    #[test]
    fn worker_sequences_wrap_without_losing_a_job() {
        let config = test_config(false);
        let worker = AnalysisWorker::new_at_sequence(u64::MAX - 1);
        let analyzer_config = config.analyzer_config(48_000.0);
        let mut left_analyzer = BetterAnalyzer::new(analyzer_config.clone());
        let mut right_analyzer = BetterAnalyzer::new(analyzer_config);
        let left = samples(16, 0.07);
        let right = samples(16, 0.11);

        for _ in 0..2 {
            worker.analyze(&mut right_analyzer, &right, None, || {
                left_analyzer.analyze(left.iter().copied(), None);
            });
            worker.finish_batch();
        }

        assert_eq!(worker.submitted_sequence.get(), 0);
        assert!(worker.is_idle());
        assert_eq!(worker.wake_count(), 2);
    }

    #[test]
    fn rapid_back_to_back_batches_do_not_miss_wakeups() {
        const BATCHES: u64 = 256;

        let config = test_config(false);
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        let mut left = samples(16, 0.07);
        let mut right = samples(16, 0.11);

        for _ in 0..BATCHES {
            chain.analyze(&mut [&mut left, &mut right], &mut sink);
        }

        let worker = chain.analysis_worker.as_ref().unwrap();
        assert_eq!(worker.wake_count(), BATCHES);
        assert!(worker.is_idle());
    }

    #[test]
    fn quiescent_worker_allows_immediate_analyzer_replacement() {
        let config = test_config(false);
        let mut chain = AnalysisChain::new(&config, 48_000.0, false);
        let mut spectrogram = Spectrogram::new(2, config.resolution);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };
        let mut left = samples(16, 0.07);
        let mut right = samples(16, 0.11);

        chain.analyze(&mut [&mut left, &mut right], &mut sink);
        let old = chain.replace_analyzers(Box::new(AnalyzerPair::new(&config, 48_000.0)));

        drop(old);
        assert!(chain.analysis_worker.as_ref().unwrap().is_idle());
    }

    #[test]
    fn parked_worker_shuts_down_cleanly() {
        let worker = AnalysisWorker::new();
        std::thread::yield_now();
        drop(worker);
    }

    #[test]
    fn idle_worker_reparks_after_stale_wake_token() {
        let worker = AnalysisWorker::new();
        let deadline = Instant::now() + Duration::from_secs(5);

        while worker.park_count() < 1 {
            assert!(Instant::now() < deadline, "worker did not initially park");
            std::thread::yield_now();
        }

        // Model an `unpark()` issued for a batch whose work the worker observed
        // without sleeping. The retained token must not leave it spinning.
        worker.thread.unpark();

        while worker.park_count() < 2 {
            assert!(Instant::now() < deadline, "worker did not park again");
            std::thread::yield_now();
        }
    }
}
