use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::{FairMutex, Mutex, RwLock};
use threadpool::ThreadPool;

use crate::{
    AnalysisMetrics,
    analyzer::{BetterAnalyzer, BetterAnalyzerConfiguration, BetterSpectrogram},
    chain::chunker::{StftHelper, StftInput},
};

mod chunker;

#[derive(Clone)]
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

impl Default for AnalysisChainConfig {
    fn default() -> Self {
        Self {
            gain: 0.0,
            listening_volume: 90.0,
            normalize_amplitude: true,
            masking: true,
            approximate_masking: false,
            internal_buffering: true,
            strict_synchronization: true,
            update_rate_hz: 2048.0,
            resolution: 1024,
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

#[allow(clippy::type_complexity)]
pub(crate) struct AnalysisChain {
    chunker: StftHelper<0>,
    left_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
    right_analyzer: Arc<Mutex<(Vec<f32>, BetterAnalyzer)>>,
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
    analyzer_pool: ThreadPool,
    pub(crate) frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
}

impl AnalysisChain {
    pub(crate) fn new(
        config: &AnalysisChainConfig,
        sample_rate: f32,
        single_input: bool,
        frequency_list_container: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    ) -> Self {
        let analyzer_config = BetterAnalyzerConfiguration {
            resolution: config.resolution,
            start_frequency: config.start_frequency,
            end_frequency: config.end_frequency,
            erb_frequency_scale: config.erb_frequency_scale,
            sample_rate,
            erb_time_resolution: config.erb_time_resolution,
            erb_bandwidth_divisor: config.erb_bandwidth_divisor,
            time_resolution_clamp: config.time_resolution_clamp,
            q_time_resolution: config.q_time_resolution,
            nc_method: config.nc_method,
            strict_nc: config.strict_nc,
            masking: config.masking,
            approximate_masking: config.approximate_masking,
        };

        let left_analyzer = BetterAnalyzer::new(analyzer_config.clone());
        let right_analyzer = BetterAnalyzer::new(analyzer_config);

        let mut chunker = StftHelper::new(2, sample_rate.ceil() as usize, 0);
        let chunk_size = (sample_rate as f64 / config.update_rate_hz).round() as usize;
        chunker.set_block_size(chunk_size);

        {
            let mut frequencies = frequency_list_container.write();
            frequencies.clear();
            frequencies.extend(
                left_analyzer
                    .frequencies()
                    .iter()
                    .map(|(a, b, c)| (*a, *b, *c)),
            );
        }

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
            frequencies: frequency_list_container,
            left_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], left_analyzer))),
            right_analyzer: Arc::new(Mutex::new((vec![0.0; chunk_size], right_analyzer))),
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
            analyzer_pool: ThreadPool::new(2),
        }
    }
    pub(crate) fn analyze(
        &mut self,
        buffer: &mut [&mut [f32]],
        output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
    ) {
        assert!(buffer.num_channels() == 1 || buffer.num_channels() == 2);

        if self.internal_buffering {
            self.analyze_buffered(buffer, output);
        } else {
            self.analyze_unbuffered(buffer, output);
        }
    }
    fn analyze_buffered(
        &mut self,
        buffer: &mut [&mut [f32]],
        output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
    ) {
        let mut finished = Instant::now();

        let mut callback = |channel_idx, buffer: &[f32]| {
            if channel_idx == 1 && self.single_input {
                return;
            }

            if self.single_input {
                let mut lock = self.left_analyzer.lock();
                let (ref _buffer, ref mut analyzer) = *lock;

                analyzer.analyze(buffer.iter().copied(), self.listening_volume);
            } else {
                let analyzer = if channel_idx == 0 {
                    self.left_analyzer.clone()
                } else {
                    self.right_analyzer.clone()
                };
                let listening_volume = self.listening_volume;

                analyzer.lock().0.copy_from_slice(buffer);

                self.analyzer_pool.execute(move || {
                    let mut lock = analyzer.lock();
                    let (ref mut buffer, ref mut analyzer) = *lock;

                    analyzer.analyze(buffer.iter().copied(), listening_volume);
                });
            }

            if channel_idx == 1 || (channel_idx == 0 && self.single_input) {
                let (ref mut spectrogram, ref mut metrics) = *output.lock();

                self.analyzer_pool.join();
                let left_lock = self.left_analyzer.lock();
                let right_lock = self.right_analyzer.lock();
                let left_analyzer = &left_lock.1;
                let right_analyzer = &right_lock.1;

                spectrogram.update_fn(|analysis_output| {
                    if self.single_input {
                        analysis_output.update_mono(
                            left_analyzer,
                            self.gain,
                            self.listening_volume,
                            self.chunk_duration,
                        );
                    } else {
                        analysis_output.update_stereo(
                            left_analyzer,
                            right_analyzer,
                            self.gain,
                            self.listening_volume,
                            self.chunk_duration,
                        );
                    }
                });

                let now = Instant::now();
                metrics.processing = now.duration_since(finished);
                metrics.finished = now;

                finished = now;
            }
        };

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
    }
    fn analyze_unbuffered(
        &mut self,
        buffer: &mut [&mut [f32]],
        output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
    ) {
        let finished = Instant::now();

        if self.single_input {
            let mut lock = self.left_analyzer.lock();
            let (ref _buffer, ref mut analyzer) = *lock;

            analyzer.analyze(buffer[0].iter().copied(), self.listening_volume);
        } else {
            for (channel_idx, buffer) in buffer.iter().enumerate() {
                let analyzer = if channel_idx == 0 {
                    self.left_analyzer.clone()
                } else {
                    self.right_analyzer.clone()
                };

                {
                    let mut analyzer = analyzer.lock();
                    if buffer.len() == analyzer.0.len() {
                        analyzer.0.copy_from_slice(buffer);
                    } else {
                        analyzer.0.clear();
                        analyzer.0.extend_from_slice(buffer);
                    }
                }

                let listening_volume = self.listening_volume;

                self.analyzer_pool.execute(move || {
                    let mut lock = analyzer.lock();
                    let (ref mut buffer, ref mut analyzer) = *lock;

                    analyzer.analyze(buffer.iter().copied(), listening_volume);
                });
            }
        }

        let chunk_duration =
            Duration::from_secs_f64(buffer.num_samples() as f64 / self.sample_rate as f64);

        let (ref mut spectrogram, ref mut metrics) = *output.lock();

        let (left_ref, right_ref) = (self.left_analyzer.clone(), self.right_analyzer.clone());

        self.analyzer_pool.join();
        let mut left_lock = left_ref.lock();
        let mut right_lock = right_ref.lock();
        let left_analyzer = &mut left_lock.1;
        let right_analyzer = &mut right_lock.1;

        spectrogram.update_fn(|analysis_output| {
            if self.single_input {
                analysis_output.update_mono(
                    left_analyzer,
                    self.gain,
                    self.listening_volume,
                    chunk_duration,
                );
            } else {
                analysis_output.update_stereo(
                    left_analyzer,
                    right_analyzer,
                    self.gain,
                    self.listening_volume,
                    chunk_duration,
                );
            }
        });

        let now = Instant::now();
        metrics.processing = now.duration_since(finished);
        metrics.finished = now;
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
    pub(crate) fn update_config(&mut self, config: &AnalysisChainConfig) {
        self.gain = config.gain;
        self.listening_volume = if config.normalize_amplitude {
            Some(config.listening_volume)
        } else {
            None
        };
        self.masking = config.masking;

        let old_left_analyzer = self.left_analyzer.lock();
        let old_analyzer_config = old_left_analyzer.1.config();

        if self.update_rate != config.update_rate_hz {
            self.chunk_size = (self.sample_rate as f64 / config.update_rate_hz).round() as usize;
            self.chunker.set_block_size(self.chunk_size);
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
            self.chunk_duration =
                Duration::from_secs_f64(self.chunk_size as f64 / self.sample_rate as f64);
        } else if self.additional_latency != config.latency_offset {
            self.additional_latency = config.latency_offset;
            self.latency_samples = if config.internal_buffering {
                self.chunker.latency_samples()
            } else {
                0
            } + (self.additional_latency.as_secs_f64()
                * self.sample_rate as f64) as u32;
        }

        if old_analyzer_config.resolution != config.resolution
            || old_analyzer_config.start_frequency != config.start_frequency
            || old_analyzer_config.end_frequency != config.end_frequency
            || old_analyzer_config.erb_frequency_scale != config.erb_frequency_scale
            || old_analyzer_config.erb_time_resolution != config.erb_time_resolution
            || old_analyzer_config.time_resolution_clamp != config.time_resolution_clamp
            || old_analyzer_config.erb_bandwidth_divisor != config.erb_bandwidth_divisor
            || old_analyzer_config.q_time_resolution != config.q_time_resolution
            || old_analyzer_config.nc_method != config.nc_method
            || old_analyzer_config.strict_nc != config.strict_nc
            || old_analyzer_config.masking != config.masking
            || old_analyzer_config.approximate_masking != config.approximate_masking
        {
            let analyzer_config = BetterAnalyzerConfiguration {
                resolution: config.resolution,
                start_frequency: config.start_frequency,
                end_frequency: config.end_frequency,
                erb_frequency_scale: config.erb_frequency_scale,
                sample_rate: self.sample_rate,
                erb_time_resolution: config.erb_time_resolution,
                erb_bandwidth_divisor: config.erb_bandwidth_divisor,
                time_resolution_clamp: config.time_resolution_clamp,
                q_time_resolution: config.q_time_resolution,
                nc_method: config.nc_method,
                strict_nc: config.strict_nc,
                masking: config.masking,
                approximate_masking: config.approximate_masking,
            };
            drop(old_left_analyzer);
            let left_analyzer = BetterAnalyzer::new(analyzer_config.clone());
            let right_analyzer = BetterAnalyzer::new(analyzer_config);

            let mut frequencies = self.frequencies.write();
            frequencies.clear();
            frequencies.extend(
                left_analyzer
                    .frequencies()
                    .iter()
                    .map(|(a, b, c)| (*a, *b, *c)),
            );

            self.left_analyzer = Arc::new(Mutex::new((vec![0.0; self.chunk_size], left_analyzer)));
            self.right_analyzer =
                Arc::new(Mutex::new((vec![0.0; self.chunk_size], right_analyzer)));
        } else if self.update_rate != config.update_rate_hz
            || self.internal_buffering != config.internal_buffering
        {
            drop(old_left_analyzer);

            self.left_analyzer.lock().0 = vec![0.0; self.chunk_size];
            self.right_analyzer.lock().0 = vec![0.0; self.chunk_size];
        }

        self.internal_buffering = config.internal_buffering;
        self.strict_synchronization = config.strict_synchronization;
        self.update_rate = config.update_rate_hz;
    }
}
