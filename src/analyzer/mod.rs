use std::{
    collections::VecDeque,
    f64::consts::{PI, SQRT_2},
    simd::{f64x64, num::SimdFloat},
    sync::Arc,
    time::Duration,
};

use serde::{Deserialize, Serialize};

mod gammatone;
mod masker;
mod vqsdft;

use vqsdft::{VQsDFT, Window};

use crate::analyzer::masker::Masker;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BetterAnalyzerConfiguration {
    pub resolution: usize,
    pub start_frequency: f64,
    pub end_frequency: f64,
    pub erb_frequency_scale: bool,

    pub sample_rate: f32,
    pub q_time_resolution: f64,
    pub erb_time_resolution: bool,
    pub erb_bandwidth_divisor: f64,
    pub time_resolution_clamp: (f64, f64),
    pub nc_method: bool,

    pub masking: bool,
}

impl Default for BetterAnalyzerConfiguration {
    fn default() -> Self {
        Self {
            resolution: 512,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            erb_frequency_scale: true,
            sample_rate: 48000.0,
            //q_time_resolution: 14.0,
            q_time_resolution: 17.30993,
            erb_time_resolution: true,
            erb_bandwidth_divisor: 2.0,
            time_resolution_clamp: (0.0, 37.0),
            nc_method: true,
            masking: true,
        }
    }
}

#[derive(Clone)]
pub struct BetterAnalyzer {
    config: BetterAnalyzerConfiguration,
    transform: VQsDFT,
    masker: Masker,
    masking: Vec<f64>,
    frequency_bands: Vec<(f64, f64, f64)>,
    frequency_indices: Vec<(usize, usize)>,
    normalizers: Vec<PrecomputedNormalizer>,
    hearing_threshold: Vec<f64>,
}

impl BetterAnalyzer {
    pub fn new(config: BetterAnalyzerConfiguration) -> Self {
        let frequency_scale = if config.erb_frequency_scale {
            FrequencyScale::Erb
        } else {
            FrequencyScale::Logarithmic
        };

        let frequency_bands = frequency_scale.generate_bands(
            config.resolution,
            config.start_frequency,
            config.end_frequency,
            |center| {
                if config.erb_time_resolution {
                    (24.7 + (0.108 * center)) / config.erb_bandwidth_divisor
                } else {
                    center / config.q_time_resolution
                }
                .min(1.0 / (config.time_resolution_clamp.0 / 1000.0))
                .max(1.0 / (config.time_resolution_clamp.1 / 1000.0))
            },
        );

        let band_count = frequency_bands.len();

        assert!(band_count.is_multiple_of(64));

        let frequency_indices = frequency_bands
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let lower = (0..i.saturating_sub(1))
                    .rev()
                    .find(|i| frequency_bands[*i].high <= f.low)
                    .unwrap_or(0);
                let upper = (i..band_count)
                    .find(|i| frequency_bands[*i].low >= f.high)
                    .unwrap_or(band_count - 1);

                ((lower + 1).min(i), upper.saturating_sub(1))
            })
            .collect();

        let normalizers: Vec<_> = frequency_bands
            .iter()
            .map(|band| PrecomputedNormalizer::new(band.center))
            .collect();

        let hearing_threshold: Vec<_> = frequency_bands
            .iter()
            .map(|band| approximate_hearing_threshold(band.center))
            .collect();

        let transform = VQsDFT::new(
            &frequency_bands,
            Window::Hann,
            config.sample_rate as f64,
            config.nc_method,
            false,
        );

        let masker = Masker::new(&frequency_bands);

        let frequency_bands: Vec<_> = frequency_bands
            .iter()
            .map(|band| (band.low, band.center, band.high))
            .collect();

        assert!(frequency_bands.len().is_multiple_of(64));

        Self {
            config,
            masker,
            masking: vec![0.0; frequency_bands.len()],
            transform,
            frequency_bands,
            frequency_indices,
            normalizers,
            hearing_threshold,
        }
    }
    #[inline(always)]
    pub fn config(&self) -> &BetterAnalyzerConfiguration {
        &self.config
    }
    #[inline(always)]
    pub fn frequencies(&self) -> &[(f64, f64, f64)] {
        &self.frequency_bands
    }
    #[inline(always)]
    pub fn clear_buffers(&mut self) {
        self.transform.reset();
    }
    pub fn analyze(
        &mut self,
        samples: impl ExactSizeIterator<Item = f64>,
        listening_volume: Option<f64>,
    ) {
        self.transform.analyze(samples);

        /*let flatness = if spectrum.len() > 128 {
            0.0
        } else {
            map_value_f64(spectral_flatness(spectrum), -60.0, 0.0, 0.0, 1.0)
        };*/

        if self.config.masking {
            self.masker.calculate_masking_threshold(
                self.transform.spectrum_data.iter().copied(),
                listening_volume,
                //0.0,
                &mut self.masking,
            );

            unsafe { self.transform.spectrum_data.as_chunks_unchecked_mut::<64>() }
                .iter_mut()
                .zip(unsafe { self.masking.as_chunks_unchecked::<64>() })
                .for_each(|(spectrum, masking)| {
                    *spectrum = f64x64::from_array(*spectrum)
                        .simd_max(f64x64::from_array(*masking))
                        .to_array();
                });
        }
    }
    #[inline(always)]
    pub fn raw_analysis(&self) -> &[f64] {
        &self.transform.spectrum_data
    }
    #[inline(always)]
    pub fn raw_masking(&self) -> &[f64] {
        &self.masking
    }
    pub fn remove_masked_components(&mut self) {
        self.transform
            .spectrum_data
            .iter_mut()
            .zip(self.masking.iter().copied())
            .for_each(|(amplitude, masking_amplitude)| {
                if masking_amplitude >= *amplitude {
                    *amplitude = f64::NEG_INFINITY;
                }
            });
    }
}

#[derive(Clone)]
pub struct BetterAnalysis {
    pub duration: Duration,
    pub data: Vec<(f32, f32)>,
    pub masking: Vec<(f32, f32)>,
    pub min: f32,
    pub mean: f32,
    pub max: f32,
    pub masking_mean: f32,
    peak_scratchpad: Vec<bool>,
    sorting_scratchpad: Vec<(f32, usize)>,
}

impl BetterAnalysis {
    pub fn new(capacity: usize) -> Self {
        Self {
            duration: Duration::ZERO,
            data: Vec::with_capacity(capacity),
            masking: Vec::with_capacity(capacity),
            min: f32::NEG_INFINITY,
            mean: f32::NEG_INFINITY,
            max: f32::NEG_INFINITY,
            masking_mean: f32::NEG_INFINITY,
            sorting_scratchpad: Vec::with_capacity(capacity),
            peak_scratchpad: Vec::with_capacity(capacity),
        }
    }
    pub fn update_stereo(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f64,
        normalization_volume: Option<f64>,
        duration: Duration,
    ) {
        assert_eq!(left.raw_analysis().len(), right.raw_analysis().len());

        let new_length = left.raw_analysis().len();

        let mut sum = 0.0;
        self.max = f32::NEG_INFINITY;

        if left.config.masking {
            if self.data.len() != new_length {
                self.masking.clear();

                for _ in 0..new_length {
                    self.masking.push((0.0, 0.0));
                }
            }

            let masking_data = left
                .raw_masking()
                .iter()
                .copied()
                .zip(right.raw_masking().iter().copied())
                .map(|(left, right)| calculate_pan_and_volume_from_amplitude(left, right));

            let mut masking_sum = 0.0;

            if let Some(listening_volume) = normalization_volume {
                let hearing_threshold = left
                    .hearing_threshold
                    .iter()
                    .copied()
                    .map(|h| h - listening_volume);

                left.normalizers
                    .iter()
                    .zip(hearing_threshold.zip(masking_data.zip(self.masking.iter_mut())))
                    .for_each(
                        |(normalizer, (threshold, ((mask_pan, mask_volume), masking_result)))| {
                            let masking_norm_db = normalizer
                                .spl_to_phon((mask_volume + gain).max(threshold) + listening_volume)
                                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                                - listening_volume;

                            *masking_result = (mask_pan as f32, masking_norm_db as f32);
                            masking_sum += dbfs_to_amplitude(masking_norm_db);
                        },
                    );
            } else {
                masking_data.zip(self.masking.iter_mut()).for_each(
                    |((mask_pan, mask_volume), masking_result)| {
                        let masking_norm_db = mask_volume + gain;

                        *masking_result = (mask_pan as f32, masking_norm_db as f32);
                        masking_sum += dbfs_to_amplitude(masking_norm_db);
                    },
                );
            }

            self.masking_mean = amplitude_to_dbfs(masking_sum / self.masking.len() as f64) as f32;
        } else if self.data.len() != new_length {
            self.masking.clear();

            for _ in 0..new_length {
                self.masking.push((0.0, f32::NEG_INFINITY));
            }

            self.masking_mean = f32::NEG_INFINITY;
        } else {
            self.masking.fill((0.0, f32::NEG_INFINITY));
            self.masking_mean = f32::NEG_INFINITY;
        }

        if self.data.len() == new_length {
            if let Some(listening_volume) = normalization_volume {
                for ((left, right), (normalizer, result)) in left
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(right.raw_analysis().iter().copied())
                    .zip(left.normalizers.iter().zip(self.data.iter_mut()))
                {
                    let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);

                    let volume = normalizer
                        .spl_to_phon(volume + gain + listening_volume)
                        //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                        .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                        - listening_volume;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    *result = (pan as f32, volume);
                    self.max = self.max.max(volume);
                }
            } else {
                for (left, (right, result)) in left.raw_analysis().iter().copied().zip(
                    right
                        .raw_analysis()
                        .iter()
                        .copied()
                        .zip(self.data.iter_mut()),
                ) {
                    let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);
                    let volume = volume + gain;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    *result = (pan as f32, volume);
                    self.max = self.max.max(volume);
                }
            }
        } else {
            assert!(self.data.capacity() >= new_length);

            self.data.clear();

            if let Some(listening_volume) = normalization_volume {
                for ((left, right), normalizer) in left
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(right.raw_analysis().iter().copied())
                    .zip(left.normalizers.iter())
                {
                    let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);

                    let volume = normalizer
                        .spl_to_phon(volume + gain + listening_volume)
                        //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                        .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                        - listening_volume;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    self.data.push((pan as f32, volume));
                    self.max = self.max.max(volume);
                }
            } else {
                let gain_amplitude = dbfs_to_amplitude(gain);

                for (left, right) in left
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(right.raw_analysis().iter().copied())
                {
                    let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);
                    let volume = volume + gain;

                    sum += (left + right) * gain_amplitude;
                    let volume = volume as f32;

                    self.data.push((pan as f32, volume));
                    self.max = self.max.max(volume);
                }
            }
        }

        if let Some(listening_volume) = normalization_volume {
            self.min = (3.0 - listening_volume) as f32;
        } else {
            self.min = f32::NEG_INFINITY;
        }

        self.mean = amplitude_to_dbfs(sum / self.data.len() as f64) as f32;

        self.duration = duration;
    }
    pub fn update_mono(
        &mut self,
        center: &BetterAnalyzer,
        gain: f64,
        normalization_volume: Option<f64>,
        duration: Duration,
    ) {
        let new_length = center.raw_analysis().len();

        let mut sum = 0.0;
        self.max = f32::NEG_INFINITY;

        if center.config.masking {
            if self.data.len() != new_length {
                self.masking.clear();

                for _ in 0..new_length {
                    self.masking.push((0.0, 0.0));
                }
            }

            let masking_data = center
                .raw_masking()
                .iter()
                .copied()
                .map(|amplitude| amplitude * 2.0);

            let gain_amplitude = dbfs_to_amplitude(gain);
            let mut masking_sum = 0.0;

            if let Some(listening_volume) = normalization_volume {
                let hearing_threshold = center
                    .hearing_threshold
                    .iter()
                    .copied()
                    .map(|h| h - listening_volume);

                center
                    .normalizers
                    .iter()
                    .zip(hearing_threshold.zip(masking_data.zip(self.masking.iter_mut())))
                    .for_each(
                        |(normalizer, (threshold, (mask_amplitude, masking_result)))| {
                            let masking_norm_db = normalizer
                                .spl_to_phon(
                                    amplitude_to_dbfs(mask_amplitude * gain_amplitude)
                                        .max(threshold)
                                        + listening_volume,
                                )
                                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                                - listening_volume;

                            *masking_result = (0.0, masking_norm_db as f32);
                            masking_sum += dbfs_to_amplitude(masking_norm_db);
                        },
                    );
            } else {
                masking_data.zip(self.masking.iter_mut()).for_each(
                    |(mask_amplitude, masking_result)| {
                        let masking_amplitude = mask_amplitude * gain_amplitude;

                        *masking_result = (0.0, amplitude_to_dbfs(masking_amplitude) as f32);
                        masking_sum += masking_amplitude;
                    },
                );
            }

            self.masking_mean = amplitude_to_dbfs(masking_sum / self.masking.len() as f64) as f32;
        } else if self.data.len() != new_length {
            self.masking.clear();

            for _ in 0..new_length {
                self.masking.push((0.0, f32::NEG_INFINITY));
            }

            self.masking_mean = f32::NEG_INFINITY;
        } else {
            self.masking.fill((0.0, f32::NEG_INFINITY));
            self.masking_mean = f32::NEG_INFINITY;
        }

        if self.data.len() == new_length {
            if let Some(listening_volume) = normalization_volume {
                for (amplitude, (normalizer, result)) in center
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(center.normalizers.iter().zip(self.data.iter_mut()))
                {
                    let volume = normalizer
                        .spl_to_phon(amplitude_to_dbfs(amplitude * 2.0) + gain + listening_volume)
                        //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                        .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                        - listening_volume;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    *result = (0.0, volume);
                    self.max = self.max.max(volume);
                }
            } else {
                for (amplitude, result) in center
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(self.data.iter_mut())
                {
                    let volume = amplitude_to_dbfs(amplitude * 2.0) + gain;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    *result = (0.0, volume);
                    self.max = self.max.max(volume);
                }
            }
        } else {
            assert!(self.data.capacity() >= new_length);

            self.data.clear();

            if let Some(listening_volume) = normalization_volume {
                for (amplitude, normalizer) in center
                    .raw_analysis()
                    .iter()
                    .copied()
                    .zip(center.normalizers.iter())
                {
                    let volume = normalizer
                        .spl_to_phon(amplitude_to_dbfs(amplitude * 2.0) + gain + listening_volume)
                        //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                        .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                        - listening_volume;

                    sum += dbfs_to_amplitude(volume);
                    let volume = volume as f32;

                    self.data.push((0.0, volume));
                    self.max = self.max.max(volume);
                }
            } else {
                let gain_amplitude = dbfs_to_amplitude(gain);

                for amplitude in center.raw_analysis().iter().copied() {
                    let amplitude = amplitude * 2.0 * gain_amplitude;

                    sum += amplitude;
                    let volume = amplitude_to_dbfs(amplitude) as f32;

                    self.data.push((0.0, volume));
                    self.max = self.max.max(volume);
                }
            }
        }

        if let Some(listening_volume) = normalization_volume {
            self.min = (3.0 - listening_volume) as f32;
        } else {
            self.min = f32::NEG_INFINITY;
        }

        self.mean = amplitude_to_dbfs(sum / self.data.len() as f64) as f32;

        self.duration = duration;
    }
    pub fn peaks(
        &mut self,
        volume_threshold: f32,
        max_count: usize,
        analyzer: &BetterAnalyzer,
    ) -> impl Iterator<Item = f32> {
        self.sorting_scratchpad.clear();

        let min = volume_threshold.max(self.min);

        if self.masking_mean.is_finite() {
            self.data
                .iter()
                .copied()
                .zip(self.masking.iter().copied())
                .enumerate()
                .for_each(|(i, ((_, a), (_, m)))| {
                    if a > min && a > m {
                        self.sorting_scratchpad.push((a - m, i));
                    }
                });
        } else {
            self.data
                .iter()
                .copied()
                .enumerate()
                .for_each(|(i, (_, a))| {
                    if a > min {
                        self.sorting_scratchpad.push((a, i));
                    }
                });
        }
        self.sorting_scratchpad
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        if self.peak_scratchpad.len() == self.data.len() {
            self.peak_scratchpad.fill(false);
        } else {
            self.peak_scratchpad.clear();

            for _ in 0..self.data.len() {
                self.peak_scratchpad.push(false);
            }
        }

        self.sorting_scratchpad
            .iter()
            .copied()
            .rev()
            .filter_map(|(_stm, i)| {
                if !self.peak_scratchpad[i] {
                    let (min, max) = analyzer.frequency_indices[i];

                    (min..=max).for_each(|i| {
                        self.peak_scratchpad[i] = true;
                    });

                    Some(analyzer.frequency_bands[i].1 as f32)
                } else {
                    None
                }
            })
            .take(max_count)
    }
}

#[derive(Clone)]
pub struct BetterSpectrogram {
    pub data: VecDeque<Arc<BetterAnalysis>>,
}

impl BetterSpectrogram {
    pub fn new(length: usize, slice_capacity: usize) -> Self {
        Self {
            data: VecDeque::from_iter((0..length).map(|_| {
                Arc::new(BetterAnalysis {
                    duration: Duration::from_secs(1),
                    data: vec![(0.0, f32::NEG_INFINITY); slice_capacity],
                    masking: vec![(0.0, f32::NEG_INFINITY); slice_capacity],
                    min: f32::NEG_INFINITY,
                    mean: f32::NEG_INFINITY,
                    max: f32::NEG_INFINITY,
                    masking_mean: f32::NEG_INFINITY,
                    sorting_scratchpad: vec![(f32::NEG_INFINITY, 0); slice_capacity],
                    peak_scratchpad: vec![false; slice_capacity],
                })
            })),
        }
    }
    pub fn update(&mut self, analysis: &BetterAnalysis) {
        self.update_fn(|buffer| {
            if buffer.data.len() == analysis.data.len() {
                buffer.data.copy_from_slice(&analysis.data);
                buffer.masking.copy_from_slice(&analysis.masking);
            } else {
                buffer.data.clone_from(&analysis.data);
                buffer.masking.clone_from(&analysis.masking);
            }
            buffer.duration = analysis.duration;
            buffer.min = analysis.min;
            buffer.mean = analysis.mean;
            buffer.max = analysis.max;
            buffer.masking_mean = analysis.masking_mean;
        });
    }
    pub fn update_fn<F>(&mut self, callback: F)
    where
        F: Fn(&mut BetterAnalysis),
    {
        let buffer = self.data.pop_back().unwrap();

        callback(unsafe { &mut *Arc::as_ptr(&buffer).cast_mut() }); // *Intentional* race condition; This can only cause visual glitches, as all other references are read-only

        // TODO: Add measures to ensure safety in the future! This is currently an ugly hack

        /*loop {
            if let Some(buffer) = Arc::get_mut(&mut buffer) {
                callback(buffer);
                break;
            }
        }*/

        self.data.push_front(buffer);
    }
    pub fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}

// ----- Below formula is based on https://stackoverflow.com/a/35614871 -----

const NEG_SQRT_2: f64 = -SQRT_2;

pub fn calculate_pan_and_volume_from_amplitude(
    left_amplitude: f64,
    right_amplitude: f64,
) -> (f64, f64) {
    let ratio = left_amplitude.algebraic_div(right_amplitude);

    let pan = if ratio == 1.0 {
        0.0
    } else if left_amplitude == 0.0 && right_amplitude > 0.0 {
        1.0
    } else if right_amplitude == 0.0 && left_amplitude > 0.0 {
        -1.0
    } else if ratio.is_nan() {
        0.0
    } else {
        const COEFF: f64 = (180.0 / PI) / 22.5;

        (f64::atan(
            (NEG_SQRT_2
                .algebraic_mul(f64::sqrt(ratio.algebraic_mul(ratio).algebraic_add(1.0)))
                .algebraic_add(ratio)
                .algebraic_add(1.0))
            .algebraic_div(ratio.algebraic_sub(1.0)),
        ))
        .algebraic_mul(COEFF)
    };

    (
        pan,
        amplitude_to_dbfs(left_amplitude.algebraic_add(right_amplitude)),
    )
}

// ----- Below formulas are taken from ISO 226:2023 -----

#[derive(Clone)]
struct PrecomputedNormalizer {
    alpha_f: f64,
    l_u: f64,
    param_1: f64,
    param_2: f64,
}

impl PrecomputedNormalizer {
    fn new(frequency: f64) -> Self {
        let (alpha_f, l_u, t_f) = approximate_coefficients(frequency);

        Self {
            alpha_f,
            l_u,
            param_1: 10.0_f64.powf(alpha_f * ((t_f + l_u) / 10.0)),
            param_2: (4.0e-10_f64).powf(0.3 - alpha_f),
        }
    }
    fn spl_to_phon(&self, db_spl: f64) -> f64 {
        NORM_MULTIPLE.algebraic_mul(f64::log10(
            ((10.0_f64
                .powf(
                    self.alpha_f
                        .algebraic_mul((db_spl.algebraic_add(self.l_u)).algebraic_div(10.0)),
                )
                .algebraic_sub(self.param_1))
            .algebraic_div(self.param_2))
            .algebraic_add(NORM_OFFSET),
        ))
    }
}

/*fn spl_to_phon(frequency: f64, db_spl: f64) -> f64 {
    let (alpha_f, l_u, t_f) = approximate_coefficients(frequency);

    NORM_MULTIPLE
        * f64::log10(
            ((10.0_f64.powf(alpha_f * ((db_spl + l_u) / 10.0))
                - 10.0_f64.powf(alpha_f * ((t_f + l_u) / 10.0)))
                / (4.0e-10_f64).powf(0.3 - alpha_f))
                + 10.0_f64.powf(0.072),
        )
}*/

/*const MIN_COMPLETE_NORM_PHON: f64 = 20.0;
const MAX_COMPLETE_NORM_PHON: f64 = 80.0;*/
const MIN_INFORMATIVE_NORM_PHON: f64 = 0.0;
const MAX_INFORMATIVE_NORM_PHON: f64 = 100.0;
const NORM_MULTIPLE: f64 = 100.0 / 3.0;
const NORM_OFFSET: f64 = 1.180_320_635_651_729_7; // 10.0_f64.powf(0.072)

const NORM_FREQUENCIES: &[f64] = &[
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0,
];

const MIN_NORM_FREQUENCY: f64 = NORM_FREQUENCIES[0];
const MAX_NORM_FREQUENCY: f64 = NORM_FREQUENCIES[NORM_FREQUENCIES.len() - 1];
const NORM_FREQUENCY_COUNT: usize = NORM_FREQUENCIES.len();

const ALPHA_F: &[f64] = &[
    0.635, 0.602, 0.569, 0.537, 0.509, 0.482, 0.456, 0.433, 0.412, 0.391, 0.373, 0.357, 0.343,
    0.330, 0.320, 0.311, 0.303, 0.300, 0.295, 0.292, 0.290, 0.290, 0.289, 0.289, 0.289, 0.293,
    0.303, 0.323, 0.354,
];

const L_U: &[f64] = &[
    -31.5, -27.2, -23.1, -19.3, -16.1, -13.1, -10.4, -8.2, -6.3, -4.6, -3.2, -2.1, -1.2, -0.5, 0.0,
    0.4, 0.5, 0.0, -2.7, -4.2, -1.2, 1.4, 2.3, 1.0, -2.3, -7.2, -11.2, -10.9, -3.5,
];

const T_F: &[f64] = &[
    78.1, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0,
    2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
];

fn approximate_coefficients(frequency: f64) -> (f64, f64, f64) {
    let frequency = frequency.clamp(MIN_NORM_FREQUENCY, MAX_NORM_FREQUENCY);

    for i in 0..NORM_FREQUENCY_COUNT {
        let ii = i + 1;

        if NORM_FREQUENCIES[i] == frequency {
            return (ALPHA_F[i], L_U[i], T_F[i]);
        }

        if NORM_FREQUENCIES[i] < frequency && frequency < NORM_FREQUENCIES[ii] {
            let k =
                (frequency - NORM_FREQUENCIES[i]) / (NORM_FREQUENCIES[ii] - NORM_FREQUENCIES[i]);

            return (
                (ALPHA_F[ii] - ALPHA_F[i]) * k + ALPHA_F[i],
                (L_U[ii] - L_U[i]) * k + L_U[i],
                (T_F[ii] - T_F[i]) * k + T_F[i],
            );
        }
    }

    panic!()
}

fn approximate_hearing_threshold(frequency: f64) -> f64 {
    let frequency = frequency.clamp(MIN_NORM_FREQUENCY, MAX_NORM_FREQUENCY);

    for i in 0..NORM_FREQUENCY_COUNT {
        let ii = i + 1;

        if NORM_FREQUENCIES[i] == frequency {
            return T_F[i];
        }

        if NORM_FREQUENCIES[i] < frequency && frequency < NORM_FREQUENCIES[ii] {
            let k =
                (frequency - NORM_FREQUENCIES[i]) / (NORM_FREQUENCIES[ii] - NORM_FREQUENCIES[i]);

            return (T_F[ii] - T_F[i]) * k + T_F[i];
        }
    }

    panic!()
}

// ----- Below algorithms are taken from https://codepen.io/TF3RDL/pen/MWLzPoO -----

#[inline(always)]
pub fn amplitude_to_dbfs(amplitude: f64) -> f64 {
    20.0_f64.algebraic_mul(f64::log10(amplitude))
}

#[inline(always)]
pub fn dbfs_to_amplitude(decibels: f64) -> f64 {
    10.0_f64.powf(decibels.algebraic_div(20.0))
}

#[inline(always)]
pub fn map_value_f64(x: f64, min: f64, max: f64, target_min: f64, target_max: f64) -> f64 {
    (x.algebraic_sub(min))
        .algebraic_div(max.algebraic_sub(min))
        .algebraic_mul(target_max.algebraic_sub(target_min))
        .algebraic_add(target_min)
}

#[inline(always)]
pub fn map_value_f32(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
    (x.algebraic_sub(min))
        .algebraic_div(max.algebraic_sub(min))
        .algebraic_mul(target_max.algebraic_sub(target_min))
        .algebraic_add(target_min)
}

pub enum FrequencyScale {
    Logarithmic,
    Erb,
    Bark,
    Mel,
}

impl FrequencyScale {
    pub fn scale(&self, x: f64) -> f64 {
        match self {
            Self::Logarithmic => x.log2(),
            Self::Erb => (1.0 + 0.00437 * x).log2(),
            Self::Bark => (26.81 * x) / (1960.0 + x) - 0.53,
            Self::Mel => (1.0 + x / 700.0).log2(),
        }
    }
    pub fn inv_scale(&self, x: f64) -> f64 {
        match self {
            Self::Logarithmic => 2.0_f64.powf(x),
            Self::Erb => (1.0 / 0.00437) * ((2.0_f64.powf(x)) - 1.0),
            Self::Bark => 1960.0 / (26.81 / (x + 0.53) - 1.0),
            Self::Mel => 700.0 * ((2.0_f64.powf(x)) - 1.0),
        }
    }
    fn generate_bands<F>(&self, n: usize, low: f64, high: f64, bandwidth: F) -> Vec<FrequencyBand>
    where
        F: Fn(f64) -> f64,
    {
        (0..n)
            .map(|i| {
                let i = i as f64;
                let target_max = (n - 1) as f64;

                let center = self.inv_scale(map_value_f64(
                    i,
                    0.0,
                    target_max,
                    self.scale(low),
                    self.scale(high),
                ));
                let lower = self.inv_scale(map_value_f64(
                    i - 0.5,
                    0.0,
                    target_max,
                    self.scale(low),
                    self.scale(high),
                ));
                let higher = self.inv_scale(map_value_f64(
                    i + 0.5,
                    0.0,
                    target_max,
                    self.scale(low),
                    self.scale(high),
                ));
                let bandwidth = bandwidth(center);

                FrequencyBand {
                    low: (center - (bandwidth / 2.0)).min(lower),
                    center,
                    high: (center + (bandwidth / 2.0)).max(higher),
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
struct FrequencyBand {
    low: f64,
    center: f64,
    high: f64,
}
