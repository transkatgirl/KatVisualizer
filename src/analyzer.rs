use core::f64;
use std::{collections::VecDeque, f64::consts::PI, time::Duration};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BetterAnalyzerConfiguration {
    pub resolution: usize,
    pub start_frequency: f64,
    pub end_frequency: f64,
    pub erb_frequency_scale: bool,

    pub sample_rate: usize,
    pub time_resolution: f64,
    pub erb_time_resolution: bool,
    pub erb_time_resolution_clamp: (f64, f64),
    pub erb_bandwidth_divisor: f64,
    pub nc_method: bool,
}

impl Default for BetterAnalyzerConfiguration {
    fn default() -> Self {
        Self {
            resolution: 512,
            start_frequency: 20.0,
            end_frequency: 20000.0,
            erb_frequency_scale: true,
            sample_rate: 48000,
            time_resolution: 37.0,
            erb_time_resolution: true,
            erb_time_resolution_clamp: (0.0, 37.0),
            erb_bandwidth_divisor: 2.0,
            nc_method: true,
        }
    }
}

#[derive(Clone)]
pub struct BetterAnalyzer {
    config: BetterAnalyzerConfiguration,
    transform: VQsDFT,
    buffer_size: usize,
    frequency_bands: Vec<(f64, f64, f64)>,
    normalizers: Vec<PrecomputedNormalizer>,
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
                    ((24.7 * (0.00437 * center + 1.0)) / config.erb_bandwidth_divisor)
                        .min(1.0 / (config.erb_time_resolution_clamp.0 / 1000.0))
                        .max(1.0 / (config.erb_time_resolution_clamp.1 / 1000.0))
                } else {
                    1.0 / (config.time_resolution / 1000.0)
                }
            },
        );

        let normalizers: Vec<_> = frequency_bands
            .iter()
            .map(|band| PrecomputedNormalizer::new(band.center))
            .collect();

        let transform = VQsDFT::new(
            &frequency_bands,
            HANN_WINDOW,
            1000.0,
            1.0,
            1000.0,
            config.sample_rate,
            config.nc_method,
        );

        let frequency_bands: Vec<_> = frequency_bands
            .iter()
            .map(|band| (band.low, band.center, band.high))
            .collect();

        Self {
            config,
            buffer_size: transform.buffer.len(),
            transform,
            frequency_bands,
            normalizers,
        }
    }
    pub fn config(&self) -> &BetterAnalyzerConfiguration {
        &self.config
    }
    pub fn chunk_size(&self) -> usize {
        self.buffer_size
    }
    pub fn frequencies(&self) -> &[(f64, f64, f64)] {
        &self.frequency_bands
    }
    pub fn clear_buffers(&mut self) {
        self.transform.reset();
    }
    pub fn analyze(
        &mut self,
        samples: impl Iterator<Item = f64>,
        gain: f64,
        normalization_volume: Option<f64>,
    ) -> &[f64] {
        self.transform.analyze(samples);

        if let Some(listening_volume) = normalization_volume {
            for (output, normalizer) in self
                .transform
                .spectrum_data
                .iter_mut()
                .zip(self.normalizers.iter())
            {
                *output = normalizer
                    .spl_to_phon(amplitude_to_dbfs(*output) + gain + listening_volume)
                    //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                    .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                    - listening_volume
            }
        } else {
            for output in self.transform.spectrum_data.iter_mut() {
                *output = amplitude_to_dbfs(*output) + gain
            }
        }

        &self.transform.spectrum_data
    }
    pub fn last_analysis(&self) -> &[f64] {
        &self.transform.spectrum_data
    }
}

#[derive(Clone)]
pub struct BetterAnalysis {
    pub duration: Duration,
    pub data: Vec<(f32, f32)>,
}

impl BetterAnalysis {
    pub fn new(capacity: usize) -> Self {
        Self {
            duration: Duration::ZERO,
            data: Vec::with_capacity(capacity),
        }
    }
    pub fn update_stereo(&mut self, left: &[f64], right: &[f64], duration: Duration) {
        let new_length = left.len().min(right.len());

        if self.data.len() == new_length {
            for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                let (pan, volume) = calculate_pan_and_volume(*left, *right);

                self.data[index] = ((pan * 2.0) as f32, volume as f32);
            }
        } else {
            assert!(self.data.capacity() >= new_length);

            self.data.clear();

            for (left, right) in left.iter().zip(right.iter()) {
                let (pan, volume) = calculate_pan_and_volume(*left, *right);

                self.data.push(((pan * 2.0) as f32, volume as f32));
            }
        }

        self.duration = duration;
    }
    pub fn update_mono(&mut self, data: &[f64], duration: Duration) {
        let new_length = data.len();

        if self.data.len() == new_length {
            for (index, volume) in data.iter().enumerate() {
                self.data[index] = (0.0, *volume as f32);
            }
        } else {
            assert!(self.data.capacity() >= new_length);

            self.data.clear();

            for volume in data {
                self.data.push((0.0, *volume as f32));
            }
        }

        self.duration = duration;
    }
}

#[derive(Clone)]
pub struct BetterSpectrogram {
    pub data: VecDeque<BetterAnalysis>,
}

impl BetterSpectrogram {
    pub fn new(length: usize, slice_capacity: usize) -> Self {
        Self {
            data: VecDeque::from(vec![
                BetterAnalysis {
                    duration: Duration::from_secs(1),
                    data: vec![(0.0, -f32::INFINITY); slice_capacity],
                };
                length
            ]),
        }
    }
    pub fn update(&mut self, analysis: &BetterAnalysis) {
        self.update_fn(|buffer| {
            if buffer.data.len() == analysis.data.len() {
                buffer.data.copy_from_slice(&analysis.data);
            } else {
                buffer.data.clone_from(&analysis.data);
            }
        });
    }
    pub fn update_fn<F>(&mut self, callback: F)
    where
        F: Fn(&mut BetterAnalysis),
    {
        let mut buffer = self.data.pop_back().unwrap();
        callback(&mut buffer);
        self.data.push_front(buffer);
    }
    pub fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}

// ----- Below formula is based on https://stackoverflow.com/a/35614871 -----

pub fn calculate_pan_and_volume(left_db: f64, right_db: f64) -> (f64, f64) {
    let left_amplitude = dbfs_to_amplitude(left_db);
    let right_amplitude = dbfs_to_amplitude(right_db);

    let ratio = left_amplitude / right_amplitude;

    let pan = if ratio == 1.0 {
        0.0
    } else {
        (f64::atan(
            (-f64::sqrt(2.0) * f64::sqrt(ratio * ratio + 1.0) + ratio + 1.0) / (ratio - 1.0),
        ))
        .to_degrees()
            / 45.0
    };

    (pan, amplitude_to_dbfs(left_amplitude + right_amplitude))
}

// ----- Below formulas are taken from ISO 226:2023 -----

#[derive(Clone)]
struct PrecomputedNormalizer {
    alpha_f: f64,
    l_u: f64,
    param_1: f64,
    param_2: f64,
    param_3: f64,
}

impl PrecomputedNormalizer {
    fn new(frequency: f64) -> Self {
        let (alpha_f, l_u, t_f) = approximate_coefficients(frequency);

        Self {
            alpha_f,
            l_u,
            param_1: 10.0_f64.powf(alpha_f * ((t_f + l_u) / 10.0)),
            param_2: (4.0e-10_f64).powf(0.3 - alpha_f),
            param_3: 10.0_f64.powf(0.072),
        }
    }
    fn spl_to_phon(&self, db_spl: f64) -> f64 {
        NORM_MULTIPLE
            * f64::log10(
                ((10.0_f64.powf(self.alpha_f * ((db_spl + self.l_u) / 10.0)) - self.param_1)
                    / self.param_2)
                    + self.param_3,
            )
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

// ----- Below algorithms are taken from https://codepen.io/TF3RDL/pen/MWLzPoO -----

pub fn amplitude_to_dbfs(amplitude: f64) -> f64 {
    20.0 * f64::log10(amplitude)
}

pub fn dbfs_to_amplitude(decibels: f64) -> f64 {
    10.0_f64.powf(decibels / 20.0)
}

pub fn map_value_f64(x: f64, min: f64, max: f64, target_min: f64, target_max: f64) -> f64 {
    (x - min) / (max - min) * (target_max - target_min) + target_min
}

pub fn map_value_f32(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
    (x - min) / (max - min) * (target_max - target_min) + target_min
}

enum FrequencyScale {
    Logarithmic,
    Erb,
    /*Bark,
    Mel,*/
}

impl FrequencyScale {
    fn scale(&self, x: f64) -> f64 {
        match self {
            Self::Logarithmic => x.log2(),
            Self::Erb => (1.0 + 0.00437 * x).log2(),
            /*Self::Bark => (26.81 * x) / (1960.0 + x) - 0.53,
            Self::Mel => (1.0 + x / 700.0).log2(),*/
        }
    }
    fn inv_scale(&self, x: f64) -> f64 {
        match self {
            Self::Logarithmic => 2.0_f64.powf(x),
            Self::Erb => (1.0 / 0.00437) * ((2.0_f64.powf(x)) - 1.0),
            /*Self::Bark => 1960.0 / (26.81 / (x + 0.53) - 1.0),
            Self::Mel => 700.0 * ((2.0_f64.powf(x)) - 1.0),*/
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

const HANN_WINDOW: &[f64] = &[1.0, 0.5];
/*#[allow(clippy::excessive_precision)]
const HAMMING_WINDOW: &[f64] = &[1.0, 0.4259434938430786];
#[allow(clippy::excessive_precision)]
const BLACKMAN_WINDOW: &[f64] = &[1.0, 0.595257580280304, 0.0952545627951622];
#[allow(clippy::excessive_precision)]
const NUTTALL_WINDOW: &[f64] = &[
    1.0,
    0.6850073933601379,
    0.20272639393806458,
    0.017719272524118423,
];
#[allow(clippy::excessive_precision)]
const FLAT_TOP_WINDOW: &[f64] = &[
    1.0,
    0.966312825679779,
    0.6430955529212952,
    0.19387830793857574,
    0.016120079904794693,
];*/
#[derive(Clone)]
struct VQsDFT {
    coeffs: Vec<VQsDFTCoeffs>,
    gains: Vec<f64>,
    buffer: Vec<f64>,
    buffer_index: usize,
    spectrum_data: Vec<f64>,
    use_nc: bool,
}

#[derive(Clone)]
struct VQsDFTCoeffs {
    period: f64,
    twiddles: Vec<(f64, f64)>,
    fiddles: Vec<(f64, f64)>,
    reson_coeffs: Vec<f64>,
    coeffs1: Vec<(f64, f64)>,
    coeffs2: Vec<(f64, f64)>,
    coeffs3: Vec<(f64, f64)>,
    coeffs4: Vec<(f64, f64)>,
    coeffs5: Vec<(f64, f64)>,
}

struct FrequencyBand {
    low: f64,
    center: f64,
    high: f64,
}

impl VQsDFT {
    fn new(
        freq_bands: &[FrequencyBand],
        window: &[f64],
        time_res: f64,
        bandwidth: f64,
        max_time_res: f64,
        sample_rate: usize,
        use_nc: bool,
    ) -> Self {
        let sample_rate_f64 = sample_rate as f64;
        let buffer_size = (sample_rate_f64 * max_time_res / 1000.0).round() as usize;
        let buffer_size_f64 = buffer_size as f64;

        let (min_idx, max_idx) = if use_nc {
            (0, 2)
        } else {
            (-(window.len() as isize) + 1, window.len() as isize)
        };
        let items_per_band = (max_idx - min_idx) as usize;

        let gains = if use_nc {
            vec![0.0; 2]
        } else {
            (min_idx..max_idx)
                .map(|i| window[i.unsigned_abs()] * (-((i as f64).abs() % 2.0) * 2.0 + 1.0))
                .collect()
        };

        let k_offset = if use_nc { -0.5 } else { 0.0 };

        VQsDFT {
            spectrum_data: vec![0.0; freq_bands.len()],
            gains,
            coeffs: freq_bands
                .iter()
                .map(|x| {
                    let mut fiddles = Vec::with_capacity(items_per_band);
                    let mut twiddles = Vec::with_capacity(items_per_band);
                    let mut reson_coeffs = Vec::with_capacity(items_per_band);

                    let period = (f64::min(
                        buffer_size_f64,
                        sample_rate_f64
                            / (bandwidth * (x.high - x.low).abs() + 1.0 / (time_res / 1000.0)),
                    ))
                    .trunc();

                    for i in min_idx..max_idx {
                        let i = i as f64;

                        let k = (x.center * period) / sample_rate_f64 + i + k_offset;
                        let fid = -2.0 * PI * k;
                        let twid = (2.0 * PI * k) / period;
                        let reson = 2.0 * f64::cos(twid);

                        fiddles.push((f64::cos(fid), f64::sin(fid)));
                        twiddles.push((f64::cos(twid), f64::sin(twid)));
                        reson_coeffs.push(reson);
                    }

                    VQsDFTCoeffs {
                        period,
                        twiddles,
                        fiddles,
                        reson_coeffs,
                        coeffs1: vec![(0.0, 0.0); items_per_band],
                        coeffs2: vec![(0.0, 0.0); items_per_band],
                        coeffs3: vec![(0.0, 0.0); items_per_band],
                        coeffs4: vec![(0.0, 0.0); items_per_band],
                        coeffs5: vec![(0.0, 0.0); items_per_band],
                    }
                })
                .collect(),
            buffer: vec![0.0; buffer_size + 1],
            buffer_index: buffer_size,
            use_nc,
        }
    }
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buffer_index = self.buffer.len() - 1;
        self.coeffs.iter_mut().for_each(|coeff| {
            coeff.coeffs1.fill((0.0, 0.0));
            coeff.coeffs2.fill((0.0, 0.0));
            coeff.coeffs3.fill((0.0, 0.0));
            coeff.coeffs4.fill((0.0, 0.0));
            coeff.coeffs5.fill((0.0, 0.0));
        });
    }
    fn analyze(&mut self, samples: impl Iterator<Item = f64>) -> &[f64] {
        self.spectrum_data.fill(0.0);

        let buffer_len = self.buffer.len();
        let buffer_len_int = buffer_len as isize;

        for sample in samples {
            self.buffer_index = (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
            self.buffer[self.buffer_index] = sample;

            for i in 0..self.coeffs.len() {
                let coeff = &mut self.coeffs[i];
                let kernel_length = coeff.coeffs1.len();
                let oldest = (((self.buffer_index as isize - coeff.period as isize)
                    % buffer_len_int)
                    + buffer_len_int) as usize
                    % buffer_len;
                let latest = self.buffer_index;
                let mut sum = (0.0, 0.0);

                for j in 0..kernel_length {
                    let fiddle = coeff.fiddles[j];
                    let twiddle = coeff.twiddles[j];

                    let comb_x = self.buffer[latest] * fiddle.0 - self.buffer[oldest];
                    let comb_y = self.buffer[latest] * fiddle.1;

                    coeff.coeffs1[j].0 =
                        comb_x * twiddle.0 - comb_y * twiddle.1 - coeff.coeffs2[j].0;
                    coeff.coeffs1[j].1 =
                        comb_x * twiddle.1 + comb_y * twiddle.0 - coeff.coeffs2[j].1;

                    coeff.coeffs2[j].0 = comb_x;
                    coeff.coeffs2[j].1 = comb_y;

                    coeff.coeffs3[j].0 = coeff.coeffs1[j].0
                        + coeff.reson_coeffs[j] * coeff.coeffs4[j].0
                        - coeff.coeffs5[j].0;
                    coeff.coeffs3[j].1 = coeff.coeffs1[j].1
                        + coeff.reson_coeffs[j] * coeff.coeffs4[j].1
                        - coeff.coeffs5[j].1;

                    coeff.coeffs5[j].0 = coeff.coeffs4[j].0;
                    coeff.coeffs5[j].1 = coeff.coeffs4[j].1;

                    coeff.coeffs4[j].0 = coeff.coeffs3[j].0;
                    coeff.coeffs4[j].1 = coeff.coeffs3[j].1;

                    sum.0 += coeff.coeffs3[j].0 * self.gains[j] / coeff.period;
                    sum.1 += coeff.coeffs3[j].1 * self.gains[j] / coeff.period;
                }
                self.spectrum_data[i] = f64::max(
                    self.spectrum_data[i],
                    if self.use_nc {
                        -(coeff.coeffs3[0].0 / coeff.period * coeff.coeffs3[1].0 / coeff.period)
                            - (coeff.coeffs3[0].1 / coeff.period * coeff.coeffs3[1].1
                                / coeff.period)
                    } else {
                        sum.0.powi(2) + sum.1.powi(2)
                    },
                );
            }
        }
        self.spectrum_data.iter_mut().for_each(|x| *x = x.sqrt());

        &self.spectrum_data
    }
}
