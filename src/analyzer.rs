use std::{collections::VecDeque, f32::consts::PI};

pub(crate) struct BetterAnalyzer {}

// ----- Below formulas are taken from ISO 226:2023 -----

// ----- Below algorithms are taken from https://codepen.io/TF3RDL/pen/MWLzPoO -----

fn map_value(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
    (x - min) / (max - min) * (target_max - target_min) + target_min
}

enum FrequencyScale {
    Logarithmic,
    Erb,
    Bark,
    Mel,
}

impl FrequencyScale {
    fn scale(&self, x: f32) -> f32 {
        match self {
            Self::Logarithmic => x.log2(),
            Self::Erb => (1.0 + 0.00437 * x).log2(),
            Self::Bark => (26.81 * x) / (1960.0 + x) - 0.53,
            Self::Mel => (1.0 + x / 700.0).log2(),
        }
    }
    fn inv_scale(&self, x: f32) -> f32 {
        match self {
            Self::Logarithmic => 2.0_f32.powf(x),
            Self::Erb => (1.0 / 0.00437) * ((x.log2()) - 1.0),
            Self::Bark => 1960.0 / (26.81 / (x + 0.53) - 1.0),
            Self::Mel => 700.0 * ((x.log2()) - 1.0),
        }
    }
    fn generate_bands(&self, n: usize, low: f32, high: f32, bandwidth: f32) -> Vec<FrequencyBand> {
        (0..n)
            .map(|i| {
                let i = i as f32;
                let target_max = (n - 1) as f32;

                FrequencyBand {
                    low: self.inv_scale(map_value(
                        i - bandwidth,
                        0.0,
                        target_max,
                        self.scale(low),
                        self.scale(high),
                    )),
                    center: self.inv_scale(map_value(
                        i,
                        0.0,
                        target_max,
                        self.scale(low),
                        self.scale(high),
                    )),
                    high: self.inv_scale(map_value(
                        i + bandwidth,
                        0.0,
                        target_max,
                        self.scale(low),
                        self.scale(high),
                    )),
                }
            })
            .collect()
    }
}

struct VQsDFT {
    coeffs: Vec<VQsDFTCoeffs>,
    buffer: Vec<f32>,
    buffer_index: usize,
    spectrum_data: Vec<f32>,
}

//#[derive(Clone, Copy)]
struct VQsDFTCoeffs {
    period: f32,
    twiddles: Vec<(f32, f32)>,
    fiddles: Vec<(f32, f32)>,
    reson_coeffs: Vec<f32>,
    coeffs1: Vec<(f32, f32)>,
    coeffs2: Vec<(f32, f32)>,
    coeffs3: Vec<(f32, f32)>,
    coeffs4: Vec<(f32, f32)>,
    coeffs5: Vec<(f32, f32)>,
    gains: Vec<f32>,
}

struct FrequencyBand {
    low: f32,
    center: f32,
    high: f32,
}

impl VQsDFT {
    fn new(
        freq_bands: &[FrequencyBand],
        window: &[f32],
        time_res: f32,
        bandwidth: f32,
        max_time_res: f32,
        sample_rate: usize,
    ) -> Self {
        let sample_rate_f32 = sample_rate as f32;
        let buffer_size = (sample_rate_f32 * max_time_res / 1000.0).round() as usize;
        let buffer_size_f32 = buffer_size as f32;

        let min_idx = -(window.len() as isize) + 1;
        let max_idx = window.len() as isize;
        let items_per_band = (max_idx - min_idx) as usize;

        VQsDFT {
            spectrum_data: vec![0.0; freq_bands.len()],
            coeffs: freq_bands
                .iter()
                .map(|x| {
                    let mut fiddles = Vec::with_capacity(items_per_band);
                    let mut twiddles = Vec::with_capacity(items_per_band);
                    let mut reson_coeffs = Vec::with_capacity(items_per_band);
                    let mut gains = Vec::with_capacity(items_per_band);

                    let period = (f32::min(
                        buffer_size_f32,
                        sample_rate_f32
                            / (bandwidth * (x.high - x.low).abs() + 1.0 / (time_res / 1000.0)),
                    ))
                    .trunc();

                    for i in min_idx..max_idx {
                        let i_f32 = i as f32;

                        let amplitude =
                            window[i.unsigned_abs()] * (-(i_f32.abs() % 2.0) * 2.0 + 1.0);
                        let k = (x.center * period) / sample_rate_f32 + i_f32;
                        let fid = -2.0 * PI * k;
                        let twid = (2.0 * PI * k) / period;
                        let reson = 2.0 * f32::cos((2.0 * PI * k) / period);

                        fiddles.push((f32::cos(fid), f32::sin(fid)));
                        twiddles.push((f32::cos(twid), f32::sin(twid)));
                        reson_coeffs.push(reson);
                        gains.push(amplitude);
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
                        gains,
                    }
                })
                .collect(),
            buffer: vec![0.0; buffer_size + 1],
            buffer_index: buffer_size,
        }
    }

    fn analyze(&mut self, samples: &[f32]) -> &[f32] {
        self.spectrum_data.fill(0.0);

        let buffer_len = self.buffer.len();

        for sample in samples {
            self.buffer_index = ((self.buffer_index + 1) % buffer_len + buffer_len) % buffer_len;
            self.buffer[self.buffer_index] = *sample;

            for i in 0..self.coeffs.len() {
                let coeff = &mut self.coeffs[i];
                let kernel_length = coeff.coeffs1.len();
                let oldest = ((self.buffer_index - coeff.period as usize) % buffer_len
                    + buffer_len)
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

                    sum.0 += coeff.coeffs3[j].0 * coeff.gains[j] / coeff.period;
                    sum.1 += coeff.coeffs3[j].1 * coeff.gains[j] / coeff.period;
                }
                self.spectrum_data[i] =
                    f32::max(self.spectrum_data[i], sum.0.powi(2) + sum.1.powi(2));
            }
        }
        self.spectrum_data.iter_mut().for_each(|x| *x = x.sqrt());

        &self.spectrum_data
    }
}
