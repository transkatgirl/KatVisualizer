use std::{
    f64::consts::PI,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    simd::{LaneCount, Simd, SupportedLaneCount, f64x2, f64x4, f64x64, num::SimdFloat},
};

use super::FrequencyBand;

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Window {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Nuttall,
    FlatTop,
}

#[derive(Clone)]
pub(super) struct VQsDFT {
    coeffs: VQsDFTCoeffWrapper,
    pub(super) buffer: Vec<f64>,
    buffer_index: usize,
    pub(super) spectrum_data: Vec<f64>,
}

#[derive(Clone)]
enum VQsDFTCoeffWrapper {
    Rectangular(Vec<(f64, VQsDFTCoeffSet<2>)>),
    NC(Vec<(f64, VQsDFTCoeffSet<4>)>),
    ShortWindow((Simd<f64, 6>, Vec<(f64, VQsDFTCoeffSet<6>)>)),
    LongWindow((Vec<f64>, Vec<(f64, Vec<VQsDFTCoeffSet<2>>)>)),
}

#[derive(Default, Clone)]
struct VQsDFTCoeffSet<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    twiddle: Simd<f64, N>,
    flipped_twiddle: Simd<f64, N>,
    fiddle: Simd<f64, N>,
    reson: Simd<f64, N>,
    coeff1: Simd<f64, N>,
    coeff2: Simd<f64, N>,
    coeff3: Simd<f64, N>,
    coeff4: Simd<f64, N>,
    coeff5: Simd<f64, N>,
}

impl<const N: usize> VQsDFTCoeffSet<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn reset(&mut self) {
        self.coeff1 = Simd::<f64, N>::splat(0.0);
        self.coeff2 = Simd::<f64, N>::splat(0.0);
        self.coeff3 = Simd::<f64, N>::splat(0.0);
        self.coeff4 = Simd::<f64, N>::splat(0.0);
        self.coeff5 = Simd::<f64, N>::splat(0.0);
    }
}

impl VQsDFTCoeffSet<2> {
    fn calculate(&mut self, latest: f64, oldest: f64, gain: f64, period: f64) -> Simd<f64, 2> {
        let comb = f64x2::splat(latest)
            .mul(self.fiddle)
            .sub(f64x2::from_array([oldest, 0.0]));

        let [comb_x, comb_y] = comb.to_array();

        self.coeff1 = f64x2::splat(comb_x)
            .mul(self.twiddle)
            .add(f64x2::from_array([-comb_y, comb_y]).mul(self.flipped_twiddle))
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self
            .coeff1
            .add(self.reson.mul(self.coeff4))
            .sub(self.coeff5);

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3
            .mul(f64x2::splat(gain))
            .div(f64x2::splat(period))
    }
    fn calculate_gainless(&mut self, latest: f64, oldest: f64, period: f64) -> Simd<f64, 2> {
        let comb = f64x2::splat(latest)
            .mul(self.fiddle)
            .sub(f64x2::from_array([oldest, 0.0]));

        let [comb_x, comb_y] = comb.to_array();

        self.coeff1 = f64x2::splat(comb_x)
            .mul(self.twiddle)
            .add(f64x2::from_array([-comb_y, comb_y]).mul(self.flipped_twiddle))
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self
            .coeff1
            .add(self.reson.mul(self.coeff4))
            .sub(self.coeff5);

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.div(f64x2::splat(period))
    }
}

impl VQsDFTCoeffSet<4> {
    fn calculate_gainless(&mut self, latest: f64, oldest: f64, period: f64) -> Simd<f64, 4> {
        let comb = f64x4::splat(latest)
            .mul(self.fiddle)
            .sub(f64x4::from_array([oldest, 0.0, oldest, 0.0]));

        let [comb_0_x, comb_0_y, comb_1_x, comb_1_y] = comb.to_array();

        self.coeff1 = f64x4::from_array([comb_0_x, comb_0_x, comb_1_x, comb_1_x])
            .mul(self.twiddle)
            .add(
                f64x4::from_array([-comb_0_y, comb_0_y, -comb_1_y, comb_1_y])
                    .mul(self.flipped_twiddle),
            )
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self
            .coeff1
            .add(self.reson.mul(self.coeff4))
            .sub(self.coeff5);

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.div(f64x4::splat(period))
    }
}

impl VQsDFTCoeffSet<6> {
    fn calculate(
        &mut self,
        latest: f64,
        oldest: f64,
        gain: Simd<f64, 6>,
        period: f64,
    ) -> Simd<f64, 6> {
        let comb = Simd::<f64, 6>::splat(latest)
            .mul(self.fiddle)
            .sub(Simd::<f64, 6>::from_array([
                oldest, 0.0, oldest, 0.0, oldest, 0.0,
            ]));

        let [comb_0_x, comb_0_y, comb_1_x, comb_1_y, comb_2_x, comb_2_y] = comb.to_array();

        self.coeff1 = Simd::<f64, 6>::from_array([
            comb_0_x, comb_0_x, comb_1_x, comb_1_x, comb_2_x, comb_2_x,
        ])
        .mul(self.twiddle)
        .add(
            Simd::<f64, 6>::from_array([
                -comb_0_y, comb_0_y, -comb_1_y, comb_1_y, -comb_2_y, comb_2_y,
            ])
            .mul(self.flipped_twiddle),
        )
        .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self
            .coeff1
            .add(self.reson.mul(self.coeff4))
            .sub(self.coeff5);

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(gain).div(Simd::<f64, 6>::splat(period))
    }
}

impl VQsDFT {
    pub(super) fn new(
        freq_bands: &[FrequencyBand],
        window: Window,
        sample_rate: f64,
        use_nc: bool,
    ) -> Self {
        let buffer_size = sample_rate.trunc() as usize;
        let buffer_size_f64 = buffer_size as f64;

        let window_coeffs: &[f64] = match window {
            Window::Rectangular => &[1.0],
            Window::Hann => &[1.0, 0.5],
            Window::Hamming => &[1.0, 0.4259434938430786],
            Window::Blackman => &[1.0, 0.595257580280304, 0.0952545627951622],
            Window::Nuttall => &[
                1.0,
                0.6850073933601379,
                0.20272639393806458,
                0.017719272524118423,
            ],
            Window::FlatTop => &[
                1.0,
                0.966312825679779,
                0.6430955529212952,
                0.19387830793857574,
                0.016120079904794693,
            ],
        };

        let (min_idx, max_idx) = if use_nc {
            (0, 2)
        } else {
            (
                -(window_coeffs.len() as isize) + 1,
                window_coeffs.len() as isize,
            )
        };
        let items_per_band = (max_idx - min_idx) as usize;

        let gains: Vec<f64> = if use_nc {
            //vec![0.0; 2]
            vec![]
        } else {
            (min_idx..max_idx)
                .map(|i| window_coeffs[i.unsigned_abs()] * (-((i as f64).abs() % 2.0) * 2.0 + 1.0))
                .collect()
        };

        assert!(freq_bands.len().is_multiple_of(64));

        VQsDFT {
            spectrum_data: vec![0.0; freq_bands.len()],
            coeffs: if use_nc {
                VQsDFTCoeffWrapper::NC(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let k_0 = q - 0.5;
                            let k_1 = q + 0.5;
                            let fid_0 = -2.0 * PI * k_0;
                            let twid_0 = (2.0 * PI * k_0) / period;
                            let reson_0 = 2.0 * f64::cos(twid_0);
                            let fid_1 = -2.0 * PI * k_1;
                            let twid_1 = (2.0 * PI * k_1) / period;
                            let reson_1 = 2.0 * f64::cos(twid_1);

                            (
                                period,
                                VQsDFTCoeffSet {
                                    twiddle: f64x4::from_array([
                                        f64::cos(twid_0),
                                        f64::sin(twid_0),
                                        f64::cos(twid_1),
                                        f64::sin(twid_1),
                                    ]),
                                    flipped_twiddle: f64x4::from_array([
                                        f64::sin(twid_0),
                                        f64::cos(twid_0),
                                        f64::sin(twid_1),
                                        f64::cos(twid_1),
                                    ]),
                                    fiddle: f64x4::from_array([
                                        f64::cos(fid_0),
                                        f64::sin(fid_0),
                                        f64::cos(fid_1),
                                        f64::sin(fid_1),
                                    ]),
                                    reson: f64x4::from_array([reson_0, reson_0, reson_1, reson_1]),
                                    coeff1: f64x4::splat(0.0),
                                    coeff2: f64x4::splat(0.0),
                                    coeff3: f64x4::splat(0.0),
                                    coeff4: f64x4::splat(0.0),
                                    coeff5: f64x4::splat(0.0),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::Rectangular {
                VQsDFTCoeffWrapper::Rectangular(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let fid = -2.0 * PI * q;
                            let twid = (2.0 * PI * q) / period;
                            let reson = 2.0 * f64::cos(twid);

                            (
                                period,
                                VQsDFTCoeffSet {
                                    twiddle: f64x2::from_array([f64::cos(twid), f64::sin(twid)]),
                                    flipped_twiddle: f64x2::from_array([
                                        f64::sin(twid),
                                        f64::cos(twid),
                                    ]),
                                    fiddle: f64x2::from_array([f64::cos(fid), f64::sin(fid)]),
                                    reson: f64x2::splat(reson),
                                    coeff1: f64x2::splat(0.0),
                                    coeff2: f64x2::splat(0.0),
                                    coeff3: f64x2::splat(0.0),
                                    coeff4: f64x2::splat(0.0),
                                    coeff5: f64x2::splat(0.0),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::Hann || window == Window::Hamming {
                VQsDFTCoeffWrapper::ShortWindow((
                    Simd::<f64, 6>::from_array([
                        gains[0], gains[0], gains[1], gains[1], gains[2], gains[2],
                    ]),
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let k_0 = q - 1.0;
                            let fid_0 = -2.0 * PI * k_0;
                            let twid_0 = (2.0 * PI * k_0) / period;
                            let reson_0 = 2.0 * f64::cos(twid_0);
                            let k_1 = q;
                            let fid_1 = -2.0 * PI * k_1;
                            let twid_1 = (2.0 * PI * k_1) / period;
                            let reson_1 = 2.0 * f64::cos(twid_1);
                            let k_2 = q + 1.0;
                            let fid_2 = -2.0 * PI * k_2;
                            let twid_2 = (2.0 * PI * k_2) / period;
                            let reson_2 = 2.0 * f64::cos(twid_2);

                            (
                                period,
                                VQsDFTCoeffSet {
                                    twiddle: Simd::<f64, 6>::from_array([
                                        f64::cos(twid_0),
                                        f64::sin(twid_0),
                                        f64::cos(twid_1),
                                        f64::sin(twid_1),
                                        f64::cos(twid_2),
                                        f64::sin(twid_2),
                                    ]),
                                    flipped_twiddle: Simd::<f64, 6>::from_array([
                                        f64::sin(twid_0),
                                        f64::cos(twid_0),
                                        f64::sin(twid_1),
                                        f64::cos(twid_1),
                                        f64::sin(twid_2),
                                        f64::cos(twid_2),
                                    ]),
                                    fiddle: Simd::<f64, 6>::from_array([
                                        f64::cos(fid_0),
                                        f64::sin(fid_0),
                                        f64::cos(fid_1),
                                        f64::sin(fid_1),
                                        f64::cos(fid_2),
                                        f64::sin(fid_2),
                                    ]),
                                    reson: Simd::<f64, 6>::from_array([
                                        reson_0, reson_0, reson_1, reson_1, reson_2, reson_2,
                                    ]),
                                    coeff1: Simd::<f64, 6>::splat(0.0),
                                    coeff2: Simd::<f64, 6>::splat(0.0),
                                    coeff3: Simd::<f64, 6>::splat(0.0),
                                    coeff4: Simd::<f64, 6>::splat(0.0),
                                    coeff5: Simd::<f64, 6>::splat(0.0),
                                },
                            )
                        })
                        .collect(),
                ))
            } else {
                VQsDFTCoeffWrapper::LongWindow((
                    gains,
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let mut fiddles = Vec::with_capacity(items_per_band);
                            let mut twiddles = Vec::with_capacity(items_per_band);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddles.push(f64x2::from_array([f64::cos(fid), f64::sin(fid)]));
                                twiddles.push(f64x2::from_array([f64::cos(twid), f64::sin(twid)]));
                                reson_coeffs.push(reson);
                            }

                            (
                                period,
                                twiddles
                                    .into_iter()
                                    .zip(fiddles.into_iter().zip(reson_coeffs))
                                    .map(|(twiddle, (fiddle, reson))| VQsDFTCoeffSet {
                                        twiddle,
                                        flipped_twiddle: twiddle.reverse(),
                                        fiddle,
                                        reson: f64x2::splat(reson),
                                        coeff1: f64x2::splat(0.0),
                                        coeff2: f64x2::splat(0.0),
                                        coeff3: f64x2::splat(0.0),
                                        coeff4: f64x2::splat(0.0),
                                        coeff5: f64x2::splat(0.0),
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                ))
            },
            buffer: vec![0.0; buffer_size + 1],
            buffer_index: buffer_size,
        }
    }
    pub(super) fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buffer_index = self.buffer.len() - 1;
        match &mut self.coeffs {
            VQsDFTCoeffWrapper::Rectangular(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::NC(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::ShortWindow((_, coeffs)) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::LongWindow((_, coeffs)) => {
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
                });
            }
        }
    }
    pub(super) fn analyze(&mut self, samples: impl ExactSizeIterator<Item = f64>) -> &[f64] {
        self.spectrum_data.fill(0.0);

        let buffer_len = self.buffer.len();
        let buffer_len_int = buffer_len as isize;

        let sample_count = samples.len().max(1) as f64;

        match &mut self.coeffs {
            VQsDFTCoeffWrapper::NC(coeffs) => {
                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for ((period, coeffs), spectrum_data) in
                        coeffs.iter_mut().zip(self.spectrum_data.iter_mut())
                    {
                        let period = *period;

                        let oldest = unsafe {
                            *self.buffer.get_unchecked(
                                (((self.buffer_index as isize - period as isize) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let [coeff3_0_0, coeff3_0_1, coeff3_1_0, coeff3_1_1] =
                            coeffs.calculate_gainless(latest, oldest, period).to_array();

                        *spectrum_data += f64x2::from_array([coeff3_0_0, coeff3_0_1])
                            .mul(f64x2::from_array([coeff3_1_0, coeff3_1_1]))
                            .neg()
                            .reduce_sum()
                            .max(0.0)
                            .sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::Rectangular(coeffs) => {
                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for ((period, coeffs), spectrum_data) in
                        coeffs.iter_mut().zip(self.spectrum_data.iter_mut())
                    {
                        let period = *period;

                        let oldest = unsafe {
                            *self.buffer.get_unchecked(
                                (((self.buffer_index as isize - period as isize) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate_gainless(latest, oldest, period);

                        *spectrum_data += result.mul(result).reduce_sum().sqrt()
                    }
                }
            }
            VQsDFTCoeffWrapper::ShortWindow((gains, coeffs)) => {
                let gains = *gains;

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for ((period, coeffs), spectrum_data) in
                        coeffs.iter_mut().zip(self.spectrum_data.iter_mut())
                    {
                        let period = *period;

                        let oldest = unsafe {
                            *self.buffer.get_unchecked(
                                (((self.buffer_index as isize - period as isize) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest, gains, period).to_array();

                        let sum = f64x2::from_array([result[0], result[1]])
                            .add(f64x2::from_array([result[2], result[3]]))
                            .add(f64x2::from_array([result[4], result[5]]));

                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::LongWindow((gains, coeffs)) => {
                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for ((period, coeffs), spectrum_data) in
                        coeffs.iter_mut().zip(self.spectrum_data.iter_mut())
                    {
                        let period = *period;

                        let oldest = unsafe {
                            *self.buffer.get_unchecked(
                                (((self.buffer_index as isize - period as isize) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };
                        let mut sum = f64x2::splat(0.0);

                        for (coeff, gain) in coeffs.iter_mut().zip(gains.iter().copied()) {
                            sum.add_assign(coeff.calculate(latest, oldest, gain, period));
                        }
                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt()
                    }
                }
            }
        }

        unsafe { self.spectrum_data.as_chunks_unchecked_mut::<64>() }
            .iter_mut()
            .for_each(|chunk| {
                *chunk = f64x64::from_array(*chunk)
                    .div(f64x64::splat(sample_count))
                    .to_array();
            });

        &self.spectrum_data
    }
}
