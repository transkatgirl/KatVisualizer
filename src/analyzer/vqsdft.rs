use std::{
    f64::consts::PI,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    simd::{Simd, f64x2, f64x4, f64x64, num::SimdFloat, simd_swizzle},
};

use super::FrequencyBand;

// ----- Below algorithm is based on https://codepen.io/TF3RDL/pen/MWLzPoO -----

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
    buffer: Vec<f64>,
    buffer_index: usize,
    pub(super) spectrum_data: Vec<f64>,
}

#[derive(Clone)]
enum VQsDFTCoeffWrapper {
    Rectangular(Vec<(isize, VQsDFTCoeffSet<2>)>),
    NC(Vec<(isize, VQsDFTCoeffSet<4>)>),
    Window2Term(Vec<(isize, VQsDFTCoeffSet<6>)>),
    Window3Term(Vec<(isize, VQsDFTCoeffSet<10>)>),
    Window4Term(Vec<(isize, VQsDFTCoeffSet<14>)>),
    Window5Term(Vec<(isize, VQsDFTCoeffSet<18>)>),
    ArbitraryWindow(Vec<(isize, Vec<VQsDFTCoeffSet<2>>)>),
}

#[derive(Default, Clone)]
struct VQsDFTCoeffSet<const N: usize> {
    twiddle: Simd<f64, N>,
    fiddle: Simd<f64, N>,
    reson: Simd<f64, N>,
    coeff1: Simd<f64, N>,
    coeff2: Simd<f64, N>,
    coeff3: Simd<f64, N>,
    coeff4: Simd<f64, N>,
    coeff5: Simd<f64, N>,
    period_gain: Simd<f64, N>,
}

impl<const N: usize> VQsDFTCoeffSet<N> {
    fn reset(&mut self) {
        self.coeff1 = Simd::<f64, N>::splat(0.0);
        self.coeff2 = Simd::<f64, N>::splat(0.0);
        self.coeff3 = Simd::<f64, N>::splat(0.0);
        self.coeff4 = Simd::<f64, N>::splat(0.0);
        self.coeff5 = Simd::<f64, N>::splat(0.0);
    }
}

impl VQsDFTCoeffSet<2> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 2> {
        let batch = simd_swizzle!(f64x2::splat(latest), self.reson, [0, 1, 2, 3])
            .mul(simd_swizzle!(self.fiddle, self.coeff4, [0, 1, 2, 3]))
            .sub(simd_swizzle!(
                f64x2::from_array([oldest, 0.0]),
                self.coeff5,
                [0, 1, 2, 3]
            ));

        let comb = simd_swizzle!(batch, [0, 1]);

        let twiddled =
            simd_swizzle!(comb, [0, 0, 1, 1]).mul(simd_swizzle!(self.twiddle, [0, 1, 1, 0]));

        self.coeff1 = simd_swizzle!(twiddled, [0, 1])
            .add(simd_swizzle!(twiddled, [2, 3]).mul(f64x2::from_array([-1.0, 1.0])))
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(batch, [2, 3]));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFTCoeffSet<4> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 4> {
        let batch = simd_swizzle!(f64x4::splat(latest), self.reson, [0, 1, 2, 3, 4, 5, 6, 7])
            .mul(simd_swizzle!(
                self.fiddle,
                self.coeff4,
                [0, 1, 2, 3, 4, 5, 6, 7]
            ))
            .sub(simd_swizzle!(
                f64x4::from_array([oldest, 0.0, oldest, 0.0]),
                self.coeff5,
                [0, 1, 2, 3, 4, 5, 6, 7]
            ));

        let comb = simd_swizzle!(batch, [0, 1, 2, 3]);

        let twiddled = simd_swizzle!(comb, [0, 0, 2, 2, 1, 1, 3, 3])
            .mul(simd_swizzle!(self.twiddle, [0, 1, 2, 3, 1, 0, 3, 2]));

        self.coeff1 = simd_swizzle!(twiddled, [0, 1, 2, 3])
            .add(
                simd_swizzle!(twiddled, [4, 5, 6, 7])
                    .mul(f64x4::from_array([-1.0, 1.0, -1.0, 1.0])),
            )
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(batch, [4, 5, 6, 7]));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFTCoeffSet<6> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 6> {
        let batch = simd_swizzle!(
            Simd::<f64, 6>::splat(latest),
            self.reson,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )
        .mul(simd_swizzle!(
            self.fiddle,
            self.coeff4,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ))
        .sub(simd_swizzle!(
            Simd::<f64, 6>::from_array([oldest, 0.0, oldest, 0.0, oldest, 0.0]),
            self.coeff5,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ));

        let comb = simd_swizzle!(batch, [0, 1, 2, 3, 4, 5]);

        let twiddled = simd_swizzle!(comb, [0, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5]).mul(
            simd_swizzle!(self.twiddle, [0, 1, 2, 3, 4, 5, 1, 0, 3, 2, 5, 4]),
        );

        self.coeff1 = simd_swizzle!(twiddled, [0, 1, 2, 3, 4, 5])
            .add(
                simd_swizzle!(twiddled, [6, 7, 8, 9, 10, 11]).mul(Simd::<f64, 6>::from_array([
                    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                ])),
            )
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(batch, [6, 7, 8, 9, 10, 11]));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFTCoeffSet<10> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 10> {
        let batch = simd_swizzle!(
            Simd::<f64, 10>::splat(latest),
            self.reson,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
            ]
        )
        .mul(simd_swizzle!(
            self.fiddle,
            self.coeff4,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
            ]
        ))
        .sub(simd_swizzle!(
            Simd::<f64, 10>::from_array([
                oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0
            ]),
            self.coeff5,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
            ]
        ));

        let comb = simd_swizzle!(batch, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let twiddled = simd_swizzle!(
            comb,
            [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9]
        )
        .mul(simd_swizzle!(
            self.twiddle,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
        ));

        self.coeff1 =
            simd_swizzle!(twiddled, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                .add(
                    simd_swizzle!(twiddled, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).mul(
                        Simd::<f64, 10>::from_array([
                            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                        ]),
                    ),
                )
                .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(
            batch,
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        ));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFTCoeffSet<14> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 14> {
        let batch = simd_swizzle!(
            Simd::<f64, 14>::splat(latest),
            self.reson,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27
            ]
        )
        .mul(simd_swizzle!(
            self.fiddle,
            self.coeff4,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27
            ]
        ))
        .sub(simd_swizzle!(
            Simd::<f64, 14>::from_array([
                oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0,
                oldest, 0.0
            ]),
            self.coeff5,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27
            ]
        ));

        let comb = simd_swizzle!(batch, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);

        let twiddled = simd_swizzle!(
            comb,
            [
                0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11,
                13, 13
            ]
        )
        .mul(simd_swizzle!(
            self.twiddle,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
                13, 12
            ]
        ));

        self.coeff1 = simd_swizzle!(twiddled, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            .add(
                simd_swizzle!(
                    twiddled,
                    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
                )
                .mul(Simd::<f64, 14>::from_array([
                    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                ])),
            )
            .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(
            batch,
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        ));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFTCoeffSet<18> {
    fn calculate(&mut self, latest: f64, oldest: f64) -> Simd<f64, 18> {
        let batch = simd_swizzle!(
            Simd::<f64, 18>::splat(latest),
            self.reson,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
            ]
        )
        .mul(simd_swizzle!(
            self.fiddle,
            self.coeff4,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
            ]
        ))
        .sub(simd_swizzle!(
            Simd::<f64, 18>::from_array([
                oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0, oldest, 0.0,
                oldest, 0.0, oldest, 0.0, oldest, 0.0
            ]),
            self.coeff5,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
            ]
        ));

        let comb = simd_swizzle!(
            batch,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        let twiddled = simd_swizzle!(
            comb,
            [
                0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 1, 1, 3, 3, 5, 5, 7,
                7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17
            ]
        )
        .mul(simd_swizzle!(
            self.twiddle,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 0, 3, 2, 5, 4, 7,
                6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16
            ]
        ));

        self.coeff1 = simd_swizzle!(
            twiddled,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        )
        .add(
            simd_swizzle!(
                twiddled,
                [
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
                ]
            )
            .mul(Simd::<f64, 18>::from_array([
                -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                1.0, -1.0, 1.0,
            ])),
        )
        .sub(self.coeff2);

        self.coeff2 = comb;

        self.coeff3 = self.coeff1.add(simd_swizzle!(
            batch,
            [
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
            ]
        ));

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        self.coeff3.mul(self.period_gain)
    }
}

impl VQsDFT {
    pub(super) fn new(
        freq_bands: &[FrequencyBand],
        window: Window,
        sample_rate: f64,
        use_nc: bool,
        strict_nc: bool,
    ) -> Self {
        assert!(sample_rate > 0.0);

        let max_period = freq_bands
            .iter()
            .map(|x| {
                let q = x.center / (x.high - x.low).abs();
                let period = if strict_nc {
                    // Alternate formula from https://arxiv.org/html/2410.07982v2#S3.E3
                    (((x.center * 2.0) / (x.center * (1.0 / q))).round()
                        * (sample_rate / (x.center * 2.0)))
                        .round()
                        .max(1.0)
                } else {
                    ((sample_rate / x.center) * q).ceil()
                };

                assert!(period >= 1.0);

                period as usize
            })
            .max()
            .unwrap();

        let buffer_size = max_period;
        let buffer_size_f64 = max_period as f64;

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
            vec![0.0; 2]
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
                            let period = if strict_nc {
                                (((x.center * 2.0) / (x.center * (1.0 / q))).round()
                                    * (sample_rate / (x.center * 2.0)))
                                    .round()
                                    .max(1.0)
                            } else {
                                ((sample_rate / x.center) * q).ceil()
                            };

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
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: f64x4::from_array([
                                        f64::cos(twid_0),
                                        f64::sin(twid_0),
                                        f64::cos(twid_1),
                                        f64::sin(twid_1),
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
                                    period_gain: f64x4::splat(1.0 / period),
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
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: f64x2::from_array([f64::cos(twid), f64::sin(twid)]),
                                    fiddle: f64x2::from_array([f64::cos(fid), f64::sin(fid)]),
                                    reson: f64x2::splat(reson),
                                    coeff1: f64x2::splat(0.0),
                                    coeff2: f64x2::splat(0.0),
                                    coeff3: f64x2::splat(0.0),
                                    coeff4: f64x2::splat(0.0),
                                    coeff5: f64x2::splat(0.0),
                                    period_gain: f64x2::splat(1.0 / period),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::Hann || window == Window::Hamming {
                VQsDFTCoeffWrapper::Window2Term(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let mut fiddle = Vec::with_capacity(items_per_band * 2);
                            let mut twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut flipped_twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band * 2);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddle.extend([f64::cos(fid), f64::sin(fid)]);

                                let twid_cos = f64::cos(twid);
                                let twid_sin = f64::sin(twid);

                                twiddle.extend([twid_cos, twid_sin]);
                                flipped_twiddle.extend([twid_sin, twid_cos]);
                                reson_coeffs.extend([reson, reson]);
                            }

                            let period_gains: Vec<f64> = gains
                                .iter()
                                .copied()
                                .map(|g| g / period)
                                .flat_map(|g| [g, g])
                                .collect();

                            (
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: Simd::<f64, 6>::from_array(
                                        twiddle.try_into().unwrap(),
                                    ),
                                    fiddle: Simd::<f64, 6>::from_array(fiddle.try_into().unwrap()),
                                    reson: Simd::<f64, 6>::from_array(
                                        reson_coeffs.try_into().unwrap(),
                                    ),
                                    coeff1: Simd::<f64, 6>::splat(0.0),
                                    coeff2: Simd::<f64, 6>::splat(0.0),
                                    coeff3: Simd::<f64, 6>::splat(0.0),
                                    coeff4: Simd::<f64, 6>::splat(0.0),
                                    coeff5: Simd::<f64, 6>::splat(0.0),
                                    period_gain: Simd::<f64, 6>::from_array(
                                        period_gains.try_into().unwrap(),
                                    ),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::Blackman {
                VQsDFTCoeffWrapper::Window3Term(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let mut fiddle = Vec::with_capacity(items_per_band * 2);
                            let mut twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut flipped_twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band * 2);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddle.extend([f64::cos(fid), f64::sin(fid)]);

                                let twid_cos = f64::cos(twid);
                                let twid_sin = f64::sin(twid);

                                twiddle.extend([twid_cos, twid_sin]);
                                flipped_twiddle.extend([twid_sin, twid_cos]);
                                reson_coeffs.extend([reson, reson]);
                            }

                            let period_gains: Vec<f64> = gains
                                .iter()
                                .copied()
                                .map(|g| g / period)
                                .flat_map(|g| [g, g])
                                .collect();

                            (
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: Simd::<f64, 10>::from_array(
                                        twiddle.try_into().unwrap(),
                                    ),
                                    fiddle: Simd::<f64, 10>::from_array(fiddle.try_into().unwrap()),
                                    reson: Simd::<f64, 10>::from_array(
                                        reson_coeffs.try_into().unwrap(),
                                    ),
                                    coeff1: Simd::<f64, 10>::splat(0.0),
                                    coeff2: Simd::<f64, 10>::splat(0.0),
                                    coeff3: Simd::<f64, 10>::splat(0.0),
                                    coeff4: Simd::<f64, 10>::splat(0.0),
                                    coeff5: Simd::<f64, 10>::splat(0.0),
                                    period_gain: Simd::<f64, 10>::from_array(
                                        period_gains.try_into().unwrap(),
                                    ),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::Nuttall {
                VQsDFTCoeffWrapper::Window4Term(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let mut fiddle = Vec::with_capacity(items_per_band * 2);
                            let mut twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut flipped_twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band * 2);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddle.extend([f64::cos(fid), f64::sin(fid)]);

                                let twid_cos = f64::cos(twid);
                                let twid_sin = f64::sin(twid);

                                twiddle.extend([twid_cos, twid_sin]);
                                flipped_twiddle.extend([twid_sin, twid_cos]);
                                reson_coeffs.extend([reson, reson]);
                            }

                            let period_gains: Vec<f64> = gains
                                .iter()
                                .copied()
                                .map(|g| g / period)
                                .flat_map(|g| [g, g])
                                .collect();

                            (
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: Simd::<f64, 14>::from_array(
                                        twiddle.try_into().unwrap(),
                                    ),
                                    fiddle: Simd::<f64, 14>::from_array(fiddle.try_into().unwrap()),
                                    reson: Simd::<f64, 14>::from_array(
                                        reson_coeffs.try_into().unwrap(),
                                    ),
                                    coeff1: Simd::<f64, 14>::splat(0.0),
                                    coeff2: Simd::<f64, 14>::splat(0.0),
                                    coeff3: Simd::<f64, 14>::splat(0.0),
                                    coeff4: Simd::<f64, 14>::splat(0.0),
                                    coeff5: Simd::<f64, 14>::splat(0.0),
                                    period_gain: Simd::<f64, 14>::from_array(
                                        period_gains.try_into().unwrap(),
                                    ),
                                },
                            )
                        })
                        .collect(),
                )
            } else if window == Window::FlatTop {
                VQsDFTCoeffWrapper::Window5Term(
                    freq_bands
                        .iter()
                        .map(|x| {
                            let q = x.center / (x.high - x.low).abs();
                            let period = (sample_rate / x.center) * q;

                            let period = period.ceil();
                            assert!(period >= 1.0 && period <= buffer_size_f64);
                            let q = (x.center * period) / sample_rate;

                            let mut fiddle = Vec::with_capacity(items_per_band * 2);
                            let mut twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut flipped_twiddle = Vec::with_capacity(items_per_band * 2);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band * 2);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddle.extend([f64::cos(fid), f64::sin(fid)]);

                                let twid_cos = f64::cos(twid);
                                let twid_sin = f64::sin(twid);

                                twiddle.extend([twid_cos, twid_sin]);
                                flipped_twiddle.extend([twid_sin, twid_cos]);
                                reson_coeffs.extend([reson, reson]);
                            }

                            let period_gains: Vec<f64> = gains
                                .iter()
                                .copied()
                                .map(|g| g / period)
                                .flat_map(|g| [g, g])
                                .collect();

                            (
                                period as isize,
                                VQsDFTCoeffSet {
                                    twiddle: Simd::<f64, 18>::from_array(
                                        twiddle.try_into().unwrap(),
                                    ),
                                    fiddle: Simd::<f64, 18>::from_array(fiddle.try_into().unwrap()),
                                    reson: Simd::<f64, 18>::from_array(
                                        reson_coeffs.try_into().unwrap(),
                                    ),
                                    coeff1: Simd::<f64, 18>::splat(0.0),
                                    coeff2: Simd::<f64, 18>::splat(0.0),
                                    coeff3: Simd::<f64, 18>::splat(0.0),
                                    coeff4: Simd::<f64, 18>::splat(0.0),
                                    coeff5: Simd::<f64, 18>::splat(0.0),
                                    period_gain: Simd::<f64, 18>::from_array(
                                        period_gains.try_into().unwrap(),
                                    ),
                                },
                            )
                        })
                        .collect(),
                )
            } else {
                VQsDFTCoeffWrapper::ArbitraryWindow(
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

                            let period_gains = gains.iter().copied().map(|g| g / period);

                            (
                                period as isize,
                                twiddles
                                    .into_iter()
                                    .zip(
                                        fiddles
                                            .into_iter()
                                            .zip(reson_coeffs.into_iter().zip(period_gains)),
                                    )
                                    .map(|(twiddle, (fiddle, (reson, period_gain)))| {
                                        VQsDFTCoeffSet {
                                            twiddle,
                                            fiddle,
                                            reson: f64x2::splat(reson),
                                            coeff1: f64x2::splat(0.0),
                                            coeff2: f64x2::splat(0.0),
                                            coeff3: f64x2::splat(0.0),
                                            coeff4: f64x2::splat(0.0),
                                            coeff5: f64x2::splat(0.0),
                                            period_gain: f64x2::splat(period_gain),
                                        }
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                )
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
            VQsDFTCoeffWrapper::Window2Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::Window3Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::Window4Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::Window5Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeff)| {
                    coeff.reset();
                });
            }
            VQsDFTCoeffWrapper::ArbitraryWindow(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        *spectrum_data += simd_swizzle!(result, [0, 1])
                            .mul(simd_swizzle!(result, [2, 3]))
                            .reduce_sum()
                            .neg()
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        *spectrum_data += result.mul(result).reduce_sum().sqrt()
                    }
                }
            }
            VQsDFTCoeffWrapper::Window2Term(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        let sum = simd_swizzle!(result, [0, 1])
                            .add(simd_swizzle!(result, [2, 3]))
                            .add(simd_swizzle!(result, [4, 5]));

                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::Window3Term(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        let sum4 = simd_swizzle!(result, [0, 1, 2, 3])
                            .add(simd_swizzle!(result, [4, 5, 6, 7]));

                        let sum = simd_swizzle!(sum4, [0, 1])
                            .add(simd_swizzle!(sum4, [2, 3]))
                            .add(simd_swizzle!(result, [8, 9]));

                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::Window4Term(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        let sum4 = simd_swizzle!(result, [0, 1, 2, 3])
                            .add(simd_swizzle!(result, [4, 5, 6, 7]))
                            .add(simd_swizzle!(result, [8, 9, 10, 11]));

                        let sum = simd_swizzle!(sum4, [0, 1])
                            .add(simd_swizzle!(sum4, [2, 3]))
                            .add(simd_swizzle!(result, [12, 13]));

                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::Window5Term(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };

                        let result = coeffs.calculate(latest, oldest);

                        let sum8 = simd_swizzle!(result, [0, 1, 2, 3, 4, 5, 6, 7])
                            .add(simd_swizzle!(result, [8, 9, 10, 11, 12, 13, 14, 15]));

                        let sum4 = simd_swizzle!(sum8, [0, 1, 2, 3])
                            .add(simd_swizzle!(sum8, [4, 5, 6, 7]));

                        let sum = simd_swizzle!(sum4, [0, 1])
                            .add(simd_swizzle!(sum4, [2, 3]))
                            .add(simd_swizzle!(result, [16, 17]));

                        *spectrum_data += sum.mul(sum).reduce_sum().sqrt();
                    }
                }
            }
            VQsDFTCoeffWrapper::ArbitraryWindow(coeffs) => {
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
                                (((self.buffer_index as isize - period) % buffer_len_int)
                                    + buffer_len_int) as usize
                                    % buffer_len,
                            )
                        };
                        let mut sum = f64x2::splat(0.0);

                        for coeff in coeffs.iter_mut() {
                            sum.add_assign(coeff.calculate(latest, oldest));
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
