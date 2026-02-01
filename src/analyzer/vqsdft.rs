use std::f64::consts::PI;

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
    Rectangular(Vec<(isize, VQsDFTCoeffSet)>),
    NC(Vec<(isize, [VQsDFTCoeffSet; 2])>),
    Window2Term(Vec<(isize, [VQsDFTCoeffSet; 3])>),
    Window3Term(Vec<(isize, [VQsDFTCoeffSet; 5])>),
    Window4Term(Vec<(isize, [VQsDFTCoeffSet; 7])>),
    Window5Term(Vec<(isize, [VQsDFTCoeffSet; 9])>),
    ArbitraryWindow(Vec<(isize, Vec<VQsDFTCoeffSet>)>),
}

#[derive(Default, Clone)]
struct VQsDFTCoeffSet {
    twiddle: (f64, f64),
    fiddle: (f64, f64),
    reson: f64,
    coeff1: (f64, f64),
    coeff2: (f64, f64),
    coeff3: (f64, f64),
    coeff4: (f64, f64),
    coeff5: (f64, f64),
    period_gain: f64,
}

impl VQsDFTCoeffSet {
    fn reset(&mut self) {
        self.coeff1 = (0.0, 0.0);
        self.coeff2 = (0.0, 0.0);
        self.coeff3 = (0.0, 0.0);
        self.coeff4 = (0.0, 0.0);
        self.coeff5 = (0.0, 0.0);
    }
    #[inline(always)]
    fn calculate(&mut self, latest: f64, oldest: f64) -> (f64, f64) {
        let comb_x = latest.algebraic_mul(self.fiddle.0).algebraic_sub(oldest);
        let comb_y = latest.algebraic_mul(self.fiddle.1);

        self.coeff1.0 = comb_x
            .algebraic_mul(self.twiddle.0)
            .algebraic_sub(comb_y.algebraic_mul(self.twiddle.1))
            .algebraic_sub(self.coeff2.0);
        self.coeff1.1 = comb_x
            .algebraic_mul(self.twiddle.1)
            .algebraic_add(comb_y.algebraic_mul(self.twiddle.0))
            .algebraic_sub(self.coeff2.1);

        self.coeff2 = (comb_x, comb_y);

        self.coeff3.0 = self
            .coeff1
            .0
            .algebraic_add(self.reson.algebraic_mul(self.coeff4.0))
            .algebraic_sub(self.coeff5.0);
        self.coeff3.1 = self
            .coeff1
            .1
            .algebraic_add(self.reson.algebraic_mul(self.coeff4.1))
            .algebraic_sub(self.coeff5.1);

        self.coeff5 = self.coeff4;
        self.coeff4 = self.coeff3;

        (
            self.coeff3.0.algebraic_mul(self.period_gain),
            self.coeff3.1.algebraic_mul(self.period_gain),
        )
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

                            let period_gain = 1.0 / period;

                            (
                                period as isize,
                                [
                                    VQsDFTCoeffSet {
                                        twiddle: (f64::cos(twid_0), f64::sin(twid_0)),
                                        fiddle: (f64::cos(fid_0), f64::sin(fid_0)),
                                        reson: reson_0,
                                        coeff1: (0.0, 0.0),
                                        coeff2: (0.0, 0.0),
                                        coeff3: (0.0, 0.0),
                                        coeff4: (0.0, 0.0),
                                        coeff5: (0.0, 0.0),
                                        period_gain,
                                    },
                                    VQsDFTCoeffSet {
                                        twiddle: (f64::cos(twid_1), f64::sin(twid_1)),
                                        fiddle: (f64::cos(fid_1), f64::sin(fid_1)),
                                        reson: reson_1,
                                        coeff1: (0.0, 0.0),
                                        coeff2: (0.0, 0.0),
                                        coeff3: (0.0, 0.0),
                                        coeff4: (0.0, 0.0),
                                        coeff5: (0.0, 0.0),
                                        period_gain,
                                    },
                                ],
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
                                    twiddle: (f64::cos(twid), f64::sin(twid)),
                                    fiddle: (f64::cos(fid), f64::sin(fid)),
                                    reson,
                                    coeff1: (0.0, 0.0),
                                    coeff2: (0.0, 0.0),
                                    coeff3: (0.0, 0.0),
                                    coeff4: (0.0, 0.0),
                                    coeff5: (0.0, 0.0),
                                    period_gain: 1.0 / period,
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

                            let mut fiddles = Vec::with_capacity(items_per_band);
                            let mut twiddles = Vec::with_capacity(items_per_band);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddles.push((f64::cos(fid), f64::sin(fid)));
                                twiddles.push((f64::cos(twid), f64::sin(twid)));
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
                                            reson,
                                            coeff1: (0.0, 0.0),
                                            coeff2: (0.0, 0.0),
                                            coeff3: (0.0, 0.0),
                                            coeff4: (0.0, 0.0),
                                            coeff5: (0.0, 0.0),
                                            period_gain,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .unwrap_or_else(|_| panic!()),
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

                            let mut fiddles = Vec::with_capacity(items_per_band);
                            let mut twiddles = Vec::with_capacity(items_per_band);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddles.push((f64::cos(fid), f64::sin(fid)));
                                twiddles.push((f64::cos(twid), f64::sin(twid)));
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
                                            reson,
                                            coeff1: (0.0, 0.0),
                                            coeff2: (0.0, 0.0),
                                            coeff3: (0.0, 0.0),
                                            coeff4: (0.0, 0.0),
                                            coeff5: (0.0, 0.0),
                                            period_gain,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .unwrap_or_else(|_| panic!()),
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

                            let mut fiddles = Vec::with_capacity(items_per_band);
                            let mut twiddles = Vec::with_capacity(items_per_band);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddles.push((f64::cos(fid), f64::sin(fid)));
                                twiddles.push((f64::cos(twid), f64::sin(twid)));
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
                                            reson,
                                            coeff1: (0.0, 0.0),
                                            coeff2: (0.0, 0.0),
                                            coeff3: (0.0, 0.0),
                                            coeff4: (0.0, 0.0),
                                            coeff5: (0.0, 0.0),
                                            period_gain,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .unwrap_or_else(|_| panic!()),
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

                            let mut fiddles = Vec::with_capacity(items_per_band);
                            let mut twiddles = Vec::with_capacity(items_per_band);
                            let mut reson_coeffs = Vec::with_capacity(items_per_band);

                            for i in min_idx..max_idx {
                                let i = i as f64;

                                let k = q + i;
                                let fid = -2.0 * PI * k;
                                let twid = (2.0 * PI * k) / period;
                                let reson = 2.0 * f64::cos(twid);

                                fiddles.push((f64::cos(fid), f64::sin(fid)));
                                twiddles.push((f64::cos(twid), f64::sin(twid)));
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
                                            reson,
                                            coeff1: (0.0, 0.0),
                                            coeff2: (0.0, 0.0),
                                            coeff3: (0.0, 0.0),
                                            coeff4: (0.0, 0.0),
                                            coeff5: (0.0, 0.0),
                                            period_gain,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .unwrap_or_else(|_| panic!()),
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

                                fiddles.push((f64::cos(fid), f64::sin(fid)));
                                twiddles.push((f64::cos(twid), f64::sin(twid)));
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
                                            reson,
                                            coeff1: (0.0, 0.0),
                                            coeff2: (0.0, 0.0),
                                            coeff3: (0.0, 0.0),
                                            coeff4: (0.0, 0.0),
                                            coeff5: (0.0, 0.0),
                                            period_gain,
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
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
                });
            }
            VQsDFTCoeffWrapper::Window2Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
                });
            }
            VQsDFTCoeffWrapper::Window3Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
                });
            }
            VQsDFTCoeffWrapper::Window4Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
                });
            }
            VQsDFTCoeffWrapper::Window5Term(coeffs) => {
                coeffs.iter_mut().for_each(|(_, coeffs)| {
                    coeffs.iter_mut().for_each(|coeff| {
                        coeff.reset();
                    });
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

        let spectrum_chunks = unsafe { self.spectrum_data.as_chunks_unchecked_mut::<64>() };

        match &mut self.coeffs {
            VQsDFTCoeffWrapper::NC(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let left_result = coeffs[0].calculate(latest, oldest);
                            let right_result = coeffs[1].calculate(latest, oldest);

                            *spectrum_data = spectrum_data.algebraic_add(
                                left_result
                                    .0
                                    .algebraic_mul(right_result.0)
                                    .algebraic_add(left_result.1.algebraic_mul(right_result.1))
                                    .algebraic_mul(-1.0)
                                    .max(0.0)
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::Rectangular(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let result = coeffs.calculate(latest, oldest);

                            *spectrum_data = spectrum_data.algebraic_add(
                                result
                                    .0
                                    .algebraic_mul(result.0)
                                    .algebraic_add(result.1.algebraic_mul(result.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::Window2Term(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let mut sum: (f64, f64) = (0.0, 0.0);

                            for coeff in coeffs {
                                let result = coeff.calculate(latest, oldest);

                                sum.0 = sum.0.algebraic_add(result.0);
                                sum.1 = sum.1.algebraic_add(result.1);
                            }

                            *spectrum_data = spectrum_data.algebraic_add(
                                sum.0
                                    .algebraic_mul(sum.0)
                                    .algebraic_add(sum.1.algebraic_mul(sum.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::Window3Term(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let mut sum: (f64, f64) = (0.0, 0.0);

                            for coeff in coeffs {
                                let result = coeff.calculate(latest, oldest);

                                sum.0 = sum.0.algebraic_add(result.0);
                                sum.1 = sum.1.algebraic_add(result.1);
                            }

                            *spectrum_data = spectrum_data.algebraic_add(
                                sum.0
                                    .algebraic_mul(sum.0)
                                    .algebraic_add(sum.1.algebraic_mul(sum.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::Window4Term(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let mut sum: (f64, f64) = (0.0, 0.0);

                            for coeff in coeffs {
                                let result = coeff.calculate(latest, oldest);

                                sum.0 = sum.0.algebraic_add(result.0);
                                sum.1 = sum.1.algebraic_add(result.1);
                            }

                            *spectrum_data = spectrum_data.algebraic_add(
                                sum.0
                                    .algebraic_mul(sum.0)
                                    .algebraic_add(sum.1.algebraic_mul(sum.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::Window5Term(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let mut sum: (f64, f64) = (0.0, 0.0);

                            for coeff in coeffs {
                                let result = coeff.calculate(latest, oldest);

                                sum.0 = sum.0.algebraic_add(result.0);
                                sum.1 = sum.1.algebraic_add(result.1);
                            }

                            *spectrum_data = spectrum_data.algebraic_add(
                                sum.0
                                    .algebraic_mul(sum.0)
                                    .algebraic_add(sum.1.algebraic_mul(sum.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
            VQsDFTCoeffWrapper::ArbitraryWindow(coeffs) => {
                let coeff_chunks = unsafe { coeffs.as_chunks_unchecked_mut::<64>() };

                for sample in samples {
                    self.buffer_index =
                        (((self.buffer_index + 1) % buffer_len) + buffer_len) % buffer_len;
                    let latest = unsafe { self.buffer.get_unchecked_mut(self.buffer_index) };
                    *latest = sample;
                    let latest = sample;

                    for (coeff_chunk, spectrum_chunk) in
                        coeff_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
                    {
                        for ((period, coeffs), spectrum_data) in
                            coeff_chunk.iter_mut().zip(spectrum_chunk.iter_mut())
                        {
                            let period = *period;

                            let oldest = unsafe {
                                *self.buffer.get_unchecked(
                                    (((self.buffer_index as isize - period) % buffer_len_int)
                                        + buffer_len_int)
                                        as usize
                                        % buffer_len,
                                )
                            };

                            let mut sum: (f64, f64) = (0.0, 0.0);

                            for coeff in coeffs {
                                let result = coeff.calculate(latest, oldest);

                                sum.0 = sum.0.algebraic_add(result.0);
                                sum.1 = sum.1.algebraic_add(result.1);
                            }

                            *spectrum_data = spectrum_data.algebraic_add(
                                sum.0
                                    .algebraic_mul(sum.0)
                                    .algebraic_add(sum.1.algebraic_mul(sum.1))
                                    .sqrt(),
                            );
                        }
                    }
                }
            }
        }

        spectrum_chunks.iter_mut().for_each(|chunk| {
            chunk.iter_mut().for_each(|s| {
                *s = s.algebraic_div(sample_count);
            });
        });

        &self.spectrum_data
    }
}
