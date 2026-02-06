#![allow(clippy::excessive_precision)]

use super::{FrequencyBand, FrequencyScale, amplitude_to_dbfs_f32, dbfs_to_amplitude_f32};

// ----- Below algorithms are taken from https://www.gammaelectronics.xyz/poda_6e_11b.html -----

// Only recommended for use on small arrays (len <= 128) due to underflow/overflow issues
/*fn spectral_flatness(spectrum: &[f64]) -> f64 {
    let (count, product, sum): (usize, f64, f64) = spectrum
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| *v * 8192.0) // Helps with underflow/overflow issues
        .fold((0, 1.0, 0.0), |acc, v| (acc.0 + 1, acc.1 * v, acc.2 + v));
    let count = count as f64;

    let geometric_mean = product.powf(1.0 / count);
    let arithmetic_mean = sum / count;

    let flatness = amplitude_to_dbfs(geometric_mean / arithmetic_mean);

    if flatness.is_normal() { flatness } else { 0.0 }
}*/

/*#[inline(always)]
fn masking_threshold_offset(center_bark: f32, flatness: f32) -> f32 {
    let tonal_masking_threshold = -6.025 - (0.275 * center_bark);
    let nontonal_masking_threshold = -2.025 - (0.175 * center_bark);

    tonal_masking_threshold * (1.0 - flatness) + (nontonal_masking_threshold * flatness)
}*/

// ----- Below algorithm is based on the following: -----
// https://link.springer.com/chapter/10.1007/978-3-319-07974-5_2 chapter 2.4
// http://www.mp3-tech.org/programmer/docs/di042001.pdf
// https://dn790006.ca.archive.org/0/items/05shlacpsychacousticsmodelsws201718gs/05_shl_AC_Psychacoustics_Models_WS-2017-18_gs.pdf

const MAX_MASKING_DYNAMIC_RANGE: f32 = 97.0;
const MAX_APPROX_MASKING_DYNAMIC_RANGE: f32 = 85.0;

#[derive(Clone)]
struct MaskerCoeff {
    bark: f32,
    masking_offset_amplitude: f32,
    /*tonal_masking_threshold: f32,
    nontonal_masking_threshold: f32,*/
    lookup: Vec<f32>,
    masking_coeff_1: f32,
    range: (usize, usize),
}

#[derive(Clone)]
pub(super) struct Masker {
    coeffs: Vec<MaskerCoeff>,
    bark_set: Vec<f32>,
    average_width: usize,
    approximate: bool,
}

impl Masker {
    pub(super) fn new(frequency_bands: &[FrequencyBand], approximate: bool) -> Self {
        let frequency_set: Vec<f32> = frequency_bands.iter().map(|f| f.center).collect();

        let mut bark_set: Vec<f32> = frequency_set
            .iter()
            .copied()
            .map(|f| FrequencyScale::Bark.scale(f))
            .collect();

        let band_count = frequency_bands.len();
        let range_indices = frequency_bands.iter().enumerate().map(|(i, f)| {
            let center_bark = FrequencyScale::Bark.scale(f.center);

            let min_masking_spread = (22.0 + (230.0 / f.center).min(10.0)).min(27.0);
            let bark_spread = if approximate {
                MAX_APPROX_MASKING_DYNAMIC_RANGE
            } else {
                MAX_MASKING_DYNAMIC_RANGE
            } / min_masking_spread;

            let lower = (0..i.saturating_sub(1))
                .rev()
                .find(|i| {
                    FrequencyScale::Bark.scale(frequency_bands[*i].high)
                        <= (center_bark - bark_spread)
                })
                .unwrap_or(0);
            let upper = (i..band_count)
                .find(|i| {
                    FrequencyScale::Bark.scale(frequency_bands[*i].low)
                        >= (center_bark + bark_spread)
                })
                .unwrap_or(band_count - 1);

            ((lower + 1).min(i), i, upper.saturating_sub(1))
        });

        const LOWER_SPREAD: f32 = -27.0;
        const AMPLITUDE_GUESS: f32 = -32.39315062; // amplitude_to_dbfs(-21.4 * f64::log10(1 + 0.00437 * 20000))

        let coeffs: Vec<MaskerCoeff> = frequency_set
            .into_iter()
            .zip(bark_set.iter().copied().zip(range_indices))
            .map(|(frequency, (bark, range))| {
                let masking_coeff_1 = 22.0 + (230.0 / frequency).min(10.0);
                let upper_spread = -(masking_coeff_1 - 0.2 * AMPLITUDE_GUESS);

                MaskerCoeff {
                    bark,
                    masking_offset_amplitude: dbfs_to_amplitude_f32(-6.025 - (0.275 * bark))
                        / (band_count as f32 / 41.65407847),
                    /*tonal_masking_threshold: -6.025 - (0.275 * bark),
                    nontonal_masking_threshold: -2.025 - (0.175 * bark),*/
                    lookup: if approximate {
                        (range.0..range.1)
                            .map(|i| dbfs_to_amplitude_f32(LOWER_SPREAD * (bark - bark_set[i])))
                            .chain((range.1..=range.2).map(|i| {
                                dbfs_to_amplitude_f32(upper_spread * (bark_set[i] - bark))
                            }))
                            .collect()
                    } else {
                        (range.0..range.1)
                            .map(|i| dbfs_to_amplitude_f32(LOWER_SPREAD * (bark - bark_set[i])))
                            .collect()
                    },
                    masking_coeff_1,
                    range: (range.0, range.2),
                }
            })
            .collect();

        let average_width = coeffs
            .iter()
            .map(|c| (c.range.1 - c.range.0) as f32 / coeffs.len() as f32)
            .sum::<f32>()
            .round() as usize;

        if approximate {
            bark_set.clear();
            bark_set.shrink_to_fit();
        }

        Self {
            coeffs,
            bark_set,
            average_width,
            approximate,
        }
    }
    pub(super) fn calculate_masking_threshold(
        &self,
        spectrum: impl Iterator<Item = f32>,
        listening_volume: Option<f32>,
        //flatness: f32,
        masking_threshold: &mut [f32],
    ) {
        assert_eq!(masking_threshold.len(), self.coeffs.len());

        masking_threshold.fill(0.0);

        if self.approximate {
            if self.average_width >= 128 {
                /*if self.average_width >= 512 {
                    self.calculate_masking_threshold_inner::<32>(
                        spectrum,
                        listening_volume,
                        masking_threshold,
                        approximate,
                    );
                } else {*/
                self.calculate_masking_threshold_inner_approx::<16>(spectrum, masking_threshold);
                //}
            } else {
                self.calculate_masking_threshold_inner_approx::<8>(spectrum, masking_threshold);
            }
        } else {
            if self.average_width >= 256 {
                /*if self.average_width >= 1024 {
                    self.calculate_masking_threshold_inner::<32>(
                        spectrum,
                        listening_volume,
                        masking_threshold,
                        approximate,
                    );
                } else {*/
                self.calculate_masking_threshold_inner_exact::<16>(
                    spectrum,
                    listening_volume,
                    masking_threshold,
                );
                //}
            } else {
                self.calculate_masking_threshold_inner_exact::<8>(
                    spectrum,
                    listening_volume,
                    masking_threshold,
                );
            }
        }
    }
    fn calculate_masking_threshold_inner_approx<const N: usize>(
        &self,
        spectrum: impl Iterator<Item = f32>,
        masking_threshold: &mut [f32],
    ) {
        for (component, coeff) in spectrum.zip(self.coeffs.iter()) {
            let amplitude = component;

            if amplitude == 0.0 {
                continue;
            }

            let adjusted_amplitude = coeff.masking_offset_amplitude * amplitude;

            {
                let (masking_chunks, masking_rem) =
                    unsafe { masking_threshold.get_unchecked_mut(coeff.range.0..=coeff.range.1) }
                        .as_chunks_mut::<N>();
                let (lookup_chunks, lookup_rem) = unsafe {
                    coeff
                        .lookup
                        .get_unchecked(0..=(coeff.range.1 - coeff.range.0))
                }
                .as_chunks::<N>();

                assert_eq!(masking_chunks.len(), lookup_chunks.len());
                assert_eq!(masking_rem.len(), lookup_rem.len());

                for (masking_chunk, lookup_chunk) in masking_chunks.iter_mut().zip(lookup_chunks) {
                    for (t, x) in masking_chunk.iter_mut().zip(lookup_chunk) {
                        *t = t.algebraic_add(x.algebraic_mul(adjusted_amplitude));
                    }
                }

                for (t, x) in masking_rem.iter_mut().zip(lookup_rem) {
                    *t = t.algebraic_add(x.algebraic_mul(adjusted_amplitude));
                }
            }
        }
    }
    fn calculate_masking_threshold_inner_exact<const N: usize>(
        &self,
        spectrum: impl Iterator<Item = f32>,
        listening_volume: Option<f32>,
        //flatness: f32,
        masking_threshold: &mut [f32],
    ) {
        let amplitude_correction_offset = if let Some(listening_volume) = listening_volume {
            listening_volume - 90.0 // Assume the spreading function was calculated for -0dBFS = 90dBSPL
        } else {
            0.0
        };

        for (i, (component, coeff)) in spectrum.zip(self.coeffs.iter()).enumerate() {
            let amplitude = component;
            let amplitude_db = amplitude_to_dbfs_f32(component);

            if amplitude == 0.0 {
                continue;
            }

            let upper_spread =
                -(coeff.masking_coeff_1 - 0.2 * (amplitude_db + amplitude_correction_offset));

            /*let threshold_offset = masking_threshold_offset(bark, flatness);
            let offset = coeff.tonal_masking_threshold - simultaneous;

            let adjusted_amplitude = dbfs_to_amplitude(offset) * amplitude;*/

            let adjusted_amplitude = coeff.masking_offset_amplitude * amplitude;

            {
                let (masking_chunks, masking_rem) =
                    unsafe { masking_threshold.get_unchecked_mut(coeff.range.0..i) }
                        .as_chunks_mut::<N>();
                let (lookup_chunks, lookup_rem) =
                    unsafe { coeff.lookup.get_unchecked(0..(i - coeff.range.0)) }.as_chunks::<N>();

                assert_eq!(masking_chunks.len(), lookup_chunks.len());
                assert_eq!(masking_rem.len(), lookup_rem.len());

                for (masking_chunk, lookup_chunk) in masking_chunks.iter_mut().zip(lookup_chunks) {
                    for (t, x) in masking_chunk.iter_mut().zip(lookup_chunk) {
                        *t = t.algebraic_add(x.algebraic_mul(adjusted_amplitude));
                    }
                }

                for (t, x) in masking_rem.iter_mut().zip(lookup_rem) {
                    *t = t.algebraic_add(x.algebraic_mul(adjusted_amplitude));
                }
            }

            unsafe { masking_threshold.get_unchecked_mut(i..=coeff.range.1) }
                .iter_mut()
                .zip(
                    unsafe { self.bark_set.get_unchecked(i..=coeff.range.1) }
                        .iter()
                        .copied(),
                )
                .for_each(|(t, b)| {
                    *t = t.algebraic_add(
                        dbfs_to_amplitude_f32(
                            upper_spread.algebraic_mul(b.algebraic_sub(coeff.bark)),
                        )
                        .algebraic_mul(adjusted_amplitude),
                    );
                });
        }
    }
}

/*pub(super) fn bulk_multiply(data: &mut [f32], multiplier: f32) {
    assert!(data.len().is_multiple_of(64));

    unsafe { data.as_chunks_unchecked_mut::<64>() }
        .iter_mut()
        .for_each(|chunk| {
            for i in chunk {
                *i = i.algebraic_mul(multiplier);
            }
        });
}
*/
