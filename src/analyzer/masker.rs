use super::{FrequencyBand, FrequencyScale, amplitude_to_dbfs, dbfs_to_amplitude};

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
fn masking_threshold_offset(center_bark: f64, flatness: f64) -> f64 {
    let tonal_masking_threshold = -6.025 - (0.275 * center_bark);
    let nontonal_masking_threshold = -2.025 - (0.175 * center_bark);

    tonal_masking_threshold * (1.0 - flatness) + (nontonal_masking_threshold * flatness)
}*/

// ----- Below algorithm is based on the following: -----
// https://link.springer.com/chapter/10.1007/978-3-319-07974-5_2 chapter 2.4
// https://www.mp3-tech.org/programmer/docs/di042001.pdf
// https://dn790006.ca.archive.org/0/items/05shlacpsychacousticsmodelsws201718gs/05_shl_AC_Psychacoustics_Models_WS-2017-18_gs.pdf

const MAX_MASKING_DYNAMIC_RANGE: f64 = 100.0;

#[derive(Clone)]
struct MaskerCoeff {
    bark: f64,
    masking_offset_amplitude: f64,
    /*tonal_masking_threshold: f64,
    nontonal_masking_threshold: f64,*/
    lower_lookup: Vec<f64>,
    masking_coeff_1: f64,
    range: (usize, usize),
}

#[derive(Clone)]
pub(super) struct Masker {
    coeffs: Vec<MaskerCoeff>,
    bark_set: Vec<f64>,
}

impl Masker {
    pub(super) fn new(frequency_bands: &[FrequencyBand]) -> Self {
        let frequency_set: Vec<f64> = frequency_bands.iter().map(|f| f.center).collect();

        let bark_set: Vec<f64> = frequency_set
            .iter()
            .copied()
            .map(|f| FrequencyScale::Bark.scale(f))
            .collect();

        let band_count = frequency_bands.len();
        let range_indices = frequency_bands.iter().enumerate().map(|(i, f)| {
            let center_bark = FrequencyScale::Bark.scale(f.center);

            let min_masking_spread = (22.0 + (230.0 / f.center).min(10.0)).min(27.0);
            let bark_spread = MAX_MASKING_DYNAMIC_RANGE / min_masking_spread;

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

        const LOWER_SPREAD: f64 = -27.0;

        Self {
            coeffs: frequency_set
                .into_iter()
                .zip(bark_set.iter().copied().zip(range_indices))
                .map(|(frequency, (bark, range))| MaskerCoeff {
                    bark,
                    masking_offset_amplitude: dbfs_to_amplitude(-6.025 - (0.275 * bark))
                        / (band_count as f64 / 41.65407847),
                    /*tonal_masking_threshold: -6.025 - (0.275 * bark),
                    nontonal_masking_threshold: -2.025 - (0.175 * bark),*/
                    lower_lookup: (range.0..range.1)
                        .map(|i| {
                            dbfs_to_amplitude(
                                LOWER_SPREAD.algebraic_mul(bark.algebraic_sub(bark_set[i])),
                            )
                        })
                        .collect(),
                    masking_coeff_1: 22.0 + (230.0 / frequency).min(10.0),
                    range: (range.0, range.2),
                })
                .collect(),
            bark_set,
        }
    }
    pub(super) fn calculate_masking_threshold(
        &self,
        spectrum: impl Iterator<Item = f64>,
        listening_volume: Option<f64>,
        //flatness: f64,
        masking_threshold: &mut [f64],
    ) {
        assert_eq!(masking_threshold.len(), self.bark_set.len());

        masking_threshold.fill(0.0);

        let amplitude_correction_offset = if let Some(listening_volume) = listening_volume {
            listening_volume - 90.0 // Assume the spreading function was calculated for -0dBFS = 90dBSPL
        } else {
            0.0
        };

        for (i, (component, coeff)) in spectrum.zip(self.coeffs.iter()).enumerate() {
            let amplitude = component;
            let amplitude_db = amplitude_to_dbfs(component);

            if amplitude == 0.0 {
                continue;
            }

            let upper_spread =
                -(coeff.masking_coeff_1 - 0.2 * (amplitude_db + amplitude_correction_offset));

            /*let threshold_offset = masking_threshold_offset(bark, flatness);
            let offset = coeff.tonal_masking_threshold - simultaneous;

            let adjusted_amplitude = dbfs_to_amplitude(offset) * amplitude;*/

            let adjusted_amplitude = coeff.masking_offset_amplitude * amplitude;

            (coeff.range.0..i).for_each(|i| {
                let t = unsafe { masking_threshold.get_unchecked_mut(i) };
                let x = unsafe { *coeff.lower_lookup.get_unchecked(i - coeff.range.0) };

                *t = t.algebraic_add(x.algebraic_mul(adjusted_amplitude));
            });

            (i..=coeff.range.1).for_each(|i| {
                let t = unsafe { masking_threshold.get_unchecked_mut(i) };
                let b = unsafe { *self.bark_set.get_unchecked(i) };

                *t = t.algebraic_add(
                    dbfs_to_amplitude(upper_spread.algebraic_mul(b.algebraic_sub(coeff.bark)))
                        .algebraic_mul(adjusted_amplitude),
                );
            });
        }
    }
}

/*pub(super) fn bulk_multiply(data: &mut [f64], multiplier: f64) {
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
