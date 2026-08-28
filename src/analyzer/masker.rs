#![allow(clippy::excessive_precision)]
#![allow(clippy::too_many_arguments)]

use super::{FrequencyBand, FrequencyScale, amplitude_to_dbfs, dbfs_to_amplitude};
use std::f32::consts::{LOG2_10, LOG10_2};

// ----- Below algorithm is based on the following: -----
// https://link.springer.com/chapter/10.1007/978-3-319-07974-5_2 chapter 2.4
// http://www.mp3-tech.org/programmer/docs/di042001.pdf
// https://dn790006.ca.archive.org/0/items/05shlacpsychacousticsmodelsws201718gs/05_shl_AC_Psychacoustics_Models_WS-2017-18_gs.pdf

const MAX_MASKING_DYNAMIC_RANGE: f32 = 97.0;
const MAX_APPROX_MASKING_DYNAMIC_RANGE: f32 = 85.0;
const BARK_INDEX_BUCKETS_PER_BARK: f32 = 64.0;
const DBFS_TO_LOG2_SCALE: f32 = LOG2_10 / 20.0;
const AMPLITUDE_DB_EXPONENT_SCALE: f32 = 0.2 * DBFS_TO_LOG2_SCALE;
const MAX_MASKING_LOG2_DYNAMIC_RANGE: f32 = MAX_MASKING_DYNAMIC_RANGE * DBFS_TO_LOG2_SCALE;

#[inline(always)]
fn fast_amplitude_to_dbfs(amplitude: f32) -> f32 {
    if amplitude.is_sign_positive() && amplitude.is_normal() {
        20.0_f32
            .algebraic_mul(LOG10_2)
            .algebraic_mul(fast_math::log2_raw(amplitude))
    } else {
        amplitude_to_dbfs(amplitude)
    }
}

#[derive(Clone)]
struct BarkIndex {
    first_bark: f32,
    lookup: Vec<u32>,
}

impl BarkIndex {
    fn new(bark_set: &[f32]) -> Self {
        let first_bark = bark_set[0];
        let last_bark = bark_set[bark_set.len() - 1];
        let bucket_count =
            (((last_bark - first_bark) * BARK_INDEX_BUCKETS_PER_BARK).ceil() as usize) + 1;
        let mut lookup = Vec::with_capacity(bucket_count);
        let mut cursor = 0;

        for bucket in 0..bucket_count {
            let bucket_bark = first_bark + bucket as f32 / BARK_INDEX_BUCKETS_PER_BARK;
            while cursor < bark_set.len() && bark_set[cursor] < bucket_bark {
                cursor += 1;
            }
            lookup.push(u32::try_from(cursor).expect("too many masking bands"));
        }

        Self { first_bark, lookup }
    }

    #[inline(always)]
    fn partition_point(&self, bark_set: &[f32], bark: f32, minimum: usize) -> usize {
        let bucket = (((bark.algebraic_sub(self.first_bark))
            .algebraic_mul(BARK_INDEX_BUCKETS_PER_BARK)) as usize)
            .min(self.lookup.len() - 1);
        let mut index = (unsafe { *self.lookup.get_unchecked(bucket) } as usize).max(minimum);

        // Floating point rounding can put the initial estimate on either side of the exact
        // partition. Correcting in both directions preserves slice::partition_point semantics.
        while index > minimum && unsafe { *bark_set.get_unchecked(index - 1) } >= bark {
            index -= 1;
        }
        while index < bark_set.len() && unsafe { *bark_set.get_unchecked(index) } < bark {
            index += 1;
        }

        index
    }
}

#[derive(Clone)]
enum UpperMasker {
    Approximate {
        target_counts: Vec<u32>,
        lookup: Vec<f32>,
    },
    Exact {
        bark_set: Vec<f32>,
        low_bark_set: Vec<f32>,
        low_bark_index: BarkIndex,
        exponent_bases: Vec<f32>,
    },
}

#[derive(Clone)]
pub(super) struct Masker {
    masking_offset_amplitudes: Vec<f32>,
    lower_bounds: Vec<u32>,
    upper: UpperMasker,
    lower_source_weight_scale: Vec<f32>,
    lower_target_scale: Vec<f32>,
}

impl Masker {
    pub(super) fn new(frequency_bands: &[FrequencyBand], approximate: bool) -> Self {
        const LOWER_SPREAD: f32 = -27.0;
        const AMPLITUDE_GUESS: f32 = -32.39315062; // amplitude_to_dbfs(-21.4 * f64::log10(1 + 0.00437 * 20000))

        assert!(
            frequency_bands.len() >= 2,
            "masking requires at least two bands"
        );

        let bark_set: Vec<f32> = frequency_bands
            .iter()
            .map(|band| FrequencyScale::Bark.scale(band.center))
            .collect();
        let low_bark_set: Vec<f32> = frequency_bands
            .iter()
            .map(|band| FrequencyScale::Bark.scale(band.low))
            .collect();
        let high_bark_set: Vec<f32> = frequency_bands
            .iter()
            .map(|band| FrequencyScale::Bark.scale(band.high))
            .collect();

        debug_assert!(low_bark_set.windows(2).all(|values| values[0] <= values[1]));
        debug_assert!(
            high_bark_set
                .windows(2)
                .all(|values| values[0] <= values[1])
        );
        assert!(
            low_bark_set
                .iter()
                .zip(&bark_set)
                .all(|(&low, &center)| low < center),
            "masking bands must have a low edge below their center"
        );

        let band_count = frequency_bands.len();
        let masking_dynamic_range = if approximate {
            MAX_APPROX_MASKING_DYNAMIC_RANGE
        } else {
            MAX_MASKING_DYNAMIC_RANGE
        };
        let lower_bark_spread = masking_dynamic_range / 27.0;
        let mut lower_bounds = Vec::with_capacity(band_count);
        let mut upper_bounds = Vec::with_capacity(if approximate { band_count } else { 0 });

        for (i, band) in frequency_bands.iter().enumerate() {
            let center_bark = bark_set[i];
            let lower_search_end = i.saturating_sub(1);
            let lower = high_bark_set[..lower_search_end]
                .partition_point(|&bark| bark <= center_bark - lower_bark_spread)
                .saturating_sub(1);
            lower_bounds.push(u32::try_from((lower + 1).min(i)).expect("too many masking bands"));

            if approximate {
                let masking_coeff_1 = 22.0 + (230.0 / band.center).min(10.0);
                let approximate_upper_spread = masking_coeff_1 - 0.2 * AMPLITUDE_GUESS;
                let upper_bark_spread = masking_dynamic_range / approximate_upper_spread;
                let upper = i + low_bark_set[i..]
                    .partition_point(|&bark| bark < center_bark + upper_bark_spread);
                upper_bounds.push(upper.min(band_count - 1).saturating_sub(1));
            }
        }

        assert!(lower_bounds.windows(2).all(|values| values[0] <= values[1]));
        debug_assert!(upper_bounds.windows(2).all(|values| values[0] <= values[1]));

        let masking_offset_amplitudes: Vec<f32> = bark_set
            .iter()
            .map(|&bark| {
                dbfs_to_amplitude(-6.025 - (0.275 * bark)) / (band_count as f32 / 41.65407847)
            })
            .collect();
        let exact_exponent_bases: Vec<f32> = if approximate {
            Vec::new()
        } else {
            frequency_bands
                .iter()
                .map(|band| {
                    let masking_coeff_1 = 22.0 + (230.0 / band.center).min(10.0);
                    -masking_coeff_1 * DBFS_TO_LOG2_SCALE
                })
                .collect()
        };

        let lower_scale: Vec<f32> = bark_set
            .iter()
            .map(|&bark| dbfs_to_amplitude(LOWER_SPREAD * bark))
            .collect();
        let lower_source_weight_scale: Vec<f32> = lower_scale
            .iter()
            .zip(&masking_offset_amplitudes)
            .zip(&lower_bounds)
            .enumerate()
            .map(|(source, ((&scale, &masking_offset_amplitude), &lower))| {
                if (lower as usize) < source {
                    scale.algebraic_mul(masking_offset_amplitude)
                } else {
                    0.0
                }
            })
            .collect();
        let lower_target_scale: Vec<f32> = lower_scale.iter().map(|&scale| scale.recip()).collect();

        let upper = if approximate {
            let lookup_len: usize = upper_bounds
                .iter()
                .enumerate()
                .map(|(source, &upper)| upper.saturating_sub(source) + usize::from(upper >= source))
                .sum();
            let mut target_counts = Vec::with_capacity(band_count);
            let mut lookup = Vec::with_capacity(lookup_len);
            for (source, (band, &upper)) in frequency_bands.iter().zip(&upper_bounds).enumerate() {
                let target_count = upper.saturating_sub(source) + usize::from(upper >= source);
                target_counts.push(u32::try_from(target_count).expect("too many masking bands"));
                let source_bark = bark_set[source];
                let masking_coeff_1 = 22.0 + (230.0 / band.center).min(10.0);
                let upper_spread = -(masking_coeff_1 - 0.2 * AMPLITUDE_GUESS);
                let masking_offset_amplitude = masking_offset_amplitudes[source];
                lookup.extend((source..source + target_count).map(|target| {
                    dbfs_to_amplitude(upper_spread * (bark_set[target] - source_bark))
                        .algebraic_mul(masking_offset_amplitude)
                }));
            }
            debug_assert_eq!(lookup.len(), lookup_len);

            UpperMasker::Approximate {
                target_counts,
                lookup,
            }
        } else {
            let low_bark_index = BarkIndex::new(&low_bark_set);
            UpperMasker::Exact {
                bark_set,
                low_bark_set,
                low_bark_index,
                exponent_bases: exact_exponent_bases,
            }
        };

        Self {
            masking_offset_amplitudes,
            lower_bounds,
            upper,
            lower_source_weight_scale,
            lower_target_scale,
        }
    }
    pub(super) fn calculate_masking_threshold(
        &self,
        spectrum: &[f64],
        listening_volume: Option<f32>,
        masking_threshold: &mut [f32],
    ) {
        assert_eq!(
            masking_threshold.len(),
            self.masking_offset_amplitudes.len()
        );
        assert_eq!(spectrum.len(), self.masking_offset_amplitudes.len());

        self.calculate_lower_masking_threshold(spectrum, masking_threshold);

        match &self.upper {
            UpperMasker::Approximate {
                target_counts,
                lookup,
            } => self.calculate_masking_threshold_inner_approx(
                spectrum,
                masking_threshold,
                target_counts,
                lookup,
            ),
            UpperMasker::Exact {
                bark_set,
                low_bark_set,
                low_bark_index,
                exponent_bases,
            } => self.calculate_masking_threshold_inner_exact(
                spectrum,
                listening_volume,
                masking_threshold,
                bark_set,
                low_bark_set,
                low_bark_index,
                exponent_bases,
            ),
        }
    }
    #[inline(never)]
    fn calculate_masking_threshold_inner_approx(
        &self,
        spectrum: &[f64],
        masking_threshold: &mut [f32],
        target_counts: &[u32],
        lookup: &[f32],
    ) {
        // All source and lookup ranges are validated by construction, and the public entry point
        // verifies the input and output lengths. Raw pointers keep those checks out of this loop.
        unsafe {
            let last_source = spectrum.len() - 1;
            let spectrum_ptr = spectrum.as_ptr();
            let target_counts_ptr = target_counts.as_ptr();
            let lookup_ptr = lookup.as_ptr();
            let threshold_ptr = masking_threshold.as_mut_ptr();
            let mut lookup_cursor = lookup_ptr;

            // Construction guarantees that the final source has no upper targets. Peeling it
            // avoids an empty inner loop (and its amplitude branch) on every invocation.
            for source in 0..last_source {
                let target_count = *target_counts_ptr.add(source) as usize;
                let source_lookup = lookup_cursor;
                lookup_cursor = lookup_cursor.add(target_count);
                let amplitude = *spectrum_ptr.add(source) as f32;
                if amplitude == 0.0 {
                    continue;
                }
                let masking = threshold_ptr.add(source);

                for target in 0..target_count {
                    let threshold = masking.add(target);
                    *threshold = (*threshold)
                        .algebraic_add((*source_lookup.add(target)).algebraic_mul(amplitude));
                }
            }
        }
    }
    #[inline(never)]
    fn calculate_masking_threshold_inner_exact(
        &self,
        spectrum: &[f64],
        listening_volume: Option<f32>,
        masking_threshold: &mut [f32],
        bark_set: &[f32],
        low_bark_set: &[f32],
        low_bark_index: &BarkIndex,
        exponent_bases: &[f32],
    ) {
        let listening_volume_exponent_offset = if let Some(listening_volume) = listening_volume {
            // Assume the spreading function was calculated for -0dBFS = 90dBSPL.
            listening_volume
                .algebraic_sub(90.0)
                .algebraic_mul(AMPLITUDE_DB_EXPONENT_SCALE)
        } else {
            0.0
        };

        // The entry point verifies that all per-band arrays have the same length. Keep their
        // bounds checks out of the source and target loops after that one validation.
        unsafe {
            let last_source = spectrum.len() - 1;
            let spectrum_ptr = spectrum.as_ptr();
            let masking_offset_ptr = self.masking_offset_amplitudes.as_ptr();
            let bark_ptr = bark_set.as_ptr();
            let exponent_base_ptr = exponent_bases.as_ptr();
            let threshold_ptr = masking_threshold.as_mut_ptr();

            // The final source cannot mask a higher band. For every preceding source,
            // low_bark[source] < bark[source] guarantees at least the source target itself.
            for i in 0..last_source {
                let amplitude = *spectrum_ptr.add(i) as f32;

                if amplitude == 0.0 {
                    continue;
                }

                let amplitude_db = fast_amplitude_to_dbfs(amplitude);

                let exponent_scale = (*exponent_base_ptr.add(i))
                    .algebraic_add(AMPLITUDE_DB_EXPONENT_SCALE.algebraic_mul(amplitude_db))
                    .algebraic_add(listening_volume_exponent_offset);

                let adjusted_amplitude = (*masking_offset_ptr.add(i)).algebraic_mul(amplitude);
                let source_bark = *bark_ptr.add(i);

                let threshold = threshold_ptr.add(i);
                *threshold = (*threshold).algebraic_add(adjusted_amplitude);

                // Keep the sign decision outside the target loop and make it only once. Negative
                // spreading exponents have a bounded range and can use the unchecked exp2 path.
                if exponent_scale < 0.0 {
                    let upper_bark = source_bark.algebraic_add(
                        MAX_MASKING_LOG2_DYNAMIC_RANGE
                            .algebraic_div(exponent_scale.algebraic_mul(-1.0)),
                    );
                    let upper = low_bark_index.partition_point(low_bark_set, upper_bark, i);
                    let upper_end = upper.min(last_source);
                    let target_count = upper_end - i;
                    debug_assert!({
                        let last_bark = *bark_ptr.add(upper_end - 1);
                        let minimum_exponent =
                            exponent_scale.algebraic_mul(last_bark.algebraic_sub(source_bark));
                        (-151.0..=0.0).contains(&minimum_exponent)
                    });
                    for target in 1..target_count {
                        let threshold = threshold_ptr.add(i + target);
                        let bark = *bark_ptr.add(i + target);
                        let exponent =
                            exponent_scale.algebraic_mul(bark.algebraic_sub(source_bark));
                        *threshold = (*threshold).algebraic_add(
                            fast_math::exp2_raw(exponent).algebraic_mul(adjusted_amplitude),
                        );
                    }
                } else {
                    let target_count = last_source - i;
                    for target in 1..target_count {
                        let threshold = threshold_ptr.add(i + target);
                        let bark = *bark_ptr.add(i + target);
                        let exponent =
                            exponent_scale.algebraic_mul(bark.algebraic_sub(source_bark));
                        *threshold = (*threshold).algebraic_add(
                            fast_math::exp2(exponent).algebraic_mul(adjusted_amplitude),
                        );
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn calculate_lower_masking_threshold(&self, spectrum: &[f64], masking_threshold: &mut [f32]) {
        let mut active_sum = 0.0_f32;

        // The highest band has no lower-masking source above it. Peeling it also makes source =
        // target + 1 valid for every iteration below. A zero precomputed weight represents a
        // source whose lower range is empty, avoiding another per-band branch.
        unsafe {
            let band_count = self.masking_offset_amplitudes.len();
            let spectrum_ptr = spectrum.as_ptr();
            let source_scale_ptr = self.lower_source_weight_scale.as_ptr();
            let target_scale_ptr = self.lower_target_scale.as_ptr();
            let lower_bounds_ptr = self.lower_bounds.as_ptr();
            let threshold_ptr = masking_threshold.as_mut_ptr();
            let mut lower_cursor = band_count;

            *threshold_ptr.add(band_count - 1) = 0.0;

            for target in (0..band_count - 1).rev() {
                let source = target + 1;
                let source_contribution =
                    (*spectrum_ptr.add(source) as f32).algebraic_mul(*source_scale_ptr.add(source));
                active_sum = active_sum.algebraic_add(source_contribution);

                *threshold_ptr.add(target) = active_sum
                    .max(0.0)
                    .algebraic_mul(*target_scale_ptr.add(target));

                // Bounds are monotonic, so the sources expiring here form one contiguous group.
                // Find it backwards, then subtract in ascending source order to stay close to the
                // pairwise accumulation order.
                while lower_cursor > source
                    && (*lower_bounds_ptr.add(lower_cursor - 1) as usize) > target
                {
                    lower_cursor -= 1;
                }
                let expiration_end = lower_cursor;
                while lower_cursor > source
                    && (*lower_bounds_ptr.add(lower_cursor - 1) as usize) == target
                {
                    lower_cursor -= 1;
                }
                for expired_source in lower_cursor..expiration_end {
                    let source = expired_source;
                    let source_contribution = (*spectrum_ptr.add(source) as f32)
                        .algebraic_mul(*source_scale_ptr.add(source));
                    active_sum = active_sum.algebraic_sub(source_contribution);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frequency_bands() -> Vec<FrequencyBand> {
        let ratio = 16_000.0_f32 / 30.0;
        (0..64)
            .map(|i| {
                let center = 30.0 * ratio.powf(i as f32 / 63.0);
                let bandwidth = (center / 17.0).max(12.0);
                FrequencyBand {
                    low: center - bandwidth / 2.0,
                    center,
                    high: center + bandwidth / 2.0,
                }
            })
            .collect()
    }

    fn spectrum() -> Vec<f64> {
        (0..64)
            .map(|i| {
                let x = i as f64;
                ((x * 0.17).sin().abs() * 0.4) + 0.01
            })
            .collect()
    }

    fn reference_threshold_with_math(
        masker: &Masker,
        spectrum: &[f64],
        listening_volume: Option<f32>,
        amplitude_to_dbfs: fn(f32) -> f32,
        exact_exp2: fn(f32) -> f32,
    ) -> Vec<f32> {
        const LOWER_SPREAD: f32 = -27.0;

        let bark_set: Vec<_> = frequency_bands()
            .iter()
            .map(|band| FrequencyScale::Bark.scale(band.center))
            .collect();
        let mut threshold = vec![0.0_f32; spectrum.len()];

        for (source, &component) in spectrum.iter().enumerate() {
            let amplitude = component as f32;
            if amplitude == 0.0 {
                continue;
            }
            let adjusted_amplitude = amplitude * masker.masking_offset_amplitudes[source];

            for target in masker.lower_bounds[source] as usize..source {
                threshold[target] +=
                    dbfs_to_amplitude(LOWER_SPREAD * (bark_set[source] - bark_set[target]))
                        * adjusted_amplitude;
            }

            match &masker.upper {
                UpperMasker::Approximate {
                    target_counts,
                    lookup,
                } => {
                    let lookup_start = target_counts[..source]
                        .iter()
                        .map(|&count| count as usize)
                        .sum::<usize>();
                    let target_end = source + target_counts[source] as usize;
                    for (lookup_index, target) in (source..target_end).enumerate() {
                        threshold[target] += lookup[lookup_start + lookup_index] * amplitude;
                    }
                }
                UpperMasker::Exact {
                    low_bark_set,
                    exponent_bases,
                    ..
                } => {
                    let amplitude_db = amplitude_to_dbfs(amplitude);
                    let listening_volume_exponent_offset = listening_volume
                        .map(|volume| (volume - 90.0).algebraic_mul(AMPLITUDE_DB_EXPONENT_SCALE))
                        .unwrap_or(0.0);
                    let exponent_scale = exponent_bases[source]
                        .algebraic_add(AMPLITUDE_DB_EXPONENT_SCALE.algebraic_mul(amplitude_db))
                        .algebraic_add(listening_volume_exponent_offset);
                    let upper = if exponent_scale < 0.0 {
                        let upper_bark =
                            bark_set[source] + MAX_MASKING_LOG2_DYNAMIC_RANGE / -exponent_scale;
                        let upper = source
                            + low_bark_set[source..].partition_point(|&bark| bark < upper_bark);
                        upper.min(spectrum.len() - 1).saturating_sub(1)
                    } else {
                        spectrum.len().saturating_sub(2)
                    };
                    if source <= upper {
                        threshold[source] += adjusted_amplitude;
                        for target in source + 1..=upper {
                            threshold[target] +=
                                exact_exp2(exponent_scale * (bark_set[target] - bark_set[source]))
                                    * adjusted_amplitude;
                        }
                    }
                }
            }
        }

        threshold
    }

    fn reference_threshold(
        masker: &Masker,
        spectrum: &[f64],
        listening_volume: Option<f32>,
    ) -> Vec<f32> {
        reference_threshold_with_math(
            masker,
            spectrum,
            listening_volume,
            fast_amplitude_to_dbfs,
            fast_math::exp2_raw,
        )
    }

    fn assert_matches_pairwise_reference(
        approximate: bool,
        spectrum: &[f64],
        listening_volume: Option<f32>,
    ) {
        let bands = frequency_bands();
        let masker = Masker::new(&bands, approximate);
        assert_eq!(masker.masking_offset_amplitudes.len(), bands.len());
        assert_eq!(masker.lower_bounds.len(), bands.len());
        let expected = reference_threshold(&masker, spectrum, listening_volume);
        let mut actual = vec![f32::NAN; spectrum.len()];
        masker.calculate_masking_threshold(spectrum, listening_volume, &mut actual);

        for (band, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "band {band}, expected {expected}, actual {actual}"
            );
        }
    }

    #[test]
    fn masking_matches_pairwise_reference_for_spectrum_shapes() {
        let mut one_hot = vec![0.0; 64];
        one_hot[24] = 0.125;
        let mixed: Vec<f64> = spectrum()
            .into_iter()
            .enumerate()
            .map(|(i, amplitude)| if i % 5 == 0 { 0.0 } else { amplitude })
            .collect();
        let spectra = [
            ("dense", spectrum()),
            ("zero", vec![0.0; 64]),
            ("one-hot", one_hot),
            ("mixed", mixed),
        ];

        for approximate in [true, false] {
            let masker = Masker::new(&frequency_bands(), approximate);
            if let UpperMasker::Approximate { target_counts, .. } = &masker.upper {
                assert_eq!(target_counts.len(), 64);
                assert_eq!(target_counts.last(), Some(&0));
            }
            for listening_volume in [Some(96.0), None] {
                for (_name, spectrum) in &spectra {
                    assert_matches_pairwise_reference(approximate, spectrum, listening_volume);
                }
            }
        }
    }

    #[test]
    fn exact_dbfs_conversion_falls_back_for_unusual_amplitudes() {
        for amplitude in [0.0, -1.0, f32::from_bits(1), f32::INFINITY, f32::NAN] {
            let expected = amplitude_to_dbfs(amplitude);
            let actual = fast_amplitude_to_dbfs(amplitude);
            assert!(
                (expected.is_nan() && actual.is_nan()) || expected.to_bits() == actual.to_bits(),
                "amplitude {amplitude}, expected {expected}, actual {actual}"
            );
        }
    }

    #[test]
    fn exact_masking_approximations_stay_within_error_bound() {
        const MAX_RELATIVE_ERROR: f32 = 0.017;

        let bands = frequency_bands();
        let masker = Masker::new(&bands, false);
        let mut maximum_absolute_error = 0.0_f32;
        let mut maximum_relative_error = 0.0_f32;

        for listening_volume in [60.0, 90.0, 120.0] {
            for source in [0, 8, 24, 40, 56, 63] {
                for amplitude in [1.0, 0.1, 0.001] {
                    let mut spectrum = vec![0.0; bands.len()];
                    spectrum[source] = amplitude;
                    let expected = reference_threshold_with_math(
                        &masker,
                        &spectrum,
                        Some(listening_volume),
                        amplitude_to_dbfs,
                        f32::exp2,
                    );
                    let mut actual = vec![f32::NAN; bands.len()];
                    masker.calculate_masking_threshold(
                        &spectrum,
                        Some(listening_volume),
                        &mut actual,
                    );

                    for (target, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                        if expected == 0.0 {
                            assert!(
                                actual.abs() <= f32::EPSILON,
                                "volume {listening_volume}, source {source}, target {target}, \
                                 amplitude {amplitude}, expected zero, actual {actual}"
                            );
                            continue;
                        }
                        let absolute_error = (expected - actual).abs();
                        let relative_error = (expected - actual).abs() / expected.abs();
                        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
                        maximum_relative_error = maximum_relative_error.max(relative_error);
                        assert!(
                            relative_error <= MAX_RELATIVE_ERROR,
                            "volume {listening_volume}, source {source}, target {target}, amplitude \
                             {amplitude}, expected {expected}, actual {actual}, relative error \
                             {relative_error}"
                        );
                    }
                }
            }
        }

        eprintln!(
            "exact masking approximation error: max absolute {maximum_absolute_error:e}, \
             max relative {maximum_relative_error:e}"
        );
    }

    #[test]
    fn bark_index_matches_partition_point() {
        let low_bark_set: Vec<_> = frequency_bands()
            .iter()
            .map(|band| FrequencyScale::Bark.scale(band.low))
            .collect();
        let index = BarkIndex::new(&low_bark_set);
        let mut queries = vec![f32::NEG_INFINITY, f32::INFINITY, f32::NAN];
        queries.extend(low_bark_set.iter().copied());
        queries.extend(
            low_bark_set
                .windows(2)
                .map(|values| (values[0] + values[1]) * 0.5),
        );

        for query in queries {
            for minimum in [0, low_bark_set.len() / 3, low_bark_set.len() - 1] {
                let expected =
                    minimum + low_bark_set[minimum..].partition_point(|&bark| bark < query);
                let actual = index.partition_point(&low_bark_set, query, minimum);
                assert_eq!(expected, actual, "query={query}, minimum={minimum}");
            }
        }
    }
}
