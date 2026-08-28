#![allow(clippy::too_many_arguments)]

use std::{f64::consts::PI, mem::MaybeUninit};

use super::FrequencyBand;

// ----- Below algorithm is based on https://codepen.io/TF3RDL/pen/MWLzPoO -----

const BAND_TILE_LEN: usize = 64;
const SAMPLE_TILE_LEN: usize = 8; // 4096 * 8 = 32768
#[cfg(not(target_arch = "wasm32"))]
const WINDOW_LANE_TILE_LEN: usize = 32;
// A full band tile lets LLVM vectorize the temporally blocked loop for WASM;
// smaller subtiles benchmark better on native targets but inhibit that pass.
#[cfg(target_arch = "wasm32")]
const WINDOW_LANE_TILE_LEN: usize = BAND_TILE_LEN;
const MIN_REUSED_PERIOD_RUN_LEN: usize = 4;

/// Returns a fixed-width view without checking the range in release builds.
///
/// # Safety
///
/// `start..start + BAND_TILE_LEN` must be contained in `slice`.
#[inline(always)]
unsafe fn get_band_tile<T>(slice: &[T], start: usize) -> &[T; BAND_TILE_LEN] {
    debug_assert!(
        start
            .checked_add(BAND_TILE_LEN)
            .is_some_and(|end| end <= slice.len())
    );
    // SAFETY: upheld by the caller. The pointer is aligned because it comes
    // from a slice of `T`, and the array has the same element layout.
    unsafe { &*slice.as_ptr().add(start).cast::<[T; BAND_TILE_LEN]>() }
}

/// Returns a mutable fixed-width view without checking the range in release
/// builds.
///
/// # Safety
///
/// `start..start + BAND_TILE_LEN` must be contained in `slice`.
#[inline(always)]
unsafe fn get_band_tile_mut<T>(slice: &mut [T], start: usize) -> &mut [T; BAND_TILE_LEN] {
    debug_assert!(
        start
            .checked_add(BAND_TILE_LEN)
            .is_some_and(|end| end <= slice.len())
    );
    // SAFETY: upheld by the caller. The exclusive slice borrow guarantees
    // that the returned array is not aliased.
    unsafe { &mut *slice.as_mut_ptr().add(start).cast::<[T; BAND_TILE_LEN]>() }
}

/// Returns two disjoint mutable fixed-width views without checking their
/// ranges in release builds.
///
/// # Safety
///
/// Both tile ranges must be contained in `slice` and must not overlap.
#[inline(always)]
unsafe fn get_two_band_tiles_mut<T>(
    slice: &mut [T],
    first_start: usize,
    second_start: usize,
) -> (&mut [T; BAND_TILE_LEN], &mut [T; BAND_TILE_LEN]) {
    debug_assert!(
        first_start
            .checked_add(BAND_TILE_LEN)
            .is_some_and(|end| end <= slice.len())
    );
    debug_assert!(
        second_start
            .checked_add(BAND_TILE_LEN)
            .is_some_and(|end| end <= slice.len())
    );
    debug_assert!(
        first_start.abs_diff(second_start) >= BAND_TILE_LEN,
        "mutable band tiles must not overlap"
    );

    let pointer = slice.as_mut_ptr();
    // SAFETY: the caller guarantees that both ranges are in bounds and
    // disjoint, so creating both exclusive array references is valid.
    unsafe {
        (
            &mut *pointer.add(first_start).cast::<[T; BAND_TILE_LEN]>(),
            &mut *pointer.add(second_start).cast::<[T; BAND_TILE_LEN]>(),
        )
    }
}

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

pub(super) struct VQsDFT {
    coeffs: VQsDFTCoefficients,
    /// A mirrored ring buffer. The second half always duplicates the first so
    /// delayed samples can be loaded without per-band wraparound arithmetic.
    buffer: Vec<f64>,
    buffer_len: usize,
    buffer_index: usize,
    pub(super) spectrum_data: Vec<f64>,
}

/// Coefficients and live recurrence state in term-major structure-of-arrays form.
/// Every range `term * band_count..(term + 1) * band_count` is contiguous so LLVM
/// can vectorize the independent per-band recurrence.
struct VQsDFTCoefficients {
    band_count: usize,
    term_count: usize,
    use_nc: bool,
    min_period: usize,
    periods: Vec<usize>,
    /// Absolute end of each comb-preparation segment, stored at the segment's
    /// first band. Equal-period runs form reuse segments, while consecutive
    /// short runs are merged so their arithmetic remains vectorizable.
    period_segment_ends: Vec<usize>,
    period_segment_reuses: Vec<bool>,
    inverse_periods: Vec<f64>,
    term_gains: Vec<f64>,
    twiddle_x: Vec<f64>,
    twiddle_y: Vec<f64>,
    fiddle_x: Vec<f64>,
    fiddle_y: Vec<f64>,
    coeff2_x: Vec<f64>,
    coeff2_y: Vec<f64>,
    coeff4_x: Vec<f64>,
    coeff4_y: Vec<f64>,
    coeff5_x: Vec<f64>,
    coeff5_y: Vec<f64>,
}

impl VQsDFT {
    pub(super) fn new(
        freq_bands: &[FrequencyBand],
        window: Window,
        sample_rate: f32,
        use_nc: bool,
        strict_nc: bool,
    ) -> Self {
        assert!(sample_rate > 0.0);
        assert!(
            !freq_bands.is_empty() && freq_bands.len().is_multiple_of(BAND_TILE_LEN),
            "VQsDFT band count must be a nonzero multiple of {BAND_TILE_LEN}"
        );

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

        let min_term = if use_nc {
            0
        } else {
            -(window_coeffs.len() as isize) + 1
        };
        let term_count = if use_nc {
            2
        } else {
            window_coeffs.len() * 2 - 1
        };
        let band_count = freq_bands.len();
        let coefficient_count = term_count
            .checked_mul(band_count)
            .expect("VQsDFT coefficient count is too large");
        let sample_rate_f64 = f64::from(sample_rate);

        let periods: Vec<usize> = freq_bands
            .iter()
            .map(|band| {
                let center = f64::from(band.center);
                let q = center / (f64::from(band.high) - f64::from(band.low)).abs();
                let period = if use_nc && strict_nc {
                    (((center * 2.0) / (center * (1.0 / q))).round()
                        * (sample_rate_f64 / (center * 2.0)))
                        .round()
                        .max(1.0)
                } else {
                    ((sample_rate_f64 / center) * q).ceil()
                };
                assert!(period >= 1.0);
                period as usize
            })
            .collect();
        let max_period = periods.iter().copied().max().unwrap();
        let min_period = periods.iter().copied().min().unwrap();
        let mut period_segment_ends = vec![0; band_count];
        let mut period_segment_reuses = vec![false; band_count];
        for tile_start in (0..band_count).step_by(BAND_TILE_LEN) {
            let tile_end = tile_start + BAND_TILE_LEN;
            let mut segment_start = tile_start;
            while segment_start < tile_end {
                let period = periods[segment_start];
                let mut run_end = segment_start + 1;
                while run_end < tile_end && periods[run_end] == period {
                    run_end += 1;
                }

                let reuse_period = run_end - segment_start >= MIN_REUSED_PERIOD_RUN_LEN;
                let segment_end = if reuse_period {
                    run_end
                } else {
                    // Preserve one long vectorizable loop across neighboring
                    // short runs. Stop before the next reusable run.
                    let mut singleton_end = run_end;
                    while singleton_end < tile_end {
                        let next_period = periods[singleton_end];
                        let mut next_run_end = singleton_end + 1;
                        while next_run_end < tile_end && periods[next_run_end] == next_period {
                            next_run_end += 1;
                        }
                        if next_run_end - singleton_end >= MIN_REUSED_PERIOD_RUN_LEN {
                            break;
                        }
                        singleton_end = next_run_end;
                    }
                    singleton_end
                };

                period_segment_ends[segment_start] = segment_end;
                period_segment_reuses[segment_start] = reuse_period;
                segment_start = segment_end;
            }
        }
        let inverse_periods: Vec<f64> = periods.iter().map(|&period| 1.0 / period as f64).collect();

        let mut twiddle_x = Vec::with_capacity(coefficient_count);
        let mut twiddle_y = Vec::with_capacity(coefficient_count);
        let mut term_gains = Vec::with_capacity(term_count);

        for term in 0..term_count {
            let offset = if use_nc {
                term as f64 - 0.5
            } else {
                (min_term + term as isize) as f64
            };
            let window_gain = if use_nc {
                1.0
            } else {
                window_coeffs[offset.abs() as usize] * (-((offset.abs() % 2.0) * 2.0 + 1.0))
            };
            term_gains.push(window_gain);

            for (band, &period) in freq_bands.iter().zip(&periods) {
                let period = period as f64;
                let q = (f64::from(band.center) * period) / sample_rate_f64;
                let k = q + offset;
                let twiddle = (2.0 * PI * k) / period;

                twiddle_x.push(twiddle.cos());
                twiddle_y.push(twiddle.sin());
            }
        }

        // All supported term offsets differ by integers, so `fiddle` is the
        // same complex value for every term. Use the first term's phase to
        // preserve the reference implementation's argument reduction.
        let first_offset = if use_nc { -0.5 } else { min_term as f64 };
        let (fiddle_x, fiddle_y): (Vec<_>, Vec<_>) = freq_bands
            .iter()
            .zip(&periods)
            .map(|(band, &period)| {
                let q = (f64::from(band.center) * period as f64) / sample_rate_f64;
                let fiddle = -2.0 * PI * (q + first_offset);
                (fiddle.cos(), fiddle.sin())
            })
            .unzip();

        let coeffs = VQsDFTCoefficients {
            band_count,
            term_count,
            use_nc,
            min_period,
            periods,
            period_segment_ends,
            period_segment_reuses,
            inverse_periods,
            term_gains,
            twiddle_x,
            twiddle_y,
            fiddle_x,
            fiddle_y,
            coeff2_x: vec![0.0; band_count],
            coeff2_y: vec![0.0; band_count],
            coeff4_x: vec![0.0; coefficient_count],
            coeff4_y: vec![0.0; coefficient_count],
            coeff5_x: vec![0.0; coefficient_count],
            coeff5_y: vec![0.0; coefficient_count],
        };

        let buffer_len = max_period
            .checked_add(1)
            .expect("VQsDFT period is too large");
        let mirrored_buffer_len = buffer_len
            .checked_mul(2)
            .expect("VQsDFT period is too large");

        Self {
            coeffs,
            buffer: vec![0.0; mirrored_buffer_len],
            buffer_len,
            buffer_index: buffer_len - 1,
            spectrum_data: vec![0.0; band_count],
        }
    }

    pub(super) fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buffer_index = self.buffer_len - 1;
        self.coeffs.coeff2_x.fill(0.0);
        self.coeffs.coeff2_y.fill(0.0);
        self.coeffs.coeff4_x.fill(0.0);
        self.coeffs.coeff4_y.fill(0.0);
        self.coeffs.coeff5_x.fill(0.0);
        self.coeffs.coeff5_y.fill(0.0);
    }

    pub(super) fn analyze(&mut self, samples: impl ExactSizeIterator<Item = f64>) -> &[f64] {
        self.spectrum_data.fill(0.0);
        let sample_count = samples.len().max(1) as f64;

        // Select a const-generic implementation once per block. Each arm is
        // monomorphized with a fixed mode and term count, allowing LLVM to
        // remove the mode branch and unroll the short term loops.
        match (self.coeffs.use_nc, self.coeffs.term_count) {
            (true, 2) => self.analyze_samples::<true, 2>(samples),
            (false, 1) => self.analyze_samples::<false, 1>(samples),
            (false, 3) => self.analyze_samples::<false, 3>(samples),
            (false, 5) => self.analyze_samples::<false, 5>(samples),
            (false, 7) => self.analyze_samples::<false, 7>(samples),
            (false, 9) => self.analyze_samples::<false, 9>(samples),
            _ => unreachable!("invalid VQsDFT mode and term count"),
        }

        for value in &mut self.spectrum_data {
            *value = value.algebraic_div(sample_count);
        }

        &self.spectrum_data
    }

    #[inline(always)]
    fn analyze_samples<const USE_NC: bool, const TERM_COUNT: usize>(
        &mut self,
        mut samples: impl Iterator<Item = f64>,
    ) {
        let mut sample_tile = [0.0_f64; SAMPLE_TILE_LEN];

        loop {
            let mut sample_count = 0;
            while sample_count < SAMPLE_TILE_LEN {
                let Some(sample) = samples.next() else {
                    break;
                };
                sample_tile[sample_count] = sample;
                sample_count += 1;
            }

            if sample_count == 0 {
                break;
            }

            macro_rules! analyze_tile {
                ($sample_start:literal, $sample_count:literal) => {{
                    let samples: &[f64; $sample_count] = sample_tile
                        [$sample_start..$sample_start + $sample_count]
                        .try_into()
                        .expect("sample tile has the selected length");
                    self.analyze_sample_tile::<USE_NC, TERM_COUNT, $sample_count>(samples);
                }};
            }

            match sample_count {
                1 => analyze_tile!(0, 1),
                2 => analyze_tile!(0, 2),
                3 => {
                    analyze_tile!(0, 2);
                    analyze_tile!(2, 1);
                }
                4 => analyze_tile!(0, 4),
                5 => {
                    analyze_tile!(0, 4);
                    analyze_tile!(4, 1);
                }
                6 => {
                    analyze_tile!(0, 4);
                    analyze_tile!(4, 2);
                }
                7 => {
                    analyze_tile!(0, 4);
                    analyze_tile!(4, 2);
                    analyze_tile!(6, 1);
                }
                8 => analyze_tile!(0, 8),
                _ => unreachable!("sample tile is larger than SAMPLE_TILE_LEN"),
            }
        }
    }

    #[inline(always)]
    fn analyze_sample_tile<
        const USE_NC: bool,
        const TERM_COUNT: usize,
        const SAMPLE_COUNT: usize,
    >(
        &mut self,
        samples: &[f64; SAMPLE_COUNT],
    ) {
        debug_assert_eq!(self.coeffs.band_count % BAND_TILE_LEN, 0);
        debug_assert!(SAMPLE_COUNT > 0 && SAMPLE_COUNT <= SAMPLE_TILE_LEN);
        let initial_buffer_index = self.buffer_index;
        let mut virtual_buffer_indices = [0; SAMPLE_COUNT];
        let mut virtual_buffer_index = initial_buffer_index;

        // History is not committed until every band has consumed the tile, so
        // compute the virtual ring cursor for each staged sample once here.
        // This replaces a runtime modulo in every band/sample combination.
        for index in &mut virtual_buffer_indices {
            let next = virtual_buffer_index + 1;
            virtual_buffer_index = next - self.buffer_len * usize::from(next == self.buffer_len);
            *index = virtual_buffer_index;
        }

        if self.coeffs.min_period >= SAMPLE_COUNT {
            self.analyze_sample_tile_inner::<USE_NC, TERM_COUNT, SAMPLE_COUNT, true>(
                samples,
                &virtual_buffer_indices,
            );
        } else {
            self.analyze_sample_tile_inner::<USE_NC, TERM_COUNT, SAMPLE_COUNT, false>(
                samples,
                &virtual_buffer_indices,
            );
        }

        // Commit staged history after every band has consumed the pre-tile
        // state. Short delays were read directly from `samples` above.
        for &latest in samples {
            append_history_sample(
                &mut self.buffer,
                self.buffer_len,
                &mut self.buffer_index,
                latest,
            );
        }
    }

    #[inline(always)]
    fn analyze_sample_tile_inner<
        const USE_NC: bool,
        const TERM_COUNT: usize,
        const SAMPLE_COUNT: usize,
        const HISTORY_ONLY: bool,
    >(
        &mut self,
        samples: &[f64; SAMPLE_COUNT],
        virtual_buffer_indices: &[usize; SAMPLE_COUNT],
    ) {
        debug_assert!(!HISTORY_ONLY || self.coeffs.min_period >= SAMPLE_COUNT);

        for band_start in (0..self.coeffs.band_count).step_by(BAND_TILE_LEN) {
            let mut comb_x = MaybeUninit::<[[f64; BAND_TILE_LEN]; SAMPLE_COUNT]>::uninit();
            let mut comb_y = MaybeUninit::<[[f64; BAND_TILE_LEN]; SAMPLE_COUNT]>::uninit();

            // SAFETY: both destinations contain exactly SAMPLE_COUNT complete
            // band tiles. `prepare_comb_tiles()` initializes every element.
            unsafe {
                prepare_comb_tiles::<SAMPLE_COUNT, HISTORY_ONLY>(
                    &self.coeffs,
                    &self.buffer,
                    self.buffer_len,
                    samples,
                    virtual_buffer_indices,
                    band_start,
                    comb_x.as_mut_ptr().cast::<f64>(),
                    comb_y.as_mut_ptr().cast::<f64>(),
                );
            }
            // SAFETY: the preparation loop initialized both complete arrays.
            let comb_x = unsafe { comb_x.assume_init_ref() };
            let comb_y = unsafe { comb_y.assume_init_ref() };

            // SAFETY: the band loop advances only in complete tiles and the
            // spectrum has exactly `band_count` elements.
            let spectrum = unsafe { get_band_tile_mut(&mut self.spectrum_data, band_start) };

            if USE_NC {
                debug_assert_eq!(TERM_COUNT, 2);
                calculate_nc_tile_block::<SAMPLE_COUNT>(
                    &mut self.coeffs,
                    band_start,
                    comb_x,
                    comb_y,
                    spectrum,
                );
            } else {
                calculate_window_tile_block::<TERM_COUNT, SAMPLE_COUNT>(
                    &mut self.coeffs,
                    band_start,
                    comb_x,
                    comb_y,
                    spectrum,
                );
            }
        }
    }
}

#[inline(always)]
fn append_history_sample(
    buffer: &mut [f64],
    buffer_len: usize,
    buffer_index: &mut usize,
    latest: f64,
) {
    debug_assert_eq!(buffer.len(), buffer_len * 2);
    debug_assert!(*buffer_index < buffer_len);

    let next = *buffer_index + 1;
    *buffer_index = next - buffer_len * usize::from(next == buffer_len);
    let mirror_index = *buffer_index + buffer_len;
    // SAFETY: the wrapped cursor is in the first half and adding
    // `buffer_len` selects the corresponding element in the second half.
    unsafe {
        *buffer.get_unchecked_mut(*buffer_index) = latest;
        *buffer.get_unchecked_mut(mirror_index) = latest;
    }
}

/// Builds all comb values for a staged sample tile directly in caller-owned
/// scratch storage.
///
/// # Safety
///
/// `comb_x_pointer` and `comb_y_pointer` must each address writable storage for
/// `SAMPLE_COUNT * BAND_TILE_LEN` `f64` values. The regions must not overlap.
#[inline(always)]
unsafe fn prepare_comb_tiles<const SAMPLE_COUNT: usize, const HISTORY_ONLY: bool>(
    coeffs: &VQsDFTCoefficients,
    buffer: &[f64],
    buffer_len: usize,
    samples: &[f64; SAMPLE_COUNT],
    virtual_buffer_indices: &[usize; SAMPLE_COUNT],
    band_start: usize,
    comb_x_pointer: *mut f64,
    comb_y_pointer: *mut f64,
) {
    debug_assert_eq!(buffer.len(), buffer_len * 2);
    debug_assert!(band_start + BAND_TILE_LEN <= coeffs.band_count);
    debug_assert!(!HISTORY_ONLY || coeffs.min_period >= SAMPLE_COUNT);

    // SAFETY: callers advance through complete band tiles in arrays whose
    // length is exactly `band_count`.
    let (periods, period_segment_ends, period_segment_reuses, fiddle_x, fiddle_y) = unsafe {
        (
            get_band_tile(&coeffs.periods, band_start),
            get_band_tile(&coeffs.period_segment_ends, band_start),
            get_band_tile(&coeffs.period_segment_reuses, band_start),
            get_band_tile(&coeffs.fiddle_x, band_start),
            get_band_tile(&coeffs.fiddle_y, band_start),
        )
    };

    for sample_offset in 0..SAMPLE_COUNT {
        let latest = samples[sample_offset];
        let virtual_buffer_index = virtual_buffer_indices[sample_offset];
        debug_assert!(virtual_buffer_index < buffer_len);
        let output_offset = sample_offset * BAND_TILE_LEN;

        macro_rules! delayed_sample {
            ($period:expr) => {{
                let period = $period;
                debug_assert!(period < buffer_len);
                if HISTORY_ONLY {
                    // Mirroring guarantees this range is contiguous even when
                    // the virtual cursor precedes the delayed sample.
                    let index = virtual_buffer_index + buffer_len - period;
                    unsafe { *buffer.get_unchecked(index) }
                } else if sample_offset >= period {
                    unsafe { *samples.get_unchecked(sample_offset - period) }
                } else {
                    let index = virtual_buffer_index + buffer_len - period;
                    unsafe { *buffer.get_unchecked(index) }
                }
            }};
        }

        macro_rules! write_comb {
            ($lane:expr, $oldest:expr) => {{
                let lane = $lane;
                // SAFETY: both output arrays contain every sample/band pair
                // and each pair is visited exactly once.
                unsafe {
                    comb_x_pointer.add(output_offset + lane).write(
                        latest
                            .algebraic_mul(*fiddle_x.get_unchecked(lane))
                            .algebraic_sub($oldest),
                    );
                    comb_y_pointer
                        .add(output_offset + lane)
                        .write(latest.algebraic_mul(*fiddle_y.get_unchecked(lane)));
                }
            }};
        }

        let mut segment_lane = 0;
        while segment_lane < BAND_TILE_LEN {
            let segment_end = period_segment_ends[segment_lane] - band_start;
            debug_assert!(segment_end > segment_lane && segment_end <= BAND_TILE_LEN);

            if period_segment_reuses[segment_lane] {
                let period = periods[segment_lane];
                debug_assert!(
                    periods[segment_lane..segment_end]
                        .iter()
                        .all(|&run_period| run_period == period)
                );
                let oldest = delayed_sample!(period);
                for lane in segment_lane..segment_end {
                    write_comb!(lane, oldest);
                }
            } else {
                for lane in segment_lane..segment_end {
                    // SAFETY: segment construction keeps every end within
                    // the complete band tile. Using the unchecked access is
                    // significant here: LLVM otherwise retains one bounds
                    // check in every varying-period lane iteration.
                    let period = unsafe { *periods.get_unchecked(lane) };
                    let oldest = delayed_sample!(period);
                    write_comb!(lane, oldest);
                }
            }

            segment_lane = segment_end;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn recurrence_step_values(
    twiddle_x: f64,
    twiddle_y: f64,
    comb_x: f64,
    comb_y: f64,
    previous_coeff2_x: f64,
    previous_coeff2_y: f64,
    previous_coeff4_x: f64,
    previous_coeff4_y: f64,
    previous_coeff5_x: f64,
    previous_coeff5_y: f64,
) -> (f64, f64) {
    let coeff1_x = comb_x
        .algebraic_mul(twiddle_x)
        .algebraic_sub(comb_y.algebraic_mul(twiddle_y))
        .algebraic_sub(previous_coeff2_x);
    let coeff1_y = comb_x
        .algebraic_mul(twiddle_y)
        .algebraic_add(comb_y.algebraic_mul(twiddle_x))
        .algebraic_sub(previous_coeff2_y);
    let resonance = twiddle_x.algebraic_mul(2.0);
    (
        coeff1_x
            .algebraic_add(resonance.algebraic_mul(previous_coeff4_x))
            .algebraic_sub(previous_coeff5_x),
        coeff1_y
            .algebraic_add(resonance.algebraic_mul(previous_coeff4_y))
            .algebraic_sub(previous_coeff5_y),
    )
}

/// Processes one NC band tile across the complete staged sample tile. The
/// per-band recurrence state remains local across time and is written back
/// only once.
#[inline(never)]
fn calculate_nc_tile_block<const SAMPLE_COUNT: usize>(
    coeffs: &mut VQsDFTCoefficients,
    band_start: usize,
    comb_x: &[[f64; BAND_TILE_LEN]; SAMPLE_COUNT],
    comb_y: &[[f64; BAND_TILE_LEN]; SAMPLE_COUNT],
    spectrum: &mut [f64; BAND_TILE_LEN],
) {
    debug_assert_eq!(coeffs.term_count, 2);
    let right_start = coeffs.band_count + band_start;

    // SAFETY: NC contains two complete, disjoint term rows, and the caller
    // advances through complete band tiles.
    let (
        left_twiddle_x,
        left_twiddle_y,
        right_twiddle_x,
        right_twiddle_y,
        inverse_periods,
        coeff2_x,
        coeff2_y,
        left_coeff4_x,
        right_coeff4_x,
        left_coeff4_y,
        right_coeff4_y,
        left_coeff5_x,
        right_coeff5_x,
        left_coeff5_y,
        right_coeff5_y,
    ) = unsafe {
        let (left_coeff4_x, right_coeff4_x) =
            get_two_band_tiles_mut(&mut coeffs.coeff4_x, band_start, right_start);
        let (left_coeff4_y, right_coeff4_y) =
            get_two_band_tiles_mut(&mut coeffs.coeff4_y, band_start, right_start);
        let (left_coeff5_x, right_coeff5_x) =
            get_two_band_tiles_mut(&mut coeffs.coeff5_x, band_start, right_start);
        let (left_coeff5_y, right_coeff5_y) =
            get_two_band_tiles_mut(&mut coeffs.coeff5_y, band_start, right_start);
        (
            get_band_tile(&coeffs.twiddle_x, band_start),
            get_band_tile(&coeffs.twiddle_y, band_start),
            get_band_tile(&coeffs.twiddle_x, right_start),
            get_band_tile(&coeffs.twiddle_y, right_start),
            get_band_tile(&coeffs.inverse_periods, band_start),
            get_band_tile_mut(&mut coeffs.coeff2_x, band_start),
            get_band_tile_mut(&mut coeffs.coeff2_y, band_start),
            left_coeff4_x,
            right_coeff4_x,
            left_coeff4_y,
            right_coeff4_y,
            left_coeff5_x,
            right_coeff5_x,
            left_coeff5_y,
            right_coeff5_y,
        )
    };

    for lane in 0..BAND_TILE_LEN {
        let mut previous_coeff2_x = coeff2_x[lane];
        let mut previous_coeff2_y = coeff2_y[lane];
        let mut left_previous_coeff4_x = left_coeff4_x[lane];
        let mut left_previous_coeff4_y = left_coeff4_y[lane];
        let mut left_previous_coeff5_x = left_coeff5_x[lane];
        let mut left_previous_coeff5_y = left_coeff5_y[lane];
        let mut right_previous_coeff4_x = right_coeff4_x[lane];
        let mut right_previous_coeff4_y = right_coeff4_y[lane];
        let mut right_previous_coeff5_x = right_coeff5_x[lane];
        let mut right_previous_coeff5_y = right_coeff5_y[lane];
        let mut magnitude_sum = spectrum[lane];

        for sample_offset in 0..SAMPLE_COUNT {
            let comb_x = comb_x[sample_offset][lane];
            let comb_y = comb_y[sample_offset][lane];
            let (left_x, left_y) = recurrence_step_values(
                left_twiddle_x[lane],
                left_twiddle_y[lane],
                comb_x,
                comb_y,
                previous_coeff2_x,
                previous_coeff2_y,
                left_previous_coeff4_x,
                left_previous_coeff4_y,
                left_previous_coeff5_x,
                left_previous_coeff5_y,
            );
            let (right_x, right_y) = recurrence_step_values(
                right_twiddle_x[lane],
                right_twiddle_y[lane],
                comb_x,
                comb_y,
                previous_coeff2_x,
                previous_coeff2_y,
                right_previous_coeff4_x,
                right_previous_coeff4_y,
                right_previous_coeff5_x,
                right_previous_coeff5_y,
            );

            left_previous_coeff5_x = left_previous_coeff4_x;
            left_previous_coeff5_y = left_previous_coeff4_y;
            left_previous_coeff4_x = left_x;
            left_previous_coeff4_y = left_y;
            right_previous_coeff5_x = right_previous_coeff4_x;
            right_previous_coeff5_y = right_previous_coeff4_y;
            right_previous_coeff4_x = right_x;
            right_previous_coeff4_y = right_y;
            previous_coeff2_x = comb_x;
            previous_coeff2_y = comb_y;

            let magnitude = left_x
                .algebraic_mul(right_x)
                .algebraic_add(left_y.algebraic_mul(right_y))
                .algebraic_mul(-1.0)
                .max(0.0)
                .sqrt()
                .algebraic_mul(inverse_periods[lane]);
            magnitude_sum = magnitude_sum.algebraic_add(magnitude);
        }

        coeff2_x[lane] = previous_coeff2_x;
        coeff2_y[lane] = previous_coeff2_y;
        left_coeff4_x[lane] = left_previous_coeff4_x;
        left_coeff4_y[lane] = left_previous_coeff4_y;
        left_coeff5_x[lane] = left_previous_coeff5_x;
        left_coeff5_y[lane] = left_previous_coeff5_y;
        right_coeff4_x[lane] = right_previous_coeff4_x;
        right_coeff4_y[lane] = right_previous_coeff4_y;
        right_coeff5_x[lane] = right_previous_coeff5_x;
        right_coeff5_y[lane] = right_previous_coeff5_y;
        spectrum[lane] = magnitude_sum;
    }
}

/// Processes a standard-window tile in small lane subtiles. Each term's live
/// recurrence state remains local across the full sample tile, reducing state
/// memory traffic without trying to keep every term live simultaneously. The
/// smaller accumulation scratch also stays cache-local for wide windows.
#[inline(never)]
fn calculate_window_tile_block<const TERM_COUNT: usize, const SAMPLE_COUNT: usize>(
    coeffs: &mut VQsDFTCoefficients,
    band_start: usize,
    comb_x: &[[f64; BAND_TILE_LEN]; SAMPLE_COUNT],
    comb_y: &[[f64; BAND_TILE_LEN]; SAMPLE_COUNT],
    spectrum: &mut [f64; BAND_TILE_LEN],
) {
    debug_assert_eq!(coeffs.term_count, TERM_COUNT);
    debug_assert!(BAND_TILE_LEN.is_multiple_of(WINDOW_LANE_TILE_LEN));
    let band_count = coeffs.band_count;

    for lane_start in (0..BAND_TILE_LEN).step_by(WINDOW_LANE_TILE_LEN) {
        let mut sum_x = [[0.0_f64; WINDOW_LANE_TILE_LEN]; SAMPLE_COUNT];
        let mut sum_y = [[0.0_f64; WINDOW_LANE_TILE_LEN]; SAMPLE_COUNT];

        // Process one term through the complete sample tile before loading the
        // next term. This writes each term's state once per tile instead of
        // once per sample.
        for term in 0..TERM_COUNT {
            let term_start = term * band_count + band_start;
            // SAFETY: every term contains a complete band row and the caller
            // advances through full band tiles.
            let (
                twiddle_x,
                twiddle_y,
                previous_coeff2_x,
                previous_coeff2_y,
                coeff4_x,
                coeff4_y,
                coeff5_x,
                coeff5_y,
            ) = unsafe {
                (
                    get_band_tile(&coeffs.twiddle_x, term_start),
                    get_band_tile(&coeffs.twiddle_y, term_start),
                    get_band_tile(&coeffs.coeff2_x, band_start),
                    get_band_tile(&coeffs.coeff2_y, band_start),
                    get_band_tile_mut(&mut coeffs.coeff4_x, term_start),
                    get_band_tile_mut(&mut coeffs.coeff4_y, term_start),
                    get_band_tile_mut(&mut coeffs.coeff5_x, term_start),
                    get_band_tile_mut(&mut coeffs.coeff5_y, term_start),
                )
            };
            // SAFETY: `TERM_COUNT` is selected from `coeffs.term_count` once
            // per block, and construction creates exactly one gain per term.
            // LLVM cannot derive that relationship from the Vec length and
            // otherwise emits a bounds check in this hot term loop.
            let term_gain = unsafe { *coeffs.term_gains.get_unchecked(term) };

            for local_lane in 0..WINDOW_LANE_TILE_LEN {
                let lane = lane_start + local_lane;
                let mut previous_coeff2_x = previous_coeff2_x[lane];
                let mut previous_coeff2_y = previous_coeff2_y[lane];
                let mut previous_coeff4_x = coeff4_x[lane];
                let mut previous_coeff4_y = coeff4_y[lane];
                let mut previous_coeff5_x = coeff5_x[lane];
                let mut previous_coeff5_y = coeff5_y[lane];

                for sample_offset in 0..SAMPLE_COUNT {
                    let current_comb_x = comb_x[sample_offset][lane];
                    let current_comb_y = comb_y[sample_offset][lane];
                    let (x, y) = recurrence_step_values(
                        twiddle_x[lane],
                        twiddle_y[lane],
                        current_comb_x,
                        current_comb_y,
                        previous_coeff2_x,
                        previous_coeff2_y,
                        previous_coeff4_x,
                        previous_coeff4_y,
                        previous_coeff5_x,
                        previous_coeff5_y,
                    );
                    previous_coeff5_x = previous_coeff4_x;
                    previous_coeff5_y = previous_coeff4_y;
                    previous_coeff4_x = x;
                    previous_coeff4_y = y;
                    previous_coeff2_x = current_comb_x;
                    previous_coeff2_y = current_comb_y;
                    sum_x[sample_offset][local_lane] =
                        sum_x[sample_offset][local_lane].algebraic_add(x.algebraic_mul(term_gain));
                    sum_y[sample_offset][local_lane] =
                        sum_y[sample_offset][local_lane].algebraic_add(y.algebraic_mul(term_gain));
                }

                coeff4_x[lane] = previous_coeff4_x;
                coeff4_y[lane] = previous_coeff4_y;
                coeff5_x[lane] = previous_coeff5_x;
                coeff5_y[lane] = previous_coeff5_y;
            }
        }

        // SAFETY: these are band-major arrays containing the complete output
        // tile and shared comb-delay state.
        let (inverse_periods, coeff2_x, coeff2_y) = unsafe {
            (
                get_band_tile(&coeffs.inverse_periods, band_start),
                get_band_tile_mut(&mut coeffs.coeff2_x, band_start),
                get_band_tile_mut(&mut coeffs.coeff2_y, band_start),
            )
        };

        for sample_offset in 0..SAMPLE_COUNT {
            for local_lane in 0..WINDOW_LANE_TILE_LEN {
                let lane = lane_start + local_lane;
                let magnitude = sum_x[sample_offset][local_lane]
                    .algebraic_mul(sum_x[sample_offset][local_lane])
                    .algebraic_add(
                        sum_y[sample_offset][local_lane]
                            .algebraic_mul(sum_y[sample_offset][local_lane]),
                    )
                    .sqrt()
                    .algebraic_mul(inverse_periods[lane]);
                spectrum[lane] = spectrum[lane].algebraic_add(magnitude);
            }
        }

        for local_lane in 0..WINDOW_LANE_TILE_LEN {
            let lane = lane_start + local_lane;
            coeff2_x[lane] = comb_x[SAMPLE_COUNT - 1][lane];
            coeff2_y[lane] = comb_y[SAMPLE_COUNT - 1][lane];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABSOLUTE_TOLERANCE: f64 = 1.0e-12;
    const RELATIVE_TOLERANCE: f64 = 1.0e-10;

    #[derive(Clone)]
    struct ReferenceCoeff {
        twiddle: (f64, f64),
        fiddle: (f64, f64),
        resonance: f64,
        coeff2: (f64, f64),
        coeff4: (f64, f64),
        coeff5: (f64, f64),
        gain: f64,
    }

    impl ReferenceCoeff {
        fn calculate(&mut self, latest: f64, oldest: f64) -> (f64, f64) {
            let comb_x = latest.algebraic_mul(self.fiddle.0).algebraic_sub(oldest);
            let comb_y = latest.algebraic_mul(self.fiddle.1);
            let coeff1_x = comb_x
                .algebraic_mul(self.twiddle.0)
                .algebraic_sub(comb_y.algebraic_mul(self.twiddle.1))
                .algebraic_sub(self.coeff2.0);
            let coeff1_y = comb_x
                .algebraic_mul(self.twiddle.1)
                .algebraic_add(comb_y.algebraic_mul(self.twiddle.0))
                .algebraic_sub(self.coeff2.1);
            self.coeff2 = (comb_x, comb_y);

            let coeff3_x = coeff1_x
                .algebraic_add(self.resonance.algebraic_mul(self.coeff4.0))
                .algebraic_sub(self.coeff5.0);
            let coeff3_y = coeff1_y
                .algebraic_add(self.resonance.algebraic_mul(self.coeff4.1))
                .algebraic_sub(self.coeff5.1);
            self.coeff5 = self.coeff4;
            self.coeff4 = (coeff3_x, coeff3_y);

            (
                coeff3_x.algebraic_mul(self.gain),
                coeff3_y.algebraic_mul(self.gain),
            )
        }

        fn reset(&mut self) {
            self.coeff2 = (0.0, 0.0);
            self.coeff4 = (0.0, 0.0);
            self.coeff5 = (0.0, 0.0);
        }
    }

    struct ScalarReference {
        periods: Vec<isize>,
        coeffs: Vec<Vec<ReferenceCoeff>>,
        use_nc: bool,
        buffer: Vec<f64>,
        buffer_index: usize,
        spectrum: Vec<f64>,
    }

    impl ScalarReference {
        fn new(
            bands: &[FrequencyBand],
            window: Window,
            sample_rate: f32,
            use_nc: bool,
            strict_nc: bool,
        ) -> Self {
            let max_period = bands
                .iter()
                .map(|band| {
                    let q = band.center / (band.high - band.low).abs();
                    let period = if strict_nc {
                        (((band.center * 2.0) / (band.center * (1.0 / q))).round()
                            * (sample_rate / (band.center * 2.0)))
                            .round()
                            .max(1.0)
                    } else {
                        ((sample_rate / band.center) * q).ceil()
                    };
                    period as usize
                })
                .max()
                .unwrap();
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
            let min_term = if use_nc {
                0
            } else {
                -(window_coeffs.len() as isize) + 1
            };
            let term_count = if use_nc {
                2
            } else {
                window_coeffs.len() * 2 - 1
            };
            let sample_rate_f64 = f64::from(sample_rate);

            let mut periods = Vec::with_capacity(bands.len());
            let mut coeffs = Vec::with_capacity(bands.len());
            for band in bands {
                let center = f64::from(band.center);
                let q = center / (f64::from(band.high) - f64::from(band.low)).abs();
                let period = if use_nc && strict_nc {
                    (((center * 2.0) / (center * (1.0 / q))).round()
                        * (sample_rate_f64 / (center * 2.0)))
                        .round()
                        .max(1.0)
                } else {
                    ((sample_rate_f64 / center) * q).ceil()
                };
                periods.push(period as isize);

                let adjusted_q = center * period / sample_rate_f64;
                let mut band_coeffs = Vec::with_capacity(term_count);
                for term in 0..term_count {
                    let offset = if use_nc {
                        term as f64 - 0.5
                    } else {
                        (min_term + term as isize) as f64
                    };
                    let window_gain = if use_nc {
                        1.0
                    } else {
                        window_coeffs[offset.abs() as usize] * (-((offset.abs() % 2.0) * 2.0 + 1.0))
                    };
                    let k = adjusted_q + offset;
                    let fiddle = -2.0 * PI * k;
                    let twiddle = 2.0 * PI * k / period;
                    band_coeffs.push(ReferenceCoeff {
                        twiddle: (twiddle.cos(), twiddle.sin()),
                        fiddle: (fiddle.cos(), fiddle.sin()),
                        resonance: 2.0 * twiddle.cos(),
                        coeff2: (0.0, 0.0),
                        coeff4: (0.0, 0.0),
                        coeff5: (0.0, 0.0),
                        gain: window_gain / period,
                    });
                }
                coeffs.push(band_coeffs);
            }

            Self {
                periods,
                coeffs,
                use_nc,
                buffer: vec![0.0; max_period + 1],
                buffer_index: max_period,
                spectrum: vec![0.0; bands.len()],
            }
        }

        fn reset(&mut self) {
            self.buffer.fill(0.0);
            self.buffer_index = self.buffer.len() - 1;
            for coeff in self.coeffs.iter_mut().flatten() {
                coeff.reset();
            }
        }

        fn analyze(&mut self, samples: &[f64]) -> &[f64] {
            self.spectrum.fill(0.0);
            let buffer_len = self.buffer.len();
            let buffer_len_signed = buffer_len as isize;

            for &latest in samples {
                self.buffer_index = (self.buffer_index + 1) % buffer_len;
                self.buffer[self.buffer_index] = latest;

                for band in 0..self.coeffs.len() {
                    let period = self.periods[band];
                    let oldest_index = (((self.buffer_index as isize - period) % buffer_len_signed
                        + buffer_len_signed) as usize)
                        % buffer_len;
                    let oldest = self.buffer[oldest_index];

                    if self.use_nc {
                        let left = self.coeffs[band][0].calculate(latest, oldest);
                        let right = self.coeffs[band][1].calculate(latest, oldest);
                        self.spectrum[band] = self.spectrum[band].algebraic_add(
                            left.0
                                .algebraic_mul(right.0)
                                .algebraic_add(left.1.algebraic_mul(right.1))
                                .algebraic_mul(-1.0)
                                .max(0.0)
                                .sqrt(),
                        );
                    } else {
                        let mut sum: (f64, f64) = (0.0, 0.0);
                        for coeff in &mut self.coeffs[band] {
                            let result = coeff.calculate(latest, oldest);
                            sum.0 = sum.0.algebraic_add(result.0);
                            sum.1 = sum.1.algebraic_add(result.1);
                        }
                        self.spectrum[band] = self.spectrum[band].algebraic_add(
                            sum.0
                                .algebraic_mul(sum.0)
                                .algebraic_add(sum.1.algebraic_mul(sum.1))
                                .sqrt(),
                        );
                    }
                }
            }

            let sample_count = samples.len().max(1) as f64;
            for value in &mut self.spectrum {
                *value = value.algebraic_div(sample_count);
            }
            &self.spectrum
        }
    }

    fn test_bands(band_count: usize) -> Vec<FrequencyBand> {
        let low = 180.0_f32;
        let high = 9_000.0_f32;
        (0..band_count)
            .map(|index| {
                let fraction = index as f32 / (band_count - 1) as f32;
                let center = low * (high / low).powf(fraction);
                let q = 6.0 + fraction * 6.0;
                let bandwidth = center / q;
                FrequencyBand {
                    low: center - bandwidth * 0.5,
                    center,
                    high: center + bandwidth * 0.5,
                }
            })
            .collect()
    }

    fn deterministic_stream(length: usize, sample_rate: f32) -> Vec<f64> {
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        (0..length)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = (((state >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5) * 0.08;
                let time = index as f64 / f64::from(sample_rate);
                let mut sample = if index < 73 {
                    0.0
                } else {
                    (2.0 * PI * 311.0 * time).sin() * 0.31
                        + (2.0 * PI * 1_337.0 * time).cos() * 0.17
                        + noise
                };
                if matches!(index, 73 | 521 | 2_047) {
                    sample += if index == 521 { -0.75 } else { 1.0 };
                }
                sample
            })
            .collect()
    }

    fn assert_close(label: &str, reference: &[f64], actual: &[f64]) {
        assert_eq!(reference.len(), actual.len());
        for (band, (&reference, &actual)) in reference.iter().zip(actual).enumerate() {
            let difference = (actual - reference).abs();
            let tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs();
            assert!(
                difference <= tolerance,
                "{label}, band {band}: actual {actual:.17e}, reference {reference:.17e}, difference {difference:.17e}, tolerance {tolerance:.17e}"
            );
        }
    }

    fn compare_bands(
        bands: &[FrequencyBand],
        window: Window,
        sample_rate: f32,
        use_nc: bool,
        strict_nc: bool,
    ) {
        let mut transform = VQsDFT::new(bands, window, sample_rate, use_nc, strict_nc);
        let mut reference = ScalarReference::new(bands, window, sample_rate, use_nc, strict_nc);
        let stream_length = transform.buffer_len * 2 + 257;
        let samples = deterministic_stream(stream_length, sample_rate);

        assert_close(
            "empty input",
            reference.analyze(&[]),
            transform.analyze([].into_iter()),
        );

        let block_sizes = [1, 17, 64, 3, 129, 8, 257, 31];
        let mut offset = 0;
        let mut block = 0;
        while offset < samples.len() {
            let end = (offset + block_sizes[block % block_sizes.len()]).min(samples.len());
            let expected = reference.analyze(&samples[offset..end]);
            let actual = transform.analyze(samples[offset..end].iter().copied());
            assert_close(&format!("stream block {block}"), expected, actual);
            offset = end;
            block += 1;
        }

        transform.reset();
        reference.reset();
        for (block, chunk) in samples[..samples.len().min(401)].chunks(23).enumerate() {
            let expected = reference.analyze(chunk);
            let actual = transform.analyze(chunk.iter().copied());
            assert_close(&format!("after reset block {block}"), expected, actual);
        }
    }

    fn compare_configuration(window: Window, sample_rate: f32, use_nc: bool, strict_nc: bool) {
        compare_bands(
            &test_bands(BAND_TILE_LEN),
            window,
            sample_rate,
            use_nc,
            strict_nc,
        );
    }

    #[test]
    fn standard_windows_match_scalar_reference() {
        let windows = [
            Window::Rectangular,
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::Nuttall,
            Window::FlatTop,
        ];
        for (index, window) in windows.into_iter().enumerate() {
            let sample_rate = if index.is_multiple_of(2) {
                44_100.0
            } else {
                48_000.0
            };
            compare_configuration(window, sample_rate, false, false);
        }
    }

    #[test]
    fn nc_modes_match_scalar_reference() {
        for sample_rate in [44_100.0, 48_000.0] {
            compare_configuration(Window::Hann, sample_rate, true, false);
            compare_configuration(Window::Hann, sample_rate, true, true);
        }
    }

    #[test]
    fn multiple_band_tiles_match_scalar_reference() {
        compare_bands(
            &test_bands(BAND_TILE_LEN * 2),
            Window::Hann,
            48_000.0,
            false,
            false,
        );
        compare_bands(
            &test_bands(BAND_TILE_LEN * 3),
            Window::Hann,
            48_000.0,
            true,
            true,
        );
    }

    #[test]
    fn short_periods_match_scalar_reference() {
        let bands: Vec<_> = (0..BAND_TILE_LEN * 2)
            .map(|index| {
                let fraction = index as f32 / (BAND_TILE_LEN * 2 - 1) as f32;
                let center = 4_000.0 + fraction * 14_000.0;
                let bandwidth = center / (2.0 + fraction * 2.0);
                FrequencyBand {
                    low: center - bandwidth * 0.5,
                    center,
                    high: center + bandwidth * 0.5,
                }
            })
            .collect();
        compare_bands(&bands, Window::Hann, 48_000.0, true, false);
    }

    #[test]
    fn period_segments_cover_complete_band_tiles() {
        let bands = test_bands(BAND_TILE_LEN * 3);
        let transform = VQsDFT::new(&bands, Window::Hann, 48_000.0, true, false);
        let coeffs = &transform.coeffs;

        for tile_start in (0..coeffs.band_count).step_by(BAND_TILE_LEN) {
            let tile_end = tile_start + BAND_TILE_LEN;
            let mut segment_start = tile_start;
            while segment_start < tile_end {
                let segment_end = coeffs.period_segment_ends[segment_start];
                assert!(segment_end > segment_start && segment_end <= tile_end);

                if coeffs.period_segment_reuses[segment_start] {
                    assert!(segment_end - segment_start >= MIN_REUSED_PERIOD_RUN_LEN);
                    let period = coeffs.periods[segment_start];
                    assert!(
                        coeffs.periods[segment_start..segment_end]
                            .iter()
                            .all(|&run_period| run_period == period)
                    );
                } else {
                    let mut run_start = segment_start;
                    while run_start < segment_end {
                        let period = coeffs.periods[run_start];
                        let mut run_end = run_start + 1;
                        while run_end < segment_end && coeffs.periods[run_end] == period {
                            run_end += 1;
                        }
                        assert!(run_end - run_start < MIN_REUSED_PERIOD_RUN_LEN);
                        run_start = run_end;
                    }
                }

                segment_start = segment_end;
            }
            assert_eq!(segment_start, tile_end);
        }
    }

    #[test]
    #[should_panic(expected = "band count must be a nonzero multiple of 64")]
    fn rejects_partial_band_tile() {
        let bands = test_bands(BAND_TILE_LEN - 1);
        let _ = VQsDFT::new(&bands, Window::Hann, 48_000.0, true, false);
    }
}
