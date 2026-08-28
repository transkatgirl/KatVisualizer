use std::{collections::VecDeque, f32::consts::PI, sync::Arc, time::Duration};

mod masker;
mod vqsdft;

use vqsdft::{VQsDFT, Window};

use crate::analyzer::masker::Masker;

#[derive(Debug, Clone)]
pub struct BetterAnalyzerConfiguration {
    pub resolution: usize,
    pub start_frequency: f32,
    pub end_frequency: f32,
    pub erb_frequency_scale: bool,

    pub sample_rate: f32,
    pub q_time_resolution: f32,
    pub erb_time_resolution: bool,
    pub erb_bandwidth_divisor: f32,
    pub time_resolution_clamp: (f32, f32),
    pub nc_method: bool,
    pub strict_nc: bool,

    pub masking: bool,
    pub approximate_masking: bool,
}

#[allow(clippy::excessive_precision)]
impl Default for BetterAnalyzerConfiguration {
    fn default() -> Self {
        Self {
            resolution: 512,        // MUST be a multiple of 64
            start_frequency: 20.0,  // Lower end of typical human hearing range
            end_frequency: 20000.0, // Upper end of typical human hearing range
            erb_frequency_scale: true,
            sample_rate: 48000.0,
            q_time_resolution: 17.30993405, // Determined using distances between MIDI notes
            erb_time_resolution: true,
            erb_bandwidth_divisor: 1.4, // Critical bands are asymmetrical, and this is taken into account by the ERB formula. However, we want our bandwidth to be based on the *narrowest side* of the critical band, so we need to adjust the formula accordingly. TODO: This constant is a rough approximation and should be replaced with a more exact value.
            time_resolution_clamp: (0.0, 37.23177300), // Upper limit is determined using 1s / ERB(20 Hz)
            nc_method: true,
            strict_nc: false,
            masking: true,
            approximate_masking: false,
        }
    }
}

pub struct BetterAnalyzer {
    config: BetterAnalyzerConfiguration,
    transform: VQsDFT,
    masker: Masker,
    masking: Vec<f32>,
    processed_spectrum: Vec<f32>,
    frequency_bands: Vec<(f32, f32, f32)>,
    normalizers: PrecomputedNormalizers,
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
                    (24.7 * ((0.00437 * center) + 1.0)) / config.erb_bandwidth_divisor
                } else {
                    center / config.q_time_resolution
                }
                .min(1.0 / (config.time_resolution_clamp.0 / 1000.0))
                .max(1.0 / (config.time_resolution_clamp.1 / 1000.0))
            },
        );

        let band_count = frequency_bands.len();

        assert!(band_count.is_multiple_of(64));

        let normalizers = PrecomputedNormalizers::new(&frequency_bands);

        let transform = VQsDFT::new(
            &frequency_bands,
            Window::Hann,
            config.sample_rate,
            config.nc_method,
            config.strict_nc,
        );

        let masker = Masker::new(&frequency_bands, config.approximate_masking);

        let frequency_bands: Vec<_> = frequency_bands
            .iter()
            .map(|band| (band.low, band.center, band.high))
            .collect();

        assert!(frequency_bands.len().is_multiple_of(64));

        Self {
            config,
            masker,
            masking: vec![0.0; frequency_bands.len()],
            processed_spectrum: vec![0.0; frequency_bands.len()],
            transform,
            frequency_bands,
            normalizers,
        }
    }
    #[inline(always)]
    pub fn config(&self) -> &BetterAnalyzerConfiguration {
        &self.config
    }
    #[inline(always)]
    pub fn frequencies(&self) -> &[(f32, f32, f32)] {
        &self.frequency_bands
    }
    #[inline(always)]
    pub fn clear_buffers(&mut self) {
        self.transform.reset();
    }
    pub fn analyze(
        &mut self,
        samples: impl ExactSizeIterator<Item = f32>,
        listening_volume: Option<f32>,
    ) {
        self.transform.analyze_into(
            samples.map(|sample| sample as f64),
            &mut self.processed_spectrum,
        );

        /*let flatness = if spectrum.len() > 128 {
            0.0
        } else {
            map_value_f64(spectral_flatness(spectrum), -60.0, 0.0, 0.0, 1.0)
        };*/

        if self.config.masking {
            self.masker.calculate_masking_threshold(
                &self.processed_spectrum,
                listening_volume,
                //0.0,
                &mut self.masking,
            );

            debug_assert_eq!(self.processed_spectrum.len(), self.masking.len());

            // Both buffers are created with the immutable band count. The normalized transform
            // is already in `processed_spectrum`, so masking can be applied in place.
            unsafe {
                let processed = self.processed_spectrum.as_mut_ptr();
                let masking = self.masking.as_ptr();

                for index in 0..self.processed_spectrum.len() {
                    let output = processed.add(index);
                    output.write((*output).max(*masking.add(index)));
                }
            }
        }
    }
    #[inline(always)]
    pub fn raw_analysis(&self) -> &[f32] {
        &self.processed_spectrum
    }
    #[inline(always)]
    pub fn raw_masking(&self) -> &[f32] {
        &self.masking
    }
}

#[derive(Clone)]
pub struct BetterAnalysis {
    pub duration: Duration,
    pub data: Vec<(f32, f32)>,
    pub masking: Vec<(f32, f32)>,
    pub masking_mean: f32,
}

impl BetterAnalysis {
    pub fn new(capacity: usize) -> Self {
        Self {
            duration: Duration::ZERO,
            data: Vec::with_capacity(capacity),
            masking: Vec::with_capacity(capacity),
            masking_mean: f32::NEG_INFINITY,
        }
    }
    pub fn update_mono(
        &mut self,
        center: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
        duration: Duration,
    ) {
        let new_length = center.raw_analysis().len();

        if self.data.len() != new_length {
            self.data.resize(new_length, (0.0, 0.0));
            self.masking.resize(new_length, (0.0, f32::NEG_INFINITY));
            self.masking_mean = f32::NEG_INFINITY;
        }

        if center.config.masking {
            self.update_mono_masking_and_data(center, gain, normalization_volume);
        } else {
            if self.masking_mean != f32::NEG_INFINITY {
                self.masking.fill((0.0, f32::NEG_INFINITY));
                self.masking_mean = f32::NEG_INFINITY;
            }
            self.update_mono_data(center, gain, normalization_volume);
        }

        self.duration = duration;
    }
    pub fn update_stereo(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
        duration: Duration,
    ) {
        assert_eq!(left.raw_analysis().len(), right.raw_analysis().len());

        let new_length = left.raw_analysis().len();

        if self.data.len() != new_length {
            self.data.resize(new_length, (0.0, 0.0));
            self.masking.resize(new_length, (0.0, f32::NEG_INFINITY));
            self.masking_mean = f32::NEG_INFINITY;
        }

        if left.config.masking {
            self.update_stereo_masking_and_data(left, right, gain, normalization_volume);
        } else {
            if self.masking_mean != f32::NEG_INFINITY {
                self.masking.fill((0.0, f32::NEG_INFINITY));
                self.masking_mean = f32::NEG_INFINITY;
            }
            self.update_stereo_data(left, right, gain, normalization_volume);
        }

        self.duration = duration;
    }
    fn update_stereo_data(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
    ) {
        if let Some(listening_volume) = normalization_volume {
            self.update_stereo_data_normalized(left, right, gain, listening_volume);
        } else {
            self.update_stereo_data_unnormalized(left, right, gain);
        }
    }
    fn update_stereo_data_normalized(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
        listening_volume: f32,
    ) {
        let left_analysis_chunks = unsafe { left.raw_analysis().as_chunks_unchecked::<64>() };
        let right_analysis_chunks = unsafe { right.raw_analysis().as_chunks_unchecked::<64>() };
        let output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let total_gain = gain.algebraic_add(listening_volume);

        let exponent_scale_chunks =
            unsafe { left.normalizers.exponent_scales.as_chunks_unchecked::<64>() };
        let exponent_bias_chunks =
            unsafe { left.normalizers.exponent_biases.as_chunks_unchecked::<64>() };
        let param_1_chunks = unsafe { left.normalizers.params_1.as_chunks_unchecked::<64>() };
        let inverse_param_2_chunks = unsafe {
            left.normalizers
                .inverse_params_2
                .as_chunks_unchecked::<64>()
        };

        for chunk in 0..output_chunks.len() {
            let output_chunk = unsafe { output_chunks.get_unchecked_mut(chunk) };
            let left_chunk = unsafe { left_analysis_chunks.get_unchecked(chunk) };
            let right_chunk = unsafe { right_analysis_chunks.get_unchecked(chunk) };
            let exponent_scale_chunk = unsafe { exponent_scale_chunks.get_unchecked(chunk) };
            let exponent_bias_chunk = unsafe { exponent_bias_chunks.get_unchecked(chunk) };
            let param_1_chunk = unsafe { param_1_chunks.get_unchecked(chunk) };
            let inverse_param_2_chunk = unsafe { inverse_param_2_chunks.get_unchecked(chunk) };

            for lane in 0..64 {
                let output = unsafe { output_chunk.get_unchecked_mut(lane) };
                let left = unsafe { *left_chunk.get_unchecked(lane) };
                let right = unsafe { *right_chunk.get_unchecked(lane) };
                let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);

                let volume = spl_to_phon(
                    volume.algebraic_add(total_gain),
                    unsafe { *exponent_scale_chunk.get_unchecked(lane) },
                    unsafe { *exponent_bias_chunk.get_unchecked(lane) },
                    unsafe { *param_1_chunk.get_unchecked(lane) },
                    unsafe { *inverse_param_2_chunk.get_unchecked(lane) },
                )
                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);

                *output = (pan, volume);
            }
        }
    }
    fn update_stereo_data_unnormalized(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
    ) {
        let left_analysis_chunks = unsafe { left.raw_analysis().as_chunks_unchecked::<64>() };
        let right_analysis_chunks = unsafe { right.raw_analysis().as_chunks_unchecked::<64>() };
        let output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        for (output_chunk, (left_chunk, right_chunk)) in output_chunks
            .iter_mut()
            .zip(left_analysis_chunks.iter().zip(right_analysis_chunks))
        {
            for lane in 0..64 {
                let output = unsafe { output_chunk.get_unchecked_mut(lane) };
                let left = unsafe { *left_chunk.get_unchecked(lane) };
                let right = unsafe { *right_chunk.get_unchecked(lane) };
                let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);
                *output = (pan, volume.algebraic_add(gain));
            }
        }
    }
    fn update_stereo_masking_and_data(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
    ) {
        if let Some(listening_volume) = normalization_volume {
            self.update_stereo_masking_and_data_normalized(left, right, gain, listening_volume);
        } else {
            self.update_stereo_masking_and_data_unnormalized(left, right, gain);
        }
    }
    fn update_stereo_masking_and_data_normalized(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
        listening_volume: f32,
    ) {
        let left_masking_chunks = unsafe { left.raw_masking().as_chunks_unchecked::<64>() };
        let right_masking_chunks = unsafe { right.raw_masking().as_chunks_unchecked::<64>() };
        let masking_output_chunks = unsafe { self.masking.as_chunks_unchecked_mut::<64>() };

        let left_analysis_chunks = unsafe { left.raw_analysis().as_chunks_unchecked::<64>() };
        let right_analysis_chunks = unsafe { right.raw_analysis().as_chunks_unchecked::<64>() };
        let data_output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let mut sum: f32 = 0.0;

        let total_gain = gain.algebraic_add(listening_volume);

        let exponent_scale_chunks =
            unsafe { left.normalizers.exponent_scales.as_chunks_unchecked::<64>() };
        let exponent_bias_chunks =
            unsafe { left.normalizers.exponent_biases.as_chunks_unchecked::<64>() };
        let param_1_chunks = unsafe { left.normalizers.params_1.as_chunks_unchecked::<64>() };
        let inverse_param_2_chunks = unsafe {
            left.normalizers
                .inverse_params_2
                .as_chunks_unchecked::<64>()
        };

        for chunk in 0..data_output_chunks.len() {
            let masking_output_chunk = unsafe { masking_output_chunks.get_unchecked_mut(chunk) };
            let data_output_chunk = unsafe { data_output_chunks.get_unchecked_mut(chunk) };
            let left_masking_chunk = unsafe { left_masking_chunks.get_unchecked(chunk) };
            let right_masking_chunk = unsafe { right_masking_chunks.get_unchecked(chunk) };
            let left_chunk = unsafe { left_analysis_chunks.get_unchecked(chunk) };
            let right_chunk = unsafe { right_analysis_chunks.get_unchecked(chunk) };
            let exponent_scale_chunk = unsafe { exponent_scale_chunks.get_unchecked(chunk) };
            let exponent_bias_chunk = unsafe { exponent_bias_chunks.get_unchecked(chunk) };
            let param_1_chunk = unsafe { param_1_chunks.get_unchecked(chunk) };
            let inverse_param_2_chunk = unsafe { inverse_param_2_chunks.get_unchecked(chunk) };

            for lane in 0..64 {
                let exponent_scale = unsafe { *exponent_scale_chunk.get_unchecked(lane) };
                let exponent_bias = unsafe { *exponent_bias_chunk.get_unchecked(lane) };
                let param_1 = unsafe { *param_1_chunk.get_unchecked(lane) };
                let inverse_param_2 = unsafe { *inverse_param_2_chunk.get_unchecked(lane) };
                let left_masking = unsafe { *left_masking_chunk.get_unchecked(lane) };
                let right_masking = unsafe { *right_masking_chunk.get_unchecked(lane) };
                let (masking_pan, masking_volume) =
                    calculate_pan_and_volume_from_amplitude(left_masking, right_masking);

                let masking_volume = spl_to_phon(
                    masking_volume.algebraic_add(total_gain),
                    exponent_scale,
                    exponent_bias,
                    param_1,
                    inverse_param_2,
                )
                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                .clamp(HEARING_THRESHOLD_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);

                sum = sum.algebraic_add(dbfs_to_amplitude(masking_volume));
                *unsafe { masking_output_chunk.get_unchecked_mut(lane) } =
                    (masking_pan, masking_volume);

                let left = unsafe { *left_chunk.get_unchecked(lane) };
                let right = unsafe { *right_chunk.get_unchecked(lane) };
                let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);
                let volume = spl_to_phon(
                    volume.algebraic_add(total_gain),
                    exponent_scale,
                    exponent_bias,
                    param_1,
                    inverse_param_2,
                )
                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);
                *unsafe { data_output_chunk.get_unchecked_mut(lane) } = (pan, volume);
            }
        }

        self.masking_mean = amplitude_to_dbfs(sum.algebraic_div(self.masking.len() as f32));
    }
    fn update_stereo_masking_and_data_unnormalized(
        &mut self,
        left: &BetterAnalyzer,
        right: &BetterAnalyzer,
        gain: f32,
    ) {
        let left_masking_chunks = unsafe { left.raw_masking().as_chunks_unchecked::<64>() };
        let right_masking_chunks = unsafe { right.raw_masking().as_chunks_unchecked::<64>() };
        let masking_output_chunks = unsafe { self.masking.as_chunks_unchecked_mut::<64>() };

        let left_analysis_chunks = unsafe { left.raw_analysis().as_chunks_unchecked::<64>() };
        let right_analysis_chunks = unsafe { right.raw_analysis().as_chunks_unchecked::<64>() };
        let data_output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let mut sum: f32 = 0.0;

        for (masking_output_chunk, (data_output_chunk, inputs)) in
            masking_output_chunks.iter_mut().zip(
                data_output_chunks.iter_mut().zip(
                    left_masking_chunks
                        .iter()
                        .zip(right_masking_chunks)
                        .zip(left_analysis_chunks.iter().zip(right_analysis_chunks)),
                ),
            )
        {
            let ((left_masking_chunk, right_masking_chunk), (left_chunk, right_chunk)) = inputs;

            for lane in 0..64 {
                let left_masking = unsafe { *left_masking_chunk.get_unchecked(lane) };
                let right_masking = unsafe { *right_masking_chunk.get_unchecked(lane) };
                let masking_amplitude = left_masking.algebraic_add(right_masking);

                sum = sum.algebraic_add(masking_amplitude);
                *unsafe { masking_output_chunk.get_unchecked_mut(lane) } = (
                    calculate_pan_from_amplitude_sum(
                        left_masking,
                        right_masking,
                        masking_amplitude,
                    ),
                    amplitude_to_dbfs(masking_amplitude).algebraic_add(gain),
                );

                let left = unsafe { *left_chunk.get_unchecked(lane) };
                let right = unsafe { *right_chunk.get_unchecked(lane) };
                let (pan, volume) = calculate_pan_and_volume_from_amplitude(left, right);
                *unsafe { data_output_chunk.get_unchecked_mut(lane) } =
                    (pan, volume.algebraic_add(gain));
            }
        }

        self.masking_mean =
            amplitude_to_dbfs(sum.algebraic_div(self.masking.len() as f32)).algebraic_add(gain);
    }
    fn update_mono_masking_and_data(
        &mut self,
        center: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
    ) {
        if let Some(listening_volume) = normalization_volume {
            self.update_mono_masking_and_data_normalized(center, gain, listening_volume);
        } else {
            self.update_mono_masking_and_data_unnormalized(center, gain);
        }
    }
    fn update_mono_masking_and_data_normalized(
        &mut self,
        center: &BetterAnalyzer,
        gain: f32,
        listening_volume: f32,
    ) {
        let masking_chunks = unsafe { center.raw_masking().as_chunks_unchecked::<64>() };
        let masking_output_chunks = unsafe { self.masking.as_chunks_unchecked_mut::<64>() };

        let analysis_chunks = unsafe { center.raw_analysis().as_chunks_unchecked::<64>() };
        let data_output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let mut sum: f32 = 0.0;

        let total_gain = gain
            .algebraic_add(listening_volume)
            .algebraic_add(MONO_AMPLITUDE_DB);

        let exponent_scale_chunks = unsafe {
            center
                .normalizers
                .exponent_scales
                .as_chunks_unchecked::<64>()
        };
        let exponent_bias_chunks = unsafe {
            center
                .normalizers
                .exponent_biases
                .as_chunks_unchecked::<64>()
        };
        let param_1_chunks = unsafe { center.normalizers.params_1.as_chunks_unchecked::<64>() };
        let inverse_param_2_chunks = unsafe {
            center
                .normalizers
                .inverse_params_2
                .as_chunks_unchecked::<64>()
        };

        for chunk in 0..data_output_chunks.len() {
            let masking_output_chunk = unsafe { masking_output_chunks.get_unchecked_mut(chunk) };
            let data_output_chunk = unsafe { data_output_chunks.get_unchecked_mut(chunk) };
            let masking_chunk = unsafe { masking_chunks.get_unchecked(chunk) };
            let analysis_chunk = unsafe { analysis_chunks.get_unchecked(chunk) };
            let exponent_scale_chunk = unsafe { exponent_scale_chunks.get_unchecked(chunk) };
            let exponent_bias_chunk = unsafe { exponent_bias_chunks.get_unchecked(chunk) };
            let param_1_chunk = unsafe { param_1_chunks.get_unchecked(chunk) };
            let inverse_param_2_chunk = unsafe { inverse_param_2_chunks.get_unchecked(chunk) };

            for lane in 0..64 {
                let exponent_scale = unsafe { *exponent_scale_chunk.get_unchecked(lane) };
                let exponent_bias = unsafe { *exponent_bias_chunk.get_unchecked(lane) };
                let param_1 = unsafe { *param_1_chunk.get_unchecked(lane) };
                let inverse_param_2 = unsafe { *inverse_param_2_chunk.get_unchecked(lane) };
                let masking_amplitude = unsafe { *masking_chunk.get_unchecked(lane) };
                let masking_volume = spl_to_phon(
                    amplitude_to_dbfs(masking_amplitude).algebraic_add(total_gain),
                    exponent_scale,
                    exponent_bias,
                    param_1,
                    inverse_param_2,
                )
                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                .clamp(HEARING_THRESHOLD_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);

                sum = sum.algebraic_add(dbfs_to_amplitude(masking_volume));
                *unsafe { masking_output_chunk.get_unchecked_mut(lane) } = (0.0, masking_volume);

                let amplitude = unsafe { *analysis_chunk.get_unchecked(lane) };
                let volume = spl_to_phon(
                    amplitude_to_dbfs(amplitude).algebraic_add(total_gain),
                    exponent_scale,
                    exponent_bias,
                    param_1,
                    inverse_param_2,
                )
                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);
                *unsafe { data_output_chunk.get_unchecked_mut(lane) } = (0.0, volume);
            }
        }

        self.masking_mean = amplitude_to_dbfs(sum.algebraic_div(self.masking.len() as f32));
    }
    fn update_mono_masking_and_data_unnormalized(&mut self, center: &BetterAnalyzer, gain: f32) {
        let masking_chunks = unsafe { center.raw_masking().as_chunks_unchecked::<64>() };
        let masking_output_chunks = unsafe { self.masking.as_chunks_unchecked_mut::<64>() };

        let analysis_chunks = unsafe { center.raw_analysis().as_chunks_unchecked::<64>() };
        let data_output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let mut sum: f32 = 0.0;

        let total_gain = gain.algebraic_add(MONO_AMPLITUDE_DB);

        for (masking_output_chunk, (data_output_chunk, (masking_chunk, analysis_chunk))) in
            masking_output_chunks.iter_mut().zip(
                data_output_chunks
                    .iter_mut()
                    .zip(masking_chunks.iter().zip(analysis_chunks)),
            )
        {
            for lane in 0..64 {
                let masking_amplitude = unsafe { *masking_chunk.get_unchecked(lane) };

                sum = sum.algebraic_add(masking_amplitude);
                *unsafe { masking_output_chunk.get_unchecked_mut(lane) } = (
                    0.0,
                    amplitude_to_dbfs(masking_amplitude).algebraic_add(total_gain),
                );

                let amplitude = unsafe { *analysis_chunk.get_unchecked(lane) };
                *unsafe { data_output_chunk.get_unchecked_mut(lane) } =
                    (0.0, amplitude_to_dbfs(amplitude).algebraic_add(total_gain));
            }
        }

        self.masking_mean = amplitude_to_dbfs(sum.algebraic_div(self.masking.len() as f32))
            .algebraic_add(total_gain);
    }
    fn update_mono_data(
        &mut self,
        center: &BetterAnalyzer,
        gain: f32,
        normalization_volume: Option<f32>,
    ) {
        if let Some(listening_volume) = normalization_volume {
            self.update_mono_data_normalized(center, gain, listening_volume);
        } else {
            self.update_mono_data_unnormalized(center, gain);
        }
    }
    fn update_mono_data_normalized(
        &mut self,
        center: &BetterAnalyzer,
        gain: f32,
        listening_volume: f32,
    ) {
        let analysis_chunks = unsafe { center.raw_analysis().as_chunks_unchecked::<64>() };
        let output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let total_gain = gain
            .algebraic_add(listening_volume)
            .algebraic_add(MONO_AMPLITUDE_DB);

        let exponent_scale_chunks = unsafe {
            center
                .normalizers
                .exponent_scales
                .as_chunks_unchecked::<64>()
        };
        let exponent_bias_chunks = unsafe {
            center
                .normalizers
                .exponent_biases
                .as_chunks_unchecked::<64>()
        };
        let param_1_chunks = unsafe { center.normalizers.params_1.as_chunks_unchecked::<64>() };
        let inverse_param_2_chunks = unsafe {
            center
                .normalizers
                .inverse_params_2
                .as_chunks_unchecked::<64>()
        };

        for chunk in 0..output_chunks.len() {
            let output_chunk = unsafe { output_chunks.get_unchecked_mut(chunk) };
            let analysis_chunk = unsafe { analysis_chunks.get_unchecked(chunk) };
            let exponent_scale_chunk = unsafe { exponent_scale_chunks.get_unchecked(chunk) };
            let exponent_bias_chunk = unsafe { exponent_bias_chunks.get_unchecked(chunk) };
            let param_1_chunk = unsafe { param_1_chunks.get_unchecked(chunk) };
            let inverse_param_2_chunk = unsafe { inverse_param_2_chunks.get_unchecked(chunk) };

            for lane in 0..64 {
                let output = unsafe { output_chunk.get_unchecked_mut(lane) };
                let amplitude = unsafe { *analysis_chunk.get_unchecked(lane) };
                let volume = spl_to_phon(
                    amplitude_to_dbfs(amplitude).algebraic_add(total_gain),
                    unsafe { *exponent_scale_chunk.get_unchecked(lane) },
                    unsafe { *exponent_bias_chunk.get_unchecked(lane) },
                    unsafe { *param_1_chunk.get_unchecked(lane) },
                    unsafe { *inverse_param_2_chunk.get_unchecked(lane) },
                )
                //.clamp(MIN_COMPLETE_NORM_PHON, MAX_COMPLETE_NORM_PHON)
                .clamp(MIN_INFORMATIVE_NORM_PHON, MAX_INFORMATIVE_NORM_PHON)
                .algebraic_sub(listening_volume);

                *output = (0.0, volume);
            }
        }
    }
    fn update_mono_data_unnormalized(&mut self, center: &BetterAnalyzer, gain: f32) {
        let analysis_chunks = unsafe { center.raw_analysis().as_chunks_unchecked::<64>() };
        let output_chunks = unsafe { self.data.as_chunks_unchecked_mut::<64>() };

        let total_gain = gain.algebraic_add(MONO_AMPLITUDE_DB);

        for (output_chunk, analysis_chunk) in output_chunks.iter_mut().zip(analysis_chunks) {
            for lane in 0..64 {
                let output = unsafe { output_chunk.get_unchecked_mut(lane) };
                let amplitude = unsafe { *analysis_chunk.get_unchecked(lane) };
                *output = (0.0, amplitude_to_dbfs(amplitude).algebraic_add(total_gain));
            }
        }
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
                    masking_mean: f32::NEG_INFINITY,
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

const MONO_AMPLITUDE_DB: f32 = 20.0 * std::f32::consts::LOG10_2;

pub fn calculate_pan_and_volume_from_amplitude(
    left_amplitude: f32,
    right_amplitude: f32,
) -> (f32, f32) {
    let amplitude = left_amplitude.algebraic_add(right_amplitude);

    (
        calculate_pan_from_amplitude_sum(left_amplitude, right_amplitude, amplitude),
        amplitude_to_dbfs(amplitude),
    )
}

pub fn calculate_pan_from_amplitude(left_amplitude: f32, right_amplitude: f32) -> f32 {
    let amplitude = left_amplitude.algebraic_add(right_amplitude);
    calculate_pan_from_amplitude_sum(left_amplitude, right_amplitude, amplitude)
}

#[inline(always)]
fn calculate_pan_from_amplitude_sum(
    left_amplitude: f32,
    right_amplitude: f32,
    amplitude: f32,
) -> f32 {
    if amplitude == 0.0 || amplitude.is_nan() {
        0.0
    } else {
        // This is algebraically equivalent to the original inverse constant-power panning
        // formula for amplitudes. Its input is bounded to [-1, 1], and the shared sum avoids
        // both the original square root and duplicate volume arithmetic.
        f32::atan(
            left_amplitude
                .algebraic_sub(right_amplitude)
                .algebraic_div(amplitude),
        )
        .algebraic_mul((-4.0_f32).algebraic_div(PI))
    }
}

// ----- Below formulas are taken from ISO 226:2023 -----

#[derive(Clone)]
struct PrecomputedNormalizers {
    exponent_scales: Vec<f32>,
    exponent_biases: Vec<f32>,
    params_1: Vec<f32>,
    inverse_params_2: Vec<f32>,
}

impl PrecomputedNormalizers {
    fn new(frequency_bands: &[FrequencyBand]) -> Self {
        let mut normalizers = Self {
            exponent_scales: Vec::with_capacity(frequency_bands.len()),
            exponent_biases: Vec::with_capacity(frequency_bands.len()),
            params_1: Vec::with_capacity(frequency_bands.len()),
            inverse_params_2: Vec::with_capacity(frequency_bands.len()),
        };

        for band in frequency_bands {
            let (alpha_f, l_u, t_f) = approximate_coefficients(band.center);
            let exponent_scale = alpha_f * std::f32::consts::LOG2_10 / 10.0;
            let exponent_bias = exponent_scale * l_u;

            normalizers.exponent_scales.push(exponent_scale);
            normalizers.exponent_biases.push(exponent_bias);
            normalizers
                .params_1
                .push(f32::exp2(exponent_scale * t_f + exponent_bias));
            normalizers
                .inverse_params_2
                .push(1.0 / (4.0e-10_f32).powf(0.3 - alpha_f));
        }

        normalizers
    }
}

#[inline(always)]
fn spl_to_phon(
    db_spl: f32,
    exponent_scale: f32,
    exponent_bias: f32,
    param_1: f32,
    inverse_param_2: f32,
) -> f32 {
    let power = f32::exp2(
        db_spl
            .algebraic_mul(exponent_scale)
            .algebraic_add(exponent_bias),
    );
    let normalized = power
        .algebraic_sub(param_1)
        .algebraic_mul(inverse_param_2)
        .algebraic_add(NORM_OFFSET);

    NORM_LOG2_MULTIPLE.algebraic_mul(f32::log2(normalized))
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

pub const MIN_COMPLETE_NORM_PHON: f32 = 20.0;
pub const MAX_COMPLETE_NORM_PHON: f32 = 80.0;
pub const MIN_INFORMATIVE_NORM_PHON: f32 = 0.0;
pub const MAX_INFORMATIVE_NORM_PHON: f32 = 100.0;
#[allow(clippy::excessive_precision)]
pub const HEARING_THRESHOLD_PHON: f32 = 2.4000000000000012;
/* Calculated using:
let mut hearing_threshold_phon: f64 = 0.0;
for f in 20..=20000 {
    hearing_threshold_phon = hearing_threshold_phon.max(spl_to_phon(
        f as f64,
        approximate_hearing_threshold(f as f64),
    ));
}*/
const NORM_MULTIPLE: f32 = 100.0 / 3.0;
const NORM_LOG2_MULTIPLE: f32 = NORM_MULTIPLE * std::f32::consts::LOG10_2;
#[allow(clippy::excessive_precision)]
const NORM_OFFSET: f32 = 1.180_320_635_651_729_7; // 10.0_f64.powf(0.072)

const NORM_FREQUENCIES: &[f32] = &[
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0,
];

const MIN_NORM_FREQUENCY: f32 = NORM_FREQUENCIES[0];
const MAX_NORM_FREQUENCY: f32 = NORM_FREQUENCIES[NORM_FREQUENCIES.len() - 1];
const NORM_FREQUENCY_COUNT: usize = NORM_FREQUENCIES.len();

const ALPHA_F: &[f32] = &[
    0.635, 0.602, 0.569, 0.537, 0.509, 0.482, 0.456, 0.433, 0.412, 0.391, 0.373, 0.357, 0.343,
    0.330, 0.320, 0.311, 0.303, 0.300, 0.295, 0.292, 0.290, 0.290, 0.289, 0.289, 0.289, 0.293,
    0.303, 0.323, 0.354,
];

const L_U: &[f32] = &[
    -31.5, -27.2, -23.1, -19.3, -16.1, -13.1, -10.4, -8.2, -6.3, -4.6, -3.2, -2.1, -1.2, -0.5, 0.0,
    0.4, 0.5, 0.0, -2.7, -4.2, -1.2, 1.4, 2.3, 1.0, -2.3, -7.2, -11.2, -10.9, -3.5,
];

const T_F: &[f32] = &[
    78.1, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0,
    2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
];

fn approximate_coefficients(frequency: f32) -> (f32, f32, f32) {
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

/*fn approximate_hearing_threshold(frequency: f32) -> f32 {
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
}*/

// ----- Below algorithms are taken from https://codepen.io/TF3RDL/pen/MWLzPoO -----

#[inline(always)]
pub fn amplitude_to_dbfs(amplitude: f32) -> f32 {
    20.0_f32.algebraic_mul(f32::log10(amplitude))
}

#[inline(always)]
pub fn dbfs_to_amplitude(decibels: f32) -> f32 {
    f32::exp2(decibels.algebraic_mul(std::f32::consts::LOG2_10.algebraic_div(20.0)))
}

#[inline(always)]
pub fn map_value(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
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

// Used by render thread
#[inline(always)]
pub fn scale_bark(x: f32) -> f32 {
    (26.81_f32.algebraic_mul(x)).algebraic_div((1960.0_f32.algebraic_add(x)).algebraic_sub(0.53))
}

impl FrequencyScale {
    pub fn scale(&self, x: f32) -> f32 {
        match self {
            Self::Logarithmic => x.log2(),
            Self::Erb => (1.0 + 0.00437 * x).log2(),
            Self::Bark => (26.81 * x) / (1960.0 + x) - 0.53,
            Self::Mel => (1.0 + x / 700.0).log2(),
        }
    }
    pub fn inv_scale(&self, x: f32) -> f32 {
        match self {
            Self::Logarithmic => 2.0_f32.powf(x),
            Self::Erb => (1.0 / 0.00437) * ((2.0_f32.powf(x)) - 1.0),
            Self::Bark => 1960.0 / (26.81 / (x + 0.53) - 1.0),
            Self::Mel => 700.0 * ((2.0_f32.powf(x)) - 1.0),
        }
    }
    fn generate_bands<F>(&self, n: usize, low: f32, high: f32, bandwidth: F) -> Vec<FrequencyBand>
    where
        F: Fn(f32) -> f32,
    {
        let scaled_low = self.scale(low);
        let scaled_high = self.scale(high);
        let target_max = (n - 1) as f32;

        (0..n)
            .map(|i| {
                let i = i as f32;

                let center = self.inv_scale(map_value(i, 0.0, target_max, scaled_low, scaled_high));
                let lower =
                    self.inv_scale(map_value(i - 0.5, 0.0, target_max, scaled_low, scaled_high));
                let higher =
                    self.inv_scale(map_value(i + 0.5, 0.0, target_max, scaled_low, scaled_high));
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
    low: f32,
    center: f32,
    high: f32,
}
