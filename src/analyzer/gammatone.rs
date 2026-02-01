use std::f64::consts::PI;

use crate::analyzer::FrequencyBand;

// ----- Below algorithm is based on https://github.com/dfl/cortix -----

#[derive(Clone)]
pub(super) struct GammatoneFilterbank {
    filters: Vec<Filter>,
    pub(super) spectrum_data: Vec<f64>,
}

impl GammatoneFilterbank {
    pub(super) fn new(freq_bands: &[FrequencyBand], sample_rate: f64) -> Self {
        assert!(freq_bands.len().is_multiple_of(64));

        let band_count = freq_bands.len();

        Self {
            filters: freq_bands
                .iter()
                .copied()
                .map(|x| Filter::new(x, sample_rate))
                .collect(),
            spectrum_data: vec![0.0; band_count],
        }
    }
    pub(super) fn analyze(&mut self, samples: impl ExactSizeIterator<Item = f64>) -> &[f64] {
        let filter_chunks = unsafe { self.filters.as_chunks_unchecked_mut::<64>() };
        let spectrum_chunks = unsafe { self.spectrum_data.as_chunks_unchecked_mut::<64>() };

        for sample in samples {
            for (filter_chunk, spectrum_chunk) in
                filter_chunks.iter_mut().zip(spectrum_chunks.iter_mut())
            {
                for (filter, spectrum_data) in filter_chunk.iter_mut().zip(spectrum_chunk) {
                    let result = filter.process(sample);

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

        &self.spectrum_data
    }
}

#[derive(Clone)]
pub(super) struct Filter {
    r: f64,
    cos_omega: f64,
    sin_omega: f64,
    gain: f64,
    state: [(f64, f64); 4],
}

impl Filter {
    fn new(band: FrequencyBand, sample_rate: f64) -> Self {
        let omega = 2.0 * PI * band.center / sample_rate;
        let bw = 2.0 * PI * (band.high - band.low).abs() / sample_rate;
        let r = (-bw).exp();

        Self {
            r,
            cos_omega: f64::cos(omega),
            sin_omega: f64::sin(omega),
            gain: (1.0 - r).powi(4) * 2.0,
            state: [(0.0, 0.0); 4],
        }
    }
    fn reset(&mut self) {
        self.state = [(0.0, 0.0); 4];
    }
    fn process(&mut self, input: f64) -> (f64, f64) {
        let mut result: (f64, f64) = (input.algebraic_mul(self.gain), 0.0);

        for stage in self.state.iter_mut() {
            result = (
                result.0.algebraic_add(
                    self.r.algebraic_mul(
                        self.cos_omega
                            .algebraic_mul(stage.0)
                            .algebraic_sub(self.sin_omega.algebraic_mul(stage.1)),
                    ),
                ),
                result.1.algebraic_add(
                    self.r.algebraic_mul(
                        self.sin_omega
                            .algebraic_mul(stage.0)
                            .algebraic_add(self.cos_omega.algebraic_mul(stage.1)),
                    ),
                ),
            );
            *stage = result;
        }

        result
    }
}
