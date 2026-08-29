use std::{
    any::Any,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytemuck::cast_slice;
use egui_glow::{
    CallbackFn,
    glow::{self, HasContext as _},
};
use half::f16;

use crate::analyzer::{Spectrogram, SpectrogramRenderState, scale_bark};

use super::{ColorTable, RenderSettings};

const VERTEX_SHADER: &str = r#"
const vec2 POSITIONS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
);
out vec2 v_uv;
void main() {
    vec2 position = POSITIONS[gl_VertexID];
    v_uv = vec2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER: &str = r#"
#ifdef GL_ES
precision highp float;
precision highp int;
#endif

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_history;
uniform sampler2D u_bar;
uniform sampler2D u_lut;
uniform sampler2D u_masking_ranges;
uniform ivec2 u_history_size;
uniform int u_history_head;
uniform int u_valid_rows;
uniform int u_horizontal;
uniform int u_nearest;
uniform int u_show_spectrogram;
uniform int u_show_bar;
uniform int u_use_smr;
uniform int u_show_masking;
uniform float u_bar_proportion;
uniform vec2 u_db_range;
uniform float u_smr_strength;
uniform vec4 u_masking_color;

float map_amplitude(float amplitude) {
    return clamp((amplitude - u_db_range.x) / (u_db_range.y - u_db_range.x), 0.0, 1.0);
}

vec3 lookup_color(float pan, float intensity) {
    ivec2 size = textureSize(u_lut, 0);
    int x = int(round(clamp(intensity, 0.0, 1.0) * float(size.x - 1)));
    int y = int(round(clamp(pan * 0.5 + 0.5, 0.0, 1.0) * float(size.y - 1)));
    return texelFetch(u_lut, ivec2(x, y), 0).rgb;
}

vec4 color_history_texel(int frequency, int age) {
    if (u_valid_rows <= 0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    frequency = clamp(frequency, 0, u_history_size.x - 1);
    age = clamp(age, 0, u_history_size.y - 1);
    if (age >= u_valid_rows) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    int physical_row = (u_history_head + age) % u_history_size.y;
    vec4 analysis = texelFetch(u_history, ivec2(frequency, physical_row), 0);
    if (analysis.a < 0.5) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float volume_intensity = map_amplitude(analysis.g);
    float intensity = volume_intensity;
    if (u_use_smr != 0) {
        float range = texelFetch(u_masking_ranges, ivec2(frequency, 0), 0).r;
        float smr_intensity = clamp((analysis.g - analysis.b) / range, 0.0, 1.0);
        intensity = min(volume_intensity,
            mix(volume_intensity, smr_intensity, u_smr_strength));
    }
    return vec4(lookup_color(analysis.r, intensity), 1.0);
}

vec4 sample_history(vec2 uv) {
    if (u_nearest != 0) {
        ivec2 coordinate = ivec2(floor(uv * vec2(u_history_size)));
        coordinate = clamp(coordinate, ivec2(0), u_history_size - ivec2(1));
        return color_history_texel(coordinate.x, coordinate.y);
    }

    vec2 position = uv * vec2(u_history_size) - vec2(0.5);
    ivec2 low = ivec2(floor(position));
    vec2 weight = fract(position);
    vec4 top = mix(color_history_texel(low.x, low.y),
                   color_history_texel(low.x + 1, low.y), weight.x);
    vec4 bottom = mix(color_history_texel(low.x, low.y + 1),
                      color_history_texel(low.x + 1, low.y + 1), weight.x);
    return mix(top, bottom, weight.y);
}

float interleaved_gradient_noise(vec2 n) {
    float f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

vec3 dither(vec3 rgb) {
    float noise = (interleaved_gradient_noise(gl_FragCoord.xy) - 0.5) * 0.95;
    return rgb + noise / 255.0;
}

vec4 draw_bar(vec2 uv) {
    int bins = textureSize(u_bar, 0).x;
    float frequency_uv = u_horizontal != 0 ? 1.0 - uv.y : uv.x;
    int frequency = clamp(int(floor(frequency_uv * float(bins))), 0, bins - 1);
    vec4 analysis = texelFetch(u_bar, ivec2(frequency, 0), 0);
    if (analysis.a < 0.5) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float along = u_horizontal != 0
        ? (u_bar_proportion - uv.x) / u_bar_proportion
        : (u_bar_proportion - uv.y) / u_bar_proportion;
    float amplitude = map_amplitude(analysis.g);
    vec3 bar_color = along <= amplitude
        ? lookup_color(analysis.r, amplitude)
        : vec3(0.0);
    if (u_show_masking != 0 && along <= map_amplitude(analysis.b)) {
        // Color32 stores premultiplied sRGBA. Match egui's ONE,
        // ONE_MINUS_SRC_ALPHA blending while this pass has GL blending disabled.
        vec3 blended = u_masking_color.rgb
            + bar_color * (1.0 - u_masking_color.a);
        return vec4(blended, 1.0);
    }
    return vec4(bar_color, 1.0);
}

void main() {
    bool in_bar = u_horizontal != 0
        ? v_uv.x < u_bar_proportion
        : v_uv.y < u_bar_proportion;
    vec4 color;
    if (in_bar && u_show_bar != 0) {
        color = draw_bar(v_uv);
    } else if (!in_bar && u_show_spectrogram != 0) {
        vec2 spectrogram_uv = u_horizontal != 0
            ? vec2(1.0 - v_uv.y,
                   (v_uv.x - u_bar_proportion) / (1.0 - u_bar_proportion))
            : vec2(v_uv.x,
                   (v_uv.y - u_bar_proportion) / (1.0 - u_bar_proportion));
        color = sample_history(clamp(spectrogram_uv, 0.0, 1.0));
    } else {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    color.rgb = dither(color.rgb);
    f_color = color;
}
"#;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FrameUniforms {
    horizontal: bool,
    nearest: bool,
    show_spectrogram: bool,
    show_bar: bool,
    use_smr: bool,
    show_masking: bool,
    bar_proportion: f32,
    min_db: f32,
    max_db: f32,
    smr_strength: f32,
    masking_color: [f32; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UploadRange {
    source_row: usize,
    target_row: usize,
    rows: usize,
}

#[derive(Default)]
struct PendingHistory {
    width: usize,
    height: usize,
    head: usize,
    valid_rows: usize,
    rebuild: bool,
    ranges: [Option<UploadRange>; 2],
    target_state: Option<SpectrogramRenderState>,
    target_duration: Option<Duration>,
}

#[derive(Default)]
struct BarCache {
    state: Option<SpectrogramRenderState>,
    averaging: Duration,
    include_masking: bool,
    width: usize,
    duration: Option<Duration>,
    contiguous_rows: usize,
    window_rows: usize,
    data_sums: Vec<BarDataSum>,
    masking_sums: Vec<FiniteSum>,
}

#[derive(Clone, Copy, Default)]
struct BarDataSum {
    pan: f64,
    amplitude: FiniteSum,
}

#[derive(Clone, Copy, Default)]
struct FiniteSum {
    value: f64,
    non_finite: usize,
}

struct GlResources {
    program: glow::Program,
    uniforms: Uniforms,
    vertex_array: glow::VertexArray,
    history: glow::Texture,
    bar: glow::Texture,
    lut: glow::Texture,
    masking_ranges: glow::Texture,
    history_size: (usize, usize),
    bar_width: usize,
    lut_size: (usize, usize),
    masking_width: usize,
}

struct Uniforms {
    history: Option<glow::UniformLocation>,
    bar: Option<glow::UniformLocation>,
    lut: Option<glow::UniformLocation>,
    masking_ranges: Option<glow::UniformLocation>,
    history_size: Option<glow::UniformLocation>,
    history_head: Option<glow::UniformLocation>,
    valid_rows: Option<glow::UniformLocation>,
    horizontal: Option<glow::UniformLocation>,
    nearest: Option<glow::UniformLocation>,
    show_spectrogram: Option<glow::UniformLocation>,
    show_bar: Option<glow::UniformLocation>,
    use_smr: Option<glow::UniformLocation>,
    show_masking: Option<glow::UniformLocation>,
    bar_proportion: Option<glow::UniformLocation>,
    db_range: Option<glow::UniformLocation>,
    smr_strength: Option<glow::UniformLocation>,
    masking_color: Option<glow::UniformLocation>,
}

pub(super) struct ShaderRendererHandle {
    state: Arc<Mutex<ShaderRenderer>>,
    callback: Arc<dyn Any + Send + Sync>,
}

impl ShaderRendererHandle {
    pub(super) fn new() -> Self {
        let state = Arc::new(Mutex::new(ShaderRenderer::default()));
        let callback_state = Arc::clone(&state);
        let callback: Arc<dyn Any + Send + Sync> =
            Arc::new(CallbackFn::new(move |_info, painter| {
                let mut renderer = callback_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Err(error) = renderer.paint(painter.gl()) {
                    renderer.error = Some(error);
                }
            }));
        Self { state, callback }
    }

    pub(super) fn callback(&self) -> Arc<dyn Any + Send + Sync> {
        Arc::clone(&self.callback)
    }

    pub(super) fn prepare(
        &self,
        spectrogram: &Spectrogram,
        frequencies: &[(f32, f32, f32)],
        color_table: &ColorTable,
        color_revision: u64,
        settings: &RenderSettings,
        masking_enabled: bool,
        min_db: f32,
        max_db: f32,
    ) {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .prepare(
                spectrogram,
                frequencies,
                color_table,
                color_revision,
                settings,
                masking_enabled,
                min_db,
                max_db,
            );
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn initialize(&self, gl: &glow::Context) -> Result<(), String> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .initialize(gl)
    }

    pub(super) fn error(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .error
            .clone()
    }
}

#[derive(Default)]
struct ShaderRenderer {
    gl: Option<GlResources>,
    error: Option<String>,
    max_texture_size: Option<usize>,
    history_staging: Vec<f16>,
    bar_staging: Vec<f16>,
    bar_cache: BarCache,
    lut_staging: Vec<u8>,
    masking_staging: Vec<f16>,
    masking_centers: Vec<f32>,
    pending_history: PendingHistory,
    bar_dirty: bool,
    lut_dirty: bool,
    masking_dirty: bool,
    frame: FrameUniforms,
    staged_state: Option<SpectrogramRenderState>,
    staged_duration: Option<Duration>,
    staged_height: usize,
    staged_width: usize,
    staged_valid_rows: usize,
    history_head: usize,
    was_spectrogram_visible: bool,
    was_bar_visible: bool,
    color_revision: Option<u64>,
    lut_size: (usize, usize),
}

impl ShaderRenderer {
    fn prepare(
        &mut self,
        spectrogram: &Spectrogram,
        frequencies: &[(f32, f32, f32)],
        color_table: &ColorTable,
        color_revision: u64,
        settings: &RenderSettings,
        masking_enabled: bool,
        min_db: f32,
        max_db: f32,
    ) {
        let bins = spectrogram.newest().data.len();
        let show_spectrogram = settings.bargraph_height < 1.0 && frequencies.len() == bins;
        let show_bar = settings.bargraph_height > 0.0 && bins > 0;
        self.frame = FrameUniforms {
            horizontal: settings.horizontal,
            nearest: settings.spectrogram_nearest_neighbor,
            show_spectrogram,
            show_bar,
            use_smr: show_spectrogram
                && masking_enabled
                && settings.clamp_using_smr
                && settings.clamp_strength > 0.0,
            show_masking: show_bar && masking_enabled && settings.show_masking,
            bar_proportion: settings.bargraph_height.clamp(0.0, 1.0),
            min_db,
            max_db,
            smr_strength: settings.clamp_strength.clamp(0.0, 1.0),
            masking_color: color32_to_array(settings.masking_color),
        };

        if self.color_revision != Some(color_revision) {
            self.lut_staging.clear();
            self.lut_staging
                .reserve((color_table.table.len() * 4).saturating_sub(self.lut_staging.capacity()));
            for &(r, g, b) in &color_table.table {
                self.lut_staging.extend_from_slice(&[r, g, b, 255]);
            }
            self.color_revision = Some(color_revision);
            self.lut_size = (color_table.size.1, color_table.size.0);
            self.lut_dirty = true;
        }

        if self.frame.use_smr
            && frequencies.len() == bins
            && frequency_centers_changed(&self.masking_centers, frequencies)
        {
            self.masking_centers.clear();
            self.masking_staging.clear();
            self.masking_centers
                .reserve(bins.saturating_sub(self.masking_centers.capacity()));
            self.masking_staging
                .reserve(bins.saturating_sub(self.masking_staging.capacity()));
            for &(_, center, _) in frequencies {
                self.masking_centers.push(center);
                self.masking_staging
                    .push(f16::from_f32(27.0 - (6.025 - 0.275 * scale_bark(center))));
            }
            self.masking_dirty = true;
        }

        if show_bar {
            let bar_changed = prepare_bar_row_cached(
                spectrogram,
                settings.bargraph_averaging,
                self.frame.show_masking,
                &mut self.bar_cache,
                &mut self.bar_staging,
            );
            self.bar_dirty |= bar_changed || !self.was_bar_visible;
        } else {
            self.bar_dirty = false;
        }
        self.was_bar_visible = show_bar;

        if show_spectrogram {
            let state = spectrogram.render_state();
            let requested_height = requested_history_height(
                settings.spectrogram_duration,
                spectrogram.newest().duration,
                state.retained_rows,
                self.max_texture_size,
            );
            let force_rebuild = !self.was_spectrogram_visible;
            self.prepare_history(spectrogram, state, bins, requested_height, force_rebuild);
        } else {
            self.pending_history.ranges = [None, None];
        }
        self.was_spectrogram_visible = show_spectrogram;
    }

    fn prepare_history(
        &mut self,
        spectrogram: &Spectrogram,
        state: SpectrogramRenderState,
        width: usize,
        height: usize,
        force_rebuild: bool,
    ) {
        let previous = self.staged_state;
        let target_duration = spectrogram.newest().duration;
        let revision_overflow =
            previous.is_some_and(|old| state.revision < old.revision && state.epoch == old.epoch);
        let rebuild = force_rebuild
            || previous.is_none()
            || previous.is_some_and(|old| old.epoch != state.epoch)
            || revision_overflow
            || width != self.staged_width
            || height != self.staged_height
            || self.staged_duration != Some(target_duration);

        if rebuild {
            let valid_rows = contiguous_history_rows(
                spectrogram,
                state.valid_rows.min(height),
                width,
                target_duration,
            );
            self.history_head = 0;
            resize_zeroed(&mut self.history_staging, width * valid_rows * 4);
            for age in 0..valid_rows {
                write_history_row(
                    &mut self.history_staging,
                    age,
                    width,
                    spectrogram.at_age(age),
                );
            }
            self.pending_history = PendingHistory {
                width,
                height,
                head: 0,
                valid_rows,
                rebuild: true,
                ranges: [
                    (valid_rows > 0).then_some(UploadRange {
                        source_row: 0,
                        target_row: 0,
                        rows: valid_rows,
                    }),
                    None,
                ],
                target_state: Some(state),
                target_duration: Some(target_duration),
            };
        } else {
            let old = previous.unwrap();
            let changed = state.revision.wrapping_sub(old.revision) as usize;
            if changed == 0 {
                self.pending_history.valid_rows =
                    self.staged_valid_rows.min(state.valid_rows).min(height);
            } else if changed >= height {
                self.prepare_history(spectrogram, state, width, height, true);
            } else {
                let changed_to_check = changed.min(state.valid_rows).min(height);
                let new_valid_rows =
                    contiguous_history_rows(spectrogram, changed_to_check, width, target_duration);
                let valid_rows = if new_valid_rows == changed_to_check {
                    changed
                        .saturating_add(self.staged_valid_rows)
                        .min(state.valid_rows)
                        .min(height)
                } else {
                    new_valid_rows
                };
                resize_zeroed(&mut self.history_staging, width * new_valid_rows * 4);
                for age in 0..new_valid_rows {
                    write_history_row(
                        &mut self.history_staging,
                        age,
                        width,
                        spectrogram.at_age(age),
                    );
                }
                let target_head = (self.history_head + height - changed) % height;
                let first_rows = new_valid_rows.min(height - target_head);
                let second_rows = new_valid_rows - first_rows;
                self.pending_history = PendingHistory {
                    width,
                    height,
                    head: target_head,
                    valid_rows,
                    rebuild: false,
                    ranges: [
                        (first_rows > 0).then_some(UploadRange {
                            source_row: 0,
                            target_row: target_head,
                            rows: first_rows,
                        }),
                        (second_rows > 0).then_some(UploadRange {
                            source_row: first_rows,
                            target_row: 0,
                            rows: second_rows,
                        }),
                    ],
                    target_state: Some(state),
                    target_duration: Some(target_duration),
                };
            }
        }
    }

    fn commit_history_upload(&mut self) {
        if let Some(state) = self.pending_history.target_state.take() {
            self.staged_state = Some(state);
            self.staged_width = self.pending_history.width;
            self.staged_height = self.pending_history.height;
            self.history_head = self.pending_history.head;
            self.staged_valid_rows = self.pending_history.valid_rows;
            self.staged_duration = self.pending_history.target_duration.take();
        }
    }

    fn initialize(&mut self, gl: &glow::Context) -> Result<(), String> {
        if self.gl.is_some() {
            return Ok(());
        }
        let resources = unsafe { create_resources(gl)? };
        self.max_texture_size =
            Some(unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) }.max(1) as usize);
        self.gl = Some(resources);
        Ok(())
    }

    fn paint(&mut self, gl: &glow::Context) -> Result<(), String> {
        if self.error.is_some() {
            return Ok(());
        }
        self.initialize(gl)?;
        if let Some(limit) = self.max_texture_size {
            clamp_pending_history(&mut self.pending_history, limit);
        }
        let mut resources = self.gl.take().unwrap();
        unsafe {
            upload_pending(gl, &mut resources, self);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(resources.program));
            gl.bind_vertex_array(Some(resources.vertex_array));

            bind_texture(gl, resources.history, 0);
            bind_texture(gl, resources.bar, 1);
            bind_texture(gl, resources.lut, 2);
            bind_texture(gl, resources.masking_ranges, 3);
            let uniforms = &resources.uniforms;
            gl.uniform_1_i32(uniforms.history.as_ref(), 0);
            gl.uniform_1_i32(uniforms.bar.as_ref(), 1);
            gl.uniform_1_i32(uniforms.lut.as_ref(), 2);
            gl.uniform_1_i32(uniforms.masking_ranges.as_ref(), 3);
            gl.uniform_2_i32(
                uniforms.history_size.as_ref(),
                self.pending_history.width as i32,
                self.pending_history.height as i32,
            );
            gl.uniform_1_i32(
                uniforms.history_head.as_ref(),
                self.pending_history.head as i32,
            );
            gl.uniform_1_i32(
                uniforms.valid_rows.as_ref(),
                self.pending_history.valid_rows as i32,
            );
            gl.uniform_1_i32(uniforms.horizontal.as_ref(), self.frame.horizontal as i32);
            gl.uniform_1_i32(uniforms.nearest.as_ref(), self.frame.nearest as i32);
            gl.uniform_1_i32(
                uniforms.show_spectrogram.as_ref(),
                self.frame.show_spectrogram as i32,
            );
            gl.uniform_1_i32(uniforms.show_bar.as_ref(), self.frame.show_bar as i32);
            gl.uniform_1_i32(uniforms.use_smr.as_ref(), self.frame.use_smr as i32);
            gl.uniform_1_i32(
                uniforms.show_masking.as_ref(),
                self.frame.show_masking as i32,
            );
            gl.uniform_1_f32(uniforms.bar_proportion.as_ref(), self.frame.bar_proportion);
            gl.uniform_2_f32(
                uniforms.db_range.as_ref(),
                self.frame.min_db,
                self.frame.max_db,
            );
            gl.uniform_1_f32(uniforms.smr_strength.as_ref(), self.frame.smr_strength);
            gl.uniform_4_f32(
                uniforms.masking_color.as_ref(),
                self.frame.masking_color[0],
                self.frame.masking_color[1],
                self.frame.masking_color[2],
                self.frame.masking_color[3],
            );
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
        }
        self.gl = Some(resources);
        Ok(())
    }
}

fn clamp_pending_history(pending: &mut PendingHistory, limit: usize) {
    if pending.height <= limit {
        return;
    }
    pending.height = limit;
    pending.valid_rows = pending.valid_rows.min(limit);
    pending.head = 0;
    pending.rebuild = true;
    pending.ranges = [
        (pending.valid_rows > 0).then_some(UploadRange {
            source_row: 0,
            target_row: 0,
            rows: pending.valid_rows,
        }),
        None,
    ];
}

fn requested_history_height(
    requested: Duration,
    slice: Duration,
    retained: usize,
    hardware: Option<usize>,
) -> usize {
    let rows = if slice.is_zero() {
        1
    } else {
        (requested.as_secs_f64() / slice.as_secs_f64()).round() as usize
    };
    rows.clamp(1, retained.max(1))
        .min(hardware.unwrap_or(usize::MAX).max(1))
}

fn frequency_centers_changed(cached: &[f32], frequencies: &[(f32, f32, f32)]) -> bool {
    cached.len() != frequencies.len()
        || cached
            .iter()
            .zip(frequencies)
            .any(|(cached, frequency)| cached.to_bits() != frequency.1.to_bits())
}

fn resize_zeroed(buffer: &mut Vec<f16>, length: usize) {
    buffer.resize(length, f16::ZERO);
    buffer.fill(f16::ZERO);
}

fn contiguous_history_rows(
    spectrogram: &Spectrogram,
    limit: usize,
    width: usize,
    duration: Duration,
) -> usize {
    spectrogram
        .newest_to_oldest()
        .take(limit)
        .take_while(|analysis| {
            analysis.duration == duration
                && analysis.data.len() == width
                && analysis.masking.len() == width
        })
        .count()
}

fn write_history_row(
    buffer: &mut [f16],
    target_row: usize,
    width: usize,
    analysis: Option<&crate::analyzer::BetterAnalysis>,
) {
    let Some(analysis) =
        analysis.filter(|analysis| analysis.data.len() == width && analysis.masking.len() == width)
    else {
        return;
    };
    let output = &mut buffer[target_row * width * 4..(target_row + 1) * width * 4];
    // Treat each texel as one item so channel writes need no bounds checks.
    let (output, remainder) = output.as_chunks_mut::<4>();
    debug_assert!(remainder.is_empty());
    for (output, (&(pan, amplitude), &(_, masking))) in output
        .iter_mut()
        .zip(analysis.data.iter().zip(&analysis.masking))
    {
        *output = [
            f16::from_f32(pan),
            finite_half(amplitude),
            finite_half(masking),
            f16::ONE,
        ];
    }
}

fn finite_half(value: f32) -> f16 {
    f16::from_f32(if value.is_finite() { value } else { -65504.0 })
}

fn prepare_bar_row_cached(
    spectrogram: &Spectrogram,
    averaging: Duration,
    include_masking: bool,
    cache: &mut BarCache,
    output: &mut Vec<f16>,
) -> bool {
    let state = spectrogram.render_state();
    let front = spectrogram.newest();
    let width = front.data.len();
    let duration = front.duration;

    let Some(old_state) = cache.state else {
        rebuild_bar_cache(spectrogram, averaging, include_masking, cache, output);
        return true;
    };
    let same_key = old_state.epoch == state.epoch
        && cache.averaging == averaging
        && cache.include_masking == include_masking
        && cache.width == width
        && cache.duration == Some(duration);
    let revision_overflow = state.revision < old_state.revision && state.epoch == old_state.epoch;
    if !same_key || revision_overflow {
        rebuild_bar_cache(spectrogram, averaging, include_masking, cache, output);
        return true;
    }

    let changed_u64 = state.revision.wrapping_sub(old_state.revision);
    if changed_u64 == 0 {
        return false;
    }
    let Ok(changed) = usize::try_from(changed_u64) else {
        rebuild_bar_cache(spectrogram, averaging, include_masking, cache, output);
        return true;
    };
    if changed >= state.retained_rows {
        rebuild_bar_cache(spectrogram, averaging, include_masking, cache, output);
        return true;
    }

    let new_rows = changed.min(state.valid_rows);
    let new_rows_are_contiguous = spectrogram
        .newest_to_oldest()
        .take(new_rows)
        .all(|row| row.duration == duration && row.data.len() == width);
    if !new_rows_are_contiguous || changed.saturating_add(cache.window_rows) > state.retained_rows {
        rebuild_bar_cache(spectrogram, averaging, include_masking, cache, output);
        return true;
    }

    let contiguous_rows = changed
        .saturating_add(cache.contiguous_rows)
        .min(state.valid_rows.max(1))
        .min(state.retained_rows);
    let window_rows = bar_window_rows(duration, averaging, contiguous_rows);
    let retained_old_rows = cache.window_rows.min(window_rows.saturating_sub(changed));

    for old_age in retained_old_rows..cache.window_rows {
        let row = spectrogram
            .at_age(changed + old_age)
            .expect("incremental bar cache only uses retained rows");
        accumulate_bar_row::<false>(cache, row);
    }
    for age in 0..changed.min(window_rows) {
        let row = spectrogram
            .at_age(age)
            .expect("new spectrogram rows are retained");
        accumulate_bar_row::<true>(cache, row);
    }
    for age in changed.saturating_add(cache.window_rows)..window_rows {
        let row = spectrogram
            .at_age(age)
            .expect("expanded bar window rows are retained");
        accumulate_bar_row::<true>(cache, row);
    }

    cache.state = Some(state);
    cache.contiguous_rows = contiguous_rows;
    cache.window_rows = window_rows;
    write_cached_bar_row(cache, state.valid_rows > 0, output);
    true
}

fn rebuild_bar_cache(
    spectrogram: &Spectrogram,
    averaging: Duration,
    include_masking: bool,
    cache: &mut BarCache,
    output: &mut Vec<f16>,
) {
    let state = spectrogram.render_state();
    let front = spectrogram.newest();
    let width = front.data.len();
    let duration = front.duration;
    let contiguous_rows = spectrogram
        .newest_to_oldest()
        .take(state.valid_rows.max(1).min(state.retained_rows))
        .take_while(|row| row.duration == duration && row.data.len() == width)
        .count();
    let window_rows = bar_window_rows(duration, averaging, contiguous_rows);

    cache.include_masking = include_masking;
    cache.data_sums.resize(width, BarDataSum::default());
    cache.data_sums.fill(BarDataSum::default());
    if include_masking {
        cache.masking_sums.resize(width, FiniteSum::default());
        cache.masking_sums.fill(FiniteSum::default());
    } else {
        cache.masking_sums.clear();
    }
    for age in 0..window_rows {
        let row = spectrogram
            .at_age(age)
            .expect("bar window is bounded by retained history");
        accumulate_bar_row::<true>(cache, row);
    }

    cache.state = Some(state);
    cache.averaging = averaging;
    cache.width = width;
    cache.duration = Some(duration);
    cache.contiguous_rows = contiguous_rows;
    cache.window_rows = window_rows;
    write_cached_bar_row(cache, state.valid_rows > 0, output);
}

fn accumulate_bar_row<const ADD: bool>(
    cache: &mut BarCache,
    row: &crate::analyzer::BetterAnalysis,
) {
    // The direction is selected when this is monomorphized, keeping its branch and the old sign
    // multiplication out of the per-bin loops.
    for (sum, &(pan, amplitude)) in cache.data_sums.iter_mut().zip(&row.data) {
        if ADD {
            sum.pan = sum.pan.algebraic_add(f64::from(pan));
        } else {
            sum.pan = sum.pan.algebraic_sub(f64::from(pan));
        }
        accumulate_finite::<ADD>(&mut sum.amplitude, amplitude);
    }
    if cache.include_masking {
        for (sum, &(_, masking)) in cache.masking_sums.iter_mut().zip(&row.masking) {
            accumulate_finite::<ADD>(sum, masking);
        }
    }
}

fn accumulate_finite<const ADD: bool>(sum: &mut FiniteSum, value: f32) {
    if value.is_finite() {
        if ADD {
            sum.value = sum.value.algebraic_add(f64::from(value));
        } else {
            sum.value = sum.value.algebraic_sub(f64::from(value));
        }
    } else if ADD {
        sum.non_finite += 1;
    } else {
        debug_assert!(sum.non_finite > 0);
        sum.non_finite -= 1;
    }
}

fn write_cached_bar_row(cache: &BarCache, valid: bool, output: &mut Vec<f16>) {
    output.resize(cache.width * 4, f16::ZERO);
    let count = cache.window_rows.max(1) as f64;
    let valid = if valid { f16::ONE } else { f16::ZERO };
    // Fixed-size texels and zipped cache entries leave only data-dependent finite-value checks in
    // these loops.
    let (output, remainder) = output.as_chunks_mut::<4>();
    debug_assert!(remainder.is_empty());

    if cache.include_masking {
        debug_assert_eq!(cache.masking_sums.len(), cache.width);
        for ((output, data), &masking) in output
            .iter_mut()
            .zip(&cache.data_sums)
            .zip(&cache.masking_sums)
        {
            *output = [
                f16::from_f64(data.pan.algebraic_div(count)),
                cached_average_half(data.amplitude, count),
                cached_average_half(masking, count),
                valid,
            ];
        }
    } else {
        for (output, data) in output.iter_mut().zip(&cache.data_sums) {
            *output = [
                f16::from_f64(data.pan.algebraic_div(count)),
                cached_average_half(data.amplitude, count),
                f16::ZERO,
                valid,
            ];
        }
    }
}

fn cached_average_half(sum: FiniteSum, count: f64) -> f16 {
    if sum.non_finite == 0 {
        f16::from_f64(sum.value.algebraic_div(count))
    } else {
        f16::MIN
    }
}

fn bar_window_rows(duration: Duration, averaging: Duration, contiguous_rows: usize) -> usize {
    if averaging.is_zero() {
        return 1;
    }
    let eligible = if duration.is_zero() {
        contiguous_rows
    } else {
        let mut max_index = (averaging.as_secs_f64() / duration.as_secs_f64()).floor() as usize;
        max_index = max_index.min(contiguous_rows.saturating_sub(1));
        while max_index > 0 && duration.mul_f32(max_index as f32) > averaging {
            max_index -= 1;
        }
        while max_index + 1 < contiguous_rows
            && duration.mul_f32((max_index + 1) as f32) <= averaging
        {
            max_index += 1;
        }
        max_index + 1
    };
    if eligible > 2 { eligible } else { 1 }
}

#[cfg(test)]
fn prepare_bar_row(
    spectrogram: &Spectrogram,
    averaging: Duration,
    include_masking: bool,
    average_data: &mut Vec<(f32, f32)>,
    average_masking: &mut Vec<f32>,
    output: &mut Vec<f16>,
) {
    let front = spectrogram.newest();
    let valid = spectrogram.render_state().valid_rows > 0;
    let bins = front.data.len();
    average_data.resize(bins, (0.0, 0.0));
    average_data.fill((0.0, 0.0));
    if include_masking {
        average_masking.resize(bins, 0.0);
        average_masking.fill(0.0);
    }

    let max_index = averaging_count(spectrogram, averaging).saturating_sub(1);
    let count = (max_index + 1) as f32;
    for age in 0..=max_index {
        let row = spectrogram.at_age(age).unwrap();
        for (sum, value) in average_data.iter_mut().zip(&row.data) {
            sum.0 += value.0;
            sum.1 += value.1;
        }
        if include_masking {
            for (sum, value) in average_masking.iter_mut().zip(&row.masking) {
                *sum += value.1;
            }
        }
    }
    resize_zeroed(output, bins * 4);
    for index in 0..bins {
        let base = index * 4;
        output[base] = f16::from_f32(average_data[index].0 / count);
        output[base + 1] = finite_half(average_data[index].1 / count);
        output[base + 2] = if include_masking {
            finite_half(average_masking[index] / count)
        } else {
            f16::ZERO
        };
        output[base + 3] = if valid { f16::ONE } else { f16::ZERO };
    }
}

#[cfg(test)]
fn averaging_count(spectrogram: &Spectrogram, averaging: Duration) -> usize {
    let front = spectrogram.newest();
    if averaging.is_zero() {
        return 1;
    }
    let valid_rows = spectrogram.render_state().valid_rows.max(1);
    let max_index = spectrogram
        .newest_to_oldest()
        .take(valid_rows)
        .enumerate()
        .take_while(|(index, row)| {
            row.duration.mul_f32(*index as f32) <= averaging
                && row.data.len() == front.data.len()
                && row.duration == front.duration
        })
        .map(|(index, _)| index)
        .last()
        .unwrap_or(0);
    if max_index > 1 { max_index + 1 } else { 1 }
}

fn color32_to_array(color: super::Color32) -> [f32; 4] {
    let [r, g, b, a] = color.to_array();
    [
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
    ]
}

unsafe fn create_resources(gl: &glow::Context) -> Result<GlResources, String> {
    unsafe {
        let program = gl.create_program()?;
        let version = if cfg!(target_arch = "wasm32") {
            "#version 300 es\n"
        } else {
            "#version 150 core\n"
        };
        let vertex = compile_shader(gl, glow::VERTEX_SHADER, version, VERTEX_SHADER)?;
        let fragment = compile_shader(gl, glow::FRAGMENT_SHADER, version, FRAGMENT_SHADER)?;
        gl.attach_shader(program, vertex);
        gl.attach_shader(program, fragment);
        gl.link_program(program);
        gl.detach_shader(program, vertex);
        gl.detach_shader(program, fragment);
        gl.delete_shader(vertex);
        gl.delete_shader(fragment);
        if !gl.get_program_link_status(program) {
            let error = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(format!("spectrogram shader link failed: {error}"));
        }
        let vertex_array = gl.create_vertex_array()?;
        let history = gl.create_texture()?;
        let bar = gl.create_texture()?;
        let lut = gl.create_texture()?;
        let masking_ranges = gl.create_texture()?;
        allocate_half_texture(gl, history, 1, 1, glow::RGBA16F, glow::RGBA);
        allocate_half_texture(gl, bar, 1, 1, glow::RGBA16F, glow::RGBA);
        allocate_half_texture(gl, masking_ranges, 1, 1, glow::R16F, glow::RED);
        configure_texture(gl, lut);
        gl.bind_texture(glow::TEXTURE_2D, Some(lut));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            1,
            1,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&[0, 0, 0, 255])),
        );

        Ok(GlResources {
            program,
            uniforms: Uniforms {
                history: gl.get_uniform_location(program, "u_history"),
                bar: gl.get_uniform_location(program, "u_bar"),
                lut: gl.get_uniform_location(program, "u_lut"),
                masking_ranges: gl.get_uniform_location(program, "u_masking_ranges"),
                history_size: gl.get_uniform_location(program, "u_history_size"),
                history_head: gl.get_uniform_location(program, "u_history_head"),
                valid_rows: gl.get_uniform_location(program, "u_valid_rows"),
                horizontal: gl.get_uniform_location(program, "u_horizontal"),
                nearest: gl.get_uniform_location(program, "u_nearest"),
                show_spectrogram: gl.get_uniform_location(program, "u_show_spectrogram"),
                show_bar: gl.get_uniform_location(program, "u_show_bar"),
                use_smr: gl.get_uniform_location(program, "u_use_smr"),
                show_masking: gl.get_uniform_location(program, "u_show_masking"),
                bar_proportion: gl.get_uniform_location(program, "u_bar_proportion"),
                db_range: gl.get_uniform_location(program, "u_db_range"),
                smr_strength: gl.get_uniform_location(program, "u_smr_strength"),
                masking_color: gl.get_uniform_location(program, "u_masking_color"),
            },
            vertex_array,
            history,
            bar,
            lut,
            masking_ranges,
            history_size: (0, 0),
            bar_width: 0,
            lut_size: (0, 0),
            masking_width: 0,
        })
    }
}

unsafe fn compile_shader(
    gl: &glow::Context,
    kind: u32,
    version: &str,
    source: &str,
) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(kind)?;
        gl.shader_source(shader, &format!("{version}{source}"));
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let error = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(format!("spectrogram shader compilation failed: {error}"))
        }
    }
}

unsafe fn upload_pending(
    gl: &glow::Context,
    resources: &mut GlResources,
    renderer: &mut ShaderRenderer,
) {
    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        if renderer.pending_history.rebuild
            || resources.history_size
                != (
                    renderer.pending_history.width,
                    renderer.pending_history.height,
                )
        {
            resources.history_size = (
                renderer.pending_history.width.max(1),
                renderer.pending_history.height.max(1),
            );
            allocate_half_texture(
                gl,
                resources.history,
                resources.history_size.0,
                resources.history_size.1,
                glow::RGBA16F,
                glow::RGBA,
            );
        }
        gl.bind_texture(glow::TEXTURE_2D, Some(resources.history));
        for range in renderer.pending_history.ranges.into_iter().flatten() {
            if range.rows == 0 {
                continue;
            }
            let row_values = renderer.pending_history.width * 4;
            let values = &renderer.history_staging
                [range.source_row * row_values..(range.source_row + range.rows) * row_values];
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                range.target_row as i32,
                renderer.pending_history.width as i32,
                range.rows as i32,
                glow::RGBA,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(cast_slice(values))),
            );
        }
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];
        renderer.pending_history.rebuild = false;

        if renderer.bar_dirty && !renderer.bar_staging.is_empty() {
            let width = renderer.bar_staging.len() / 4;
            if resources.bar_width != width {
                allocate_half_texture(gl, resources.bar, width, 1, glow::RGBA16F, glow::RGBA);
                resources.bar_width = width;
            }
            gl.bind_texture(glow::TEXTURE_2D, Some(resources.bar));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                1,
                glow::RGBA,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(cast_slice(&renderer.bar_staging))),
            );
            renderer.bar_dirty = false;
        }
    }
    upload_lookup_and_masking(gl, resources, renderer);
}

fn upload_lookup_and_masking(
    gl: &glow::Context,
    resources: &mut GlResources,
    renderer: &mut ShaderRenderer,
) {
    unsafe {
        if renderer.lut_dirty && !renderer.lut_staging.is_empty() {
            // Color tables are intensity-major within each pan row.
            let dimensions = renderer.lut_size;
            configure_texture(gl, resources.lut);
            gl.bind_texture(glow::TEXTURE_2D, Some(resources.lut));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&renderer.lut_staging)),
            );
            resources.lut_size = dimensions;
            renderer.lut_dirty = false;
        }
        if renderer.masking_dirty && !renderer.masking_staging.is_empty() {
            let width = renderer.masking_staging.len();
            configure_texture(gl, resources.masking_ranges);
            gl.bind_texture(glow::TEXTURE_2D, Some(resources.masking_ranges));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R16F as i32,
                width as i32,
                1,
                0,
                glow::RED,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(cast_slice(&renderer.masking_staging))),
            );
            resources.masking_width = width;
            renderer.masking_dirty = false;
        }
    }
}

unsafe fn allocate_half_texture(
    gl: &glow::Context,
    texture: glow::Texture,
    width: usize,
    height: usize,
    internal: u32,
    format: u32,
) {
    unsafe {
        configure_texture(gl, texture);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            internal as i32,
            width as i32,
            height as i32,
            0,
            format,
            glow::HALF_FLOAT,
            glow::PixelUnpackData::Slice(None),
        );
    }
}

unsafe fn configure_texture(gl: &glow::Context, texture: glow::Texture) {
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
    }
}

unsafe fn bind_texture(gl: &glow::Context, texture: glow::Texture, unit: u32) {
    unsafe {
        gl.active_texture(glow::TEXTURE0 + unit);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_row(spectrogram: &mut Spectrogram, duration: Duration, value: f32) {
        spectrogram.update_with(|row| {
            row.duration = duration;
            row.data.resize(2, (0.0, 0.0));
            row.masking.resize(2, (0.0, 0.0));
            row.data.fill((value * 0.1, value));
            row.masking.fill((0.0, value - 1.0));
            row.masking_mean = value - 1.0;
        });
    }

    fn test_color_table() -> ColorTable {
        let mut table = ColorTable::new(4, 8);
        table.build(195.0, 328.0, 0.13, 0.818, 0.09);
        table
    }

    #[test]
    fn retained_height_is_rounded_and_capped() {
        assert_eq!(
            requested_history_height(
                Duration::from_millis(105),
                Duration::from_millis(10),
                8,
                None
            ),
            8
        );
        assert_eq!(
            requested_history_height(
                Duration::from_millis(14),
                Duration::from_millis(10),
                8,
                None
            ),
            1
        );
        assert_eq!(
            requested_history_height(
                Duration::from_secs(1),
                Duration::from_millis(10),
                500,
                Some(32)
            ),
            32
        );
    }

    #[test]
    fn wraparound_ranges_need_at_most_two_uploads() {
        let head = 1usize;
        let height = 8usize;
        let changed = 3usize;
        let new_head = (head + height - changed) % height;
        let first = changed.min(height - new_head);
        let second = changed - first;
        assert_eq!(new_head, 6);
        assert_eq!((first, second), (2, 1));
    }

    #[test]
    fn circular_head_and_upload_ranges_follow_committed_rows() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        let mut renderer = ShaderRenderer::default();
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];
        renderer.pending_history.rebuild = false;

        push_row(&mut spectrogram, Duration::from_millis(10), 2.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert_eq!(renderer.pending_history.head, 3);
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];

        push_row(&mut spectrogram, Duration::from_millis(10), 3.0);
        push_row(&mut spectrogram, Duration::from_millis(10), 4.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert_eq!(renderer.pending_history.head, 1);
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];

        push_row(&mut spectrogram, Duration::from_millis(10), 5.0);
        push_row(&mut spectrogram, Duration::from_millis(10), 6.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert_eq!(
            renderer.pending_history.ranges,
            [
                Some(UploadRange {
                    source_row: 0,
                    target_row: 3,
                    rows: 1,
                }),
                Some(UploadRange {
                    source_row: 1,
                    target_row: 0,
                    rows: 1,
                }),
            ]
        );
    }

    #[test]
    fn history_rebuilds_only_for_documented_state_changes() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        push_row(&mut spectrogram, Duration::from_millis(10), 2.0);
        let mut renderer = ShaderRenderer::default();

        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.ranges.iter().flatten().count(), 1);
        renderer.commit_history_upload();
        renderer.pending_history.rebuild = false;
        renderer.pending_history.ranges = [None, None];

        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(!renderer.pending_history.rebuild);
        assert!(renderer.pending_history.ranges.iter().all(Option::is_none));

        push_row(&mut spectrogram, Duration::from_millis(10), 3.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(!renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.ranges.iter().flatten().count(), 1);
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];

        spectrogram.clear();
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.valid_rows, 0);
    }

    #[test]
    fn history_rebuild_stages_and_uploads_only_valid_rows() {
        let mut spectrogram = Spectrogram::new(8, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        push_row(&mut spectrogram, Duration::from_millis(10), 2.0);
        let mut renderer = ShaderRenderer::default();

        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 8, false);
        assert_eq!(renderer.history_staging.len(), 2 * 2 * 4);
        assert_eq!(renderer.pending_history.ranges[0].unwrap().rows, 2);

        spectrogram.clear();
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 8, false);
        assert!(renderer.history_staging.is_empty());
        assert!(renderer.pending_history.ranges.iter().all(Option::is_none));
    }

    #[test]
    fn hardware_clamp_never_uploads_more_than_valid_rows() {
        let mut pending = PendingHistory {
            width: 2,
            height: 16,
            head: 7,
            valid_rows: 3,
            rebuild: false,
            ranges: [
                Some(UploadRange {
                    source_row: 0,
                    target_row: 7,
                    rows: 3,
                }),
                None,
            ],
            target_state: None,
            target_duration: None,
        };
        clamp_pending_history(&mut pending, 8);
        assert_eq!(pending.height, 8);
        assert_eq!(pending.head, 0);
        assert!(pending.rebuild);
        assert_eq!(pending.ranges[0].unwrap().rows, 3);

        pending.height = 16;
        pending.valid_rows = 0;
        clamp_pending_history(&mut pending, 8);
        assert!(pending.ranges.iter().all(Option::is_none));
    }

    #[test]
    fn duration_changes_rebuild_and_limit_history_to_contiguous_rows() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        push_row(&mut spectrogram, Duration::from_millis(10), 2.0);
        let mut renderer = ShaderRenderer::default();

        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert_eq!(renderer.pending_history.valid_rows, 2);
        renderer.commit_history_upload();
        renderer.pending_history.rebuild = false;
        renderer.pending_history.ranges = [None, None];

        push_row(&mut spectrogram, Duration::from_millis(20), 3.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.valid_rows, 1);
        renderer.commit_history_upload();
        renderer.pending_history.rebuild = false;
        renderer.pending_history.ranges = [None, None];

        push_row(&mut spectrogram, Duration::from_millis(20), 4.0);
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(!renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.valid_rows, 2);
    }

    #[test]
    fn revision_overflow_forces_a_full_rebuild() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        let mut renderer = ShaderRenderer::default();
        renderer.staged_state = Some(SpectrogramRenderState {
            epoch: 0,
            revision: u64::MAX,
            valid_rows: 4,
            retained_rows: 4,
        });
        renderer.staged_width = 2;
        renderer.staged_height = 4;
        renderer.prepare_history(&spectrogram, spectrogram.render_state(), 2, 4, false);
        assert!(renderer.pending_history.rebuild);
        assert_eq!(renderer.pending_history.ranges[0].unwrap().rows, 1);
    }

    #[test]
    fn bar_averaging_matches_the_previous_window_rule() {
        let mut spectrogram = Spectrogram::new(4, 2);
        for value in [1.0, 2.0, 3.0] {
            push_row(&mut spectrogram, Duration::from_millis(10), value);
        }
        let mut data = Vec::new();
        let mut masking = Vec::new();
        let mut output = Vec::new();
        prepare_bar_row(
            &spectrogram,
            Duration::from_millis(25),
            true,
            &mut data,
            &mut masking,
            &mut output,
        );
        for bin in 0..2 {
            assert!((output[bin * 4].to_f32() - 0.2).abs() < 0.001);
            assert!((output[bin * 4 + 1].to_f32() - 2.0).abs() < 0.001);
            assert!((output[bin * 4 + 2].to_f32() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn bar_averaging_excludes_rows_invalidated_by_a_clear() {
        let mut spectrogram = Spectrogram::new(4, 2);
        for value in 1..=4 {
            push_row(&mut spectrogram, Duration::from_millis(10), value as f32);
        }
        spectrogram.clear();
        push_row(&mut spectrogram, Duration::from_millis(10), 10.0);

        assert_eq!(averaging_count(&spectrogram, Duration::from_millis(30)), 1);

        let mut data = Vec::new();
        let mut masking = Vec::new();
        let mut output = Vec::new();
        prepare_bar_row(
            &spectrogram,
            Duration::from_millis(30),
            true,
            &mut data,
            &mut masking,
            &mut output,
        );
        assert_eq!(output[1].to_f32(), 10.0);
        assert_eq!(output[2].to_f32(), 9.0);
    }

    #[test]
    fn cached_bar_matches_full_aggregation_and_skips_unchanged_revisions() {
        let mut spectrogram = Spectrogram::new(8, 2);
        let mut cache = BarCache::default();
        let mut cached = Vec::new();

        for value in 1..=12 {
            push_row(&mut spectrogram, Duration::from_millis(10), value as f32);
            assert!(prepare_bar_row_cached(
                &spectrogram,
                Duration::from_millis(35),
                true,
                &mut cache,
                &mut cached,
            ));

            let mut data = Vec::new();
            let mut masking = Vec::new();
            let mut reference = Vec::new();
            prepare_bar_row(
                &spectrogram,
                Duration::from_millis(35),
                true,
                &mut data,
                &mut masking,
                &mut reference,
            );
            assert_eq!(cached.len(), reference.len());
            for (cached, reference) in cached.iter().zip(reference) {
                assert!((cached.to_f32() - reference.to_f32()).abs() <= 0.002);
            }

            let unchanged = cached.clone();
            assert!(!prepare_bar_row_cached(
                &spectrogram,
                Duration::from_millis(35),
                true,
                &mut cache,
                &mut cached,
            ));
            assert_eq!(cached, unchanged);
        }

        assert!(prepare_bar_row_cached(
            &spectrogram,
            Duration::from_millis(20),
            false,
            &mut cache,
            &mut cached,
        ));
        assert!(cache.masking_sums.is_empty());

        spectrogram.clear();
        push_row(&mut spectrogram, Duration::from_millis(10), 20.0);
        assert!(prepare_bar_row_cached(
            &spectrogram,
            Duration::from_millis(20),
            false,
            &mut cache,
            &mut cached,
        ));
        assert_eq!(cached[1].to_f32(), 20.0);
    }

    #[test]
    fn cached_bar_recovers_after_non_finite_rows_leave_the_window() {
        let mut spectrogram = Spectrogram::new(8, 2);
        let mut cache = BarCache::default();
        let mut cached = Vec::new();

        for value in [1.0, f32::NEG_INFINITY, 3.0, 4.0, 5.0] {
            spectrogram.update_with(|row| {
                row.duration = Duration::from_millis(10);
                row.data.resize(2, (0.0, value));
                row.masking.resize(2, (0.0, value));
                row.data.fill((0.0, value));
                row.masking.fill((0.0, value));
                row.masking_mean = value;
            });
            assert!(prepare_bar_row_cached(
                &spectrogram,
                Duration::from_millis(25),
                true,
                &mut cache,
                &mut cached,
            ));

            let mut data = Vec::new();
            let mut masking = Vec::new();
            let mut reference = Vec::new();
            prepare_bar_row(
                &spectrogram,
                Duration::from_millis(25),
                true,
                &mut data,
                &mut masking,
                &mut reference,
            );
            assert_eq!(cached, reference);
        }
        assert!(cached[1].to_f32().is_finite());
        assert!(cached[2].to_f32().is_finite());
    }

    #[test]
    fn cached_bar_window_count_matches_the_previous_float_boundary_rule() {
        for duration in [
            Duration::from_nanos(1),
            Duration::from_nanos(333_333),
            Duration::from_millis(10),
        ] {
            for rows in [1usize, 2, 3, 17, 257] {
                for averaging in [
                    Duration::ZERO,
                    duration
                        .saturating_mul(2)
                        .saturating_sub(Duration::from_nanos(1)),
                    duration.saturating_mul(2),
                    duration.saturating_mul(16),
                ] {
                    let max_index = (0..rows)
                        .take_while(|index| duration.mul_f32(*index as f32) <= averaging)
                        .last()
                        .unwrap_or(0);
                    let expected = if averaging.is_zero() || max_index <= 1 {
                        1
                    } else {
                        max_index + 1
                    };
                    assert_eq!(bar_window_rows(duration, averaging, rows), expected);
                }
            }
        }
    }

    #[test]
    fn unchanged_bar_does_not_dirty_a_completed_upload() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        let frequencies = [(20.0, 30.0, 40.0), (40.0, 50.0, 60.0)];
        let color_table = test_color_table();
        let settings = RenderSettings {
            bargraph_height: 1.0,
            ..RenderSettings::default()
        };
        let mut renderer = ShaderRenderer::default();

        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            true,
            -80.0,
            0.0,
        );
        assert!(renderer.bar_dirty);
        renderer.bar_dirty = false;
        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            true,
            -80.0,
            0.0,
        );
        assert!(!renderer.bar_dirty);
    }

    #[test]
    fn hidden_and_disabled_paths_do_not_stage_uploads() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        let frequencies = [(20.0, 30.0, 40.0), (40.0, 50.0, 60.0)];
        let color_table = test_color_table();
        let mut settings = RenderSettings::default();
        settings.bargraph_height = 1.0;
        let mut renderer = ShaderRenderer::default();
        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            false,
            -80.0,
            0.0,
        );
        assert!(renderer.pending_history.ranges.iter().all(Option::is_none));
        assert!(renderer.bar_dirty);
        assert!(!renderer.frame.show_masking);
        assert!(!renderer.frame.use_smr);
        assert!(renderer.bar_cache.masking_sums.is_empty());
        assert!(renderer.masking_staging.is_empty());

        let mut settings = settings;
        settings.bargraph_height = 0.0;
        renderer.bar_dirty = false;
        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            false,
            -80.0,
            0.0,
        );
        assert!(!renderer.bar_dirty);
        assert!(renderer.pending_history.ranges.iter().any(Option::is_some));
    }

    #[test]
    fn steady_state_preparation_reuses_renderer_capacities() {
        let mut spectrogram = Spectrogram::new(4, 2);
        push_row(&mut spectrogram, Duration::from_millis(10), 1.0);
        let frequencies = [(20.0, 30.0, 40.0), (40.0, 50.0, 60.0)];
        let color_table = test_color_table();
        let settings = RenderSettings::default();
        let mut renderer = ShaderRenderer::default();
        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            false,
            -80.0,
            0.0,
        );
        renderer.commit_history_upload();
        renderer.pending_history.ranges = [None, None];
        let capacities = (
            renderer.history_staging.capacity(),
            renderer.bar_staging.capacity(),
            renderer.bar_cache.data_sums.capacity(),
            renderer.bar_cache.masking_sums.capacity(),
            renderer.lut_staging.capacity(),
        );

        push_row(&mut spectrogram, Duration::from_millis(10), 2.0);
        renderer.prepare(
            &spectrogram,
            &frequencies,
            &color_table,
            1,
            &settings,
            false,
            -80.0,
            0.0,
        );
        assert_eq!(
            capacities,
            (
                renderer.history_staging.capacity(),
                renderer.bar_staging.capacity(),
                renderer.bar_cache.data_sums.capacity(),
                renderer.bar_cache.masking_sums.capacity(),
                renderer.lut_staging.capacity(),
            )
        );
    }

    #[test]
    fn lut_addressing_matches_cpu_rounding() {
        let max = (255.0_f32, 511.0_f32);
        for (pan, intensity) in [(-1.0, 0.0), (-0.2, 0.33), (0.0, 0.5), (1.0, 1.0)] {
            let cpu = (
                ((pan + 1.0) * 0.5 * max.0).round() as usize,
                (intensity * max.1).round() as usize,
            );
            let shader = (
                ((pan * 0.5 + 0.5) * max.0).round() as usize,
                (intensity * max.1).round() as usize,
            );
            assert_eq!(cpu, shader);
        }
    }

    #[test]
    fn smr_blend_matches_old_formula() {
        let volume: f32 = 0.8;
        let smr: f32 = 0.25;
        let strength: f32 = 0.2;
        let old = volume.min(smr * strength + volume * (1.0 - strength));
        let shader = volume.min(volume + (smr - volume) * strength);
        assert!((old - shader).abs() < f32::EPSILON);
    }

    #[test]
    fn amplitude_and_orientation_match_the_cpu_coordinates() {
        let amplitude = -30.0_f32;
        let mapped = ((amplitude - -80.0) / (0.0 - -80.0)).clamp(0.0, 1.0);
        assert_eq!(mapped, 0.625);

        let screen = (0.7_f32, 0.25_f32);
        let bar = 0.2_f32;
        let horizontal = (1.0 - screen.1, (screen.0 - bar) / (1.0 - bar));
        let vertical = (screen.0, (screen.1 - bar) / (1.0 - bar));
        assert_eq!(horizontal, (0.75, 0.625));
        assert!((vertical.0 - 0.7).abs() < f32::EPSILON);
        assert!((vertical.1 - 0.0625).abs() < f32::EPSILON);
    }

    #[test]
    fn bilinear_interpolation_is_applied_after_color_mapping() {
        let colors = [0.1_f32, 0.4, 0.7, 1.0];
        let weight = (0.25_f32, 0.75_f32);
        let top = colors[0] + (colors[1] - colors[0]) * weight.0;
        let bottom = colors[2] + (colors[3] - colors[2]) * weight.0;
        let result = top + (bottom - top) * weight.1;
        assert!((result - 0.625).abs() < 0.000_001);
    }
}
