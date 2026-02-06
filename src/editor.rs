#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]

use color::{ColorSpaceTag, DynamicColor, Flags, Rgba8, Srgb};
use nih_plug::editor::Editor;
use nih_plug_egui::{
    EguiSettings, GlConfig, GraphicsConfig, create_egui_editor,
    egui::{
        self, Align2, Color32, ColorImage, Context, FontId, ImageData, Mesh, Pos2, Rect, Shape,
        TextureId, TextureOptions, ThemePreference, Vec2,
        epaint::{ImageDelta, Vertex, WHITE_UV},
    },
};
use parking_lot::{FairMutex, Mutex, RwLock};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    AnalysisChain, AnalysisChainConfig, AnalysisMetrics, AudioState, MAX_FREQUENCY_BINS,
    PluginParams, SPECTROGRAM_SLICES,
    analyzer::{BetterSpectrogram, FrequencyScale, map_value},
};

fn calculate_volume_min_max(
    settings: &RenderSettings,
    spectrogram: &BetterSpectrogram,
) -> (f32, f32) {
    if !settings.automatic_gain || !spectrogram.data[0].masking_mean.is_finite() {
        return (settings.min_db, settings.max_db);
    }

    let mut elapsed = Duration::ZERO;

    let mut masking_sum = 0.0;
    let mut rows: usize = 0;

    for row in &spectrogram.data {
        elapsed += row.duration;
        if elapsed > settings.agc_duration {
            break;
        }

        if row.masking_mean.is_finite() {
            masking_sum += row.masking_mean as f64;
            rows += 1;
        }
    }

    let masking = (masking_sum / rows as f64) as f32;

    (
        (masking - settings.agc_below_masking).clamp(settings.agc_minimum, settings.agc_maximum),
        (masking + settings.agc_above_masking).clamp(settings.agc_minimum, settings.agc_maximum),
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn draw_bargraph(
    mesh: &mut Mesh,
    spectrogram: &BetterSpectrogram,
    mut buffer: (Vec<(f32, f32)>, Vec<f32>),
    bounds: Rect,
    color_table: &ColorTable,
    masking_color: Option<Color32>,
    (max_db, min_db): (f32, f32),
    averaging: Duration,
) {
    let front = &spectrogram.data.front().unwrap();

    if !averaging.is_zero() {
        assert_eq!(front.data.len(), buffer.0.len());

        let target_len = front.data.len();
        let target_duration = front.duration;

        let max_index = spectrogram
            .data
            .iter()
            .enumerate()
            .take_while(|(i, row)| {
                row.duration.mul_f32(*i as f32) <= averaging
                    && row.data.len() == target_len
                    && row.duration == target_duration
            })
            .map(|(i, _)| i)
            .last()
            .unwrap_or(1);

        if max_index > 1 {
            let count = max_index as f32 + 1.0;

            for i in 0..=max_index {
                for (spectrogram_chunk, output_chunk) in unsafe {
                    spectrogram
                        .data
                        .get(i)
                        .unwrap_unchecked()
                        .data
                        .as_chunks_unchecked::<64>()
                }
                .iter()
                .zip(unsafe { buffer.0.as_chunks_unchecked_mut::<64>() })
                {
                    for (data, output) in spectrogram_chunk.iter().zip(output_chunk) {
                        *output = (
                            output.0.algebraic_add(data.0),
                            output.1.algebraic_add(data.1),
                        )
                    }
                }
            }

            for chunk in unsafe { buffer.0.as_chunks_unchecked_mut::<64>() } {
                for item in chunk {
                    *item = (item.0.algebraic_div(count), item.1.algebraic_div(count));
                }
            }

            draw_bargraph_from(mesh, &buffer.0, bounds, color_table, (max_db, min_db));

            if let Some(masking_color) = masking_color {
                assert_eq!(front.masking.len(), buffer.1.len());

                for i in 0..=max_index {
                    for (spectrogram_chunk, output_chunk) in unsafe {
                        spectrogram
                            .data
                            .get(i)
                            .unwrap_unchecked()
                            .masking
                            .as_chunks_unchecked::<64>()
                    }
                    .iter()
                    .zip(unsafe { buffer.1.as_chunks_unchecked_mut::<64>() })
                    {
                        for (data, output) in spectrogram_chunk.iter().zip(output_chunk) {
                            *output = output.algebraic_add(data.1);
                        }
                    }
                }

                for chunk in unsafe { buffer.1.as_chunks_unchecked_mut::<64>() } {
                    for item in chunk {
                        *item = item.algebraic_div(count);
                    }
                }

                draw_secondary_bargraph(mesh, &buffer.1, bounds, masking_color, (max_db, min_db));
            }

            return;
        }
    }

    draw_bargraph_from(mesh, &front.data, bounds, color_table, (max_db, min_db));
    if let Some(masking_color) = masking_color {
        draw_secondary_bargraph_from_pairs(
            mesh,
            &front.masking,
            bounds,
            masking_color,
            (max_db, min_db),
        );
    }
}

fn draw_bargraph_from(
    mesh: &mut Mesh,
    analysis: &[(f32, f32)],
    bounds: Rect,
    color_table: &ColorTable,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let mut vertices = mesh.vertices.len() as u32;

    let band_width = width / analysis.len() as f32;

    let mut buffer = [(0, Rect::ZERO); 64];

    for (ci, chunk) in unsafe { analysis.as_chunks_unchecked::<64>() }
        .iter()
        .enumerate()
    {
        let offset = ci * 64;

        for (ii, ((pan, volume), output)) in
            chunk.iter().copied().zip(buffer.iter_mut()).enumerate()
        {
            let intensity = map_value(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

            let start_x = bounds
                .min
                .x
                .algebraic_add(((offset + ii) as f32).algebraic_mul(band_width));

            let rect = Rect {
                min: Pos2 {
                    x: start_x,
                    y: bounds.max.y.algebraic_sub(intensity.algebraic_mul(height)),
                },
                max: Pos2 {
                    x: start_x.algebraic_add(band_width),
                    y: bounds.max.y,
                },
            };

            let index = color_table.calculate_index(pan, intensity);

            *output = (index, rect);
        }

        for _ in 0..64 {
            mesh.indices.extend_from_slice(&[
                vertices,
                vertices + 1,
                vertices + 2,
                vertices + 2,
                vertices + 1,
                vertices + 3,
            ]);
            vertices += 4;
        }

        for (index, rect) in buffer {
            let color = unsafe { color_table.get_unchecked(index) };

            mesh.vertices.extend_from_slice(&[
                Vertex {
                    pos: rect.left_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.left_bottom(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_bottom(),
                    uv: WHITE_UV,
                    color,
                },
            ]);
        }
    }
}

fn draw_secondary_bargraph(
    mesh: &mut Mesh,
    analysis: &[f32],
    bounds: Rect,
    color: Color32,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let mut vertices = mesh.vertices.len() as u32;

    let band_width = width / analysis.len() as f32;

    let mut buffer = [Rect::ZERO; 64];

    for (ci, chunk) in unsafe { analysis.as_chunks_unchecked::<64>() }
        .iter()
        .enumerate()
    {
        let offset = ci * 64;

        for (ii, (volume, output)) in chunk.iter().copied().zip(buffer.iter_mut()).enumerate() {
            let intensity = map_value(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

            let start_x = bounds
                .min
                .x
                .algebraic_add(((offset + ii) as f32).algebraic_mul(band_width));

            let rect = Rect {
                min: Pos2 {
                    x: start_x,
                    y: bounds.max.y.algebraic_sub(intensity.algebraic_mul(height)),
                },
                max: Pos2 {
                    x: start_x.algebraic_add(band_width),
                    y: bounds.max.y,
                },
            };

            *output = rect;
        }

        for _ in 0..64 {
            mesh.indices.extend_from_slice(&[
                vertices,
                vertices + 1,
                vertices + 2,
                vertices + 2,
                vertices + 1,
                vertices + 3,
            ]);
            vertices += 4;
        }

        for rect in buffer {
            mesh.vertices.extend_from_slice(&[
                Vertex {
                    pos: rect.left_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.left_bottom(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_bottom(),
                    uv: WHITE_UV,
                    color,
                },
            ]);
        }
    }
}

fn draw_secondary_bargraph_from_pairs(
    mesh: &mut Mesh,
    analysis: &[(f32, f32)],
    bounds: Rect,
    color: Color32,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let mut vertices = mesh.vertices.len() as u32;

    let band_width = width / analysis.len() as f32;

    let mut buffer = [Rect::ZERO; 64];

    for (ci, chunk) in unsafe { analysis.as_chunks_unchecked::<64>() }
        .iter()
        .enumerate()
    {
        let offset = ci * 64;

        for (ii, ((_, volume), output)) in chunk.iter().copied().zip(buffer.iter_mut()).enumerate()
        {
            let intensity = map_value(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

            let start_x = bounds
                .min
                .x
                .algebraic_add(((offset + ii) as f32).algebraic_mul(band_width));

            let rect = Rect {
                min: Pos2 {
                    x: start_x,
                    y: bounds.max.y.algebraic_sub(intensity.algebraic_mul(height)),
                },
                max: Pos2 {
                    x: start_x.algebraic_add(band_width),
                    y: bounds.max.y,
                },
            };

            *output = rect;
        }

        for _ in 0..64 {
            mesh.indices.extend_from_slice(&[
                vertices,
                vertices + 1,
                vertices + 2,
                vertices + 2,
                vertices + 1,
                vertices + 3,
            ]);
            vertices += 4;
        }

        for rect in buffer {
            mesh.vertices.extend_from_slice(&[
                Vertex {
                    pos: rect.left_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_top(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.left_bottom(),
                    uv: WHITE_UV,
                    color,
                },
                Vertex {
                    pos: rect.right_bottom(),
                    uv: WHITE_UV,
                    color,
                },
            ]);
        }
    }
}

fn draw_spectrogram_image(
    image: &mut ColorImage,
    spectrogram: &BetterSpectrogram,
    frequencies: &[(f32, f32, f32)],
    color_table: &ColorTable,
    (max_db, min_db): (f32, f32),
    clamp_using_smr: bool,
) {
    let target_duration = spectrogram.data.front().unwrap().duration;

    let image_width = image.width();
    let image_height = image.height();

    assert!(image_width.is_multiple_of(64));

    let mut buffer = [0; 64];

    if clamp_using_smr {
        let masking_ranges: Vec<f32> =
            unsafe { frequencies.as_chunks_unchecked::<64>() }
                .iter()
                .flat_map(|chunk| {
                    chunk.iter().copied().map(|(_, center, _)| {
                        27.0_f32.algebraic_sub(6.025_f32.algebraic_sub(
                            0.275_f32.algebraic_mul(FrequencyScale::Bark.scale(center)),
                        ))
                    })
                })
                .collect();

        assert!(masking_ranges.len().is_multiple_of(64));

        for (y, analysis) in spectrogram.data.iter().enumerate() {
            if analysis.data.len() != image_width
                || y == image_height
                || analysis.duration != target_duration
            {
                break;
            }

            for (analysis_chunk, (masking_chunk, (masking_range_chunk, pixel_chunk))) in
                unsafe { analysis.data.as_chunks_unchecked::<64>() }
                    .iter()
                    .zip(
                        unsafe { analysis.masking.as_chunks_unchecked::<64>() }
                            .iter()
                            .zip(
                                unsafe { masking_ranges.as_chunks_unchecked::<64>() }
                                    .iter()
                                    .zip(unsafe {
                                        image
                                            .pixels
                                            .get_unchecked_mut(
                                                (image_width * y)..(image_width * (y + 1)),
                                            )
                                            .as_chunks_unchecked_mut::<64>()
                                    }),
                            ),
                    )
            {
                for ((pan, volume), (((_, masking), range), output)) in
                    analysis_chunk.iter().copied().zip(
                        masking_chunk
                            .iter()
                            .copied()
                            .zip(masking_range_chunk.iter().copied())
                            .zip(buffer.iter_mut()),
                    )
                {
                    let intensity = map_value(volume, min_db, max_db, 0.0, 1.0).min(map_value(
                        volume.algebraic_sub(masking),
                        0.0,
                        range,
                        0.0,
                        1.0,
                    ));

                    *output = color_table.calculate_index(pan, intensity);
                }

                for (index, pixel) in buffer.into_iter().zip(pixel_chunk) {
                    *pixel = unsafe { color_table.get_unchecked(index) };
                }
            }
        }
    } else {
        for (y, analysis) in spectrogram.data.iter().enumerate() {
            if analysis.data.len() != image_width
                || y == image_height
                || analysis.duration != target_duration
            {
                break;
            }

            for (analysis_chunk, pixel_chunk) in
                unsafe { analysis.data.as_chunks_unchecked::<64>() }
                    .iter()
                    .zip(unsafe {
                        image
                            .pixels
                            .get_unchecked_mut((image_width * y)..(image_width * (y + 1)))
                            .as_chunks_unchecked_mut::<64>()
                    })
            {
                for ((pan, volume), output) in analysis_chunk.iter().copied().zip(buffer.iter_mut())
                {
                    let intensity = map_value(volume, min_db, max_db, 0.0, 1.0);
                    *output = color_table.calculate_index(pan, intensity);
                }

                for (index, pixel) in buffer.into_iter().zip(pixel_chunk) {
                    *pixel = unsafe { color_table.get_unchecked(index) };
                }
            }
        }
    }
}

struct UnderCursor {
    pub frequency: (f32, f32, f32),
    pub amplitude: f32,
    pub pan: Option<f32>,
    pub time: Option<Duration>,
}

fn get_under_cursor(
    cursor: Pos2,
    spectrogram: &BetterSpectrogram,
    frequencies: &[(f32, f32, f32)],
    bargraph_bounds: Rect,
    spectrogram_bounds: Rect,
    (bargraph_max_db, bargraph_min_db): (f32, f32),
    spectrogram_height: usize,
) -> Option<UnderCursor> {
    let frequency_count = frequencies.len() as f32;

    if bargraph_bounds.contains(cursor) {
        let frequency = frequencies[map_value(
            cursor.x,
            bargraph_bounds.min.x,
            bargraph_bounds.max.x,
            0.0,
            frequency_count,
        )
        .floor() as usize];
        let amplitude = map_value(
            bargraph_bounds.max.y - cursor.y,
            bargraph_bounds.min.y,
            bargraph_bounds.max.y,
            bargraph_min_db,
            bargraph_max_db,
        );

        Some(UnderCursor {
            frequency,
            amplitude,
            pan: None,
            time: None,
        })
    } else if spectrogram_bounds.contains(cursor) {
        let x = map_value(
            cursor.x,
            spectrogram_bounds.min.x,
            spectrogram_bounds.max.x,
            0.0,
            frequency_count,
        )
        .floor() as usize;
        let y = map_value(
            cursor.y,
            spectrogram_bounds.min.y,
            spectrogram_bounds.max.y,
            0.0,
            spectrogram_height as f32,
        )
        .floor() as usize;

        let frequency = frequencies[x];
        let duration = spectrogram.data[0].duration;

        let item = if spectrogram.data.len() > y
            && spectrogram.data[y].data.len() == frequencies.len()
            && spectrogram.data[y].duration == duration
        {
            Some(spectrogram.data[y].data[x])
        } else {
            None
        };

        if let Some((pan, amplitude)) = item {
            Some(UnderCursor {
                frequency,
                amplitude,
                pan: Some(pan),
                time: Some(duration.mul_f32(y as f32)),
            })
        } else {
            None
        }
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct RenderSettings {
    left_hue: f32,
    right_hue: f32,
    minimum_lightness: f32,
    maximum_lightness: f32,
    maximum_chroma: f32,
    lookup_size: usize,
    automatic_gain: bool,
    agc_duration: Duration,
    agc_above_masking: f32,
    agc_below_masking: f32,
    agc_minimum: f32,
    agc_maximum: f32,
    min_db: f32,
    max_db: f32,
    clamp_using_smr: bool,
    bargraph_height: f32,
    spectrogram_duration: Duration,
    bargraph_averaging: Duration,
    spectrogram_nearest_neighbor: bool,
    show_performance: bool,
    show_format: bool,
    show_hover: bool,
    show_masking: bool,
    masking_color: Color32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            left_hue: 195.0,
            right_hue: 328.0,
            minimum_lightness: 0.13,
            maximum_lightness: 0.82,
            maximum_chroma: 0.09,
            automatic_gain: true,
            lookup_size: 4,
            agc_duration: Duration::from_secs_f32(1.0),
            agc_above_masking: 40.0,
            agc_below_masking: 0.0,
            agc_minimum: 3.0 - AnalysisChainConfig::default().listening_volume,
            agc_maximum: 100.0 - AnalysisChainConfig::default().listening_volume,
            min_db: 20.0 - AnalysisChainConfig::default().listening_volume,
            max_db: 80.0 - AnalysisChainConfig::default().listening_volume,
            clamp_using_smr: false,
            bargraph_height: 0.33,
            spectrogram_duration: Duration::from_secs_f32(0.67),
            bargraph_averaging: Duration::from_secs_f32(BASELINE_TARGET_FRAME_SECS),
            spectrogram_nearest_neighbor: false,
            show_performance: true,
            show_format: false,
            show_hover: true,
            show_masking: true,
            masking_color: Color32::from_rgb(33, 0, 4),
        }
    }
}

struct ColorTable {
    table: Vec<(u8, u8, u8)>,
    size: (usize, usize),
    max: (f32, f32),
}

const COLOR_TABLE_BASE_CHROMA_SIZE: usize = 64;
const COLOR_TABLE_BASE_LIGHTNESS_SIZE: usize = 128;

impl ColorTable {
    fn new(chroma_size: usize, lightness_size: usize) -> Self {
        Self {
            table: vec![(0, 0, 0); chroma_size * lightness_size],
            size: (chroma_size, lightness_size),
            max: ((chroma_size - 1) as f32, (lightness_size - 1) as f32),
        }
    }
    fn build(
        &mut self,
        left_hue: f32,
        right_hue: f32,
        min_lightness: f32,
        max_lightness: f32,
        max_chroma: f32,
    ) {
        let left_color = DynamicColor {
            cs: ColorSpaceTag::Oklch,
            flags: Flags::default(),
            components: [max_lightness, max_chroma, left_hue, 1.0],
        };
        let right_color = DynamicColor {
            cs: ColorSpaceTag::Oklch,
            flags: Flags::default(),
            components: [max_lightness, max_chroma, right_hue, 1.0],
        };

        for split_index in 0..self.size.0 {
            let split = map_value(split_index as f32, 0.0, self.max.0, -1.0, 1.0);
            for intensity_index in 0..self.size.1 {
                if intensity_index == 0 {
                    self.table[split_index * self.size.1] = (0, 0, 0);
                    continue;
                }

                let intensity = map_value(intensity_index as f32, 0.0, self.max.1, 0.0, 1.0);

                let mut color = if split >= 0.0 {
                    let mut color = right_color;
                    color.components[1] = map_value(split, 0.0, 1.0, 0.0, color.components[1]);
                    color
                } else {
                    let mut color = left_color;
                    color.components[1] = map_value(-split, 0.0, 1.0, 0.0, color.components[1]);
                    color
                };

                color.components[0] =
                    map_value(intensity, 0.0, 1.0, min_lightness, color.components[0]);

                let converted: Rgba8 = color.to_alpha_color::<Srgb>().to_rgba8();

                self.table[(split_index * self.size.1) + intensity_index] =
                    (converted.r, converted.g, converted.b);
            }
        }
    }
    /*fn lookup(&self, split: f32, intensity: f32) -> Color32 {
        let location = (
            map_value(split, -1.0, 1.0, 0.0, self.max.0)
                .round()
                .clamp(0.0, self.max.0) as usize,
            map_value(intensity, 0.0, 1.0, 0.0, self.max.1)
                .round()
                .clamp(0.0, self.max.1) as usize,
        );

        let color = unsafe {
            *self
                .table
                .get_unchecked((location.0 * self.size.1) + location.1)
        };

        //let color = self.table[(location.0 * self.size.1) + location.1];

        Color32::from_rgb(color.0, color.1, color.2)
    }*/
    fn calculate_index(&self, split: f32, intensity: f32) -> usize {
        let location = (
            map_value(split, -1.0, 1.0, 0.0, self.max.0)
                .round()
                .clamp(0.0, self.max.0) as usize,
            map_value(intensity, 0.0, 1.0, 0.0, self.max.1)
                .round()
                .clamp(0.0, self.max.1) as usize,
        );

        (location.0 * self.size.1) + location.1
    }
    unsafe fn get_unchecked(&self, index: usize) -> Color32 {
        let color = unsafe { *self.table.get_unchecked(index) };
        Color32::from_rgb(color.0, color.1, color.2)
    }
}

struct SharedState {
    settings: RwLock<RenderSettings>,
    last_size: Mutex<(usize, usize)>,
    frame_timing: Mutex<(Instant, Duration, Duration)>,
    color_table: RwLock<ColorTable>,
    cached_analysis_settings: Mutex<AnalysisChainConfig>,
    spectrogram_texture: Arc<RwLock<Option<TextureId>>>,
}

const BASELINE_TARGET_FPS: f32 = 60.0;
const BASELINE_TARGET_FRAME_SECS: f32 = 1.0 / BASELINE_TARGET_FPS;

pub fn create(
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    audio_state: Arc<RwLock<Option<AudioState>>>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let shared_state = {
        let settings = RenderSettings::default();
        let mut color_table = ColorTable::new(
            COLOR_TABLE_BASE_CHROMA_SIZE * settings.lookup_size,
            COLOR_TABLE_BASE_LIGHTNESS_SIZE * settings.lookup_size,
        );
        color_table.build(
            settings.left_hue,
            settings.right_hue,
            settings.minimum_lightness,
            settings.maximum_lightness,
            settings.maximum_chroma,
        );

        SharedState {
            settings: RwLock::new(settings),
            last_size: Mutex::new((MAX_FREQUENCY_BINS, SPECTROGRAM_SLICES)),
            frame_timing: Mutex::new((Instant::now(), Duration::ZERO, Duration::ZERO)),
            color_table: RwLock::new(color_table),
            cached_analysis_settings: Mutex::new(AnalysisChainConfig::default()),
            spectrogram_texture: Arc::new(RwLock::new(None)),
        }
    };

    let spectrogram_texture = shared_state.spectrogram_texture.clone();

    create_egui_editor(
        egui_state.clone(),
        (),
        EguiSettings {
            graphics_config: GraphicsConfig {
                dithering: true,
                shader_version: None,
            },
            enable_vsync_on_x11: false,
            gl_config: GlConfig {
                vsync: false,
                ..Default::default()
            },
        },
        move |egui_ctx, _, _| {
            build(egui_ctx, &spectrogram_texture);
        },
        move |egui_ctx, _setter, _, _| {
            render(
                egui_ctx,
                &analysis_chain,
                &analysis_output,
                &analysis_frequencies,
                &audio_state,
                &shared_state,
            );
        },
    )
}

fn build(egui_ctx: &Context, spectrogram_texture: &RwLock<Option<TextureId>>) {
    let manager = egui_ctx.tex_manager();
    let mut manager = manager.write();
    let mut spectrogram_texture = spectrogram_texture.write();

    if let Some(ref id) = *spectrogram_texture {
        if manager.meta(*id).is_some() {
            manager.free(*id);
        }
    }

    *spectrogram_texture = Some(manager.alloc(
        "spectrogram".to_string(),
        ImageData::Color(Arc::new(ColorImage::filled([1, 1], Color32::TRANSPARENT))),
        TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            wrap_mode: egui::TextureWrapMode::ClampToEdge,
            mipmap_mode: None,
        },
    ));

    egui_ctx.tessellation_options_mut(|options| {
        options.coarse_tessellation_culling = false;
    });

    egui_ctx.set_theme(ThemePreference::Dark);
}

fn render(
    egui_ctx: &Context,
    analysis_chain: &Mutex<Option<AnalysisChain>>,
    analysis_output: &FairMutex<(BetterSpectrogram, AnalysisMetrics)>,
    analysis_frequencies: &RwLock<Vec<(f32, f32, f32)>>,
    audio_state: &RwLock<Option<AudioState>>,
    shared_state: &SharedState,
    //backend_specific_ui: F,
) /*where
F: FnOnce(&Ui),*/
{
    egui_ctx.request_repaint();

    let start = Instant::now();

    egui::CentralPanel::default().show(egui_ctx, |ui| {
        let last_size = *shared_state.last_size.lock();

        let mut bargraph_mesh = Mesh::default();
        bargraph_mesh.reserve_triangles(MAX_FREQUENCY_BINS * 2 * 2);
        bargraph_mesh.reserve_vertices(MAX_FREQUENCY_BINS * 4 * 2);

        let mut spectrogram_image_pixels = vec![Color32::TRANSPARENT; last_size.0 * last_size.1];
        let mut bargraph_buffer_1 = vec![(0.0, 0.0); last_size.0];
        let mut bargraph_buffer_2 = vec![0.0; last_size.0];

        let painter = ui.painter();
        let max_x = painter.clip_rect().max.x;
        let max_y = painter.clip_rect().max.y;

        let settings = *shared_state.settings.read();

        let frequencies = analysis_frequencies.read();

        let bargraph_bounds = Rect {
            min: Pos2 { x: 0.0, y: 0.0 },
            max: Pos2 {
                x: max_x,
                y: max_y * settings.bargraph_height,
            },
        };
        let spectrogram_bounds = Rect {
            min: Pos2 {
                x: 0.0,
                y: max_y * settings.bargraph_height,
            },
            max: Pos2 { x: max_x, y: max_y },
        };

        let lock = analysis_output.lock();
        let (ref spectrogram, metrics) = *lock;

        let spectrogram = spectrogram.clone();

        drop(lock);

        let front = spectrogram.data.front().unwrap();

        let spectrogram_width = front.data.len();
        let spectrogram_height = (settings.spectrogram_duration.as_secs_f32()
            / front.duration.as_secs_f32())
        .round() as usize;

        *shared_state.last_size.lock() = (spectrogram_width, spectrogram_height);

        if (last_size.0 * last_size.1) > (spectrogram_width * spectrogram_height) {
            spectrogram_image_pixels.truncate(spectrogram_width * spectrogram_height);
        } else if (last_size.0 * last_size.1) < (spectrogram_width * spectrogram_height) {
            spectrogram_image_pixels
                .resize(spectrogram_width * spectrogram_height, Color32::TRANSPARENT);
        }

        if last_size.0 > spectrogram_width {
            bargraph_buffer_1.truncate(spectrogram_width);
            bargraph_buffer_2.truncate(spectrogram_width);
        } else if last_size.0 < spectrogram_width {
            bargraph_buffer_1.resize(spectrogram_width, (0.0, 0.0));
            bargraph_buffer_2.resize(spectrogram_width, 0.0);

            let additional = spectrogram_width - last_size.0;
            bargraph_mesh.reserve_triangles(additional * 2 * 2);
            bargraph_mesh.reserve_vertices(additional * 4 * 2);
        }

        let mut spectrogram_image = ColorImage {
            size: [spectrogram_width, spectrogram_height],
            source_size: Vec2 {
                x: spectrogram_width as f32,
                y: spectrogram_height as f32,
            },
            pixels: spectrogram_image_pixels,
        };

        let buffering_duration = metrics.finished.elapsed();
        let processing_duration = metrics.processing;
        let chunk_duration = front.duration;

        let (min_db, max_db) = calculate_volume_min_max(&settings, &spectrogram);

        {
            let color_table = &shared_state.color_table.read();

            if settings.bargraph_height != 0.0 {
                if settings.show_masking {
                    draw_bargraph(
                        &mut bargraph_mesh,
                        &spectrogram,
                        (bargraph_buffer_1, bargraph_buffer_2),
                        bargraph_bounds,
                        color_table,
                        Some(settings.masking_color),
                        (max_db, min_db),
                        settings.bargraph_averaging,
                    );
                } else {
                    draw_bargraph(
                        &mut bargraph_mesh,
                        &spectrogram,
                        (bargraph_buffer_1, bargraph_buffer_2),
                        bargraph_bounds,
                        color_table,
                        None,
                        (max_db, min_db),
                        settings.bargraph_averaging,
                    );
                }
            }

            if settings.bargraph_height != 1.0 {
                draw_spectrogram_image(
                    &mut spectrogram_image,
                    &spectrogram,
                    &frequencies,
                    color_table,
                    (max_db, min_db),
                    settings.clamp_using_smr,
                );
            }
        }

        let under_pointer = if settings.show_hover {
            if let Some(pointer) = egui_ctx.pointer_latest_pos() {
                get_under_cursor(
                    pointer,
                    &spectrogram,
                    &frequencies,
                    bargraph_bounds,
                    spectrogram_bounds,
                    (max_db, min_db),
                    spectrogram_height,
                )
            } else {
                None
            }
        } else {
            None
        };

        drop(spectrogram);

        {
            let spectrogram_texture = shared_state.spectrogram_texture.read().unwrap();

            egui_ctx.tex_manager().write().set(
                spectrogram_texture,
                ImageDelta {
                    image: ImageData::Color(Arc::new(spectrogram_image)),
                    options: if settings.spectrogram_nearest_neighbor {
                        TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Nearest,
                            wrap_mode: egui::TextureWrapMode::ClampToEdge,
                            mipmap_mode: None,
                        }
                    } else {
                        TextureOptions {
                            magnification: egui::TextureFilter::Linear,
                            minification: egui::TextureFilter::Linear,
                            wrap_mode: egui::TextureWrapMode::ClampToEdge,
                            mipmap_mode: None,
                        }
                    },
                    pos: None,
                },
            );

            painter.extend([
                Shape::Mesh(Arc::new(bargraph_mesh)),
                Shape::Mesh(Arc::new(Mesh {
                    indices: vec![0, 1, 2, 2, 1, 3],
                    vertices: vec![
                        Vertex {
                            pos: spectrogram_bounds.left_top(),
                            uv: Pos2 { x: 0.0, y: 0.0 },
                            color: Color32::WHITE,
                        },
                        Vertex {
                            pos: spectrogram_bounds.right_top(),
                            uv: Pos2 { x: 1.0, y: 0.0 },
                            color: Color32::WHITE,
                        },
                        Vertex {
                            pos: spectrogram_bounds.left_bottom(),
                            uv: Pos2 { x: 0.0, y: 1.0 },
                            color: Color32::WHITE,
                        },
                        Vertex {
                            pos: spectrogram_bounds.right_bottom(),
                            uv: Pos2 { x: 1.0, y: 1.0 },
                            color: Color32::WHITE,
                        },
                    ],
                    texture_id: spectrogram_texture,
                })),
            ]);
        }

        if let Some(under) = under_pointer {
            let analysis_settings = shared_state.cached_analysis_settings.lock();

            let amplitude_text = if analysis_settings.normalize_amplitude {
                format!(
                    "{:.0} phon",
                    under.amplitude + analysis_settings.listening_volume
                )
            } else {
                format!("{:+.0}dBFS", under.amplitude)
            };

            drop(analysis_settings);

            let text = if let (Some(pan), Some(elapsed)) = (under.pan, under.time) {
                format!(
                    "{:.0}hz, -{:.3}s\n{}, {:+.2} pan",
                    under.frequency.1,
                    elapsed.as_secs_f32(),
                    amplitude_text,
                    pan
                )
            } else {
                let resolution = (1.0 / (under.frequency.2 - under.frequency.0)) * 1000.0;
                let averaging = settings.bargraph_averaging.as_secs_f32() * 1000.0;

                if averaging > 0.0 {
                    format!(
                        "{:.0}hz, {}\n{:.0}ms res, {:.0} ms avg",
                        under.frequency.1, amplitude_text, resolution, averaging
                    )
                } else {
                    format!(
                        "{:.0}hz, {}\n{:.0}ms",
                        under.frequency.1, amplitude_text, resolution
                    )
                }
            };

            painter.text(
                Pos2 { x: 16.0, y: 16.0 },
                Align2::LEFT_TOP,
                text,
                FontId {
                    size: 12.0,
                    family: egui::FontFamily::Monospace,
                },
                Color32::from_rgb(224, 224, 224),
            );
        }

        if settings.show_format {
            if let Some(audio_state) = audio_state.read().as_ref() {
                let min_buffer_size_s =
                    audio_state.buffer_size_range.0 as f32 / audio_state.sample_rate;
                let max_buffer_size_s =
                    audio_state.buffer_size_range.1 as f32 / audio_state.sample_rate;

                let should_warn = audio_state.sample_rate < (frequencies.last().unwrap().2 * 2.0)
                    || max_buffer_size_s > 0.010
                    || !audio_state.realtime;

                painter.text(
                    Pos2 {
                        x: max_x / 2.0,
                        y: 16.0,
                    },
                    Align2::CENTER_CENTER,
                    format!(
                        "{} in -> {} out, {:.1}kHz, {:.0}ms to {:.0}ms buffer, mode {}",
                        audio_state.input_channels,
                        audio_state.output_channels,
                        audio_state.sample_rate / 1000.0,
                        min_buffer_size_s * 1000.0,
                        max_buffer_size_s * 1000.0,
                        audio_state.process_mode_title
                    ),
                    FontId {
                        size: 12.0,
                        family: egui::FontFamily::Monospace,
                    },
                    if should_warn {
                        Color32::YELLOW
                    } else {
                        Color32::from_rgb(224, 224, 224)
                    },
                );
            }
        }

        drop(frequencies);

        if settings.show_performance {
            let frame_timing = shared_state.frame_timing.lock();

            let frame_elapsed = frame_timing.1;
            let rasterize_elapsed = frame_timing.2;

            drop(frame_timing);

            let buffering_secs = buffering_duration.as_secs_f32();
            let rasterize_secs = rasterize_elapsed.as_secs_f32();
            let chunk_secs = chunk_duration.as_secs_f32();
            let frame_secs = frame_elapsed.as_secs_f32();

            /*let rasterize_processing_duration = rasterize_secs / (frame_secs / chunk_secs);
            let adjusted_processing_duration =
                processing_duration.as_secs_f32() + rasterize_processing_duration;*/
            let buffer_processing_duration = buffering_secs / (frame_secs / chunk_secs);
            let adjusted_processing_duration =
                processing_duration.as_secs_f32() + buffer_processing_duration;
            let rasterize_proportion = rasterize_secs / frame_secs;
            let processing_proportion = adjusted_processing_duration / chunk_secs;
            let buffering_proportion = buffering_secs / frame_secs;

            if buffering_duration < Duration::from_millis(500) {
                painter.text(
                    Pos2 {
                        x: max_x - 32.0,
                        y: 48.0,
                    },
                    Align2::RIGHT_TOP,
                    format!(
                        "{:.0}% ({:.1}ms) processing",
                        processing_proportion * 100.0,
                        adjusted_processing_duration * 1000.0,
                    ),
                    FontId {
                        size: 12.0,
                        family: egui::FontFamily::Monospace,
                    },
                    if processing_proportion >= 0.95 {
                        Color32::RED
                    } else if processing_proportion >= 0.8 {
                        Color32::YELLOW
                    } else {
                        Color32::from_rgb(224, 224, 224)
                    },
                );
                painter.text(
                    Pos2 {
                        x: max_x - 32.0,
                        y: 64.0,
                    },
                    Align2::RIGHT_TOP,
                    format!("{:.1}ms buffering", buffering_secs * 1000.0),
                    FontId {
                        size: 12.0,
                        family: egui::FontFamily::Monospace,
                    },
                    if buffering_proportion >= 1.0 {
                        Color32::RED
                    } else if buffering_proportion >= 0.5 {
                        Color32::YELLOW
                    } else {
                        Color32::from_rgb(224, 224, 224)
                    },
                );
            }

            painter.text(
                Pos2 {
                    x: max_x - 32.0,
                    y: 16.0,
                },
                Align2::RIGHT_TOP,
                format!("{:2}ms frame", frame_elapsed.as_millis()),
                FontId {
                    size: 12.0,
                    family: egui::FontFamily::Monospace,
                },
                if frame_elapsed >= Duration::from_secs_f32(1.0 / (BASELINE_TARGET_FPS * 0.5)) {
                    Color32::RED
                } else if frame_elapsed
                    >= Duration::from_secs_f32(1.0 / (BASELINE_TARGET_FPS * 0.9))
                {
                    Color32::YELLOW
                } else {
                    Color32::from_rgb(224, 224, 224)
                },
            );
            painter.text(
                Pos2 {
                    x: max_x - 32.0,
                    y: 32.0,
                },
                Align2::RIGHT_TOP,
                format!(
                    "{:.0}% ({:.1}ms) rasterize",
                    rasterize_proportion * 100.0,
                    rasterize_elapsed.as_secs_f32() * 1000.0,
                ),
                FontId {
                    size: 12.0,
                    family: egui::FontFamily::Monospace,
                },
                if rasterize_elapsed >= Duration::from_secs_f32(BASELINE_TARGET_FRAME_SECS * 0.6) {
                    Color32::RED
                } else if rasterize_elapsed
                    >= Duration::from_secs_f32(BASELINE_TARGET_FRAME_SECS * 0.3)
                    || rasterize_proportion >= 0.6
                {
                    Color32::YELLOW
                } else {
                    Color32::from_rgb(224, 224, 224)
                },
            );
        }
    });
    egui::Window::new("Settings")
        .id(egui::Id::new("settings"))
        .default_pos(Pos2 {
            x: 16.0,
            y: 56.0
        })
        .default_open(false)
        .show(egui_ctx, |ui| {
            let mut render_settings = shared_state.settings.write();
            let mut analysis_settings = shared_state.cached_analysis_settings.lock();

            ui.collapsing("Render Options", |ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.left_hue, 0.0..=360.0)
                            .suffix("°")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Left channel hue"),
                    )
                    .changed()
                {
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.right_hue, 0.0..=360.0)
                            .suffix("°")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Right channel hue"),
                    )
                    .changed()
                {
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.minimum_lightness, 0.0..=0.3)
                            .text("Minimum OkLCH lightness value"),
                    )
                    .changed()
                {
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.maximum_lightness, 0.5..=1.0)
                            .text("Maximum OkLCH lightness value"),
                    )
                    .changed()
                {
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.maximum_chroma, 0.0..=0.2)
                            .text("Maximum OkLCH chroma value"),
                    )
                    .changed()
                {
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if ui
                    .add(
                        egui::Slider::new(&mut render_settings.lookup_size, 1..=8)
                            .logarithmic(true)
                            .clamping(egui::SliderClamping::Always)
                            .text("Color lookup table size multiplier"),
                    )
                    .changed()
                {
                    let mut color_table = shared_state.color_table.write();

                    *color_table = ColorTable::new(
                        COLOR_TABLE_BASE_CHROMA_SIZE * render_settings.lookup_size,
                        COLOR_TABLE_BASE_LIGHTNESS_SIZE * render_settings.lookup_size,
                    );
                    color_table.build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                };

                if analysis_settings.masking {
                    ui.checkbox(&mut render_settings.automatic_gain, "Automatic amplitude ranging");
                }

                if render_settings.automatic_gain && analysis_settings.masking {
                    let mut agc_duration = render_settings.agc_duration.as_secs_f64();
                    if ui
                        .add(
                            egui::Slider::new(&mut agc_duration, 0.1..=8.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("s")
                                .text("Amplitude ranging duration"),
                        )
                        .changed()
                    {
                        if agc_duration > 0.0 {
                            render_settings.agc_duration =
                                Duration::from_secs_f64(agc_duration);
                        }
                    };

                    ui.add(
                        egui::Slider::new(&mut render_settings.agc_above_masking, 0.0..=100.0)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("dB")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Range above masking mean"),
                    );

                    ui.add(
                        egui::Slider::new(&mut render_settings.agc_below_masking, -50.0..=50.0)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("dB")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Range below masking mean"),
                    );
                } else {
                    if analysis_settings.normalize_amplitude {
                        let mut min_phon =
                            (render_settings.min_db + analysis_settings.listening_volume).clamp(0.0, 100.0);
                        let mut max_phon =
                            (render_settings.max_db + analysis_settings.listening_volume).clamp(0.0, 100.0);

                        if ui.add(
                            egui::Slider::new(&mut max_phon, 0.0..=100.0)
                                .suffix(" phon")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Maximum amplitude"),
                        ).changed() {
                            render_settings.max_db =
                                max_phon - analysis_settings.listening_volume;
                        }

                        if ui.add(
                            egui::Slider::new(&mut min_phon, 0.0..=100.0)
                                .suffix(" phon")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Minimum amplitude"),
                        ).changed() {
                            render_settings.min_db =
                                min_phon - analysis_settings.listening_volume;
                        }
                    } else {
                        ui.add(
                            egui::Slider::new(&mut render_settings.max_db, -100.0..=0.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Maximum amplitude"),
                        );

                        ui.add(
                            egui::Slider::new(&mut render_settings.min_db, -100.0..=0.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Minimum amplitude"),
                        );
                    }
                }

                if analysis_settings.masking {
                    ui.checkbox(&mut render_settings.clamp_using_smr, "Use signal-to-mask ratio when calculating spectrogram shading");
                }

                let mut spectrogram_duration = render_settings.spectrogram_duration.as_secs_f64();
                if ui
                    .add(
                        egui::Slider::new(&mut spectrogram_duration, 0.05..=8.0)
                            .logarithmic(true)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("s")
                            .text("Spectrogram duration"),
                    )
                    .changed()
                {
                    render_settings.spectrogram_duration =
                        Duration::from_secs_f64(spectrogram_duration);
                };

                let mut bargraph_averaging = render_settings.bargraph_averaging.as_secs_f64() * 1000.0;
                if ui
                    .add(
                        egui::Slider::new(&mut bargraph_averaging, 0.0..=1000.0)
                            .logarithmic(true)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("ms")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Bargraph averaging"),
                    )
                    .changed()
                {
                    render_settings.bargraph_averaging =
                        Duration::from_secs_f64(bargraph_averaging / 1000.0);
                };

                ui.add(
                    egui::Slider::new(&mut render_settings.bargraph_height, 0.0..=1.0)
                        .text("Bargraph height"),
                );

                ui.checkbox(&mut render_settings.spectrogram_nearest_neighbor, "Use nearest-neighbor scaling for spectrogram").changed();

                ui.checkbox(&mut render_settings.show_hover, "Show hover information");

                ui.checkbox(&mut render_settings.show_performance, "Show performance counters");

                ui.checkbox(&mut render_settings.show_format, "Show audio format information");

                if analysis_settings.masking {
                    ui.checkbox(&mut render_settings.show_masking, "Highlight simultaneous masking thresholds");

                    if render_settings.show_masking {
                        let label = ui.label("Masking threshold color:");
                        ui.color_edit_button_srgba(&mut render_settings.masking_color).labelled_by(label.id);
                    }
                }

                if ui.button("Reset Render Options").clicked() {
                    *render_settings = RenderSettings::default();
                    render_settings.max_db =
                            80.0 - analysis_settings.listening_volume;
                    render_settings.min_db =
                            20.0 - analysis_settings.listening_volume;
                    if analysis_settings.normalize_amplitude {
                        render_settings.agc_minimum =
                            3.0 - analysis_settings.listening_volume;
                        render_settings.agc_maximum =
                            100.0 - analysis_settings.listening_volume;
                    } else {
                        render_settings.agc_minimum =
                            f32::NEG_INFINITY;
                        render_settings.agc_maximum = f32::INFINITY;
                    }
                    shared_state.color_table.write().build(
                        render_settings.left_hue,
                        render_settings.right_hue,
                        render_settings.minimum_lightness,
                        render_settings.maximum_lightness,
                        render_settings.maximum_chroma,
                    );
                }
            });

            ui.collapsing("Analysis Options", |ui| {
                let update = |settings| {
                    let mut lock = analysis_chain.lock();
                    let analysis_chain = lock.as_mut().unwrap();
                    analysis_chain.update_config(settings);
                };

                let update_and_clear = |settings| {
                    let mut lock = analysis_chain.lock();
                    let analysis_chain = lock.as_mut().unwrap();
                    analysis_chain.update_config(settings);

                    analysis_output.lock().0 =
                        BetterSpectrogram::new(SPECTROGRAM_SLICES, MAX_FREQUENCY_BINS);
                };

                let is_outside_hearing_range = analysis_settings.start_frequency < 20.0 || analysis_settings.start_frequency > 20000.0 || analysis_settings.end_frequency > 20000.0 || analysis_settings.end_frequency < 20.0;

                ui.colored_label(
                    Color32::YELLOW,
                    "Editing these options temporarily interrupts audio analysis.",
                );

                if ui
                    .add(
                        egui::Slider::new(&mut analysis_settings.gain, -20.0..=40.0)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("dB")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Analysis gain"),
                    )
                    .on_hover_text("This setting adjusts the amplitude of the incoming signal before it is processed (but does not affect the plugin's output channels; audio is always passed through unmodified).\n\nAll internal audio processing is done using 32-bit floating point, so this can be adjusted freely without concern for clipping.")
                    .changed()
                {
                    update(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                };

                let mut latency_offset = analysis_settings.latency_offset.as_secs_f64() * 1000.0;
                if ui
                    .add(
                        egui::Slider::new(&mut latency_offset, 0.0..=500.0)
                            .clamping(egui::SliderClamping::Never)
                            .suffix("ms")
                            .step_by(1.0)
                            .fixed_decimals(0)
                            .text("Latency offset"),
                    )
                    .on_hover_text("This setting allows you to manually increase the processing latency reported to the plugin's host.\n\nBy default (when this is set to 0ms), the latency incurred by internal buffering is already accounted for.")
                    .changed()
                {
                    analysis_settings.latency_offset = Duration::from_secs_f64(latency_offset.max(0.0) / 1000.0);
                    update(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                };

                if !is_outside_hearing_range {
                    if ui
                        .checkbox(
                            &mut analysis_settings.normalize_amplitude,
                            "Perform amplitude normalization",
                        )
                        .on_hover_text("If this is enabled, amplitude values are normalized using the ISO 226:2023 equal-loudness contours, which map the amplitudes of frequency bins into phons, a psychoacoustic unit of loudness measurement.\nIf this is disabled, amplitude values are not normalized.")
                        .changed()
                    {
                        update(&analysis_settings);
                        if analysis_settings.normalize_amplitude {
                            render_settings.agc_minimum =
                                3.0 - analysis_settings.listening_volume;
                            render_settings.agc_maximum =
                                100.0 - analysis_settings.listening_volume;
                        } else {
                            render_settings.agc_minimum =
                                f32::NEG_INFINITY;
                            render_settings.agc_maximum = f32::INFINITY;
                        }
                        egui_ctx.request_discard("Changed setting");
                        return;
                    }
                }

                if analysis_settings.normalize_amplitude {
                    let old_min_phon =
                        (render_settings.min_db + analysis_settings.listening_volume).clamp(0.0, 100.0);
                    let old_max_phon =
                        (render_settings.max_db + analysis_settings.listening_volume).clamp(0.0, 100.0);

                    if ui
                        .add(
                            egui::Slider::new(&mut analysis_settings.listening_volume, 20.0..=120.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix(" dB SPL")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("0dbFS output volume"),
                        )
                        .on_hover_text("When normalizing amplitude values using an equal-loudness contour, a reference value is necessary to convert dBFS into dB SPL.\nIn order to improve the accuracy of amplitude normalization and receive accurate phon values, this value should be set to the dB SPL value corresponding to 0 dBFS on your system.")
                        .changed()
                    {
                        update_and_clear(&analysis_settings);
                        render_settings.min_db =
                            old_min_phon - analysis_settings.listening_volume;
                        render_settings.max_db =
                            old_max_phon - analysis_settings.listening_volume;
                        render_settings.agc_minimum =
                            3.0 - analysis_settings.listening_volume;
                        render_settings.agc_maximum =
                            100.0 - analysis_settings.listening_volume;
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };
                }

                if !is_outside_hearing_range {
                    if ui
                        .checkbox(
                            &mut analysis_settings.masking,
                            "Perform simultaneous masking",
                        )
                        .on_hover_text("In hearing, tones can mask the presence of other tones in a process called simultaneous masking. Most lossy audio codecs use a model of this process in order to hide compression artifacts.\nIf this is enabled, simultaneous masking thresholds are calculated using a simple tone-masking-tone model.\nIf this is disabled, simultaneous masking thresholds are not calculated.")
                        .changed()
                    {
                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    }

                    if analysis_settings.masking {
                        if ui
                            .checkbox(&mut analysis_settings.approximate_masking, "Approximate spreading function")
                            .on_hover_text("Calculating the simultaneous masking threshold makes use of a spreading function, which can be computationally intensive to compute in real time.\nIf this is enabled, the spreading function is approximated, reducing CPU usage at the expense of outputting less psychoacoustically accurate results.\nIf this is disabled, the spreading function is computed normally, resulting in more psychoacoustically accurate results at the expense of additional CPU usage.")
                            .changed()
                        {
                            update(&analysis_settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }
                    }

                    /*if analysis_settings.masking {
                        if ui
                            .checkbox(
                                &mut analysis_settings.remove_masked_components,
                                "Remove masked components",
                            )
                            .on_hover_text("In hearing, tones can mask the presence of other tones in a process called simultaneous masking. Most lossy audio codecs use a model of this process in order to hide compression artifacts.\nIf this is enabled, tones underneath the simultaneous masking threshold are replaced with an amplitude of zero, creating a (sometimes) more readable but less psychoacoustically accurate output.\nIf this is disabled, tones underneath the simultaneous masking threshold are replaced with the simultaneous masking threshold's value.")
                            .changed()
                        {
                            update(&analysis_settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }
                    }*/
                }

                if ui
                    .checkbox(
                        &mut analysis_settings.internal_buffering,
                        "Use internal buffering",
                    )
                    .on_hover_text("In order to better capture transient signals and phase information, audio is processed in multiple overlapping windows.\nIf this is enabled, the plugin maintains its own buffer of samples, allowing the number of overlapping windows per second to be changed by the user. This adds a small amount of latency, which is reported to the plugin's host so that it can be compensated for.\nIf this is disabled, the number of overlapping windows per second is determined by the buffer size set by the host.")
                    .changed()
                {
                    update(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                }

                if analysis_settings.internal_buffering {
                    if ui
                        .checkbox(
                            &mut analysis_settings.strict_synchronization,
                            "Use strict synchronization",
                        )
                        .on_hover_text("When using internal buffering, synchronization can be done in a relaxed manner, where audio analysis can desynchronize slightly from audio output (if the host supports delay compensation, the analysis ends up running slightly ahead of the output), or in a strict manner, where samples must be analyzed before they can be outputted.\nIf this is enabled, audio synchronization is performed in a strict manner.\nIf this is disabled, audio synchronization is performed in a relaxed manner.")
                        .changed()
                    {
                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    }

                    if ui
                        .add(
                            egui::Slider::new(
                                &mut analysis_settings.update_rate_hz,
                                128.0..=4096.0,
                            )
                            .logarithmic(true)
                            .suffix("hz")
                            .step_by(128.0)
                            .fixed_decimals(0)
                            .text("Update rate"),
                        )
                        .on_hover_text("In order to better capture transient signals and phase information, audio is processed in multiple overlapping windows. This setting allows you to adjust the number of overlapping windows per second, effectively setting the spectrogram's vertical resolution (and the associated amount of CPU usage required).\n\nThe default value for the setting is roughly half the length of the just-noticeable-difference in onset time between two auditory events.\n\n(Note: This setting does not change the trade-off between time resolution and frequency resolution.)")
                        .changed()
                    {
                        update_and_clear(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };
                }

                if ui
                    .add(
                        egui::Slider::new(
                            &mut analysis_settings.resolution,
                            128..=MAX_FREQUENCY_BINS,
                        )
                        .suffix(" bins")
                        .step_by(64.0)
                        .fixed_decimals(0)
                        .text("Resolution"),
                    )
                    .on_hover_text("In order to convert data into frequency domain, the selected frequency range needs to be split into a set number of frequency bins. This setting allows you to adjust the number of bins used, effectively setting the horizontal resolution of the spectrogram and bargraph (and the associated amount of CPU usage required).\n\nThis setting does not increase the width of the transform's filters beyond the time resolution setting.")
                    .changed()
                {
                    update_and_clear(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                };

                let mut frame = egui::Frame::group(ui.style()).inner_margin(4.0).begin(ui);

                {
                    let ui = &mut frame.content_ui;

                    let label = ui.label("Frequency range");

                    if ui
                        .add(
                            egui::DragValue::new(&mut analysis_settings.start_frequency)
                                .suffix("hz")
                                .fixed_decimals(0),
                        )
                        .labelled_by(label.id)
                        .changed()
                    {
                        if analysis_settings.start_frequency < 0.0 {
                            analysis_settings.start_frequency = 0.0;
                        }
                        if analysis_settings.end_frequency < 0.0 {
                            analysis_settings.end_frequency = 0.0;
                        }
                        if analysis_settings.end_frequency > analysis_settings.start_frequency {
                            if analysis_settings.start_frequency < 20.0 || analysis_settings.start_frequency > 20000.0  {
                                analysis_settings.normalize_amplitude = false;
                                analysis_settings.masking = false;
                            }
                            update_and_clear(&analysis_settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }
                    };

                    if ui
                        .add(
                            egui::DragValue::new(&mut analysis_settings.end_frequency)
                                .speed(20.0)
                                .suffix("hz")
                                .fixed_decimals(0),
                        )
                        .labelled_by(label.id)
                        .changed()
                    {
                        if analysis_settings.start_frequency < 0.0 {
                            analysis_settings.start_frequency = 0.0;
                        }
                        if analysis_settings.end_frequency < 0.0 {
                            analysis_settings.end_frequency = 0.0;
                        }
                        if analysis_settings.end_frequency > analysis_settings.start_frequency {
                            if analysis_settings.end_frequency > 20000.0 || analysis_settings.end_frequency < 20.0 {
                                analysis_settings.normalize_amplitude = false;
                                analysis_settings.masking = false;
                            }
                            update_and_clear(&analysis_settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }
                    };
                }
                frame.end(ui).on_hover_text("The frequency range of human hearing in healthy young individuals generally spans from 20 Hz to 20 kHz. However, this range can vary significantly, often becoming narrower as age progresses.\n\nThere are many use cases where you may want to adjust the range of processed frequencies, such as zooming in on a specific range of auditory frequencies or checking for the presence or absence of ultrasonic/subsonic sounds. These settings allow you to do so.");

                if ui
                    .checkbox(
                        &mut analysis_settings.erb_frequency_scale,
                        "Use ERB frequency scale",
                    )
                    .on_hover_text("If this is enabled, frequencies will be displayed using the Equivalent Rectangular Bandwidth scale, a psychoacoustic measure of pitch perception.\nIf this is disabled, frequencies will be displayed on a base 2 logarithmic scale, which is used by most musical scales.")
                    .changed()
                {
                    update_and_clear(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                }

                if ui
                    .checkbox(
                        &mut analysis_settings.erb_time_resolution,
                        "Use ERB time resolution",
                    )
                    .on_hover_text("Transforming time-domain data (audio samples) into the frequency domain has an inherent tradeoff between time resolution and frequency resolution.\nIf this setting is enabled, the appropriate time resolution will be determined using an adjusted version of the ERB scale.\nIf this setting is disabled, time resolution is determined based on the configured Q factor.")
                    .changed()
                {
                    update(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                }

                if analysis_settings.erb_time_resolution {
                    if ui
                        .add(
                            egui::Slider::new(&mut analysis_settings.erb_bandwidth_divisor, 0.5..=6.0)
                                .clamping(egui::SliderClamping::Never)
                                .text("ERB bandwidth divisor"),
                        )
                        .on_hover_text("Transforming time-domain data (audio samples) into the frequency domain has an inherent tradeoff between time resolution and frequency resolution.\nWhen using the Equivalent Rectangular Bandwidth model to determine this trade-off, adjusting the time resolution calculated by this function may be useful to improve visualization readability. This setting allows you to change how this adjustment is performed.")
                        .changed()
                    {
                        if analysis_settings.erb_bandwidth_divisor < 0.01 {
                            analysis_settings.erb_bandwidth_divisor = 0.01;
                        }

                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };
                } else {
                    if ui
                        .add(
                            egui::Slider::new(&mut analysis_settings.q_time_resolution, 2.0..=128.0)
                                .logarithmic(true)
                                .clamping(egui::SliderClamping::Never)
                                .suffix(" Q")
                                .fixed_decimals(1)
                                .text("Time resolution"),
                        )
                        .on_hover_text("Transforming time-domain data (audio samples) into the frequency domain has an inherent tradeoff between time resolution and frequency resolution. This setting allows you to adjust this tradeoff.")
                        .changed()
                    {
                        analysis_settings.q_time_resolution = analysis_settings.q_time_resolution.max(0.1);

                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };
                }

                if ui
                        .add(
                            egui::Slider::new(&mut analysis_settings.time_resolution_clamp.0, 0.0..=200.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("ms")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Minimum time resolution"),
                        )
                        .on_hover_text("Transforming time-domain data (audio samples) into the frequency domain has an inherent tradeoff between time resolution and frequency resolution.\nWhen determining this tradeoff, bounding the time resolution may be useful to improve visualization readability. This setting allows you to adjust this bound.")
                        .changed()
                    {
                        analysis_settings.time_resolution_clamp.0 = analysis_settings.time_resolution_clamp.0.clamp(0.0, 1000.0);
                        if analysis_settings.time_resolution_clamp.0 > analysis_settings.time_resolution_clamp.1 {
                            analysis_settings.time_resolution_clamp.1 = analysis_settings.time_resolution_clamp.0;
                        }

                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };

                    if ui
                        .add(
                            egui::Slider::new(&mut analysis_settings.time_resolution_clamp.1, 0.0..=200.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("ms")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Maximum time resolution"),
                        )
                        .on_hover_text("Transforming time-domain data (audio samples) into the frequency domain has an inherent tradeoff between time resolution and frequency resolution.\nWhen determining this tradeoff, bounding the time resolution may be useful to improve visualization readability. This setting allows you to adjust this bound.")
                        .changed()
                    {
                        analysis_settings.time_resolution_clamp.1 = analysis_settings.time_resolution_clamp.1.clamp(0.0, 1000.0);
                        if analysis_settings.time_resolution_clamp.0 > analysis_settings.time_resolution_clamp.1 {
                            analysis_settings.time_resolution_clamp.0 = analysis_settings.time_resolution_clamp.1;
                        }

                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    };

                if ui
                    .checkbox(
                        &mut analysis_settings.nc_method,
                        "Use NC method",
                    )
                    .on_hover_text("If this is enabled, windowing is performed using the NC method, a form of spectral reassignment using phase information which usually outperforms typical window functions.\nIf this is disabled, windowing is performed using the Hann window function.")
                    .changed()
                {
                    update(&analysis_settings);
                    egui_ctx.request_discard("Changed setting");
                    return;
                }

                if analysis_settings.nc_method {
                    if ui
                        .checkbox(
                            &mut analysis_settings.strict_nc,
                            "Use optimal NC method",
                        )
                        .on_hover_text("If this is enabled, the \"optimal\" version of the NC method is used, improving filter bandwidth characteristics at the expense of introducing additional banding artifacts into the spectrogram.\nIf this is disabled, an alternate implementation of the NC method is used, which prioritizes the reduction of spectrogram artifacts.")
                        .changed()
                    {
                        update(&analysis_settings);
                        egui_ctx.request_discard("Changed setting");
                        return;
                    }
                }

                if ui.button("Reset Analysis Options").clicked() {
                    *analysis_settings = AnalysisChainConfig::default();
                    render_settings.max_db =
                        80.0 - analysis_settings.listening_volume;
                    render_settings.min_db =
                        20.0 - analysis_settings.listening_volume;
                    render_settings.agc_minimum =
                        3.0 - analysis_settings.listening_volume;
                    render_settings.agc_maximum =
                        100.0 - analysis_settings.listening_volume;
                    update_and_clear(&analysis_settings);
                }
            });
        });

    let now = Instant::now();
    let mut frame_timing = shared_state.frame_timing.lock();

    *frame_timing = (
        now,
        now.duration_since(frame_timing.0),
        now.duration_since(start),
    )
}
