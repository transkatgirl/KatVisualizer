#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]

use color::{ColorSpaceTag, DynamicColor, Flags, Rgba8, Srgb};
use ndarray::Array2;
use nih_plug::prelude::*;
use nih_plug_egui::{
    create_egui_editor,
    egui::{
        self, Align2, Color32, ColorImage, FontId, ImageData, Mesh, Pos2, Rect, Shape, TextureId,
        TextureOptions, ThemePreference, Vec2,
        epaint::{ImageDelta, Vertex, WHITE_UV},
    },
};
use parking_lot::{FairMutex, Mutex, RwLock};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    AnalysisChain, AnalysisChainConfig, AnalysisMetrics, MAX_FREQUENCY_BINS,
    MAX_OSC_FREQUENCY_BINS, MyPlugin, PluginParams, PluginStateInfo, SPECTROGRAM_SLICES,
    analyzer::{BetterSpectrogram, map_value_f32},
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

fn draw_bargraph(
    mesh: &mut Mesh,
    spectrogram: &BetterSpectrogram,
    bounds: Rect,
    color_table: &ColorTable,
    masking_color: Option<Color32>,
    (max_db, min_db): (f32, f32),
    averaging: Duration,
) {
    let front = &spectrogram.data.front().unwrap();

    if !averaging.is_zero() {
        let target_len = front.data.len();
        let target_duration = front.duration;

        let max_index = spectrogram
            .data
            .iter()
            .enumerate()
            .take_while(|(i, row)| {
                row.duration.mul_f64(*i as f64) <= averaging
                    && row.data.len() == target_len
                    && row.duration == target_duration
            })
            .map(|(i, _)| i)
            .last()
            .unwrap_or(1);

        if max_index > 1 {
            let count = max_index as f32 + 1.0;

            let iterator = (0..target_len).map(move |i| {
                let sum = spectrogram
                    .data
                    .iter()
                    .take(max_index + 1)
                    .map(|row| row.data[i])
                    .fold((0.0, 0.0), |acc, d| (acc.0 + d.0, acc.1 + d.1));

                (sum.0 / count, sum.1 / count)
            });

            draw_bargraph_from_iter(
                mesh,
                iterator,
                target_len,
                bounds,
                color_table,
                (max_db, min_db),
            );

            if let Some(masking_color) = masking_color {
                let masking_iterator = (0..target_len).map(move |i| {
                    let sum = spectrogram
                        .data
                        .iter()
                        .take(max_index + 1)
                        .map(|row| row.masking[i])
                        .fold(0.0, |acc, d| acc + d.1);

                    sum / count
                });
                draw_secondary_bargraph_from_iter(
                    mesh,
                    masking_iterator,
                    front.masking.len(),
                    bounds,
                    masking_color,
                    None,
                    (max_db, min_db),
                );
            }

            return;
        }
    }

    draw_bargraph_from_iter(
        mesh,
        front.data.iter().copied().map(|d| (d.0, d.1)),
        front.data.len(),
        bounds,
        color_table,
        (max_db, min_db),
    );
    if let Some(masking_color) = masking_color {
        draw_secondary_bargraph_from_iter(
            mesh,
            front.masking.iter().map(|(_, m)| *m),
            front.masking.len(),
            bounds,
            masking_color,
            None,
            (max_db, min_db),
        );
    }
}

fn draw_bargraph_from_iter(
    mesh: &mut Mesh,
    analysis: impl Iterator<Item = (f32, f32)>,
    analysis_len: usize,
    bounds: Rect,
    color_table: &ColorTable,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let mut vertices = mesh.vertices.len() as u32;

    let band_width = width / analysis_len as f32;

    for (i, (pan, volume)) in analysis.enumerate() {
        let intensity = map_value_f32(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

        let start_x = bounds.min.x + i as f32 * band_width;

        let rect = Rect {
            min: Pos2 {
                x: start_x,
                y: bounds.max.y - intensity * height,
            },
            max: Pos2 {
                x: start_x + band_width,
                y: bounds.max.y,
            },
        };
        let color = color_table.lookup(pan, intensity);

        mesh.indices.extend_from_slice(&[
            vertices,
            vertices + 1,
            vertices + 2,
            vertices + 2,
            vertices + 1,
            vertices + 3,
        ]);
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
        vertices += 4;
    }
}

fn draw_secondary_bargraph_from_iter(
    mesh: &mut Mesh,
    analysis: impl Iterator<Item = f32>,
    analysis_len: usize,
    bounds: Rect,
    color: Color32,
    thickness: Option<f32>,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let mut vertices = mesh.vertices.len() as u32;

    let band_width = width / analysis_len as f32;

    if let Some(thickness) = thickness {
        for (i, volume) in analysis.enumerate() {
            let intensity = map_value_f32(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

            let start_x = bounds.min.x + i as f32 * band_width;

            let rect = Rect {
                min: Pos2 {
                    x: start_x,
                    y: bounds.max.y - intensity * height,
                },
                max: Pos2 {
                    x: start_x + band_width,
                    y: (bounds.max.y - (intensity - thickness) * height).min(bounds.max.y),
                },
            };

            mesh.indices.extend_from_slice(&[
                vertices,
                vertices + 1,
                vertices + 2,
                vertices + 2,
                vertices + 1,
                vertices + 3,
            ]);
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
            vertices += 4;
        }
    } else {
        for (i, volume) in analysis.enumerate() {
            let intensity = map_value_f32(volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

            let start_x = bounds.min.x + i as f32 * band_width;

            let rect = Rect {
                min: Pos2 {
                    x: start_x,
                    y: bounds.max.y - intensity * height,
                },
                max: Pos2 {
                    x: start_x + band_width,
                    y: bounds.max.y,
                },
            };

            mesh.indices.extend_from_slice(&[
                vertices,
                vertices + 1,
                vertices + 2,
                vertices + 2,
                vertices + 1,
                vertices + 3,
            ]);
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
            vertices += 4;
        }
    }
}

fn draw_spectrogram_image(
    image: &mut ColorImage,
    spectrogram: &BetterSpectrogram,
    color_table: &ColorTable,
    (max_db, min_db): (f32, f32),
) {
    let target_duration = spectrogram.data.front().unwrap().duration;

    let image_width = image.width();
    let image_height = image.height();

    for (y, analysis) in spectrogram.data.iter().enumerate() {
        if analysis.data.len() != image_width
            || y == image_height
            || analysis.duration != target_duration
        {
            break;
        }

        for ((pan, volume), pixel) in analysis
            .data
            .iter()
            .copied()
            .zip(unsafe { image.pixels.get_unchecked_mut((image_width * y)..) }.iter_mut())
        {
            let intensity = map_value_f32(volume, min_db, max_db, 0.0, 1.0);
            *pixel = color_table.lookup(pan, intensity);
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
        let frequency = frequencies[map_value_f32(
            cursor.x,
            bargraph_bounds.min.x,
            bargraph_bounds.max.x,
            0.0,
            frequency_count,
        )
        .floor() as usize];
        let amplitude = map_value_f32(
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
        let x = map_value_f32(
            cursor.x,
            spectrogram_bounds.min.x,
            spectrogram_bounds.max.x,
            0.0,
            frequency_count,
        )
        .floor() as usize;
        let y = map_value_f32(
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
                time: Some(duration.mul_f64(y as f64)),
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
    automatic_gain: bool,
    agc_duration: Duration,
    agc_above_masking: f32,
    agc_below_masking: f32,
    agc_minimum: f32,
    agc_maximum: f32,
    min_db: f32,
    max_db: f32,
    bargraph_height: f32,
    spectrogram_duration: Duration,
    bargraph_averaging: Duration,
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
            agc_duration: Duration::from_secs_f64(1.0),
            agc_above_masking: 42.0,
            agc_below_masking: 3.0,
            agc_minimum: 3.0 - AnalysisChainConfig::default().listening_volume as f32,
            agc_maximum: 100.0 - AnalysisChainConfig::default().listening_volume as f32,
            min_db: 20.0 - AnalysisChainConfig::default().listening_volume as f32,
            max_db: 80.0 - AnalysisChainConfig::default().listening_volume as f32,
            bargraph_height: 0.33,
            spectrogram_duration: Duration::from_secs_f64(0.67),
            bargraph_averaging: Duration::from_secs_f64(0.004),
            show_performance: true,
            show_format: false,
            show_hover: true,
            show_masking: true,
            masking_color: Color32::from_rgb(33, 0, 4),
        }
    }
}

struct ColorTable {
    table: ndarray::Array<(u8, u8, u8), ndarray::Dim<[usize; 2]>>,
    size: (usize, usize),
    max: (f32, f32),
}

const COLOR_TABLE_CHROMA_SIZE: usize = 512;
const COLOR_TABLE_LIGHTNESS_SIZE: usize = 1024;

impl ColorTable {
    fn new() -> Self {
        Self {
            table: Array2::default((COLOR_TABLE_CHROMA_SIZE, COLOR_TABLE_LIGHTNESS_SIZE)),
            size: (COLOR_TABLE_CHROMA_SIZE, COLOR_TABLE_LIGHTNESS_SIZE),
            max: (
                (COLOR_TABLE_CHROMA_SIZE - 1) as f32,
                (COLOR_TABLE_LIGHTNESS_SIZE - 1) as f32,
            ),
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
            let split = map_value_f32(split_index as f32, 0.0, self.max.0, -1.0, 1.0);
            for intensity_index in 0..self.size.1 {
                if intensity_index == 0 {
                    self.table[[split_index, 0]] = (0, 0, 0);
                    continue;
                }

                let intensity = map_value_f32(intensity_index as f32, 0.0, self.max.1, 0.0, 1.0);

                let mut color = if split >= 0.0 {
                    let mut color = right_color;
                    color.components[1] = map_value_f32(split, 0.0, 1.0, 0.0, color.components[1]);
                    color
                } else {
                    let mut color = left_color;
                    color.components[1] = map_value_f32(-split, 0.0, 1.0, 0.0, color.components[1]);
                    color
                };

                color.components[0] =
                    map_value_f32(intensity, 0.0, 1.0, min_lightness, color.components[0]);

                let converted: Rgba8 = color.to_alpha_color::<Srgb>().to_rgba8();

                self.table[[split_index, intensity_index]] =
                    (converted.r, converted.g, converted.b);
            }
        }
    }
    fn lookup(&self, split: f32, intensity: f32) -> Color32 {
        let color = self.table[[
            map_value_f32(split, -1.0, 1.0, 0.0, self.max.0)
                .round()
                .clamp(0.0, self.max.0) as usize,
            map_value_f32(intensity, 0.0, 1.0, 0.0, self.max.1)
                .round()
                .clamp(0.0, self.max.1) as usize,
        ]];

        Color32::from_rgb(color.0, color.1, color.2)
    }
}

struct SharedState {
    settings: RwLock<RenderSettings>,
    last_frame: Mutex<Instant>,
    color_table: RwLock<ColorTable>,
    cached_analysis_settings: Mutex<AnalysisChainConfig>,
    spectrogram_texture: Arc<RwLock<Option<TextureId>>>,
}

const PERFORMANCE_METER_TARGET_FPS: f64 = 60.0;

pub fn create(
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    analysis_output: Arc<FairMutex<(BetterSpectrogram, AnalysisMetrics)>>,
    analysis_frequencies: Arc<RwLock<Vec<(f32, f32, f32)>>>,
    plugin_state: Arc<RwLock<Option<PluginStateInfo>>>,
    _async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let shared_state = {
        let settings = RenderSettings::default();
        let mut color_table = ColorTable::new();
        color_table.build(
            settings.left_hue,
            settings.right_hue,
            settings.minimum_lightness,
            settings.maximum_lightness,
            settings.maximum_chroma,
        );

        SharedState {
            settings: RwLock::new(settings),
            last_frame: Mutex::new(Instant::now()),
            color_table: RwLock::new(color_table),
            cached_analysis_settings: Mutex::new(AnalysisChainConfig::default()),
            spectrogram_texture: Arc::new(RwLock::new(None)),
        }
    };

    let spectrogram_texture = shared_state.spectrogram_texture.clone();

    create_egui_editor(
        egui_state.clone(),
        (),
        move |egui_ctx, _| {
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
                    magnification: egui::TextureFilter::Nearest,
                    minification: egui::TextureFilter::Linear,
                    wrap_mode: egui::TextureWrapMode::ClampToEdge,
                    mipmap_mode: None,
                },
            ));

            egui_ctx.tessellation_options_mut(|options| {
                options.coarse_tessellation_culling = false;
            });

            egui_ctx.set_theme(ThemePreference::Dark);
        },
        move |egui_ctx, _setter, _| {
            egui_ctx.request_repaint();

            egui::CentralPanel::default().show(egui_ctx, |ui| {
                let mut bargraph_mesh = Mesh::default();
                bargraph_mesh.reserve_triangles(MAX_FREQUENCY_BINS * 2 * 2);
                bargraph_mesh.reserve_vertices(MAX_FREQUENCY_BINS * 4 * 2);

                let mut spectrogram_image_pixels =
                    vec![Color32::TRANSPARENT; MAX_FREQUENCY_BINS * SPECTROGRAM_SLICES];

                let painter = ui.painter();
                let max_x = painter.clip_rect().max.x;
                let max_y = painter.clip_rect().max.y;

                let settings = *shared_state.settings.read();
                let color_table = &shared_state.color_table.read();
                let spectrogram_texture = shared_state.spectrogram_texture.read().unwrap();
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

                let start = Instant::now();

                let lock = analysis_output.lock();
                let (ref spectrogram, ref metrics) = *lock;

                let front = spectrogram.data.front().unwrap();

                let spectrogram_width = front.data.len();
                let spectrogram_height = (settings.spectrogram_duration.as_secs_f64()
                    / front.duration.as_secs_f64())
                .round() as usize;

                spectrogram_image_pixels.truncate(spectrogram_width * spectrogram_height);

                let mut spectrogram_image = ColorImage {
                    size: [spectrogram_width, spectrogram_height],
                    source_size: Vec2 {
                        x: spectrogram_width as f32,
                        y: spectrogram_height as f32,
                    },
                    pixels: spectrogram_image_pixels,
                };

                let buffering_duration = start.duration_since(metrics.finished);
                let processing_duration = metrics.processing;
                let chunk_duration = front.duration;

                let (min_db, max_db) = calculate_volume_min_max(&settings, spectrogram);

                if settings.bargraph_height != 0.0 {
                    if settings.show_masking {
                        draw_bargraph(
                            &mut bargraph_mesh,
                            spectrogram,
                            bargraph_bounds,
                            color_table,
                            Some(settings.masking_color),
                            (max_db, min_db),
                            settings.bargraph_averaging,
                        );
                    } else {
                        draw_bargraph(
                            &mut bargraph_mesh,
                            spectrogram,
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
                        spectrogram,
                        color_table,
                        (max_db, min_db),
                    );
                }

                let under_pointer = if settings.show_hover {
                    if let Some(pointer) = egui_ctx.pointer_latest_pos() {
                        get_under_cursor(
                            pointer,
                            spectrogram,
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

                drop(lock);

                egui_ctx.tex_manager().write().set(
                    spectrogram_texture,
                    ImageDelta {
                        image: ImageData::Color(Arc::new(spectrogram_image)),
                        options: TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Linear,
                            wrap_mode: egui::TextureWrapMode::ClampToEdge,
                            mipmap_mode: None,
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

                if let Some(under) = under_pointer {
                    let analysis_settings = shared_state.cached_analysis_settings.lock();

                    let amplitude_text = if analysis_settings.normalize_amplitude {
                        format!(
                            "{:.0} phon",
                            under.amplitude as f64 + analysis_settings.listening_volume
                        )
                    } else {
                        format!("{:+.0}dBFS", under.amplitude)
                    };

                    drop(analysis_settings);

                    let text = if let (Some(pan), Some(elapsed)) = (under.pan, under.time) {
                        format!(
                            "{:.0}hz, -{:.3}s\n{}, {:+.2} pan",
                            under.frequency.1,
                            elapsed.as_secs_f64(),
                            amplitude_text,
                            pan
                        )
                    } else {
                        let resolution = (1.0 / (under.frequency.2 - under.frequency.0)) * 1000.0;
                        let averaging = settings.bargraph_averaging.as_secs_f64() * 1000.0;

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
                    if let Some(ref plugin_state) = *plugin_state.read() {
                        let min_buffer_size_s =
                            plugin_state.buffer_config.min_buffer_size.unwrap_or(0) as f64
                                / plugin_state.buffer_config.sample_rate as f64;
                        let max_buffer_size_s = plugin_state.buffer_config.max_buffer_size as f64
                            / plugin_state.buffer_config.sample_rate as f64;

                        let should_warn = plugin_state.buffer_config.sample_rate
                            < (frequencies.last().unwrap().2 * 2.0)
                            || max_buffer_size_s > 0.010
                            || plugin_state.buffer_config.process_mode != ProcessMode::Realtime;

                        painter.text(
                            Pos2 {
                                x: max_x / 2.0,
                                y: 16.0,
                            },
                            Align2::CENTER_CENTER,
                            format!(
                                "{} in -> {} out, {:.1}kHz, {:.0}ms to {:.0}ms buffer, mode {:?}",
                                plugin_state
                                    .audio_io_layout
                                    .main_input_channels
                                    .map(u32::from)
                                    .unwrap_or(0),
                                plugin_state
                                    .audio_io_layout
                                    .main_output_channels
                                    .map(u32::from)
                                    .unwrap_or(0),
                                plugin_state.buffer_config.sample_rate / 1000.0,
                                min_buffer_size_s * 1000.0,
                                max_buffer_size_s * 1000.0,
                                plugin_state.buffer_config.process_mode
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

                let now = Instant::now();
                let frame_elapsed = now.duration_since(*shared_state.last_frame.lock());

                if settings.show_performance {
                    let rasterize_elapsed = now.duration_since(start);

                    let rasterize_secs = rasterize_elapsed.as_secs_f64();
                    let chunk_secs = chunk_duration.as_secs_f64();
                    let frame_secs = frame_elapsed.as_secs_f64();

                    let rasterize_processing_duration = rasterize_secs / (frame_secs / chunk_secs);
                    let adjusted_processing_duration =
                        processing_duration.as_secs_f64() + rasterize_processing_duration;
                    let rasterize_proportion = rasterize_secs / frame_secs;
                    let processing_proportion = adjusted_processing_duration / chunk_secs;
                    let buffering_proportion = buffering_duration.as_secs_f64() / frame_secs;

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
                            format!(
                                "{:.1}ms buffering",
                                buffering_duration.as_secs_f64() * 1000.0
                            ),
                            FontId {
                                size: 12.0,
                                family: egui::FontFamily::Monospace,
                            },
                            if buffering_proportion >= 1.0 {
                                Color32::RED
                            } else if buffering_proportion >= 0.6 {
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
                        if frame_elapsed
                            >= Duration::from_secs_f64(1.0 / (PERFORMANCE_METER_TARGET_FPS * 0.5))
                        {
                            Color32::RED
                        } else if frame_elapsed
                            >= Duration::from_secs_f64(1.0 / (PERFORMANCE_METER_TARGET_FPS * 0.9))
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
                            rasterize_elapsed.as_secs_f64() * 1000.0,
                        ),
                        FontId {
                            size: 12.0,
                            family: egui::FontFamily::Monospace,
                        },
                        if rasterize_elapsed
                            >= Duration::from_secs_f64(1.0 / (PERFORMANCE_METER_TARGET_FPS * 4.0))
                            || rasterize_proportion > 0.25
                        {
                            Color32::RED
                        } else if rasterize_elapsed
                            >= Duration::from_secs_f64(1.0 / (PERFORMANCE_METER_TARGET_FPS * 8.0))
                            || rasterize_proportion > 0.125
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


                        if analysis_settings.masking {
                            ui.checkbox(&mut render_settings.automatic_gain, "Automatic amplitude ranging");
                        }

                        if render_settings.automatic_gain && analysis_settings.masking {
                            let mut agc_duration = render_settings.agc_duration.as_secs_f64();
                            if ui
                                .add(
                                    egui::Slider::new(&mut agc_duration, 0.2..=8.0)
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
                                    (render_settings.min_db as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);
                                let mut max_phon =
                                    (render_settings.max_db as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);

                                if ui.add(
                                    egui::Slider::new(&mut max_phon, 0.0..=100.0)
                                        .suffix(" phon")
                                        .step_by(1.0)
                                        .fixed_decimals(0)
                                        .text("Maximum amplitude"),
                                ).changed() {
                                    render_settings.max_db =
                                        (max_phon - analysis_settings.listening_volume) as f32;
                                }

                                if ui.add(
                                    egui::Slider::new(&mut min_phon, 0.0..=100.0)
                                        .suffix(" phon")
                                        .step_by(1.0)
                                        .fixed_decimals(0)
                                        .text("Minimum amplitude"),
                                ).changed() {
                                    render_settings.min_db =
                                        (min_phon - analysis_settings.listening_volume) as f32;
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

                        let mut spectrogram_duration = render_settings.spectrogram_duration.as_secs_f64();
                        if ui
                            .add(
                                egui::Slider::new(&mut spectrogram_duration, 0.05..=2.0)
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
                                egui::Slider::new(&mut bargraph_averaging, 0.0..=100.0)
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
                                    (80.0 - analysis_settings.listening_volume) as f32;
                            render_settings.min_db =
                                    (20.0 - analysis_settings.listening_volume) as f32;
                            if analysis_settings.normalize_amplitude {
                                render_settings.agc_minimum =
                                    (3.0 - analysis_settings.listening_volume) as f32;
                                render_settings.agc_maximum =
                                    (100.0 - analysis_settings.listening_volume) as f32;
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
                            .on_hover_text("This setting adjusts the amplitude of the incoming signal before it is processed (but does not affect the plugin's output channels; audio is always passed through unmodified).\n\nAll internal audio processing is done using 64-bit floating point, so this can be adjusted freely without concern for clipping.")
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
                                        (3.0 - analysis_settings.listening_volume) as f32;
                                    render_settings.agc_maximum =
                                        (100.0 - analysis_settings.listening_volume) as f32;
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
                            let old_tone_threshold_phon =
                                (analysis_settings.midi_tone_amplitude_threshold as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);
                            let old_min_phon =
                                (render_settings.min_db as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);
                            let old_max_phon =
                                (render_settings.max_db as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);

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
                                analysis_settings.midi_tone_amplitude_threshold =
                                    (old_tone_threshold_phon - analysis_settings.listening_volume) as f32;
                                render_settings.min_db =
                                    (old_min_phon - analysis_settings.listening_volume) as f32;
                                render_settings.max_db =
                                    (old_max_phon - analysis_settings.listening_volume) as f32;
                                render_settings.agc_minimum =
                                    (3.0 - analysis_settings.listening_volume) as f32;
                                render_settings.agc_maximum =
                                    (100.0 - analysis_settings.listening_volume) as f32;
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
                        }

                        if ui
                            .checkbox(
                                &mut analysis_settings.internal_buffering,
                                "Use internal buffering",
                            )
                            .on_hover_text("In order to better capture transient signals and phase information, audio is processed in multiple overlapping windows.\nIf this is enabled, the plugin maintains its own buffer of samples, allowing the number of overlapping windows per second to be changed by the user. This adds a small amount of latency, which is reported to the plugin's host so that it can be compensated for.\nIf this is disabled, the number of overlapping windows per second is determined by the buffer size set by the host.")
                            .changed()
                        {
                            analysis_settings.output_osc = false;
                            analysis_settings.output_midi = false;
                            update(&analysis_settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }

                        if analysis_settings.internal_buffering {
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
                                .on_hover_text("In order to better capture transient signals and phase information, audio is processed in multiple overlapping windows. This setting allows you to adjust the number of overlapping windows per second, effectively setting the spectrogram's vertical resolution (and the associated amount of CPU usage required).\n\nThe default value for the setting is based on the just-noticeable-difference in onset time between two auditory events.\n\n(Note: This setting does not change the trade-off between time resolution and frequency resolution.)")
                                .changed()
                            {
                                update_and_clear(&analysis_settings);
                                egui_ctx.request_discard("Changed setting");
                                return;
                            };
                        }

                        let maximum_resolution = if analysis_settings.output_osc {
                            MAX_OSC_FREQUENCY_BINS
                        } else {
                            MAX_FREQUENCY_BINS
                        };

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut analysis_settings.resolution,
                                    128..=maximum_resolution,
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

                        ui.collapsing("External Outputs", |ui| {
                            ui.colored_label(
                                Color32::YELLOW,
                                "Enabling external output requires internal buffering to be disabled.",
                            );

                            if ui
                                .checkbox(
                                    &mut analysis_settings.output_osc,
                                    "Output analysis via OSC",
                                )
                                .on_hover_text("If this is enabled, the plugin will output analysis data via OSC, which can then be used as an input for alternative visualization methods.")
                                .changed()
                            {
                                if analysis_settings.output_osc {
                                    analysis_settings.internal_buffering = false;
                                    analysis_settings.normalize_amplitude = true;
                                    analysis_settings.masking = true;
                                    if analysis_settings.resolution > MAX_OSC_FREQUENCY_BINS {
                                        analysis_settings.resolution = MAX_OSC_FREQUENCY_BINS;
                                    }
                                }
                                update(&analysis_settings);
                                egui_ctx.request_discard("Changed setting");
                                return;
                            }

                            if analysis_settings.output_osc {
                                ui.colored_label(
                                    Color32::YELLOW,
                                    "OSC packets produced by this program are intended solely for further local analysis and significantly exceed the MTU of most networks.",
                                );

                                let address_label = ui
                                    .label("OSC server UDP/IP address:")
                                    .on_hover_text("The UDP/IP address of the OSC server that analysis output will be sent to.");

                                if ui.
                                    text_edit_singleline(&mut analysis_settings.osc_socket_address)
                                    .labelled_by(address_label.id)
                                    .on_hover_text("The UDP/IP address of the OSC server that analysis output will be sent to.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                }

                                let message_label_1 = ui
                                    .label("Frequency bin data OSC message address:")
                                    .on_hover_text("The OSC address pattern that frequency bin data will be sent under.\n\nThe message format for frequency bin data is [(frequency, bandwidth, pan, volume, signalToMaskRatio)], with bins being listed in order of priority. Frequencies and bandwidths are in Hz, volume is in phon (or dBFS if amplitude normalization is disabled), pan ranges from -1 to 1, and signalToMaskRatio is in dB.");

                                if ui.
                                    text_edit_singleline(&mut analysis_settings.osc_resource_address_frequencies)
                                    .labelled_by(message_label_1.id)
                                    .on_hover_text("The OSC address pattern that frequency bin data will be sent under.\n\nThe message format for frequency bin data is [(frequency, bandwidth, pan, volume, signalToMaskRatio)], with bins being listed in order of priority. Frequencies and bandwidths are in Hz, volume is in phon (or dBFS if amplitude normalization is disabled), pan ranges from -1 to 1, and signalToMaskRatio is in dB.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                }

                                let message_label_2 = ui
                                    .label("Analysis statistics OSC message address:")
                                    .on_hover_text("The OSC address pattern that analysis statistics will be sent under.\n\nThe message format for analysis statistics is (averageMasking, averageVolume, maximumVolume, chunkDuration), with amplitude values being in phon (or dBFS if amplitude normalization is disabled) and duration values being in seconds. Analysis statistics are only applicable for the most recent slice of tone data.");

                                if ui.
                                    text_edit_singleline(&mut analysis_settings.osc_resource_address_stats)
                                    .labelled_by(message_label_2.id)
                                    .on_hover_text("The OSC address pattern that analysis statistics will be sent under.\n\nThe message format for analysis statistics is (averageMasking, averageVolume, maximumVolume, chunkDuration), with amplitude values being in phon (or dBFS if amplitude normalization is disabled) and duration values being in seconds. Analysis statistics are only applicable for the most recent slice of tone data.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                }
                            }

                            #[cfg(feature = "midi")]
                            if ui
                                .checkbox(
                                    &mut analysis_settings.output_midi,
                                    "Output analysis via MIDI",
                                )
                                .on_hover_text("If this is enabled, the plugin will use analysis data to generate a MIDI output, which can then be used as an input for alternative visualization methods.")
                                .changed()
                            {
                                if analysis_settings.output_midi {
                                    analysis_settings.internal_buffering = false;
                                    analysis_settings.normalize_amplitude = true;
                                    analysis_settings.masking = true;
                                    analysis_settings.erb_time_resolution = false;
                                    analysis_settings.q_time_resolution = 17.30993;
                                }
                                update(&analysis_settings);
                                egui_ctx.request_discard("Changed setting");
                                return;
                            }

                            #[cfg(feature = "midi")]
                            if analysis_settings.output_midi {
                                ui.colored_label(
                                    Color32::YELLOW,
                                    "Some MIDI software may struggle to handle this plugin's output rate. If you encounter issues, try lowering the number of simultaneous tones or increasing the plugin's buffer size.",
                                );

                                if ui
                                    .add(
                                        egui::Slider::new(&mut analysis_settings.midi_max_simultaneous_tones, 1..=128)
                                            .suffix(" notes")
                                            .logarithmic(true)
                                            .text("Maximum simultaneous MIDI notes"),
                                    )
                                    .on_hover_text("This setting adjusts the maximum number of simultaneous MIDI notes output by the plugin.\nValid notes are prioritized by their distance from the masking threshold, or if masking data is not available, their perceptual amplitude. If amplitude normalization is also disabled, notes will then be prioritized based on absolute amplitude.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                };

                                if analysis_settings.normalize_amplitude {
                                    let mut tone_threshold_phon =
                                        (analysis_settings.midi_tone_amplitude_threshold as f64 + analysis_settings.listening_volume).clamp(0.0, 100.0);

                                    if ui
                                        .add(
                                            egui::Slider::new(&mut tone_threshold_phon, 0.0..=100.0)
                                                .suffix(" phon")
                                                .step_by(1.0)
                                                .fixed_decimals(0)
                                                .text("MIDI note amplitude threshold"),
                                        )
                                        .on_hover_text("This setting adjusts the minimum amplitude necessary for a note to be considered valid. Notes with an amplitude below this threshold will always be considered inactive, regardless of the note limit.")
                                        .changed()
                                    {
                                        analysis_settings.midi_tone_amplitude_threshold =
                                            (tone_threshold_phon - analysis_settings.listening_volume) as f32;
                                        update(&analysis_settings);
                                        egui_ctx.request_discard("Changed setting");
                                        return;
                                    }
                                } else {
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut analysis_settings.midi_tone_amplitude_threshold, -100.0..=0.0)
                                                .clamping(egui::SliderClamping::Never)
                                                .suffix("dB")
                                                .step_by(1.0)
                                                .fixed_decimals(0)
                                                .text("MIDI note amplitude threshold"),
                                        )
                                        .on_hover_text("This setting adjusts the minimum amplitude necessary for a note to be considered valid. Notes with an amplitude below this threshold will always be considered inactive, regardless of the note limit.")
                                        .changed()
                                    {
                                        update(&analysis_settings);
                                        egui_ctx.request_discard("Changed setting");
                                        return;
                                    };
                                }

                                if ui
                                    .add(
                                        egui::Slider::new(&mut analysis_settings.midi_pressure_min_amplitude, -100.0..=0.0)
                                            .clamping(egui::SliderClamping::Never)
                                            .suffix("dB")
                                            .step_by(1.0)
                                            .fixed_decimals(0)
                                            .text("MIDI note pressure minimum amplitude"),
                                    )
                                    .on_hover_text("When converting frequency data into MIDI notes, amplitudes must be mapped to a note pressure level.\nThis setting allows you to adjust the amplitude corresponding to a note pressure of 0%.\n\nNote: Due to a quirk of the MIDI specification, a note pressure of exactly 0% will cause the note to be released.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                };

                                if ui
                                    .add(
                                        egui::Slider::new(&mut analysis_settings.midi_pressure_max_amplitude, -100.0..=0.0)
                                            .clamping(egui::SliderClamping::Never)
                                            .suffix("dB")
                                            .step_by(1.0)
                                            .fixed_decimals(0)
                                            .text("MIDI note pressure maximum amplitude"),
                                    )
                                    .on_hover_text("When converting frequency data into MIDI notes, amplitudes must be mapped to a note pressure level.\nThis setting allows you to adjust the amplitude corresponding to a note pressure of 100%.")
                                    .changed()
                                {
                                    update(&analysis_settings);
                                    egui_ctx.request_discard("Changed setting");
                                };
                            }
                        });

                        if ui.button("Reset Analysis Options").clicked() {
                            *analysis_settings = AnalysisChainConfig::default();
                            render_settings.max_db =
                                (80.0 - analysis_settings.listening_volume) as f32;
                            render_settings.min_db =
                                (20.0 - analysis_settings.listening_volume) as f32;
                            render_settings.agc_minimum =
                                (3.0 - analysis_settings.listening_volume) as f32;
                            render_settings.agc_maximum =
                                (100.0 - analysis_settings.listening_volume) as f32;
                            update_and_clear(&analysis_settings);
                        }
                    });


                });

            *shared_state.last_frame.lock() = Instant::now();
        },
    )
}
