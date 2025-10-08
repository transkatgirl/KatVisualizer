use color::{ColorSpaceTag, DynamicColor, Flags, Rgba8, Srgb};
use ndarray::Array2;
use nih_plug::prelude::*;
use nih_plug_egui::{
    create_egui_editor,
    egui::{
        self, Align2, Color32, ColorImage, FontId, ImageData, Mesh, Pos2, Rect, Shape, TextureId,
        TextureOptions,
        epaint::{ImageDelta, Vertex, WHITE_UV},
    },
};
use parking_lot::{FairMutex, Mutex, RwLock};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    AnalysisChain, AnalysisChainConfig, AnalysisMetrics, MAX_FREQUENCY_BINS, MyPlugin,
    PluginParams, SPECTROGRAM_SLICES,
    analyzer::{BetterAnalysis, BetterSpectrogram, map_value_f32},
};

fn draw_bargraph(
    mesh: &mut Mesh,
    analysis: &BetterAnalysis,
    bounds: Rect,
    color_table: &ColorTable,
    (max_db, min_db): (f32, f32),
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let band_width = width / analysis.data.len() as f32;

    let mut vertices = mesh.vertices.len() as u32;

    for (i, (pan, volume)) in analysis.data.iter().enumerate() {
        let intensity = map_value_f32(*volume, min_db, max_db, 0.0, 1.0).clamp(0.0, 1.0);

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
        let color = color_table.lookup(*pan, intensity);

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

        for (x, (pan, volume)) in analysis.data.iter().enumerate() {
            let intensity = map_value_f32(*volume, min_db, max_db, 0.0, 1.0);
            image.pixels[(image_width * y) + x] = color_table.lookup(*pan, intensity);
        }
    }
}

fn get_frequency_amplitude_pan_time(
    cursor: Pos2,
    spectrogram: &BetterSpectrogram,
    frequencies: &[(f32, f32, f32)],
    bargraph_bounds: Rect,
    spectrogram_bounds: Rect,
    (bargraph_max_db, bargraph_min_db): (f32, f32),
    spectrogram_height: usize,
) -> Option<((f32, f32, f32), f32, Option<(f32, Duration)>)> {
    if bargraph_bounds.contains(cursor) {
        let frequency = frequencies[map_value_f32(
            cursor.x,
            bargraph_bounds.min.x,
            bargraph_bounds.max.x,
            0.0,
            frequencies.len() as f32,
        )
        .floor() as usize];
        let amplitude = map_value_f32(
            bargraph_bounds.max.y - cursor.y,
            bargraph_bounds.min.y,
            bargraph_bounds.max.y,
            bargraph_min_db,
            bargraph_max_db,
        );

        Some((frequency, amplitude, None))
    } else if spectrogram_bounds.contains(cursor) {
        let x = map_value_f32(
            cursor.x,
            spectrogram_bounds.min.x,
            spectrogram_bounds.max.x,
            0.0,
            frequencies.len() as f32,
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
            Some((
                frequency,
                amplitude,
                Some((pan, duration.mul_f64(y as f64))),
            ))
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
    min_db: f32,
    max_db: f32,
    bargraph_height: f32,
    spectrogram_duration: Duration,
    show_performance: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            left_hue: 195.0,
            right_hue: 328.0,
            minimum_lightness: 0.13,
            maximum_lightness: 0.8,
            maximum_chroma: 0.1,
            min_db: -72.0,
            max_db: -12.0,
            bargraph_height: 0.4,
            spectrogram_duration: Duration::from_secs_f64(0.5),
            show_performance: true,
        }
    }
}

struct ColorTable {
    table: ndarray::Array<Color32, ndarray::Dim<[usize; 2]>>,
    size: usize,
    max: f32,
}

const COLOR_TABLE_SIZE: usize = 2048;

impl ColorTable {
    fn new(size: usize) -> Self {
        Self {
            table: Array2::default((size, size)),
            size,
            max: (size - 1) as f32,
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

        for split_index in 0..self.size {
            let split = map_value_f32(split_index as f32, 0.0, self.max, -1.0, 1.0);
            for intensity_index in 0..self.size {
                if intensity_index == 0 {
                    self.table[[split_index, 0]] = Color32::BLACK;
                    continue;
                }

                let intensity = map_value_f32(intensity_index as f32, 0.0, self.max, 0.0, 1.0);

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
                    Color32::from_rgb(converted.r, converted.g, converted.b);
            }
        }
    }
    fn lookup(&self, split: f32, intensity: f32) -> Color32 {
        self.table[[
            map_value_f32(split, -1.0, 1.0, 0.0, self.max)
                .round()
                .clamp(0.0, self.max) as usize,
            map_value_f32(intensity, 0.0, 1.0, 0.0, self.max)
                .round()
                .clamp(0.0, self.max) as usize,
        ]]
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
    _async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let shared_state = {
        let settings = RenderSettings::default();
        let mut color_table = ColorTable::new(COLOR_TABLE_SIZE);
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
                ImageData::Color(Arc::new(ColorImage::new([1, 1], Color32::TRANSPARENT))),
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
        },
        move |egui_ctx, _setter, _| {
            egui_ctx.request_repaint();

            egui::CentralPanel::default().show(egui_ctx, |ui| {
                let mut bargraph_mesh = Mesh::default();
                bargraph_mesh.reserve_triangles(MAX_FREQUENCY_BINS * 2);
                bargraph_mesh.reserve_vertices(MAX_FREQUENCY_BINS * 4);

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
                    pixels: spectrogram_image_pixels,
                };

                let buffering_duration = start.duration_since(metrics.finished);
                let processing_duration = metrics.processing;
                let chunk_duration = front.duration;

                if settings.bargraph_height != 0.0 {
                    draw_bargraph(
                        &mut bargraph_mesh,
                        &front,
                        bargraph_bounds,
                        color_table,
                        (settings.max_db, settings.min_db),
                    );
                }

                if settings.bargraph_height != 1.0 {
                    draw_spectrogram_image(
                        &mut spectrogram_image,
                        spectrogram,
                        color_table,
                        (settings.max_db, settings.min_db),
                    );
                }

                let under_pointer = if let Some(pointer) = egui_ctx.pointer_latest_pos() {
                    get_frequency_amplitude_pan_time(
                        pointer,
                        spectrogram,
                        &frequencies,
                        bargraph_bounds,
                        spectrogram_bounds,
                        (settings.max_db, settings.min_db),
                        spectrogram_height,
                    )
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

                if let Some((frequency, amplitude, additional)) = under_pointer {
                    let analysis_settings = shared_state.cached_analysis_settings.lock();

                    let amplitude_text = if analysis_settings.normalize_amplitude {
                        format!(
                            "{:.0} phon",
                            amplitude + analysis_settings.listening_volume as f32
                        )
                    } else {
                        format!("{:+.0}dBFS", amplitude)
                    };

                    drop(analysis_settings);

                    let text = if let Some((pan, elapsed)) = additional {
                        format!(
                            "{:.0}hz, -{:.3}s\n{}, {:+.2} pan",
                            frequency.1,
                            elapsed.as_secs_f64(),
                            amplitude_text,
                            pan
                        )
                    } else {
                        format!("{:.0}hz, {}", frequency.1, amplitude_text)
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
                    let buffering_proportion =
                        buffering_duration.as_secs_f64() / (1.0 / PERFORMANCE_METER_TARGET_FPS);

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
                .default_open(false)
                .show(egui_ctx, |ui| {
                    let mut settings = shared_state.settings.write();

                    ui.collapsing("Render Options", |ui| {
                        if ui
                            .add(
                                egui::Slider::new(&mut settings.left_hue, 0.0..=360.0)
                                    .suffix("°")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("Left channel hue"),
                            )
                            .changed()
                        {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.right_hue, 0.0..=360.0)
                                    .suffix("°")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("Right channel hue"),
                            )
                            .changed()
                        {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.minimum_lightness, 0.0..=0.3)
                                    .text("Minimum OkLCH lightness value"),
                            )
                            .changed()
                        {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.maximum_lightness, 0.5..=1.0)
                                    .text("Maximum OkLCH lightness value"),
                            )
                            .changed()
                        {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.maximum_chroma, 0.0..=0.2)
                                    .text("Maximum OkLCH chroma value"),
                            )
                            .changed()
                        {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                        };

                        ui.add(
                            egui::Slider::new(&mut settings.max_db, 0.0..=-100.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Maximum amplitude"),
                        );

                        ui.add(
                            egui::Slider::new(&mut settings.min_db, 0.0..=-100.0)
                                .clamping(egui::SliderClamping::Never)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Minimum amplitude"),
                        );

                        let mut spectrogram_duration = settings.spectrogram_duration.as_secs_f64();
                        if ui
                            .add(
                                egui::Slider::new(&mut spectrogram_duration, 0.05..=2.0)
                                    .clamping(egui::SliderClamping::Never)
                                    .suffix("s")
                                    .text("Spectrogram duration"),
                            )
                            .changed()
                        {
                            settings.spectrogram_duration =
                                Duration::from_secs_f64(spectrogram_duration);
                        };

                        ui.add(
                            egui::Slider::new(&mut settings.bargraph_height, 0.0..=1.0)
                                .text("Bargraph height"),
                        );

                        ui.checkbox(&mut settings.show_performance, "Show performance counters");

                        if ui.button("Reset Render Options").clicked() {
                            shared_state.color_table.write().build(
                                settings.left_hue,
                                settings.right_hue,
                                settings.minimum_lightness,
                                settings.maximum_lightness,
                                settings.maximum_chroma,
                            );
                            *settings = RenderSettings::default();
                        }
                    });

                    ui.collapsing("Analysis Options", |ui| {
                        let mut render_settings = settings;
                        let mut settings = shared_state.cached_analysis_settings.lock();

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

                        ui.colored_label(
                            Color32::YELLOW,
                            "Editing these options temporarily interrupts audio analysis.",
                        );

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.gain, 20.0..=-20.0)
                                    .clamping(egui::SliderClamping::Never)
                                    .suffix("dB")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("Input gain"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        let old_min_phon =
                            (render_settings.min_db + settings.listening_volume as f32).max(0.0);
                        let old_max_phon =
                            (render_settings.max_db + settings.listening_volume as f32).min(100.0);

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.listening_volume, 100.0..=20.0)
                                    .suffix(" dB SPL")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("0dbFS output volume"),
                            )
                            .changed()
                        {
                            if settings.normalize_amplitude {
                                update_and_clear(&settings);
                                render_settings.min_db =
                                    old_min_phon - settings.listening_volume as f32;
                                render_settings.max_db =
                                    old_max_phon - settings.listening_volume as f32;
                            } else {
                                update(&settings);
                            }
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .checkbox(
                                &mut settings.normalize_amplitude,
                                "Perform amplitude normalization",
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut settings.update_rate_hz,
                                    128.0..=SPECTROGRAM_SLICES as f64,
                                )
                                .logarithmic(true)
                                .suffix("hz")
                                .step_by(128.0)
                                .fixed_decimals(0)
                                .text("Update rate"),
                            )
                            .changed()
                        {
                            update_and_clear(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut settings.resolution,
                                    32..=MAX_FREQUENCY_BINS,
                                )
                                .suffix(" bins")
                                .step_by(32.0)
                                .fixed_decimals(0)
                                .text("Resolution"),
                            )
                            .changed()
                        {
                            update_and_clear(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        let mut frame = egui::Frame::group(ui.style()).inner_margin(4.0).begin(ui);

                        {
                            let ui = &mut frame.content_ui;

                            ui.label("Frequency range");

                            if ui
                                .add(
                                    egui::DragValue::new(&mut settings.start_frequency)
                                        .suffix("hz")
                                        .fixed_decimals(0),
                                )
                                .changed()
                            {
                                if settings.start_frequency < 0.0 {
                                    settings.start_frequency = 0.0;
                                }
                                if settings.end_frequency < 0.0 {
                                    settings.end_frequency = 0.0;
                                }
                                if settings.end_frequency > settings.start_frequency {
                                    update_and_clear(&settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                }
                            };

                            if ui
                                .add(
                                    egui::DragValue::new(&mut settings.end_frequency)
                                        .speed(20.0)
                                        .suffix("hz")
                                        .fixed_decimals(0),
                                )
                                .changed()
                            {
                                if settings.start_frequency < 0.0 {
                                    settings.start_frequency = 0.0;
                                }
                                if settings.end_frequency < 0.0 {
                                    settings.end_frequency = 0.0;
                                }
                                if settings.end_frequency > settings.start_frequency {
                                    update_and_clear(&settings);
                                    egui_ctx.request_discard("Changed setting");
                                    return;
                                }
                            };
                        }
                        frame.end(ui);

                        if ui
                            .checkbox(
                                &mut settings.log_frequency_scale,
                                "Use logarithmic (non-perceptual) frequency scale",
                            )
                            .changed()
                        {
                            update_and_clear(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.time_resolution.0, 20.0..=400.0)
                                    .suffix("ms")
                                    .step_by(5.0)
                                    .fixed_decimals(0)
                                    .text("Minimum time resolution"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.time_resolution.1, 20.0..=400.0)
                                    .suffix("ms")
                                    .step_by(5.0)
                                    .fixed_decimals(0)
                                    .text("Maximum time resolution"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .checkbox(
                                &mut settings.spectral_reassignment,
                                "Use NC method (spectral reassignment based windowing)",
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }

                        if ui.button("Reset Analysis Options").clicked() {
                            *settings = AnalysisChainConfig::default();
                            update_and_clear(&settings);
                        }
                    });
                });

            *shared_state.last_frame.lock() = Instant::now();
        },
    )
}
