use color::{ColorSpaceTag, DynamicColor, Flags, Rgba8, Srgb};
use nih_plug::prelude::*;
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Align2, Color32, CornerRadius, FontId, Painter, Pos2, Rect, Vec2, Window},
    widgets,
};
use std::{
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};
use triple_buffer::Output;

use crate::{
    AnalysisChain, AnalysisChainConfig, MyPlugin, PluginParams, Spectrogram,
    analyzer::{calculate_pan_and_volume, map_value_f32},
};

fn convert_dynamic_color(color: DynamicColor) -> Color32 {
    let converted: Rgba8 = color.to_alpha_color::<Srgb>().to_rgba8();
    Color32::from_rgba_unmultiplied(converted.r, converted.g, converted.b, converted.a)
}

fn draw_bargraph<F>(
    painter: &Painter,
    (left, right): (&[f64], &[f64]),
    bounds: Rect,
    color: &F,
    (max_db, min_db): (f32, f32),
) where
    F: Fn(f32, f32) -> Color32,
{
    let bands = left.iter().zip(right.iter()).enumerate();

    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let band_width = width / bands.len() as f32;

    for (i, (left, right)) in bands {
        let (pan, volume) = calculate_pan_and_volume(*left, *right);
        let intensity = map_value_f32(volume as f32, min_db, max_db, 0.0, 1.0);
        let color = color(pan as f32, intensity);

        painter.rect_filled(
            Rect {
                min: Pos2 {
                    x: bounds.min.x + i as f32 * band_width,
                    y: bounds.max.y - intensity * height,
                },
                max: Pos2 {
                    x: bounds.min.x + i as f32 * band_width + band_width,
                    y: bounds.max.y,
                },
            },
            CornerRadius::ZERO,
            color,
        );
    }
}

fn draw_spectrogram<F>(
    painter: &Painter,
    spectrogram: &Spectrogram,
    bounds: Rect,
    color: &F,
    (max_db, min_db): (f32, f32),
    duration: Duration,
) where
    F: Fn(f32, f32) -> Color32,
{
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let second_height = height / duration.as_secs_f32();

    let mut last_elapsed = Duration::ZERO;

    for (left, right, _, _timestamp, length) in spectrogram {
        let elapsed = last_elapsed + *length;

        if elapsed > duration {
            break;
        }

        let bands = left.iter().zip(right.iter()).enumerate();

        let band_width = width / bands.len() as f32;

        for (i, (left, right)) in bands {
            let (pan, volume) = calculate_pan_and_volume(*left, *right);
            let intensity = map_value_f32(volume as f32, min_db, max_db, 0.0, 1.0);
            let color = color(pan as f32, intensity);

            painter.rect_filled(
                Rect {
                    min: Pos2 {
                        x: bounds.min.x + i as f32 * band_width,
                        y: bounds.min.y + last_elapsed.as_secs_f32() * second_height,
                    },
                    max: Pos2 {
                        x: bounds.min.x + i as f32 * band_width + band_width,
                        y: bounds.min.y + (elapsed.as_secs_f32() * second_height).max(bounds.max.y),
                    },
                },
                CornerRadius::ZERO,
                color,
            );
        }

        last_elapsed = elapsed;
    }
}

const MAX_LIGHTNESS: f32 = 0.72;
const MAX_CHROMA: f32 = 0.12;

fn color_function(settings: &RenderSettings) -> impl Fn(f32, f32) -> Color32 {
    let left_color = DynamicColor {
        cs: ColorSpaceTag::Oklch,
        flags: Flags::default(),
        components: [MAX_LIGHTNESS, MAX_CHROMA, settings.left_hue, 1.0],
    };
    let right_color = DynamicColor {
        cs: ColorSpaceTag::Oklch,
        flags: Flags::default(),
        components: [MAX_LIGHTNESS, MAX_CHROMA, settings.right_hue, 1.0],
    };
    let minimum_lightness = settings.minimum_lightness;

    move |split: f32, intensity: f32| -> Color32 {
        if intensity - f32::EPSILON <= 0.0 {
            return Color32::BLACK;
        }

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
            map_value_f32(intensity, 0.0, 1.0, minimum_lightness, color.components[0]);

        convert_dynamic_color(color)
    }
}

#[derive(Clone, Copy)]
struct RenderSettings {
    left_hue: f32,
    right_hue: f32,
    minimum_lightness: f32,
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
            minimum_lightness: 0.14,
            min_db: -75.0,
            max_db: -5.0,
            bargraph_height: 0.4,
            spectrogram_duration: Duration::from_millis(333),
            show_performance: true,
        }
    }
}

pub fn create(
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    analyzer_output: Arc<Mutex<Output<Spectrogram>>>,
    async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let last_frame = Mutex::new(Instant::now());

    let settings = Mutex::new(RenderSettings::default());

    let cached_analysis_settings = Mutex::new(AnalysisChainConfig::default());

    create_egui_editor(
        egui_state.clone(),
        (),
        |_, _| {},
        move |egui_ctx, setter, _state| {
            egui::CentralPanel::default().show(egui_ctx, |ui| {
                egui_ctx.request_repaint();

                let start = Instant::now();

                let painter = ui.painter();
                let max_x = painter.clip_rect().max.x;
                let max_y = painter.clip_rect().max.y;
                let settings = *settings.lock().unwrap();

                let mut lock = analyzer_output.lock().unwrap();
                let spectrogram = lock.read();

                let (left, right, processing_duration, timestamp, chunk_duration) =
                    spectrogram.front().unwrap();

                let buffering_duration = start.duration_since(*timestamp);
                let processing_duration = *processing_duration;
                let chunk_duration = *chunk_duration;

                let color_function = color_function(&settings);

                if settings.bargraph_height != 0.0 {
                    draw_bargraph(
                        painter,
                        (left, right),
                        Rect {
                            min: Pos2 { x: 0.0, y: 0.0 },
                            max: Pos2 {
                                x: max_x,
                                y: max_y * settings.bargraph_height,
                            },
                        },
                        &color_function,
                        (settings.max_db, settings.min_db),
                    );
                }

                if settings.bargraph_height != 1.0 {
                    draw_spectrogram(
                        painter,
                        spectrogram,
                        Rect {
                            min: Pos2 {
                                x: 0.0,
                                y: max_y * settings.bargraph_height,
                            },
                            max: Pos2 { x: max_x, y: max_y },
                        },
                        &color_function,
                        (settings.max_db, settings.min_db),
                        settings.spectrogram_duration,
                    );
                }

                drop(lock);

                if settings.show_performance && buffering_duration < Duration::from_millis(500) {
                    let processing_proportion =
                        processing_duration.as_secs_f64() / chunk_duration.as_secs_f64();

                    painter.text(
                        Pos2 {
                            x: max_x - 32.0,
                            y: 64.0,
                        },
                        Align2::RIGHT_BOTTOM,
                        format!(
                            "{:.0}% ({:.1}ms) processing",
                            processing_proportion * 100.0,
                            processing_duration.as_secs_f64() * 1000.0,
                        ),
                        FontId {
                            size: 12.0,
                            family: egui::FontFamily::Monospace,
                        },
                        if processing_proportion >= 1.0 {
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
                            y: 80.0,
                        },
                        Align2::RIGHT_BOTTOM,
                        format!(
                            "{:.1}ms buffering",
                            buffering_duration.as_secs_f64() * 1000.0
                        ),
                        FontId {
                            size: 12.0,
                            family: egui::FontFamily::Monospace,
                        },
                        if buffering_duration >= chunk_duration.mul_f64(3.0) {
                            Color32::RED
                        } else if buffering_duration >= chunk_duration.mul_f64(2.0) {
                            Color32::YELLOW
                        } else {
                            Color32::from_rgb(224, 224, 224)
                        },
                    );
                }

                let mut last_frame = last_frame.lock().unwrap();
                let now = Instant::now();
                if settings.show_performance {
                    let frame_elapsed = now.duration_since(*last_frame);
                    painter.text(
                        Pos2 {
                            x: max_x - 32.0,
                            y: 32.0,
                        },
                        Align2::RIGHT_BOTTOM,
                        format!("{:2}ms frame", frame_elapsed.as_millis()),
                        FontId {
                            size: 12.0,
                            family: egui::FontFamily::Monospace,
                        },
                        if frame_elapsed > Duration::from_millis(33) {
                            Color32::RED
                        } else if frame_elapsed > Duration::from_millis(18) {
                            Color32::YELLOW
                        } else {
                            Color32::from_rgb(224, 224, 224)
                        },
                    );
                    let draw_elapsed = now.duration_since(start);
                    painter.text(
                        Pos2 {
                            x: max_x - 32.0,
                            y: 48.0,
                        },
                        Align2::RIGHT_BOTTOM,
                        format!("{:.1}ms composite", draw_elapsed.as_secs_f64() * 1000.0),
                        FontId {
                            size: 12.0,
                            family: egui::FontFamily::Monospace,
                        },
                        if draw_elapsed > Duration::from_millis(4) {
                            Color32::RED
                        } else if draw_elapsed > Duration::from_millis(2) {
                            Color32::YELLOW
                        } else {
                            Color32::from_rgb(224, 224, 224)
                        },
                    );
                }

                *last_frame = now;
            });
            /*egui::TopBottomPanel::bottom("my_panel").show(egui_ctx, move |ui| {
                ui.label("Hello World!");
            });*/
            egui::Window::new("Settings")
                .id(egui::Id::new("settings"))
                .default_open(false)
                .show(egui_ctx, |ui| {
                    ui.collapsing("Render Options", |ui| {
                        let mut settings = settings.lock().unwrap();

                        let mut spectrogram_duration = settings.spectrogram_duration.as_secs_f64();

                        ui.add(
                            egui::Slider::new(&mut settings.left_hue, 0.0..=360.0)
                                .suffix("°")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Left channel hue"),
                        );

                        ui.add(
                            egui::Slider::new(&mut settings.right_hue, 0.0..=360.0)
                                .suffix("°")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Right channel hue"),
                        );

                        ui.add(
                            egui::Slider::new(&mut settings.minimum_lightness, 0.0..=0.3)
                                .text("Minimum OkLCH lightness value"),
                        );

                        ui.add(
                            egui::Slider::new(&mut settings.max_db, 0.0..=-75.0)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Maximum (normalized) amplitude"),
                        );

                        ui.add(
                            egui::Slider::new(&mut settings.min_db, 0.0..=-75.0)
                                .suffix("dB")
                                .step_by(1.0)
                                .fixed_decimals(0)
                                .text("Minimum (normalized) amplitude"),
                        );

                        if ui
                            .add(
                                egui::Slider::new(&mut spectrogram_duration, 0.05..=1.0)
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
                            *settings = RenderSettings::default();
                        }
                    });

                    ui.collapsing("Analysis Options", |ui| {
                        let mut settings = cached_analysis_settings.lock().unwrap();
                        let update = |settings| {
                            let mut lock = analysis_chain.lock().unwrap();
                            let analysis_chain = lock.as_mut().unwrap();
                            analysis_chain.update_config(settings);
                        };

                        ui.colored_label(
                            Color32::YELLOW,
                            "Editing these options may temporarily interrupt playback.",
                        );

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.gain, 20.0..=-20.0)
                                    .suffix("dB")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("Gain"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

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
                            update(&settings);
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
                                egui::Slider::new(&mut settings.update_rate_hz, 64.0..=8192.0)
                                    .logarithmic(true)
                                    .suffix("hz")
                                    .step_by(64.0)
                                    .fixed_decimals(0)
                                    .text("Update rate"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.resolution, 32..=1024)
                                    .suffix(" bins")
                                    .step_by(32.0)
                                    .fixed_decimals(0)
                                    .text("Resolution"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui
                            .checkbox(
                                &mut settings.log_frequency_scale,
                                "Use logarithmic (non-perceptual) frequency scale",
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        }

                        if ui
                            .add(
                                egui::Slider::new(&mut settings.time_resolution.0, 50.0..=200.0)
                                    .suffix("ms")
                                    .step_by(1.0)
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
                                egui::Slider::new(&mut settings.time_resolution.1, 50.0..=400.0)
                                    .suffix("ms")
                                    .step_by(1.0)
                                    .fixed_decimals(0)
                                    .text("Maximum time resolution"),
                            )
                            .changed()
                        {
                            update(&settings);
                            egui_ctx.request_discard("Changed setting");
                            return;
                        };

                        if ui.button("Reset Analysis Options").clicked() {
                            *settings = AnalysisChainConfig::default();
                            update(&settings);
                        }
                    });

                    /*ui.group(|ui| {
                        ui.label("Within a frame");
                    });*/
                });
        },
    )
}
