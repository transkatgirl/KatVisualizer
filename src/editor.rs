use color::{ColorSpaceTag, DynamicColor, Flags, Rgba8, Srgb};
use nih_plug::prelude::*;
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, Align2, Color32, CornerRadius, FontId, Painter, Pos2, Rect},
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    AnalysisChain, AnalysisChainConfig, AnalysisMetrics, MAX_FREQUENCY_BINS, MyPlugin,
    PluginParams, SPECTROGRAM_SLICES,
    analyzer::{BetterAnalysis, BetterSpectrogram, map_value_f32},
};

fn convert_dynamic_color(color: DynamicColor) -> Color32 {
    let converted: Rgba8 = color.to_alpha_color::<Srgb>().to_rgba8();
    Color32::from_rgba_unmultiplied(converted.r, converted.g, converted.b, converted.a)
}

fn draw_bargraph<F>(
    painter: &Painter,
    analysis: &BetterAnalysis,
    bounds: Rect,
    color: &F,
    (max_db, min_db): (f32, f32),
) where
    F: Fn(f32, f32) -> Color32,
{
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let band_width = width / analysis.data.len() as f32;

    for (i, (pan, volume)) in analysis.data.iter().enumerate() {
        let intensity = map_value_f32(*volume, min_db, max_db, 0.0, 1.0);
        let color = color(*pan, intensity);

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
    spectrogram: &BetterSpectrogram,
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

    for (analysis, length) in &spectrogram.data {
        let elapsed = last_elapsed + *length;

        if elapsed > duration {
            break;
        }

        let band_width = width / analysis.data.len() as f32;

        for (i, (pan, volume)) in analysis.data.iter().enumerate() {
            let intensity = map_value_f32(*volume, min_db, max_db, 0.0, 1.0);
            let color = color(*pan, intensity);

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
            minimum_lightness: 0.13,
            min_db: -75.0,
            max_db: -5.0,
            bargraph_height: 0.4,
            spectrogram_duration: Duration::from_secs_f64(0.33),
            show_performance: true,
        }
    }
}

pub fn create(
    params: Arc<PluginParams>,
    analysis_chain: Arc<Mutex<Option<AnalysisChain>>>,
    main_spectrogram: Arc<Mutex<(BetterSpectrogram, AnalysisMetrics)>>,
    _async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let last_frame = Mutex::new(Instant::now());

    let settings = Mutex::new(RenderSettings::default());

    let cached_analysis_settings = Mutex::new(AnalysisChainConfig::default());

    let render_spectrogram = Mutex::new(BetterSpectrogram::new(
        SPECTROGRAM_SLICES,
        MAX_FREQUENCY_BINS,
    ));

    create_egui_editor(
        egui_state.clone(),
        (),
        |_, _| {},
        move |egui_ctx, _setter, _state| {
            egui::CentralPanel::default().show(egui_ctx, |ui| {
                egui_ctx.request_repaint();

                let start = Instant::now();

                let painter = ui.painter();
                let max_x = painter.clip_rect().max.x;
                let max_y = painter.clip_rect().max.y;
                let settings = *settings.lock().unwrap();

                let (spectrogram, metrics) = {
                    let shared = main_spectrogram.lock().unwrap();
                    let mut render = render_spectrogram.lock().unwrap();
                    render.clone_from(&shared.0);
                    (render, shared.1)
                };

                let front = spectrogram.data.front().unwrap();

                let buffering_duration = start.duration_since(metrics.finished);
                let processing_duration = metrics.processing;
                let chunk_duration = front.1;

                let color_function = color_function(&settings);

                if settings.bargraph_height != 0.0 {
                    draw_bargraph(
                        painter,
                        &front.0,
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
                        &spectrogram,
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

                drop(spectrogram);

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
                        if frame_elapsed >= Duration::from_secs_f64(1.0 / 30.0) {
                            Color32::RED
                        } else if frame_elapsed >= Duration::from_secs_f64(1.0 / 55.0) {
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
                        if draw_elapsed >= Duration::from_secs_f64(1.0 / (60.0 * 4.0)) {
                            Color32::RED
                        } else if draw_elapsed >= Duration::from_secs_f64(1.0 / (60.0 * 8.0)) {
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

                        let mut frame = egui::Frame::group(ui.style()).inner_margin(4.0).begin(ui);

                        {
                            let ui = &mut frame.content_ui;

                            ui.label("Frequency range");

                            if ui
                                .add(
                                    egui::DragValue::new(&mut settings.start_frequency)
                                        .range(0.0..=22049.0)
                                        .suffix("hz")
                                        .fixed_decimals(0),
                                )
                                .changed()
                            {
                                settings.end_frequency =
                                    settings.end_frequency.max(settings.start_frequency + 1.0);
                                update(&settings);
                                egui_ctx.request_discard("Changed setting");
                                return;
                            };

                            if ui
                                .add(
                                    egui::DragValue::new(&mut settings.end_frequency)
                                        .range(1.0..=22050.0)
                                        .speed(20.0)
                                        .suffix("hz")
                                        .fixed_decimals(0),
                                )
                                .changed()
                            {
                                settings.start_frequency =
                                    settings.start_frequency.min(settings.end_frequency - 1.0);
                                update(&settings);
                                egui_ctx.request_discard("Changed setting");
                                return;
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
                            update(&settings);
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

                        if ui.button("Reset Analysis Options").clicked() {
                            *settings = AnalysisChainConfig::default();
                            update(&settings);
                        }
                    });
                });
        },
    )
}
