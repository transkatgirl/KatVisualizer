use color::{
    AlphaColor, ColorSpace, ColorSpaceTag, DynamicColor, Flags, Interpolator, Rgba8, Srgb,
};
use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Align2, Color32, CornerRadius, FontId, Painter, Pos2, Rect, Rgba, Vec2, Window},
    resizable_window::ResizableWindow,
    widgets,
};
use std::{
    collections::{HashMap, VecDeque},
    mem,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use triple_buffer::Output;

use crate::{
    AnalyzerOutput, AnalyzerSet, AnalyzerSetWrapper, MyPlugin, PluginParams, Spectrogram,
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
    color: F,
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
    spectrogram: &VecDeque<(Vec<f64>, Vec<f64>, Duration, Instant)>,
    bounds: Rect,
    color: F,
    (max_db, min_db): (f32, f32),
    duration: Duration,
) where
    F: Fn(f32, f32) -> Color32,
{
    painter.rect_filled(bounds, CornerRadius::ZERO, Color32::BLACK);

    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let second_height = height / duration.as_secs_f32();

    //let (_, _, _, now) = spectrogram.front().unwrap();

    let mut last_elapsed = Duration::ZERO;

    for (left, right, _, _timestamp) in spectrogram {
        //let elapsed = now.duration_since(*timestamp);
        let elapsed = last_elapsed + Duration::from_secs_f64(1.0 / 256.0);

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

pub fn create(
    params: Arc<PluginParams>,
    analyzers: AnalyzerSetWrapper,
    analyzer_output: Arc<Mutex<Output<Spectrogram>>>,
    async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let last_frame = Mutex::new(Instant::now());

    // .54, .9
    // .42, .19

    let left_color = DynamicColor {
        cs: ColorSpaceTag::Oklch,
        flags: Flags::default(),
        components: [0.7, 0.16, 195.0, 1.0],
    };
    /*let middle_color = DynamicColor {
        cs: ColorSpaceTag::Oklch,
        flags: Flags::default(),
        components: [0.96, 0.0, 0.0, 1.0],
    };*/
    let right_color = DynamicColor {
        cs: ColorSpaceTag::Oklch,
        flags: Flags::default(),
        components: [0.7, 0.16, 328.0, 1.0],
    };
    /*let left_color_converted = convert_dynamic_color(left_color);
    let middle_color_converted = convert_dynamic_color(middle_color);
    let right_color_converted = convert_dynamic_color(right_color);
    let left_right_color = left_color.interpolate(
        right_color,
        ColorSpaceTag::Oklch,
        color::HueDirection::Shorter,
    );
    let left_middle_color = left_color.interpolate(
        middle_color,
        ColorSpaceTag::Oklch,
        color::HueDirection::Shorter,
    );
    let right_middle_color = right_color.interpolate(
        middle_color,
        ColorSpaceTag::Oklch,
        color::HueDirection::Shorter,
    );*/

    let color_function = move |split: f32, intensity: f32| -> Color32 {
        let mut color = if split >= 0.0 {
            let mut color = right_color;
            color.components[1] = map_value_f32(split, 0.0, 1.0, 0.0, color.components[1]);
            color
        } else {
            let mut color = left_color;
            color.components[1] = map_value_f32(-split, 0.0, 1.0, 0.0, color.components[1]);
            color
        };

        color.components[0] = map_value_f32(intensity, 0.1, 1.0, 0.0, color.components[0]);
        color.components[1] = map_value_f32(intensity, 0.05, 1.0, 0.0, color.components[1]);

        convert_dynamic_color(color)
    };

    create_egui_editor(
        egui_state.clone(),
        (),
        |_, _| {},
        move |egui_ctx, setter, _state| {
            ResizableWindow::new("res-wind")
                .min_size(Vec2::new(128.0, 128.0))
                .show(egui_ctx, egui_state.as_ref(), |ui| {
                    let start = Instant::now();

                    let painter = ui.painter();

                    let mut lock = analyzer_output.lock().unwrap();
                    let spectrogram = lock.read();

                    let (left, right, processing_duration, timestamp) =
                        spectrogram.front().unwrap();

                    let buffering_duration = start.duration_since(*timestamp);

                    let max_x = painter.clip_rect().max.x;
                    let max_y = painter.clip_rect().max.y;

                    draw_bargraph(
                        painter,
                        (left, right),
                        Rect {
                            min: Pos2 { x: 0.0, y: 0.0 },
                            max: Pos2 {
                                x: max_x,
                                y: max_y / 2.0,
                            },
                        },
                        color_function,
                        (0.0, -80.0),
                    );

                    draw_spectrogram(
                        painter,
                        spectrogram,
                        Rect {
                            min: Pos2 {
                                x: 0.0,
                                y: max_y / 2.0,
                            },
                            max: Pos2 { x: max_x, y: max_y },
                        },
                        color_function,
                        (0.0, -80.0),
                        Duration::from_millis(333),
                    );

                    if buffering_duration < Duration::from_millis(500) {
                        let processing_proportion =
                            processing_duration.as_secs_f64() / (1.0 / 256.0);

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
                                processing_duration.as_secs_f64() * 1000.0
                            ),
                            FontId {
                                size: 12.0,
                                family: egui::FontFamily::Monospace,
                            },
                            if buffering_duration > Duration::from_millis(12) {
                                Color32::RED
                            } else if buffering_duration > Duration::from_millis(8) {
                                Color32::YELLOW
                            } else {
                                Color32::from_rgb(224, 224, 224)
                            },
                        );
                    }

                    let mut last_frame = last_frame.lock().unwrap();
                    let now = Instant::now();
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

                    *last_frame = now;

                    // TODO
                });
        },
    )
}
