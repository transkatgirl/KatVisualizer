use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Align2, Color32, CornerRadius, FontId, Painter, Pos2, Rect, Vec2, Window},
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

use crate::{AnalyzerOutput, AnalyzerSet, AnalyzerSetWrapper, MyPlugin, PluginParams};

fn refresh_spectrogram_data(
    spectrogram: &mut VecDeque<(Vec<f64>, Vec<f64>, Instant)>,
    new: (&[f64], &[f64], Instant),
) {
    let (mut old_left, mut old_right, _) = spectrogram.pop_back().unwrap();

    if old_left.len() == new.0.len() {
        old_left.copy_from_slice(new.0);
    } else {
        old_left.clear();
        old_left.extend_from_slice(new.0);
    }
    if old_right.len() == new.1.len() {
        old_right.copy_from_slice(new.1);
    } else {
        old_right.clear();
        old_right.extend_from_slice(new.1);
    }
    spectrogram.push_front((old_left, old_right, new.2));
}

fn draw_bargraph(
    painter: &Painter,
    (left, right): (&[f64], &[f64]),
    bounds: Rect,
    max_db: f32,
    min_db: f32,
) {
    let bands = left.iter().zip(right.iter()).enumerate();

    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let db_height = height / (max_db - min_db);
    let band_width = width / bands.len() as f32;

    for (i, (left, right)) in bands {
        painter.rect_filled(
            Rect {
                min: Pos2 {
                    x: bounds.min.x + i as f32 * band_width,
                    y: bounds.min.y + (max_db * db_height) - (*right as f32 * db_height),
                },
                max: Pos2 {
                    x: bounds.min.x + i as f32 * band_width + band_width,
                    y: bounds.max.y,
                },
            },
            CornerRadius::ZERO,
            Color32::from_rgb(128, 0, 128),
        );
        painter.rect_filled(
            Rect {
                min: Pos2 {
                    x: bounds.min.x + i as f32 * band_width,
                    y: bounds.min.y + (max_db * db_height) - (*left as f32 * db_height),
                },
                max: Pos2 {
                    x: bounds.min.x + i as f32 * band_width + band_width,
                    y: bounds.max.y,
                },
            },
            CornerRadius::ZERO,
            Color32::from_rgb(0, 128, 128),
        );
        painter.rect_filled(
            Rect {
                min: Pos2 {
                    x: bounds.min.x + i as f32 * band_width,
                    y: bounds.min.y + (3.0 * db_height) - (left.min(*right) as f32 * db_height),
                },
                max: Pos2 {
                    x: bounds.min.x + i as f32 * band_width + band_width,
                    y: bounds.max.y,
                },
            },
            CornerRadius::ZERO,
            Color32::from_rgb(224, 224, 224),
        );
    }
}

fn draw_spectrogram(
    painter: &Painter,
    spectrogram: &VecDeque<(Vec<f64>, Vec<f64>, Instant)>,
    bounds: Rect,
    max_db: f32,
    min_db: f32,
    duration: Duration,
) {
    let width = bounds.max.x - bounds.min.x;
    let height = bounds.max.y - bounds.min.y;

    let db_color_step = 255.0 / (max_db - min_db);
    let second_height = height / duration.as_secs_f32();

    let (_, _, now) = spectrogram.front().unwrap();

    let mut last_elapsed = Duration::ZERO;

    for (left, right, timestamp) in spectrogram {
        let elapsed = now.duration_since(*timestamp);

        if elapsed > duration {
            break;
        }

        let bands = left.iter().zip(right.iter()).enumerate();

        let band_width = width / bands.len() as f32;

        for (i, (left, right)) in bands {
            let intensity =
                ((max_db - ((left + right) / 2.0) as f32) * db_color_step.round()) as u8;
            let color = Color32::from_rgb(intensity, intensity, intensity); // TODO: Improve this

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
    analyzer_output: Arc<Mutex<Output<AnalyzerOutput>>>,
    async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let last_frame = Mutex::new(Instant::now());

    let spectrogram = Mutex::new(VecDeque::from(vec![
        (
            Vec::with_capacity(24000),
            Vec::with_capacity(24000),
            Instant::now(),
        );
        256
    ]));

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

                    let mut spectrogram = spectrogram.lock().unwrap();

                    let (processing_duration, buffering_duration) = {
                        let mut lock = analyzer_output.lock().unwrap();
                        let (left, right, processing_duration, timestamp) = lock.read();

                        let processing_duration = *processing_duration;
                        let buffering_duration = start.duration_since(*timestamp);

                        refresh_spectrogram_data(&mut spectrogram, (left, right, *timestamp));

                        (processing_duration, buffering_duration)
                    };

                    let (left, right, _) = spectrogram.front().unwrap();

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
                        3.0,
                        -80.0,
                    );

                    draw_spectrogram(
                        painter,
                        &spectrogram,
                        Rect {
                            min: Pos2 {
                                x: 0.0,
                                y: max_y / 2.0,
                            },
                            max: Pos2 { x: max_x, y: max_y },
                        },
                        3.0,
                        -80.0,
                        Duration::from_millis(500),
                    );

                    if buffering_duration < Duration::from_millis(500) {
                        painter.text(
                            Pos2 {
                                x: max_x - 32.0,
                                y: 64.0,
                            },
                            Align2::RIGHT_BOTTOM,
                            format!(
                                "{:.1}ms processing",
                                processing_duration.as_secs_f64() * 1000.0
                            ),
                            FontId {
                                size: 12.0,
                                family: egui::FontFamily::Monospace,
                            },
                            if processing_duration > Duration::from_millis(4) {
                                Color32::RED
                            } else if processing_duration > Duration::from_millis(3) {
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
