use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Vec2, Window},
    resizable_window::ResizableWindow,
    widgets,
};
use std::{
    collections::HashMap,
    mem,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use triple_buffer::Output;

use crate::{AnalyzerOutput, AnalyzerSet, AnalyzerSetWrapper, MyPlugin, PluginParams};

pub fn create(
    params: Arc<PluginParams>,
    analyzers: AnalyzerSetWrapper,
    analyzer_output: Arc<Mutex<Output<AnalyzerOutput>>>,
    async_executor: AsyncExecutor<MyPlugin>,
) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();

    let last_frame = Mutex::new(Instant::now());

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
                    let (left, right, processing_duration, timestamp) = lock.read();

                    let buffering_duration = start.duration_since(*timestamp);

                    let bands = left.iter().zip(right.iter()).enumerate();

                    let max_x = painter.clip_rect().max.x;
                    let max_y = painter.clip_rect().max.y;
                    let db_width = painter.clip_rect().max.y / 80.0;
                    let band_width = painter.clip_rect().max.x / bands.len() as f32;

                    for (i, (left, right)) in bands {
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (3.0 * db_width) - (*right as f32 * db_width),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgb(128, 0, 128),
                        );
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (3.0 * db_width) - (*left as f32 * db_width),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgb(0, 128, 128),
                        );
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (3.0 * db_width) - (left.min(*right) as f32 * db_width),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgb(224, 224, 224),
                        );
                    }

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
                            if *processing_duration > Duration::from_millis(4) {
                                Color32::RED
                            } else if *processing_duration > Duration::from_millis(3) {
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
