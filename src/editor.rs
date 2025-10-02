use nih_plug::{prelude::*, util::StftHelper};
use nih_plug_egui::{
    EguiState, create_egui_editor,
    egui::{self, Color32, CornerRadius, Pos2, Rect, Vec2, Window},
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

    create_egui_editor(
        egui_state.clone(),
        (),
        |_, _| {},
        move |egui_ctx, setter, _state| {
            ResizableWindow::new("res-wind")
                .min_size(Vec2::new(128.0, 128.0))
                .show(egui_ctx, egui_state.as_ref(), |ui| {
                    let painter = ui.painter();

                    let mut lock = analyzer_output.lock().unwrap();
                    let (left, right) = lock.read();

                    let bands = left.iter().zip(right.iter()).enumerate();

                    let band_width = painter.clip_rect().max.x / bands.len() as f32;
                    let max_y = painter.clip_rect().max.y;

                    for (i, (left, right)) in bands {
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (5.0 * 10.0) - (*right as f32 * 5.0),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgba_unmultiplied(255, 0, 255, 64),
                        );
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (5.0 * 10.0) - (*left as f32 * 5.0),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgba_unmultiplied(0, 255, 255, 64),
                        );

                        /*painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * band_width,
                                    y: (5.0 * 10.0) - (left.min(*right) as f32 * 5.0),
                                },
                                max: Pos2 {
                                    x: (i as f32 * band_width) + band_width,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::from_rgb(255, 255, 255),
                        );*/
                    }

                    // TODO
                });
        },
    )
}
