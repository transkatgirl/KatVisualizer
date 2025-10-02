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

                    let max_y = painter.clip_rect().max.y;

                    for (i, (left, right)) in bands {
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * 5.0,
                                    y: (5.0 * 10.0) - (*left as f32 * 5.0),
                                },
                                max: Pos2 {
                                    x: (i as f32 * 5.0) + 5.0,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::BLUE,
                        );
                        painter.rect_filled(
                            Rect {
                                min: Pos2 {
                                    x: i as f32 * 5.0,
                                    y: (5.0 * 10.0) - (*right as f32 * 5.0),
                                },
                                max: Pos2 {
                                    x: (i as f32 * 5.0) + 5.0,
                                    y: max_y,
                                },
                            },
                            CornerRadius::ZERO,
                            Color32::RED,
                        );
                    }

                    // TODO
                });
        },
    )
}
