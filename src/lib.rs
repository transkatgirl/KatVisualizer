use baseview::{WindowHandle, WindowOpenOptions, WindowScalePolicy};
use crossbeam::atomic::AtomicCell;
use nih_plug::params::persist::PersistentField;
use nih_plug::prelude::*;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    num::NonZeroIsize,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use wgpu::SurfaceTargetUnsafe;

mod analyzer;

/// The time it takes for the peak meter to decay by 12 dB after switching to complete silence.
const PEAK_METER_DECAY_MS: f64 = 150.0;

pub struct CustomWgpuWindow {
    gui_context: Arc<dyn GuiContext>,

    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    #[allow(unused)]
    params: Arc<MyPluginParams>,
    #[allow(unused)]
    peak_meter: Arc<AtomicF32>,
}

impl CustomWgpuWindow {
    fn new(
        window: &mut baseview::Window<'_>,
        gui_context: Arc<dyn GuiContext>,
        params: Arc<MyPluginParams>,
        peak_meter: Arc<AtomicF32>,
        scaling_factor: f32,
    ) -> Self {
        let target = baseview_window_to_surface_target(window);

        pollster::block_on(Self::create(
            target,
            gui_context,
            params,
            peak_meter,
            scaling_factor,
        ))
    }

    async fn create(
        target: SurfaceTargetUnsafe,
        gui_context: Arc<dyn GuiContext>,
        params: Arc<MyPluginParams>,
        peak_meter: Arc<AtomicF32>,
        scaling_factor: f32,
    ) -> Self {
        let (unscaled_width, unscaled_height) = params.editor_state.size();
        let width = (unscaled_width as f64 * scaling_factor as f64).round() as u32;
        let height = (unscaled_height as f64 * scaling_factor as f64).round() as u32;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface = unsafe { instance.create_surface_unsafe(target) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        const SHADER: &str = "
            const VERTS = array(
                vec2<f32>(0.5, 1.0),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(1.0, 0.0)
            );

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) position: vec2<f32>,
            };

            @vertex
            fn vs_main(
                @builtin(vertex_index) in_vertex_index: u32,
            ) -> VertexOutput {
                var out: VertexOutput;
                out.position = VERTS[in_vertex_index];
                out.clip_position = vec4<f32>(out.position - 0.5, 0.0, 1.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(in.position, 0.5, 1.0);
            }
            ";

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &surface_config);

        Self {
            gui_context,
            device,
            queue,
            pipeline,
            surface,
            surface_config,
            params,
            peak_meter,
        }
    }
}

impl baseview::WindowHandler for CustomWgpuWindow {
    fn on_frame(&mut self, _window: &mut baseview::Window) {
        // Do rendering here.

        let frame = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn on_event(
        &mut self,
        _window: &mut baseview::Window,
        event: baseview::Event,
    ) -> baseview::EventStatus {
        // Use this to set parameter values.
        let _param_setter = ParamSetter::new(self.gui_context.as_ref());

        match &event {
            // Do event processing here.
            baseview::Event::Window(event) => match event {
                baseview::WindowEvent::Resized(window_info) => {
                    self.params.editor_state.size.store((
                        window_info.logical_size().width.round() as u32,
                        window_info.logical_size().height.round() as u32,
                    ));

                    self.surface_config.width = window_info.physical_size().width;
                    self.surface_config.height = window_info.physical_size().height;

                    self.surface.configure(&self.device, &self.surface_config);
                }
                _ => {}
            },
            _ => {}
        }

        baseview::EventStatus::Captured
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomWgpuEditorState {
    /// The window's size in logical pixels before applying `scale_factor`.
    #[serde(with = "nih_plug::params::persist::serialize_atomic_cell")]
    size: AtomicCell<(u32, u32)>,
    /// Whether the editor's window is currently open.
    #[serde(skip)]
    open: AtomicBool,
}

impl CustomWgpuEditorState {
    pub fn from_size(size: (u32, u32)) -> Arc<Self> {
        Arc::new(Self {
            size: AtomicCell::new(size),
            open: AtomicBool::new(false),
        })
    }

    /// Returns a `(width, height)` pair for the current size of the GUI in logical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size.load()
    }

    /// Whether the GUI is currently visible.
    // Called `is_open()` instead of `open()` to avoid the ambiguity.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }
}

impl<'a> PersistentField<'a, CustomWgpuEditorState> for Arc<CustomWgpuEditorState> {
    fn set(&self, new_value: CustomWgpuEditorState) {
        self.size.store(new_value.size.load());
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&CustomWgpuEditorState) -> R,
    {
        f(self)
    }
}

pub struct CustomWgpuEditor {
    params: Arc<MyPluginParams>,
    peak_meter: Arc<AtomicF32>,

    /// The scaling factor reported by the host, if any. On macOS this will never be set and we
    /// should use the system scaling factor instead.
    scaling_factor: AtomicCell<Option<f32>>,
}

impl Editor for CustomWgpuEditor {
    fn spawn(
        &self,
        parent: ParentWindowHandle,
        context: Arc<dyn GuiContext>,
    ) -> Box<dyn std::any::Any + Send> {
        let (unscaled_width, unscaled_height) = self.params.editor_state.size();
        let scaling_factor = self.scaling_factor.load();

        let gui_context = Arc::clone(&context);

        let params = Arc::clone(&self.params);
        let peak_meter = Arc::clone(&self.peak_meter);

        let window = baseview::Window::open_parented(
            &ParentWindowHandleAdapter(parent),
            WindowOpenOptions {
                title: String::from("WGPU Window"),
                // Baseview should be doing the DPI scaling for us
                size: baseview::Size::new(unscaled_width as f64, unscaled_height as f64),
                // NOTE: For some reason passing 1.0 here causes the UI to be scaled on macOS but
                //       not the mouse events.
                scale: scaling_factor
                    .map(|factor| WindowScalePolicy::ScaleFactor(factor as f64))
                    .unwrap_or(WindowScalePolicy::SystemScaleFactor),

                // NOTE: The OpenGL feature in baseview is not needed here, but rust-analyzer gets
                // confused when some crates do use it and others don't.
                gl_config: None,
            },
            move |window: &mut baseview::Window<'_>| -> CustomWgpuWindow {
                CustomWgpuWindow::new(
                    window,
                    gui_context,
                    params,
                    peak_meter,
                    scaling_factor.unwrap_or(1.0),
                )
            },
        );

        self.params.editor_state.open.store(true, Ordering::Release);
        Box::new(CustomWgpuEditorHandle {
            state: self.params.editor_state.clone(),
            window,
        })
    }

    fn size(&self) -> (u32, u32) {
        self.params.editor_state.size()
    }

    fn set_scale_factor(&self, factor: f32) -> bool {
        // If the editor is currently open then the host must not change the current HiDPI scale as
        // we don't have a way to handle that. Ableton Live does this.
        if self.params.editor_state.is_open() {
            return false;
        }

        self.scaling_factor.store(Some(factor));
        true
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        // As mentioned above, for now we'll always force a redraw to allow meter widgets to work
        // correctly. In the future we can use an `Arc<AtomicBool>` and only force a redraw when
        // that boolean is set.
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {}

    fn param_values_changed(&self) {
        // Same
    }
}

/// The window handle used for [`CustomWgpuEditor`].
struct CustomWgpuEditorHandle {
    state: Arc<CustomWgpuEditorState>,
    window: WindowHandle,
}

/// The window handle enum stored within 'WindowHandle' contains raw pointers. Is there a way around
/// having this requirement?
unsafe impl Send for CustomWgpuEditorHandle {}

impl Drop for CustomWgpuEditorHandle {
    fn drop(&mut self) {
        self.state.open.store(false, Ordering::Release);
        // XXX: This should automatically happen when the handle gets dropped, but apparently not
        self.window.close();
    }
}

/// This version of `baseview` uses a different version of `raw_window_handle than NIH-plug, so we
/// need to adapt it ourselves.
struct ParentWindowHandleAdapter(nih_plug::editor::ParentWindowHandle);

unsafe impl HasRawWindowHandle for ParentWindowHandleAdapter {
    fn raw_window_handle(&self) -> RawWindowHandle {
        match self.0 {
            ParentWindowHandle::X11Window(window) => {
                let mut handle = raw_window_handle::XcbWindowHandle::empty();
                handle.window = window;
                RawWindowHandle::Xcb(handle)
            }
            ParentWindowHandle::AppKitNsView(ns_view) => {
                let mut handle = raw_window_handle::AppKitWindowHandle::empty();
                handle.ns_view = ns_view;
                RawWindowHandle::AppKit(handle)
            }
            ParentWindowHandle::Win32Hwnd(hwnd) => {
                let mut handle = raw_window_handle::Win32WindowHandle::empty();
                handle.hwnd = hwnd;
                RawWindowHandle::Win32(handle)
            }
        }
    }
}

/// WGPU uses raw_window_handle v6, but baseview uses raw_window_handle v5, so manually convert it.
fn baseview_window_to_surface_target(window: &baseview::Window<'_>) -> wgpu::SurfaceTargetUnsafe {
    use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

    let raw_display_handle = window.raw_display_handle();
    let raw_window_handle = window.raw_window_handle();

    wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: match raw_display_handle {
            raw_window_handle::RawDisplayHandle::AppKit(_) => {
                raw_window_handle_06::RawDisplayHandle::AppKit(
                    raw_window_handle_06::AppKitDisplayHandle::new(),
                )
            }
            raw_window_handle::RawDisplayHandle::Xlib(handle) => {
                raw_window_handle_06::RawDisplayHandle::Xlib(
                    raw_window_handle_06::XlibDisplayHandle::new(
                        NonNull::new(handle.display),
                        handle.screen,
                    ),
                )
            }
            raw_window_handle::RawDisplayHandle::Xcb(handle) => {
                raw_window_handle_06::RawDisplayHandle::Xcb(
                    raw_window_handle_06::XcbDisplayHandle::new(
                        NonNull::new(handle.connection),
                        handle.screen,
                    ),
                )
            }
            raw_window_handle::RawDisplayHandle::Windows(_) => {
                raw_window_handle_06::RawDisplayHandle::Windows(
                    raw_window_handle_06::WindowsDisplayHandle::new(),
                )
            }
            _ => todo!(),
        },
        raw_window_handle: match raw_window_handle {
            raw_window_handle::RawWindowHandle::AppKit(handle) => {
                raw_window_handle_06::RawWindowHandle::AppKit(
                    raw_window_handle_06::AppKitWindowHandle::new(
                        NonNull::new(handle.ns_view).unwrap(),
                    ),
                )
            }
            raw_window_handle::RawWindowHandle::Xlib(handle) => {
                raw_window_handle_06::RawWindowHandle::Xlib(
                    raw_window_handle_06::XlibWindowHandle::new(handle.window),
                )
            }
            raw_window_handle::RawWindowHandle::Xcb(handle) => {
                raw_window_handle_06::RawWindowHandle::Xcb(
                    raw_window_handle_06::XcbWindowHandle::new(
                        NonZeroU32::new(handle.window).unwrap(),
                    ),
                )
            }
            raw_window_handle::RawWindowHandle::Win32(handle) => {
                let mut raw_handle = raw_window_handle_06::Win32WindowHandle::new(
                    NonZeroIsize::new(handle.hwnd as isize).unwrap(),
                );

                raw_handle.hinstance = NonZeroIsize::new(handle.hinstance as isize);

                raw_window_handle_06::RawWindowHandle::Win32(raw_handle)
            }
            _ => todo!(),
        },
    }
}

/// This is mostly identical to the gain example, minus some fluff, and with a GUI.
pub struct MyPlugin {
    params: Arc<MyPluginParams>,

    /// Needed to normalize the peak meter's response based on the sample rate.
    peak_meter_decay_weight: f32,
    /// The current data for the peak meter. This is stored as an [`Arc`] so we can share it between
    /// the GUI and the audio processing parts. If you have more state to share, then it's a good
    /// idea to put all of that in a struct behind a single `Arc`.
    ///
    /// This is stored as voltage gain.
    peak_meter: Arc<AtomicF32>,
}

#[derive(Params)]
pub struct MyPluginParams {
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    editor_state: Arc<CustomWgpuEditorState>,

    #[id = "gain"]
    pub gain: FloatParam,

    #[id = "foobar"]
    pub some_int: IntParam,
}

impl Default for MyPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(MyPluginParams::default()),

            peak_meter_decay_weight: 1.0,
            peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
        }
    }
}

impl Default for MyPluginParams {
    fn default() -> Self {
        Self {
            editor_state: CustomWgpuEditorState::from_size((400, 300)),

            // See the main gain example for more details
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            some_int: IntParam::new("Something", 3, IntRange::Linear { min: 0, max: 3 }),
        }
    }
}

impl Plugin for MyPlugin {
    const NAME: &'static str = "BYO GUI Example (WGPU)";
    const VENDOR: &'static str = "Moist Plugins GmbH";
    const URL: &'static str = "https://youtu.be/dQw4w9WgXcQ";
    const EMAIL: &'static str = "info@example.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        Some(Box::new(CustomWgpuEditor {
            params: Arc::clone(&self.params),
            peak_meter: Arc::clone(&self.peak_meter),

            // TODO: We can't get the size of the window when baseview does its own scaling, so if the
            //       host does not set a scale factor on Windows or Linux we should just use a factor of
            //       1. That may make the GUI tiny but it also prevents it from getting cut off.
            #[cfg(target_os = "macos")]
            scaling_factor: AtomicCell::new(None),
            #[cfg(not(target_os = "macos"))]
            scaling_factor: AtomicCell::new(Some(1.0)),
        }))
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // TODO: Figure out a way to disable log spam from wgpu.

        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight = 0.25f64
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for channel_samples in buffer.iter_samples() {
            let mut amplitude = 0.0;
            let num_samples = channel_samples.len();

            let gain = self.params.gain.smoothed.next();
            for sample in channel_samples {
                *sample *= gain;
                amplitude += *sample;
            }

            // To save resources, a plugin can (and probably should!) only perform expensive
            // calculations that are only displayed on the GUI while the GUI is open
            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).abs();
                let current_peak_meter = self.peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                let new_peak_meter = if amplitude > current_peak_meter {
                    amplitude
                } else {
                    current_peak_meter * self.peak_meter_decay_weight
                        + amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                self.peak_meter
                    .store(new_peak_meter, std::sync::atomic::Ordering::Relaxed)
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for MyPlugin {
    const CLAP_ID: &'static str = "com.moist-plugins-gmbh.byo-gui-wgpu";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A simple example plugin with a raw WGPU context for rendering");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for MyPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"ByoGuiWGPUWooooo";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nih_export_clap!(MyPlugin);
nih_export_vst3!(MyPlugin);
