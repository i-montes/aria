//! The dock window: a layer-shell surface on the focused monitor hosting the web UI. It executes the
//! commands `visibility::step` decides and carries the messages of `aria_core::bridge`.
//!
//! Two WebKitGTK facts that cost the first Aria prototype days: the window must be sized with
//! `set_size_request` (a default size never paints a layer surface with a WebView child), and
//! `WEBKIT_DISABLE_DMABUF_RENDERER=1` must be set before GTK starts (done in `main.rs`).

use crate::visibility::{self, Command, Event, State};
use aria_core::bridge::{DockMessage, PageMessage};
use gtk::prelude::*;
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use include_dir::{include_dir, Dir};
use javascriptcore::ValueExt;
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::time::Duration;
use webkit2gtk::{
    SecurityManagerExt, URISchemeRequest, URISchemeRequestExt, UserContentManager, UserContentManagerExt, WebContext,
    WebContextExt, WebView, WebViewExt,
};

/// The surface covers the design's 700 px veil; only the 470 px dock column takes input.
const SURFACE_WIDTH: i32 = 700;
const DOCK_WIDTH: i32 = 470;
/// The close transition (140 ms + 20 ms delay in dock.css) plus a frame of margin.
const CLOSE_MS: u64 = 180;

static UI: Dir = include_dir!("$CARGO_MANIFEST_DIR/../../ui");

type OnMessage = Box<dyn Fn(PageMessage, &Rc<Dock>)>;

pub struct Dock {
    window: gtk::Window,
    webview: WebView,
    state: Cell<State>,
    /// Bumped on every show and close, so a close timer from an earlier close is ignored.
    generation: Cell<u64>,
    problem: RefCell<Option<String>>,
    /// The page has loaded and can take messages; until then they wait in `queued`.
    ready: Cell<bool>,
    queued: RefCell<Vec<String>>,
    on_message: OnMessage,
}

impl Dock {
    /// `on_message` receives every page message the dock does not handle itself
    /// (everything except `ready`, `opened` and `closeRequest`).
    pub fn new(on_message: OnMessage) -> Rc<Dock> {
        let window = build_window();
        let ucm = UserContentManager::new();
        ucm.register_script_message_handler("aria");
        let context = WebContext::default().expect("web context");
        register_ui_scheme(&context);
        let webview = WebView::builder().web_context(&context).user_content_manager(&ucm).build();
        webview.set_background_color(&gdk::RGBA::new(0.0, 0.0, 0.0, 0.0));
        window.add(&webview);

        // Clicks pass through the clear part of the veil; only the dock column takes them.
        window.connect_size_allocate(|window, allocation| {
            let rect = gtk::cairo::RectangleInt::new(allocation.width() - DOCK_WIDTH, 0, DOCK_WIDTH, allocation.height());
            window.input_shape_combine_region(Some(&gtk::cairo::Region::create_rectangle(&rect)));
        });
        window.realize();

        let dock = Rc::new(Dock {
            window,
            webview,
            state: Cell::new(State::Hidden),
            generation: Cell::new(0),
            problem: RefCell::new(None),
            ready: Cell::new(false),
            queued: RefCell::new(Vec::new()),
            on_message,
        });

        let weak = Rc::downgrade(&dock);
        ucm.connect_script_message_received(Some("aria"), move |_, result| {
            let Some(dock) = weak.upgrade() else { return };
            let Some(json) = result.js_value().and_then(|value| value.to_json(0)) else { return };
            match serde_json::from_str::<PageMessage>(&json) {
                Ok(PageMessage::Ready) => dock.page_ready(),
                Ok(PageMessage::Opened) => dock.feed(Event::Opened),
                Ok(PageMessage::CloseRequest) => dock.feed(Event::Escape),
                Ok(message) => (dock.on_message)(message, &dock),
                Err(error) => eprintln!("aria: ignoring a page message: {error}"),
            }
        });
        let weak = Rc::downgrade(&dock);
        dock.window.connect_key_press_event(move |_, event| {
            if event.keyval() == gdk::keys::constants::Escape {
                if let Some(dock) = weak.upgrade() {
                    dock.feed(Event::Escape);
                }
            }
            glib::Propagation::Proceed
        });

        dock.webview.load_uri("aria://app/index.html");
        dock
    }

    /// Something prevents Aria from working; the page shows it every time the dock opens.
    pub fn set_problem(&self, code: &str) {
        *self.problem.borrow_mut() = Some(code.to_string());
    }

    /// Sends a message to the page; before the page has loaded it waits in a queue.
    pub fn send(&self, message: &DockMessage) {
        let Ok(json) = serde_json::to_string(message) else { return };
        if self.ready.get() {
            self.deliver(&json);
        } else {
            self.queued.borrow_mut().push(json);
        }
    }

    fn deliver(&self, json: &str) {
        self.webview.evaluate_javascript(&format!("window.aria.receive({json})"), None, None, None::<&gio::Cancellable>, |_| {});
    }

    fn page_ready(&self) {
        self.ready.set(true);
        for json in self.queued.take() {
            self.deliver(&json);
        }
    }

    pub fn feed(self: &Rc<Self>, event: Event) {
        let previous = self.state.get();
        let (next, command) = visibility::step(previous, event);
        self.state.set(next);
        if std::env::var_os("ARIA_DEBUG").is_some() {
            eprintln!("aria: {previous:?} + {event:?} -> {next:?} ({command:?})");
        }
        match command {
            Command::Show => {
                self.generation.set(self.generation.get() + 1);
                // A surface that is still mapped (re-opened mid-close) does not get the keyboard back
                // by changing the mode, so it is mapped again.
                if previous == State::Closing {
                    self.window.hide();
                }
                let monitor = self.place_on_focused_monitor();
                self.window.set_keyboard_mode(KeyboardMode::Exclusive);
                self.window.show_all();
                self.window.present();
                self.send(&DockMessage::Open { monitor });
                if let Some(code) = self.problem.borrow().clone() {
                    self.send(&DockMessage::Problem { code });
                }
            }
            Command::StartClose => {
                // The surface stays mapped for the close animation; it must not keep the keyboard.
                self.window.set_keyboard_mode(KeyboardMode::None);
                self.send(&DockMessage::Close);
                self.generation.set(self.generation.get() + 1);
                self.schedule_close_finished(self.generation.get());
            }
            Command::Unmap => self.window.hide(),
            Command::None => {}
        }
    }

    fn schedule_close_finished(self: &Rc<Self>, generation: u64) {
        let weak: Weak<Dock> = Rc::downgrade(self);
        glib::timeout_add_local_once(Duration::from_millis(CLOSE_MS), move || {
            if let Some(dock) = weak.upgrade() {
                if dock.generation.get() == generation {
                    dock.feed(Event::CloseFinished);
                }
            }
        });
    }

    /// Moves the surface to the monitor Hyprland says has the focus; returns its name.
    fn place_on_focused_monitor(&self) -> String {
        let Some(focused) = aria_hyprland::focused_monitor() else { return "(compositor's choice)".into() };
        match gdk_monitor_at(focused.x, focused.y) {
            Some(monitor) => {
                self.window.set_monitor(&monitor);
                focused.name
            }
            None => "(compositor's choice)".into(),
        }
    }
}

fn build_window() -> gtk::Window {
    let window = gtk::Window::new(gtk::WindowType::Toplevel);
    window.init_layer_shell();
    window.set_namespace("aria-dock");
    window.set_layer(Layer::Overlay);
    for edge in [Edge::Top, Edge::Bottom, Edge::Right] {
        window.set_anchor(edge, true);
    }
    window.set_keyboard_mode(KeyboardMode::None);
    // set_size_request, not set_default_size (see the module comment). The compositor sets the height.
    window.set_size_request(SURFACE_WIDTH, 100);
    window.set_app_paintable(true);
    if let Some(visual) = WidgetExt::screen(&window).and_then(|screen| screen.rgba_visual()) {
        window.set_visual(Some(&visual));
    }
    window
}

/// The GDK monitor at a layout position (Hyprland and GDK agree on logical pixels).
fn gdk_monitor_at(x: i32, y: i32) -> Option<gdk::Monitor> {
    let display = gdk::Display::default()?;
    (0..display.n_monitors()).filter_map(|i| display.monitor(i)).find(|m| {
        let geometry = m.geometry();
        geometry.x() == x && geometry.y() == y
    })
}

/// Serves the embedded `ui/` folder as `aria://app/<path>`.
fn register_ui_scheme(context: &WebContext) {
    if let Some(security) = context.security_manager() {
        security.register_uri_scheme_as_secure("aria");
        security.register_uri_scheme_as_cors_enabled("aria");
    }
    context.register_uri_scheme("aria", |request: &URISchemeRequest| {
        let path = request.path().map(|p| p.to_string()).unwrap_or_default();
        let path = path.trim_start_matches('/');
        let path = if path.is_empty() { "index.html" } else { path };
        match UI.get_file(path) {
            Some(file) => {
                let bytes = glib::Bytes::from_static(file.contents());
                let stream = gio::MemoryInputStream::from_bytes(&bytes);
                request.finish(&stream, file.contents().len() as i64, Some(content_type(path)));
            }
            None => request.finish_error(&mut glib::Error::new(gio::IOErrorEnum::NotFound, "not found")),
        }
    });
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html",
        "css" => "text/css",
        "js" | "mjs" => "text/javascript",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}
