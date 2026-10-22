//! The dock window: a layer-shell surface on the focused monitor hosting the web UI. It executes the
//! commands `visibility::step` decides and carries the messages of `aria_core::bridge`.
//!
//! Two WebKitGTK facts that cost the first Aria prototype days: the window must be sized with
//! `set_size_request` (a default size never paints a layer surface with a WebView child), and WebKit's renderer must
//! be chosen before GTK starts (`WEBKIT_DMABUF_RENDERER_FORCE_SHM=1`, done in `ui.rs`).
//!
//! Nothing animates the dock on Omarchy. WebKitGTK 2.54 shows stale frames while a transparent surface animates its
//! opacity, so the page never animates its own open and keeps its open look while hidden; and the compositor's own
//! layer fade is off too (the rule `aria_hyprland` registers), since it would run over the empty first buffers (WebKit
//! renders nothing while the surface is unmapped and its first render after a map takes 70 ms, 200 ms the first time
//! in a process) and the content would pop in mid-fade. So an open maps the surface at once and the dock shows the
//! moment WebKit has its first frame, complete (the window has its real height from the start, `build_window`); a close
//! unmaps at once. The page is also mapped once at start, opened but off the monitor's edge, once the first
//! conversation has reached it, so WebKit brings its renderer up and renders the real chat before the first open.
//!
//! The page runs in WebKit's own web process. When that process dies the surface would stay mapped and paint
//! nothing, so the dock reloads the page (after the `Recovery` delay) and brings the new page up to date: the
//! `Replay`ed `Platform` and `Version`, then `Open` and `Problem` when the dock is on screen, then what was queued.

use super::{content_type, Recovery, Replay, UI};
use crate::visibility::{self, Command, Event, State};
use aria_core::bridge::{DockMessage, PageMessage};
use gtk::prelude::*;
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use javascriptcore::ValueExt;
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use webkit2gtk::{
    NavigationPolicyDecision, NavigationPolicyDecisionExt, PolicyDecisionExt, PolicyDecisionType, SecurityManagerExt,
    SettingsExt, URIRequestExt, URISchemeRequest, URISchemeRequestExt, UserContentManager, UserContentManagerExt,
    WebContext, WebContextExt, WebView, WebViewExt,
};

/// Nothing to wait for on a close: the surface unmaps at once.
const CLOSE_MS: u64 = 0;
/// How long the surface stays mapped for the warm-up at start: WebKit's renderer coming up and a first render.
const WARM_MS: u64 = 500;
/// The warm-up waits for the first conversation to reach the page (its chat is built then), this long at most.
const WARM_WAIT_MS: u64 = 8000;
/// The page, loaded at start and again after its web process dies.
const PAGE: &str = "aria://app/index.html";
/// The corner notice fades after 8 s in the page (`NOTICE_MS` in app.js); the surface goes a bit later.
const NOTICE_UNMAP_MS: u64 = 8600;

type OnMessage = Box<dyn Fn(PageMessage, &Rc<Dock>)>;

pub struct Dock {
    window: gtk::Window,
    webview: WebView,
    state: Cell<State>,
    /// Bumped on every show and close, so a close timer from an earlier close is ignored.
    generation: Cell<u64>,
    problem: RefCell<Option<String>>,
    /// The page has loaded and can take messages; until then (and from its process's death until the reloaded page
    /// is ready) they wait in `queued`.
    ready: Cell<bool>,
    queued: RefCell<Vec<String>>,
    /// The monitor of the last open, sent again in `Open` to a page reloaded while the dock is on screen.
    monitor: RefCell<String>,
    /// How long to wait before reloading after the page's process dies.
    recovery: RefCell<Recovery>,
    /// The messages sent only once, sent again to a reloaded page.
    replay: RefCell<Replay>,
    /// Bumped on every death of the page's process, so a reload timer from an earlier death is ignored.
    reloads: Cell<u64>,
    /// The page was reloaded after a death and has not said `ready` yet: when it does, it is brought up to date.
    reloaded: Cell<bool>,
    /// While the first-run notice is on screen the surface is mapped but lets every click through.
    click_through: Rc<Cell<bool>>,
    /// The trace panel is shown: the surface is wide and the panel takes input too (`super::input_width`).
    wide: Rc<Cell<bool>>,
    notice_visible: Cell<bool>,
    notice_generation: Cell<u64>,
    notice_wanted: Cell<bool>,
    /// The surface is mapped, unseen, for the warm-up (once, at start).
    warming: Cell<bool>,
    warmed: Cell<bool>,
    /// A conversation has reached the page: the warm-up can build the chat for real.
    history_seen: Cell<bool>,
    /// Is the dock on screen? Read by the chat (a prompt that nobody can see needs a notification).
    visible: Arc<AtomicBool>,
    on_message: OnMessage,
}

impl Dock {
    /// `on_message` receives every page message the dock does not handle itself
    /// (everything except `ready`, `opened` and `closeRequest`).
    pub fn new(visible: Arc<AtomicBool>, on_message: OnMessage) -> Rc<Dock> {
        let window = build_window();
        let ucm = UserContentManager::new();
        ucm.register_script_message_handler("aria");
        let context = WebContext::default().expect("web context");
        register_ui_scheme(&context);
        let webview = WebView::builder().web_context(&context).user_content_manager(&ucm).build();
        webview.set_background_color(&gdk::RGBA::new(0.0, 0.0, 0.0, 0.0));
        if std::env::var_os("ARIA_DEBUG").is_some() {
            if let Some(settings) = WebViewExt::settings(&webview) {
                settings.set_enable_write_console_messages_to_stdout(true);
            }
        }
        // The page never leaves `aria://`: a file dropped where nothing takes it, or a link, would load in its place.
        webview.connect_decide_policy(|_, decision, kind| {
            if !matches!(kind, PolicyDecisionType::NavigationAction | PolicyDecisionType::NewWindowAction) {
                return false;
            }
            let uri = decision
                .downcast_ref::<NavigationPolicyDecision>()
                .and_then(|d| d.navigation_action())
                .and_then(|action| action.request())
                .and_then(|request| request.uri());
            if uri.as_deref().is_some_and(super::stays_in_app) {
                return false; // WebKit's default: load it
            }
            decision.ignore();
            true
        });
        window.add(&webview);

        // Clicks pass through the clear part of the veil; only the dock column takes them.
        let click_through = Rc::new(Cell::new(false));
        let wide = Rc::new(Cell::new(false));
        {
            let (click_through, wide) = (click_through.clone(), wide.clone());
            window.connect_size_allocate(move |window, allocation| {
                apply_input_region(window, allocation.width(), allocation.height(), click_through.get(), wide.get());
            });
        }
        window.realize();

        let dock = Rc::new(Dock {
            window,
            webview,
            state: Cell::new(State::Hidden),
            generation: Cell::new(0),
            problem: RefCell::new(None),
            ready: Cell::new(false),
            queued: RefCell::new(Vec::new()),
            monitor: RefCell::new(String::new()),
            recovery: RefCell::new(Recovery::default()),
            replay: RefCell::new(Replay::default()),
            reloads: Cell::new(0),
            reloaded: Cell::new(false),
            click_through,
            wide,
            notice_visible: Cell::new(false),
            notice_generation: Cell::new(0),
            notice_wanted: Cell::new(false),
            warming: Cell::new(false),
            warmed: Cell::new(false),
            history_seen: Cell::new(false),
            visible,
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
                Ok(PageMessage::Wide { on }) => dock.set_wide(on),
                Ok(message) => (dock.on_message)(message, &dock),
                Err(error) => eprintln!("aria: ignoring a page message: {error}"),
            }
        });
        let weak = Rc::downgrade(&dock);
        dock.webview.connect_web_process_terminated(move |_, reason| {
            if let Some(dock) = weak.upgrade() {
                dock.page_died(reason);
            }
        });
        // Escape is the page's to interpret (it closes a picker, stops a run, leaves a panel, and only then asks to
        // close with `closeRequest`): a window-level handler here would close the dock on every Escape.

        dock.webview.load_uri(PAGE);
        dock
    }

    /// Something prevents Aria from working; the page shows it every time the dock opens.
    pub fn set_problem(&self, code: &str) {
        *self.problem.borrow_mut() = Some(code.to_string());
    }

    pub fn clear_problem(&self) {
        *self.problem.borrow_mut() = None;
    }

    /// Sends a message to the page; before the page has loaded it waits in a queue.
    pub fn send(&self, message: &DockMessage) {
        if let DockMessage::Session { message: inner, .. } = message {
            if matches!(&**inner, DockMessage::ChatEvent { event } if event["type"] == "history") {
                self.history_seen.set(true);
            }
        }
        let Ok(json) = serde_json::to_string(message) else { return };
        self.replay.borrow_mut().remember(message, &json);
        if self.ready.get() {
            self.deliver(&json);
        } else {
            self.queued.borrow_mut().push(json);
        }
    }

    fn deliver(&self, json: &str) {
        self.webview.evaluate_javascript(
            &format!("window.aria.receive({json})"),
            None,
            None,
            None::<&gio::Cancellable>,
            |_| {},
        );
    }

    fn page_ready(self: &Rc<Self>) {
        self.ready.set(true);
        // On the first load what was sent before is all in the queue; a reloaded page also needs what was sent once
        // and, when the dock is on screen, the open (as `Command::Show` sends it).
        if self.reloaded.take() {
            for json in self.replay.borrow().messages() {
                self.deliver(&json);
            }
            if self.state.get() != State::Hidden {
                self.send(&DockMessage::Open { monitor: self.monitor.borrow().clone() });
                let code = self.problem.borrow().clone().unwrap_or_default();
                self.send(&DockMessage::Problem { code });
            }
        }
        for json in self.queued.take() {
            self.deliver(&json);
        }
        if self.notice_wanted.take() {
            self.show_notice_now();
        }
        self.warm_up_when_ready();
    }

    /// Warms up once the first conversation has reached the page (so the chat it builds is the real one), or after
    /// `WARM_WAIT_MS` without one.
    fn warm_up_when_ready(self: &Rc<Self>) {
        if self.warmed.get() {
            return;
        }
        let weak: Weak<Dock> = Rc::downgrade(self);
        let started = std::time::Instant::now();
        glib::timeout_add_local(Duration::from_millis(250), move || {
            let Some(dock) = weak.upgrade() else { return glib::ControlFlow::Break };
            if dock.warmed.get() {
                return glib::ControlFlow::Break;
            }
            if dock.history_seen.get() || started.elapsed() >= Duration::from_millis(WARM_WAIT_MS) {
                dock.warm_up();
                return glib::ControlFlow::Break;
            }
            glib::ControlFlow::Continue
        });
    }

    /// Maps the surface once, unseen, with the page opened, so WebKit brings its renderer up and renders the chat now
    /// and not on the first open (see the module doc).
    fn warm_up(self: &Rc<Self>) {
        if self.warmed.replace(true) || self.state.get() != State::Hidden || self.notice_visible.get() {
            return;
        }
        self.warming.set(true);
        self.click_through.set(true);
        // Off the monitor's right edge: mapped and rendered as the real thing, seen by nobody.
        self.window.set_layer_shell_margin(Edge::Right, -(super::SURFACE_WIDE + 40));
        self.window.set_keyboard_mode(KeyboardMode::None);
        self.window.show_all();
        self.refresh_input_region();
        // Opened as well: the chat gets built and rendered now, so the first open's render is a short one.
        self.send(&DockMessage::Open { monitor: self.monitor.borrow().clone() });
        let weak: Weak<Dock> = Rc::downgrade(self);
        glib::timeout_add_local_once(Duration::from_millis(WARM_MS), move || {
            let Some(dock) = weak.upgrade() else { return };
            if dock.end_warm_up() && dock.state.get() == State::Hidden && !dock.notice_visible.get() {
                dock.window.hide();
            }
        });
    }

    /// Ends the warm-up: the page shows itself again, and clicks are taken again unless the notice is up. True when
    /// one was going.
    fn end_warm_up(&self) -> bool {
        if !self.warming.replace(false) {
            return false;
        }
        self.window.set_layer_shell_margin(Edge::Right, 0);
        if self.state.get() == State::Hidden {
            self.send(&DockMessage::Close);
        }
        if !self.notice_visible.get() {
            self.click_through.set(false);
            self.refresh_input_region();
        }
        true
    }

    /// The page's web process died: messages wait in the queue until the page is loaded again, after the
    /// `Recovery` delay.
    fn page_died(self: &Rc<Self>, reason: webkit2gtk::WebProcessTerminationReason) {
        self.ready.set(false);
        self.reloaded.set(true);
        let delay = self.recovery.borrow_mut().died(Instant::now());
        eprintln!("aria: the page's process died ({reason:?}); reloading in {} ms", delay.as_millis());
        self.reloads.set(self.reloads.get() + 1);
        let reloads = self.reloads.get();
        let weak: Weak<Dock> = Rc::downgrade(self);
        glib::timeout_add_local_once(delay, move || {
            if let Some(dock) = weak.upgrade() {
                if dock.reloads.get() == reloads {
                    eprintln!("aria: page reloaded");
                    dock.webview.load_uri(PAGE);
                }
            }
        });
    }

    /// The first-run corner notice, shown while the dock is closed: the surface is mapped without the
    /// keyboard and without taking clicks, then unmapped again. Pressing SUPER + A meanwhile opens
    /// the dock normally.
    pub fn show_notice(self: &Rc<Self>) {
        if self.ready.get() {
            self.show_notice_now();
        } else {
            self.notice_wanted.set(true);
        }
    }

    fn show_notice_now(self: &Rc<Self>) {
        if self.state.get() != State::Hidden || self.notice_visible.get() {
            return;
        }
        self.notice_visible.set(true);
        self.click_through.set(true);
        self.notice_generation.set(self.notice_generation.get() + 1);
        let generation = self.notice_generation.get();
        self.place_on_focused_monitor();
        self.window.set_keyboard_mode(KeyboardMode::None);
        self.window.show_all();
        self.refresh_input_region();
        self.send(&DockMessage::Notice);
        let weak: Weak<Dock> = Rc::downgrade(self);
        glib::timeout_add_local_once(Duration::from_millis(NOTICE_UNMAP_MS), move || {
            let Some(dock) = weak.upgrade() else { return };
            if dock.notice_generation.get() == generation
                && dock.notice_visible.get()
                && dock.state.get() == State::Hidden
            {
                dock.notice_visible.set(false);
                dock.click_through.set(false);
                dock.window.hide();
            }
        });
    }

    fn refresh_input_region(&self) {
        let allocation = self.window.allocation();
        apply_input_region(
            &self.window,
            allocation.width(),
            allocation.height(),
            self.click_through.get(),
            self.wide.get(),
        );
    }

    /// The trace panel came or went: the surface grows or shrinks toward the left (it is anchored on the right), and
    /// the panel takes clicks while it is there.
    fn set_wide(&self, wide: bool) {
        if self.wide.replace(wide) == wide {
            return;
        }
        self.window.set_size_request(super::surface_width(wide), 100);
        self.refresh_input_region();
    }

    /// Opens the dock when it is not open (`aria show`); does nothing when it already is.
    pub fn show(self: &Rc<Self>) {
        if matches!(self.state.get(), State::Hidden | State::Closing) {
            self.feed(Event::Toggle);
        }
    }

    pub fn feed(self: &Rc<Self>, event: Event) {
        let previous = self.state.get();
        let (next, command) = visibility::step(previous, event);
        self.state.set(next);
        self.visible.store(next != State::Hidden, Ordering::SeqCst);
        if std::env::var_os("ARIA_DEBUG").is_some() {
            eprintln!("aria: {previous:?} + {event:?} -> {next:?} ({command:?})");
        }
        match command {
            Command::Show => {
                crate::face::seen(); // the person is looking: a failure has been seen, and Aria is awake
                self.generation.set(self.generation.get() + 1);
                let was_notice = self.notice_visible.replace(false);
                self.notice_generation.set(self.notice_generation.get() + 1);
                self.click_through.set(false);
                // A surface still mapped for the notice or the warm-up does not get the keyboard back by changing
                // the mode, so it is mapped again (once the page says it is painted).
                if was_notice || self.end_warm_up() {
                    self.window.hide();
                }
                let monitor = self.place_on_focused_monitor();
                *self.monitor.borrow_mut() = monitor.clone();
                // On demand, not exclusive: Hyprland still focuses the dock when it maps (so you can type at
                // once), but a click on another window takes the keyboard away. Exclusive made every other
                // window impossible to focus while Aria was open.
                self.window.set_keyboard_mode(KeyboardMode::OnDemand);
                self.window.show_all();
                self.window.present();
                // The page reads the keys (Escape included), so the keyboard must be in the web view.
                self.webview.grab_focus();
                self.send(&DockMessage::Open { monitor });
                // Always tell the page: an empty code clears a banner from an earlier open.
                let code = self.problem.borrow().clone().unwrap_or_default();
                self.send(&DockMessage::Problem { code });
            }
            Command::StartClose => {
                // Unmapped at once.
                self.window.set_keyboard_mode(KeyboardMode::None);
                self.window.hide();
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

    /// Moves the surface to the monitor Hyprland says has the focus and gives it that monitor's usable height before it
    /// maps: the compositor confirms the height only after the map, and a page laid out at the window's minimum until
    /// then hands the fade-in a half-made first frame. Returns the monitor's name.
    fn place_on_focused_monitor(&self) -> String {
        let Some(focused) = aria_hyprland::focused_monitor() else { return "(compositor's choice)".into() };
        if let Some(height) = focused.usable_height {
            self.window.set_size_request(super::surface_width(self.wide.get()), height);
        }
        match gdk_monitor_at(focused.x, focused.y) {
            Some(monitor) => {
                self.window.set_monitor(&monitor);
                focused.name
            }
            None => "(compositor's choice)".into(),
        }
    }
}

/// Only the dock column (and the trace panel while `wide`) takes input; nothing at all while `click_through`.
fn apply_input_region(window: &gtk::Window, width: i32, height: i32, click_through: bool, wide: bool) {
    let region = if click_through {
        gtk::cairo::Region::create()
    } else {
        let taken = super::input_width(wide).min(width);
        gtk::cairo::Region::create_rectangle(&gtk::cairo::RectangleInt::new(width - taken, 0, taken, height))
    };
    window.input_shape_combine_region(Some(&region));
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
    // set_size_request, not set_default_size (see the module comment). The compositor sets the height once the
    // surface maps; until then the window has this one, and the page lays out to it: the focused monitor's usable
    // height now, so the page's first paint after the first map is its real layout (at the minimum, the composer
    // would sit at the top of a 100 px page for the ~200 ms the relayout takes, after the fade-in).
    let height = aria_hyprland::focused_monitor().and_then(|monitor| monitor.usable_height).unwrap_or(100);
    window.set_size_request(super::SURFACE_NARROW, height);
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
