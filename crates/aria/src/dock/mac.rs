//! The macOS dock: a non-activating `NSPanel` on the screen under the pointer, hosting the same page
//! as the Omarchy one in a `WKWebView`.
//!
//! The page is served as `aria://app/…` through a `WKURLSchemeHandler`, and it talks to us with the
//! script message handler it already used on Linux (`window.webkit.messageHandlers.aria`). The panel
//! joins every space and full-screen app, takes the keyboard when it opens, and gives it back to
//! whichever app had it when it closes. Only the 470 px column is visible: the page draws the veil.
//!
//! A click on the veil closes the dock (the page sends `closeRequest`); click-through, which is what
//! GTK gets from an input region, is not implemented on macOS yet (docs/pending.md).
//!
//! The page runs in WebKit's own web content process. When that process dies the panel would stay on screen and paint
//! nothing, so the dock reloads the page (after the `Recovery` delay) and brings the new page up to date: the
//! `Replay`ed `Platform` and `Version`, then `Open` and `Problem` when the dock is on screen, then what was queued.

use super::{content_type, Recovery, Replay, UI};
use crate::visibility::{self, Command, Event, State};
use aria_core::bridge::{DockMessage, PageMessage};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, AnyThread, DefinedClass, MainThreadOnly};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationOptions, NSAutoresizingMaskOptions, NSBackingStoreType, NSColor, NSPanel,
    NSRunningApplication, NSStatusWindowLevel, NSWindowCollectionBehavior, NSWindowStyleMask, NSWorkspace,
};
use objc2_foundation::{
    ns_string, MainThreadMarker, NSData, NSJSONSerialization, NSJSONWritingOptions, NSNumber, NSObject,
    NSObjectProtocol, NSPoint, NSRect, NSSize, NSString, NSTimer, NSURLRequest, NSURLResponse, NSURL,
};
use objc2_web_kit::{
    WKNavigationDelegate, WKScriptMessage, WKScriptMessageHandler, WKURLSchemeHandler, WKURLSchemeTask,
    WKUserContentController, WKWebView, WKWebViewConfiguration,
};
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The surface covers the design's 700 px veil; the page draws the dock column inside it.
/// The close transition (140 ms + 20 ms delay in dock.css) plus a frame of margin.
const CLOSE_MS: u64 = 180;
/// The page, loaded at start and again after its web process dies.
const PAGE: &str = "aria://app/index.html";
/// The corner notice fades after 8 s in the page (`NOTICE_MS` in app.js); the panel goes a bit later.
const NOTICE_UNMAP_MS: u64 = 8600;

type OnMessage = Box<dyn Fn(PageMessage, &Rc<Dock>)>;

pub struct Dock {
    mtm: MainThreadMarker,
    panel: Retained<DockPanel>,
    webview: Retained<WKWebView>,
    /// The page's handlers are retained by the configuration; the fields keep the Rust side explicit.
    _controller: Retained<WKUserContentController>,
    _scheme: Retained<SchemeHandler>,
    _script: Retained<ScriptHandler>,
    /// The web view holds its navigation delegate weakly: this field is what keeps it alive.
    _navigation: Retained<NavigationDelegate>,
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
    notice_visible: Cell<bool>,
    notice_generation: Cell<u64>,
    notice_wanted: Cell<bool>,
    /// The app that had the keyboard before the dock opened; it gets it back when the dock closes.
    previous_app: RefCell<Option<Retained<NSRunningApplication>>>,
    /// Is the dock on screen? Read by the chat (a prompt that nobody can see needs a notification).
    visible: Arc<AtomicBool>,
    /// The trace panel is shown: the surface is wide (`super::surface_width`).
    wide: Cell<bool>,
    on_message: OnMessage,
}

impl Dock {
    /// `on_message` receives every page message the dock does not handle itself
    /// (everything except `ready`, `opened` and `closeRequest`).
    pub fn new(visible: Arc<AtomicBool>, on_message: OnMessage) -> Rc<Dock> {
        let mtm = MainThreadMarker::new().expect("the dock is created on the main thread");
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(super::SURFACE_NARROW as f64, 800.0));
        let panel: Retained<DockPanel> = unsafe {
            msg_send![
                DockPanel::alloc(mtm),
                initWithContentRect: frame,
                styleMask: NSWindowStyleMask::Borderless | NSWindowStyleMask::NonactivatingPanel,
                backing: NSBackingStoreType::Buffered,
                defer: false
            ]
        };
        unsafe { panel.setReleasedWhenClosed(false) };
        panel.setOpaque(false);
        panel.setBackgroundColor(Some(&NSColor::clearColor()));
        panel.setHasShadow(false);
        panel.setLevel(NSStatusWindowLevel);
        panel.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary,
        );
        panel.setHidesOnDeactivate(false);
        panel.setMovable(false);
        panel.setIgnoresMouseEvents(true);
        panel.setBecomesKeyOnlyIfNeeded(false);

        // The handler needs the dock, and the dock needs the handler: the weak slot is filled once the
        // Rc exists, before the page can load.
        let dock_cell: Rc<RefCell<Option<Weak<Dock>>>> = Rc::new(RefCell::new(None));
        let script = ScriptHandler::new(mtm, dock_cell.clone());
        let navigation = NavigationDelegate::new(mtm, dock_cell.clone());
        let scheme = SchemeHandler::new(mtm);
        let config = unsafe { WKWebViewConfiguration::new(mtm) };
        let controller = unsafe { WKUserContentController::new(mtm) };
        unsafe {
            controller.addScriptMessageHandler_name(ProtocolObject::from_ref(&*script), ns_string!("aria"));
            config.setUserContentController(&controller);
            config.setURLSchemeHandler_forURLScheme(Some(ProtocolObject::from_ref(&*scheme)), ns_string!("aria"));
        }

        let webview = unsafe {
            WKWebView::initWithFrame_configuration(
                WKWebView::alloc(mtm),
                NSRect::new(NSPoint::new(0.0, 0.0), frame.size),
                &config,
            )
        };
        webview.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        unsafe {
            webview.setUnderPageBackgroundColor(Some(&NSColor::clearColor()));
            // WKWebView paints its own white unless its (private, but long-standing) background is off.
            let draws = NSNumber::numberWithBool(false);
            let _: () = msg_send![&*webview, setValue: &*draws, forKey: ns_string!("drawsBackground")];
        }
        unsafe { webview.setNavigationDelegate(Some(ProtocolObject::from_ref(&*navigation))) };
        panel.setContentView(Some(&webview));

        let dock = Rc::new(Dock {
            mtm,
            panel,
            webview,
            _controller: controller,
            _scheme: scheme,
            _script: script,
            _navigation: navigation,
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
            notice_visible: Cell::new(false),
            notice_generation: Cell::new(0),
            notice_wanted: Cell::new(false),
            previous_app: RefCell::new(None),
            visible,
            wide: Cell::new(false),
            on_message,
        });
        *dock_cell.borrow_mut() = Some(Rc::downgrade(&dock));

        dock.load_page();
        dock
    }

    /// Loads the page into the web view: at start, and again after its web content process died.
    fn load_page(&self) {
        if let Some(url) = NSURL::URLWithString(&NSString::from_str(PAGE)) {
            unsafe { self.webview.loadRequest(&NSURLRequest::requestWithURL(&url)) };
        }
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
        let Ok(json) = serde_json::to_string(message) else { return };
        self.replay.borrow_mut().remember(message, &json);
        if self.ready.get() {
            self.deliver(&json);
        } else {
            self.queued.borrow_mut().push(json);
        }
    }

    fn deliver(&self, json: &str) {
        let js = NSString::from_str(&format!("window.aria.receive({json})"));
        unsafe { self.webview.evaluateJavaScript_completionHandler(&js, None) };
    }

    fn page_ready(self: &Rc<Self>) {
        self.ready.set(true);
        crate::note("page ready");
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
    }

    /// The page's web content process died: messages wait in the queue until the page is loaded again, after the
    /// `Recovery` delay.
    fn page_died(self: &Rc<Self>) {
        self.ready.set(false);
        self.reloaded.set(true);
        let delay = self.recovery.borrow_mut().died(Instant::now());
        crate::note(&format!("the page's process died; reloading in {} ms", delay.as_millis()));
        self.reloads.set(self.reloads.get() + 1);
        let reloads = self.reloads.get();
        let weak: Weak<Dock> = Rc::downgrade(self);
        after(delay, move || {
            if let Some(dock) = weak.upgrade() {
                if dock.reloads.get() == reloads {
                    crate::note("page reloaded");
                    dock.load_page();
                }
            }
        });
    }

    /// The first-run corner notice, shown while the dock is closed: mapped without the keyboard and
    /// without taking clicks, then unmapped again. Pressing the hotkey meanwhile opens the dock.
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
        self.notice_generation.set(self.notice_generation.get() + 1);
        let generation = self.notice_generation.get();
        self.place_on_focused_screen();
        self.panel.setIgnoresMouseEvents(true);
        self.panel.orderFrontRegardless();
        self.send(&DockMessage::Notice);
        let weak: Weak<Dock> = Rc::downgrade(self);
        after(Duration::from_millis(NOTICE_UNMAP_MS), move || {
            let Some(dock) = weak.upgrade() else { return };
            if dock.notice_generation.get() == generation
                && dock.notice_visible.get()
                && dock.state.get() == State::Hidden
            {
                dock.notice_visible.set(false);
                dock.panel.orderOut(None);
            }
        });
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
        crate::note(&format!("{previous:?} + {event:?} -> {next:?} ({command:?})"));
        match command {
            Command::Show => {
                self.generation.set(self.generation.get() + 1);
                let was_notice = self.notice_visible.replace(false);
                self.notice_generation.set(self.notice_generation.get() + 1);
                // A panel that is still mapped (re-opened mid-close, or showing the notice) is ordered out
                // and back in, so it gets the keyboard again.
                if previous == State::Closing || was_notice {
                    self.panel.orderOut(None);
                }
                let monitor = self.place_on_focused_screen();
                *self.monitor.borrow_mut() = monitor.clone();
                self.remember_previous_app();
                self.panel.setIgnoresMouseEvents(false);
                let app = NSApplication::sharedApplication(self.mtm);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);
                self.panel.makeKeyAndOrderFront(None);
                let _ = self.panel.makeFirstResponder(Some(&self.webview));
                // Whether the panel really is on screen with the keyboard a moment later (another app or a full-screen
                // space can keep it from either).
                let weak: Weak<Dock> = Rc::downgrade(self);
                after(Duration::from_millis(150), move || {
                    if let Some(dock) = weak.upgrade() {
                        crate::note(&format!(
                            "shown on {}: visible={} key={} active={} page ready={}",
                            dock.panel.screen().map(|s| s.localizedName().to_string()).unwrap_or_default(),
                            dock.panel.isVisible(),
                            dock.panel.isKeyWindow(),
                            NSApplication::sharedApplication(dock.mtm).isActive(),
                            dock.ready.get()
                        ));
                    }
                });
                self.send(&DockMessage::Open { monitor });
                // Always tell the page: an empty code clears a banner from an earlier open.
                let code = self.problem.borrow().clone().unwrap_or_default();
                self.send(&DockMessage::Problem { code });
            }
            Command::StartClose => {
                // The panel stays on screen for the page's close animation, but takes no more clicks
                // and leaves the keyboard to the app that had it.
                self.panel.setIgnoresMouseEvents(true);
                self.send(&DockMessage::Close);
                self.generation.set(self.generation.get() + 1);
                self.schedule_close_finished(self.generation.get());
            }
            Command::Unmap => self.hide(),
            Command::None => {}
        }
    }

    fn schedule_close_finished(self: &Rc<Self>, generation: u64) {
        let weak: Weak<Dock> = Rc::downgrade(self);
        after(Duration::from_millis(CLOSE_MS), move || {
            if let Some(dock) = weak.upgrade() {
                if dock.generation.get() == generation {
                    dock.feed(Event::CloseFinished);
                }
            }
        });
    }

    fn hide(&self) {
        self.panel.orderOut(None);
        let previous = self.previous_app.borrow_mut().take();
        // Only while Aria still has the focus: if the user went to another app with the dock open, that app keeps it.
        if !NSApplication::sharedApplication(self.mtm).isActive() {
            return;
        }
        if let Some(app) = previous {
            #[allow(deprecated)]
            app.activateWithOptions(NSApplicationActivationOptions::ActivateIgnoringOtherApps);
        }
    }

    fn remember_previous_app(&self) {
        let Some(app) = NSWorkspace::sharedWorkspace().frontmostApplication() else { return };
        if app.processIdentifier() == std::process::id() as i32 {
            return;
        }
        *self.previous_app.borrow_mut() = Some(app);
    }

    /// The trace panel came or went: the surface grows or shrinks toward the left, its right edge where it is.
    fn set_wide(&self, wide: bool) {
        if self.wide.replace(wide) == wide {
            return;
        }
        let frame = self.panel.frame();
        let width = super::surface_width(wide) as f64;
        let right = frame.origin.x + frame.size.width;
        let next = NSRect::new(NSPoint::new(right - width, frame.origin.y), NSSize::new(width, frame.size.height));
        self.panel.setFrame_display(next, true);
    }

    /// Moves the panel to the visible part of the screen under the pointer; returns its name.
    fn place_on_focused_screen(&self) -> String {
        let Some(screen) = aria_mac::focused_screen() else { return "(main screen)".into() };
        let visible = screen.visible;
        let width = super::surface_width(self.wide.get()) as f64;
        let frame =
            NSRect::new(NSPoint::new(visible.x + visible.width - width, visible.y), NSSize::new(width, visible.height));
        self.panel.setFrame_display(frame, true);
        screen.name
    }

    fn page_message(self: &Rc<Self>, message: PageMessage) {
        if std::env::var_os("ARIA_DEBUG").is_some() {
            eprintln!("aria: page {}", message.log_name());
        }
        match message {
            PageMessage::Ready => self.page_ready(),
            PageMessage::Opened => self.feed(Event::Opened),
            PageMessage::CloseRequest => self.feed(Event::Escape),
            PageMessage::Wide { on } => self.set_wide(on),
            other => (self.on_message)(other, self),
        }
    }
}

// A panel can only become key by itself if it says so; a borderless panel otherwise never takes the
// keyboard, which is the one thing this dock must do when it opens.
define_class!(
    // SAFETY: NSPanel has no subclassing requirements, and DockPanel does not implement Drop.
    #[unsafe(super(NSPanel))]
    #[thread_kind = MainThreadOnly]
    #[name = "AriaDockPanel"]
    struct DockPanel;

    impl DockPanel {
        #[unsafe(method(canBecomeKeyWindow))]
        fn can_become_key_window(&self) -> bool {
            true
        }

        #[unsafe(method(canBecomeMainWindow))]
        fn can_become_main_window(&self) -> bool {
            false
        }
    }
);

/// The dock, for the Objective-C objects that call back into it: a weak slot filled once the dock's `Rc` exists.
struct DockIvars {
    dock: Rc<RefCell<Option<Weak<Dock>>>>,
}

// `window.webkit.messageHandlers.aria.postMessage(message)` on the page arrives here as a property
// list; it is serialized back to JSON and parsed by the same `PageMessage` the GTK dock uses.
define_class!(
    // SAFETY: NSObject has no subclassing requirements, and ScriptHandler does not implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AriaScriptHandler"]
    #[ivars = DockIvars]
    struct ScriptHandler;

    unsafe impl NSObjectProtocol for ScriptHandler {}

    unsafe impl WKScriptMessageHandler for ScriptHandler {
        #[unsafe(method(userContentController:didReceiveScriptMessage:))]
        unsafe fn did_receive(&self, _controller: &WKUserContentController, message: &WKScriptMessage) {
            let body = message.body();
            let Ok(data) = NSJSONSerialization::dataWithJSONObject_options_error(&body, NSJSONWritingOptions(0)) else {
                eprintln!("aria: ignoring a page message that is not JSON");
                return;
            };
            let Ok(json) = String::from_utf8(data.to_vec()) else { return };
            let message = match serde_json::from_str::<PageMessage>(&json) {
                Ok(message) => message,
                Err(error) => {
                    eprintln!("aria: ignoring a page message: {error}");
                    return;
                }
            };
            let dock = {
                let cell = self.ivars().dock.borrow();
                cell.as_ref().and_then(Weak::upgrade)
            };
            if let Some(dock) = dock {
                dock.page_message(message);
            }
        }
    }
);

impl ScriptHandler {
    fn new(mtm: MainThreadMarker, dock: Rc<RefCell<Option<Weak<Dock>>>>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DockIvars { dock });
        unsafe { msg_send![super(this), init] }
    }
}

// WebKit reports the death of the page's web content process (out of memory, a WebKit bug, killed) to the web view's
// navigation delegate; the dock reloads the page. Every other navigation callback keeps WebKit's default.
define_class!(
    // SAFETY: NSObject has no subclassing requirements, and NavigationDelegate does not implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AriaNavigationDelegate"]
    #[ivars = DockIvars]
    struct NavigationDelegate;

    unsafe impl NSObjectProtocol for NavigationDelegate {}

    unsafe impl WKNavigationDelegate for NavigationDelegate {
        #[unsafe(method(webViewWebContentProcessDidTerminate:))]
        unsafe fn process_did_terminate(&self, _web_view: &WKWebView) {
            let dock = {
                let cell = self.ivars().dock.borrow();
                cell.as_ref().and_then(Weak::upgrade)
            };
            if let Some(dock) = dock {
                dock.page_died();
            }
        }
    }
);

impl NavigationDelegate {
    fn new(mtm: MainThreadMarker, dock: Rc<RefCell<Option<Weak<Dock>>>>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DockIvars { dock });
        unsafe { msg_send![super(this), init] }
    }
}

// Serves the embedded `ui/` folder as `aria://app/<path>`: the macOS counterpart of GTK's URI scheme.
struct SchemeIvars;

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and SchemeHandler does not implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AriaSchemeHandler"]
    #[ivars = SchemeIvars]
    struct SchemeHandler;

    unsafe impl NSObjectProtocol for SchemeHandler {}

    unsafe impl WKURLSchemeHandler for SchemeHandler {
        #[unsafe(method(webView:startURLSchemeTask:))]
        unsafe fn start(&self, _web_view: &WKWebView, task: &ProtocolObject<dyn WKURLSchemeTask>) {
            let request = task.request();
            let Some(url) = request.URL() else { return };
            let path = url.path().map(|path| path.to_string()).unwrap_or_default();
            let path = path.trim_start_matches('/');
            let path = if path.is_empty() { "index.html" } else { path };
            let bytes = match UI.get_file(path) {
                Some(file) => file.contents(),
                None => {
                    eprintln!("aria: missing page asset: {path}");
                    &[]
                }
            };
            let mime = NSString::from_str(content_type(path));
            let response = NSURLResponse::initWithURL_MIMEType_expectedContentLength_textEncodingName(
                NSURLResponse::alloc(),
                &url,
                Some(&mime),
                bytes.len() as isize,
                None,
            );
            task.didReceiveResponse(&response);
            task.didReceiveData(&NSData::with_bytes(bytes));
            task.didFinish();
        }

        #[unsafe(method(webView:stopURLSchemeTask:))]
        unsafe fn stop(&self, _web_view: &WKWebView, _task: &ProtocolObject<dyn WKURLSchemeTask>) {}
    }
);

impl SchemeHandler {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(SchemeIvars);
        unsafe { msg_send![super(this), init] }
    }
}

/// Schedules a one-shot block on the main run loop.
fn after(ms: Duration, callback: impl FnOnce() + 'static) {
    let callback = Rc::new(RefCell::new(Some(callback)));
    let block = RcBlock::new(move |_timer: NonNull<NSTimer>| {
        if let Some(callback) = callback.borrow_mut().take() {
            callback();
        }
    });
    unsafe { NSTimer::scheduledTimerWithTimeInterval_repeats_block(ms.as_secs_f64(), false, &block) };
}
