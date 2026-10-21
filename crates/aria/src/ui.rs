//! The UI thread and its event loop, per platform. The dock, the page and every window live here;
//! worker threads hand their replies back through channels and this loop drains them.

#[cfg(target_os = "linux")]
pub fn init() {
    // WebKitGTK's DMA-BUF renderer draws nothing in a layer-shell surface on this setup.
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    gtk::init().expect("gtk");
}

/// Calls `tick` on the UI thread every `ms` milliseconds.
#[cfg(target_os = "linux")]
pub fn every(ms: u64, mut tick: impl FnMut() + 'static) {
    glib::timeout_add_local(std::time::Duration::from_millis(ms), move || {
        tick();
        glib::ControlFlow::Continue
    });
}

#[cfg(target_os = "linux")]
pub fn run() {
    gtk::main();
}

/// An accessory app: no Dock icon and no menu bar of its own, but panels can still take the keyboard.
#[cfg(target_os = "macos")]
pub fn init() {
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    use objc2_foundation::MainThreadMarker;

    let mtm = MainThreadMarker::new().expect("the UI runs on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    install_edit_menu(mtm);
}

/// macOS matches Cmd+C/V/X/A against the main menu before anything else: without an Edit menu the
/// shortcuts never reach the web view, and pasting an API key in the setup silently does nothing. The
/// app is an accessory, so the bar is never drawn, but the menu items still handle the key equivalents.
#[cfg(target_os = "macos")]
fn install_edit_menu(mtm: objc2_foundation::MainThreadMarker) {
    use objc2::rc::Retained;
    use objc2::runtime::Sel;
    use objc2::{sel, MainThreadOnly};
    use objc2_app_kit::{NSApplication, NSEventModifierFlags, NSMenu, NSMenuItem};
    use objc2_foundation::{ns_string, NSString};

    fn item(
        mtm: objc2_foundation::MainThreadMarker,
        title: &NSString,
        action: Sel,
        key: &NSString,
    ) -> Retained<NSMenuItem> {
        unsafe { NSMenuItem::initWithTitle_action_keyEquivalent(NSMenuItem::alloc(mtm), title, Some(action), key) }
    }

    let main = NSMenu::new(mtm);

    // The app menu's Quit has no key: Cmd+Q reaches the page, which asks first and quits on a second press (app.js).
    let app_item = NSMenuItem::new(mtm);
    let app_menu = NSMenu::new(mtm);
    app_menu.addItem(&item(mtm, ns_string!("Quit Aria"), sel!(terminate:), ns_string!("")));
    app_item.setSubmenu(Some(&app_menu));
    main.addItem(&app_item);

    let edit_item = NSMenuItem::new(mtm);
    let edit = NSMenu::new(mtm);
    edit.addItem(&item(mtm, ns_string!("Undo"), sel!(undo:), ns_string!("z")));
    let redo = item(mtm, ns_string!("Redo"), sel!(redo:), ns_string!("z"));
    redo.setKeyEquivalentModifierMask(NSEventModifierFlags::Command | NSEventModifierFlags::Shift);
    edit.addItem(&redo);
    edit.addItem(&NSMenuItem::separatorItem(mtm));
    edit.addItem(&item(mtm, ns_string!("Cut"), sel!(cut:), ns_string!("x")));
    edit.addItem(&item(mtm, ns_string!("Copy"), sel!(copy:), ns_string!("c")));
    edit.addItem(&item(mtm, ns_string!("Paste"), sel!(paste:), ns_string!("v")));
    edit.addItem(&item(mtm, ns_string!("Select All"), sel!(selectAll:), ns_string!("a")));
    edit_item.setSubmenu(Some(&edit));
    main.addItem(&edit_item);

    NSApplication::sharedApplication(mtm).setMainMenu(Some(&main));
    if std::env::var_os("ARIA_DEBUG").is_some() {
        eprintln!("aria: edit menu installed (Cmd+C/V/X/A)");
    }
}

/// Calls `tick` on the UI thread every `ms` milliseconds (the run loop's default mode).
#[cfg(target_os = "macos")]
pub fn every(ms: u64, tick: impl FnMut() + 'static) {
    use block2::RcBlock;
    use objc2_foundation::NSTimer;
    use std::cell::RefCell;
    use std::rc::Rc;

    let tick = Rc::new(RefCell::new(tick));
    let block = RcBlock::new(move |_timer: std::ptr::NonNull<NSTimer>| {
        if let Ok(mut tick) = tick.try_borrow_mut() {
            tick();
        }
    });
    unsafe { NSTimer::scheduledTimerWithTimeInterval_repeats_block(ms as f64 / 1000.0, true, &block) };
}

/// Calls `stop` when AppKit is about to exit (Quit Aria, Cmd+Q): that exit sends no signal, so the quit handlers never see it.
#[cfg(target_os = "macos")]
pub fn on_terminate(stop: impl Fn() + 'static) {
    use block2::RcBlock;
    use objc2_app_kit::NSApplicationWillTerminateNotification;
    use objc2_foundation::{NSNotification, NSNotificationCenter};

    let block = RcBlock::new(move |_: std::ptr::NonNull<NSNotification>| stop());
    // SAFETY: with no queue the block runs on the thread that posts, AppKit's main thread, where `stop` belongs.
    let observer = unsafe {
        NSNotificationCenter::defaultCenter().addObserverForName_object_queue_usingBlock(
            Some(NSApplicationWillTerminateNotification),
            None,
            None,
            &block,
        )
    };
    std::mem::forget(observer); // for the life of the app
}

#[cfg(target_os = "macos")]
pub fn run() {
    use objc2_app_kit::NSApplication;
    use objc2_foundation::MainThreadMarker;

    let mtm = MainThreadMarker::new().expect("the UI runs on the main thread");
    NSApplication::sharedApplication(mtm).run();
}
