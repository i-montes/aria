//! Spike: the Aria dock owns its hotkey and opens on the focused monitor, with no edits to
//! `~/.config/hypr/`. See README.md. Throwaway code: no tests, minimal error handling.
//!
//! `dock-open-spike`         runs the dock (hidden) and registers SUPER+A with Hyprland.
//! `dock-open-spike toggle`  shows or hides it, starting it first if it is not running.

use gtk::prelude::*;
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::cell::Cell;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;
use webkit2gtk::{WebView, WebViewExt};

/// The surface covers the design's 700 px veil; only the 470 px dock column takes input.
const WIDTH: i32 = 700;
const DOCK_WIDTH: i32 = 470;
/// The closing transition (140 ms + 20 ms delay in dock.html) plus a frame of margin.
const CLOSE_MS: u64 = 180;
const HOTKEY: &str = "SUPER + A";

fn runtime_dir() -> PathBuf {
    PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into()))
}

fn control_socket() -> PathBuf {
    runtime_dir().join("aria-spike").join("control.sock")
}

fn hypr_socket(name: &str) -> PathBuf {
    let sig = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").expect("not running under Hyprland");
    runtime_dir().join("hypr").join(sig).join(name)
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("toggle") => toggle_client(),
        None => run_dock(),
        Some(other) => eprintln!("unknown command: {other} (use: toggle)"),
    }
}

// ---- client ---------------------------------------------------------------------------------

/// Sends `toggle` to the running dock; starts the dock first when nothing answers.
fn toggle_client() {
    if send("toggle").is_ok() {
        return;
    }
    let exe = std::env::current_exe().expect("own path");
    Command::new(exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start the dock");
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(50));
        if send("toggle").is_ok() {
            return;
        }
    }
    eprintln!("the dock did not start");
}

fn send(command: &str) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(control_socket())?;
    stream.write_all(format!("{command}\n").as_bytes())
}

// ---- Hyprland -------------------------------------------------------------------------------

/// Registers the hotkey and the layer rule at runtime. `unbind` first, so a restarted dock
/// never leaves two binds that would toggle twice.
fn register_with_hyprland() {
    let exe = std::env::current_exe().expect("own path");
    let lua = format!(
        r#"hl.unbind("{HOTKEY}"); hl.bind("{HOTKEY}", hl.dsp.exec_cmd("{} toggle"), {{ description = "Toggle Aria" }}); hl.layer_rule({{ name = "aria-dock", match = {{ namespace = "^aria-dock$" }}, blur = false, no_anim = true }})"#,
        exe.display()
    );
    match Command::new("hyprctl").args(["eval", &lua]).output() {
        Ok(out) => eprintln!("hyprland: {}", String::from_utf8_lossy(&out.stdout).trim()),
        Err(e) => eprintln!("hyprland: hyprctl failed: {e}"),
    }
}

/// Re-registers after every config reload: a reload drops runtime binds.
fn watch_hyprland_events() {
    std::thread::spawn(|| {
        let Ok(stream) = UnixStream::connect(hypr_socket(".socket2.sock")) else {
            eprintln!("hyprland: no event socket");
            return;
        };
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            if line.starts_with("configreloaded>>") {
                eprintln!("hyprland: config reloaded, registering again");
                register_with_hyprland();
            }
        }
    });
}

/// The focused monitor as Hyprland sees it: name and top-left corner in layout coordinates.
fn focused_monitor() -> Option<(String, i32, i32)> {
    let mut stream = UnixStream::connect(hypr_socket(".socket.sock")).ok()?;
    stream.write_all(b"j/monitors").ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;
    let monitors: serde_json::Value = serde_json::from_str(&body).ok()?;
    let m = monitors.as_array()?.iter().find(|m| m["focused"].as_bool() == Some(true))?;
    Some((m["name"].as_str()?.to_string(), m["x"].as_i64()? as i32, m["y"].as_i64()? as i32))
}

/// The GDK monitor at the same layout position (both are in logical pixels).
fn gdk_monitor_at(x: i32, y: i32) -> Option<gdk::Monitor> {
    let display = gdk::Display::default()?;
    (0..display.n_monitors())
        .filter_map(|i| display.monitor(i))
        .find(|m| {
            let g = m.geometry();
            g.x() == x && g.y() == y
        })
}

// ---- dock -----------------------------------------------------------------------------------

fn run_dock() {
    let socket = control_socket();
    if UnixStream::connect(&socket).is_ok() {
        eprintln!("the dock is already running");
        return;
    }
    std::fs::create_dir_all(socket.parent().unwrap()).expect("runtime dir");
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).expect("control socket");

    // WebKitGTK's DMA-BUF renderer draws nothing in a layer-shell surface here (found by the
    // first Aria dock); the shared-memory path works.
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    gtk::init().expect("gtk");
    let window = build_window();
    let webview = WebView::new();
    webview.set_background_color(&gdk::RGBA::new(0.0, 0.0, 0.0, 0.0));
    webview.load_html(include_str!("dock.html"), None);
    window.add(&webview);
    window.realize();

    // Clicks pass through the clear part of the veil; only the dock column takes them.
    window.connect_size_allocate(|window, alloc| {
        let rect = gtk::cairo::RectangleInt::new(alloc.width() - DOCK_WIDTH, 0, DOCK_WIDTH, alloc.height());
        window.input_shape_combine_region(Some(&gtk::cairo::Region::create_rectangle(&rect)));
    });

    // `open` is what the user asked for; the surface stays mapped while the page animates closed.
    // `generation` cancels a pending hide when the dock is reopened mid-close.
    let open = Rc::new(Cell::new(false));
    let generation = Rc::new(Cell::new(0u64));
    let hide = {
        let (window, webview, open, generation) = (window.clone(), webview.clone(), open.clone(), generation.clone());
        move || {
            if !open.get() {
                return;
            }
            open.set(false);
            window.set_keyboard_mode(KeyboardMode::None);
            webview.run_javascript("window.ariaClose()", None::<&gio::Cancellable>, |_| {});
            let current = generation.get() + 1;
            generation.set(current);
            let (window, open, generation) = (window.clone(), open.clone(), generation.clone());
            glib::timeout_add_local_once(Duration::from_millis(CLOSE_MS), move || {
                if generation.get() == current && !open.get() {
                    window.hide();
                }
            });
        }
    };
    {
        let hide = hide.clone();
        window.connect_key_press_event(move |_, event| {
            if event.keyval() == gdk::keys::constants::Escape {
                hide();
            }
            glib::Propagation::Proceed
        });
    }

    // Commands arrive on a socket thread and are handled on the GTK thread.
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for stream in listener.incoming().map_while(Result::ok) {
            if let Some(Ok(line)) = BufReader::new(stream).lines().next() {
                let _ = tx.send(line);
            }
        }
    });
    glib::timeout_add_local(Duration::from_millis(16), move || {
        while let Ok(command) = rx.try_recv() {
            if command != "toggle" {
                continue;
            }
            if open.get() {
                hide();
                continue;
            }
            open.set(true);
            generation.set(generation.get() + 1);
            let mut name = String::from("(compositor's choice)");
            if let Some((monitor, x, y)) = focused_monitor() {
                if let Some(gdk_monitor) = gdk_monitor_at(x, y) {
                    window.set_monitor(&gdk_monitor);
                    name = monitor;
                }
            }
            window.set_keyboard_mode(KeyboardMode::OnDemand);
            window.show_all();
            window.present();
            webview.run_javascript(&format!("window.ariaOpen({:?})", name), None::<&gio::Cancellable>, |_| {});
            eprintln!("dock: opened on {name}");
        }
        glib::ControlFlow::Continue
    });

    register_with_hyprland();
    watch_hyprland_events();
    eprintln!("dock: ready, press {HOTKEY}");
    gtk::main();
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
    // set_size_request, not set_default_size: with a WebView child and only a default size the
    // layer surface never paints (found by the first Aria dock). The compositor sets the height.
    window.set_size_request(WIDTH, 100);
    window.set_app_paintable(true);
    if let Some(visual) = WidgetExt::screen(&window).and_then(|s| s.rgba_visual()) {
        window.set_visual(Some(&visual));
    }
    window
}
