# Spike: open the dock on the focused monitor with SUPER+A

Throwaway. Question: can Aria own its hotkey and its placement without the user editing
`~/.config/hypr/`?

- **Hotkey:** at start, and again after every `configreloaded` event, the dock runs
  `hyprctl eval 'hl.unbind("SUPER + A"); hl.bind("SUPER + A", hl.dsp.exec_cmd("<exe> toggle"), …)'`.
  Nothing is written to disk; the bind lives only while Hyprland runs.
- **Single instance:** a Unix socket at `$XDG_RUNTIME_DIR/aria-spike/control.sock`. `toggle`
  starts the dock if it is not running.
- **Focused monitor:** on every open it asks Hyprland's IPC socket for the focused monitor and
  moves the layer surface there before mapping it.

Run: `cargo run --release` (or `target/release/dock-open-spike toggle`), then press SUPER+A.
Esc or SUPER+A again hides it.
