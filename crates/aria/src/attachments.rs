//! The files attached to a chat message: saved privately under `~/.aria/attachments/<session>/<yyyy-mm-dd>/`, and
//! named in the message Pi gets (an image the model can see also goes to it as an image).

use aria_core::bridge::Attachment;
use base64::Engine as _;
use serde_json::{json, Value};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// At most this many attachments in one message.
pub const MAX_FILES: usize = 10;
/// At most this many bytes in one attachment.
pub const MAX_EACH: u64 = 20 * 1024 * 1024;
/// At most this many bytes in all the attachments of one message.
pub const MAX_TOTAL: u64 = 50 * 1024 * 1024;
/// A safe name is cut to this many characters.
const NAME_LIMIT: usize = 80;
/// The image types a model can be sent as images.
const IMAGE_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// An attachment on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Saved {
    /// The name the person gave it (made fit for a line: no brackets, no line breaks).
    pub name: String,
    pub mime: String,
    pub size: u64,
    pub path: PathBuf,
    /// The image's media type and base64 data, for the image types a model can be sent.
    pub image: Option<(String, String)>,
}

/// A file name made of `[A-Za-z0-9._-]` only, at most 80 characters: accents lose their mark, anything else becomes
/// `_`, and the name never starts with a dot (no `..`, no hidden file). The extension is kept when the name is cut.
pub fn safe_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        let c = plain(c);
        let c = if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' };
        if c == '_' && out.ends_with('_') {
            continue;
        }
        out.push(c);
    }
    let out = out.trim_start_matches(['.', '_', '-']).trim_end_matches(['_', '.']).to_string();
    let out = if out.is_empty() { "file".to_string() } else { out };
    if out.len() <= NAME_LIMIT {
        return out;
    }
    match out.rfind('.') {
        Some(dot) if out.len() - dot <= 16 && dot > 0 => {
            let ext = &out[dot..];
            format!("{}{ext}", &out[..NAME_LIMIT - ext.len()])
        }
        _ => out[..NAME_LIMIT].to_string(),
    }
}

/// The letter without its accent, for the Latin letters a name is likely to have.
fn plain(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'È' | 'É' | 'Ê' | 'Ë' => 'E',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'O',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
        'ñ' => 'n',
        'Ñ' => 'N',
        'ç' => 'c',
        'Ç' => 'C',
        other => other,
    }
}

/// The name as it goes in a line: brackets and line breaks would end the line early or split it.
fn line_name(name: &str) -> String {
    let name: String = name
        .chars()
        .map(|c| match c {
            '[' => '(',
            ']' => ')',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    let name = name.trim();
    if name.is_empty() {
        "file".to_string()
    } else {
        name.to_string()
    }
}

/// The media type, lower case (`image/jpg` is `image/jpeg`); `application/octet-stream` when the page sent none.
fn media_type(mime: &str) -> String {
    let mime = mime.trim().to_ascii_lowercase();
    match mime.as_str() {
        "" => "application/octet-stream".into(),
        "image/jpg" => "image/jpeg".into(),
        _ => mime,
    }
}

/// The bytes of an attachment's base64 (a `data:` URL prefix is allowed).
fn decode(data: &str) -> Result<Vec<u8>, String> {
    let data = match data.strip_prefix("data:") {
        Some(rest) => rest.split_once(',').map(|(_, body)| body).unwrap_or(rest),
        None => data,
    };
    let clean: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(clean.as_bytes())
        .map_err(|_| "An attachment could not be read.".to_string())
}

/// A folder made and kept private (0700).
fn private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

/// The next number free in a folder: one more than the highest `<n>-…` there.
fn next_number(dir: &Path) -> u64 {
    let highest = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str()?.split_once('-')?.0.parse::<u64>().ok())
        .max();
    highest.unwrap_or(0) + 1
}

/// Saves a message's attachments in `root/<session>/<today>/` and says what each became. Nothing is written when one
/// is over the limits or cannot be read, and what was written goes away when a later one fails (the message is then
/// refused as a whole). `Err` is a sentence for the person.
pub fn save(root: &Path, session: &str, today: &str, list: &[Attachment]) -> Result<Vec<Saved>, String> {
    if list.is_empty() {
        return Ok(Vec::new());
    }
    if list.len() > MAX_FILES {
        return Err(format!("At most {MAX_FILES} attachments can go in one message."));
    }
    let valid =
        |part: &str| !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'));
    if !valid(session) || !valid(today) {
        return Err("The attachments could not be saved.".into());
    }
    let mut decoded = Vec::with_capacity(list.len());
    let mut total = 0u64;
    for attachment in list {
        let bytes = decode(&attachment.data)?;
        let size = bytes.len() as u64;
        if size > MAX_EACH {
            return Err(format!("{} is larger than 20 MB.", line_name(&attachment.name)));
        }
        total += size;
        if total > MAX_TOTAL {
            return Err("The attachments of one message can be 50 MB together at most.".into());
        }
        decoded.push(bytes);
    }
    let failed = |_| "The attachments could not be saved.".to_string();
    let dir = root.join(session).join(today);
    for level in [root.to_path_buf(), root.join(session), dir.clone()] {
        private_dir(&level).map_err(failed)?;
    }
    let mut saved: Vec<Saved> = Vec::with_capacity(list.len());
    for (attachment, bytes) in list.iter().zip(decoded) {
        match write_one(&dir, attachment, &bytes) {
            Ok(one) => saved.push(one),
            Err(error) => {
                for one in &saved {
                    let _ = std::fs::remove_file(&one.path);
                }
                return Err(failed(error));
            }
        }
    }
    Ok(saved)
}

/// Writes one attachment under the next free number (another send may take a number at the same time: then the next).
fn write_one(dir: &Path, attachment: &Attachment, bytes: &[u8]) -> std::io::Result<Saved> {
    let name = safe_name(&attachment.name);
    let mut number = next_number(dir);
    let (path, mut file) = loop {
        let path = dir.join(format!("{number}-{name}"));
        match std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path) {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => number += 1,
            Err(error) => return Err(error),
        }
    };
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    let mime = media_type(&attachment.mime);
    let image = IMAGE_TYPES
        .contains(&mime.as_str())
        .then(|| (mime.clone(), base64::engine::general_purpose::STANDARD.encode(bytes)));
    Ok(Saved { name: line_name(&attachment.name), mime, size: bytes.len() as u64, path, image })
}

/// A size as a person reads it: `512 B`, `12 KB`, `1.3 MB`.
pub fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{} KB", (bytes + KB / 2) / KB)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}

/// The message Pi gets and its `images`: the text, then a line per attachment. An image goes as an image (and a line
/// saying where it is saved) when the model can see images (`images_ok`); every other file, and an image for a model
/// that cannot, is a line with its name, type, size and path for the model to read with its tools.
pub fn message(text: &str, saved: &[Saved], images_ok: bool) -> (String, Vec<Value>) {
    let mut lines = Vec::with_capacity(saved.len());
    let mut images = Vec::new();
    for one in saved {
        let path = one.path.display();
        match &one.image {
            Some((mime, data)) if images_ok => {
                lines.push(format!("[attached image: {} at {path}]", one.name));
                images.push(json!({"type": "image", "data": data, "mimeType": mime}));
            }
            _ => lines.push(format!("[attached: {} ({}, {}) at {path}]", one.name, one.mime, human_size(one.size))),
        }
    }
    let message = match (text.trim().is_empty(), lines.is_empty()) {
        (_, true) => text.to_string(),
        (true, false) => lines.join("\n"),
        (false, false) => format!("{text}\n\n{}", lines.join("\n")),
    };
    (message, images)
}

/// Today's date where the person is (`yyyy-mm-dd`, local time).
pub fn today() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0);
    // SAFETY: `localtime_r` only writes the `tm` it is given.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&now, &mut tm) }.is_null() {
        return "1970-01-01".into();
    }
    format!("{:04}-{:02}-{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn attachment(name: &str, mime: &str, bytes: &[u8]) -> Attachment {
        Attachment { name: name.into(), mime: mime.into(), data: b64(bytes) }
    }

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn unsafe_names_are_made_safe() {
        assert_eq!(safe_name("../../etc/passwd"), "etc_passwd");
        assert_eq!(safe_name("..%2F..%2Fetc%2Fpasswd"), "2F.._2Fetc_2Fpasswd");
        assert_eq!(safe_name(".."), "file");
        assert_eq!(safe_name(".bashrc"), "bashrc");
        assert_eq!(safe_name("Reseña de reunión final.pdf"), "Resena_de_reunion_final.pdf");
        assert_eq!(safe_name("a  b\tc.txt"), "a_b_c.txt");
        assert_eq!(safe_name("日本語.png"), "png");
        assert_eq!(safe_name(""), "file");
        let long = format!("{}.png", "x".repeat(200));
        let cut = safe_name(&long);
        assert_eq!(cut.len(), 80);
        assert!(cut.ends_with(".png"));
        assert!(safe_name(&"y".repeat(200)).len() == 80);
        for name in ["../../etc/passwd", "Reseña de reunión final.pdf", "a b", "$(rm -rf ~).sh", "x\ny"] {
            let safe = safe_name(name);
            assert!(safe.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')), "{safe}");
            assert!(!safe.starts_with('.'));
        }
    }

    #[test]
    fn files_are_private() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        let saved = save(&root, "s1", "2026-10-21", &[attachment("../../etc/passwd", "text/plain", b"hi")]).unwrap();
        let path = &saved[0].path;
        assert_eq!(path, &root.join("s1/2026-10-21/1-etc_passwd"));
        assert_eq!(std::fs::read(path).unwrap(), b"hi");
        assert_eq!(mode(path), 0o600);
        for folder in [root.clone(), root.join("s1"), root.join("s1/2026-10-21")] {
            assert_eq!(mode(&folder), 0o700, "{}", folder.display());
        }
        // The next one takes the next number, even with the same name.
        let again = save(&root, "s1", "2026-10-21", &[attachment("passwd", "text/plain", b"x")]).unwrap();
        assert_eq!(again[0].path, root.join("s1/2026-10-21/2-passwd"));
        // A session id that is not one never becomes a folder elsewhere.
        assert!(save(&root, "../evil", "2026-10-21", &[attachment("a", "text/plain", b"x")]).is_err());
    }

    #[test]
    fn limits_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        let eleven: Vec<_> = (0..11).map(|i| attachment(&format!("{i}.txt"), "text/plain", b"x")).collect();
        assert!(save(&root, "s1", "2026-10-21", &eleven).is_err());
        let ten: Vec<_> = (0..10).map(|i| attachment(&format!("{i}.txt"), "text/plain", b"x")).collect();
        assert_eq!(save(&root, "s1", "2026-10-21", &ten).unwrap().len(), 10);
        let big = vec![0u8; 21 * 1024 * 1024];
        assert!(save(&root, "s2", "2026-10-21", &[attachment("big.bin", "", &big)]).is_err());
        let part = vec![0u8; 17 * 1024 * 1024];
        let three: Vec<_> = (0..3).map(|i| attachment(&format!("{i}.bin"), "", &part)).collect();
        assert!(save(&root, "s3", "2026-10-21", &three).is_err(), "51 MB together");
        assert!(!root.join("s2").exists() && !root.join("s3").exists(), "nothing is written for a refused message");
        let bad = Attachment { name: "a".into(), mime: "".into(), data: "not base64!".into() };
        assert!(save(&root, "s4", "2026-10-21", &[bad]).is_err());
    }

    fn png(dir: &Path) -> Vec<Saved> {
        save(&dir.join("attachments"), "s1", "2026-10-21", &[attachment("shot.png", "image/png", b"PNGDATA")]).unwrap()
    }

    #[test]
    fn an_image_for_a_text_model_goes_as_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let saved = png(dir.path());
        let (text, images) = message("look", &saved, false);
        assert!(images.is_empty());
        assert_eq!(text, format!("look\n\n[attached: shot.png (image/png, 7 B) at {}]", saved[0].path.display()));
    }

    #[test]
    fn an_image_for_a_vision_model_goes_as_an_image_and_a_path_line() {
        let dir = tempfile::tempdir().unwrap();
        let saved = png(dir.path());
        let (text, images) = message("look", &saved, true);
        assert_eq!(text, format!("look\n\n[attached image: shot.png at {}]", saved[0].path.display()));
        assert_eq!(images, vec![json!({"type": "image", "data": b64(b"PNGDATA"), "mimeType": "image/png"})]);
    }

    #[test]
    fn other_files_go_as_path_lines() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        let pdf = vec![0u8; 12 * 1024];
        let list = [
            attachment("Informe [final].pdf", "application/pdf", &pdf),
            attachment("notes.txt", "", b"n"),
            attachment("diagram.svg", "image/svg+xml", b"<svg/>"),
        ];
        let saved = save(&root, "s1", "2026-10-21", &list).unwrap();
        let (text, images) = message("", &saved, true);
        assert!(images.is_empty(), "only png, jpeg, gif and webp go as images");
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(
            lines,
            vec![
                format!("[attached: Informe (final).pdf (application/pdf, 12 KB) at {}]", saved[0].path.display()),
                format!("[attached: notes.txt (application/octet-stream, 1 B) at {}]", saved[1].path.display()),
                format!("[attached: diagram.svg (image/svg+xml, 6 B) at {}]", saved[2].path.display()),
            ]
        );
        assert_eq!(message("just text", &[], true), ("just text".to_string(), vec![]));
        assert_eq!(human_size(1_363_149), "1.3 MB");
    }

    #[test]
    fn today_is_a_date() {
        let today = today();
        assert_eq!(today.len(), 10);
        assert_eq!(&today[4..5], "-");
    }
}
