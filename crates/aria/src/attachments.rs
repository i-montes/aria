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
    /// The image's media type and base64 data, for an image the session's model is sent (png, jpeg, gif or webp, and a
    /// model that can see images).
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
    let out = match unnamed_extension(name) {
        Some(ext) => format!("file.{ext}"),
        None if out.is_empty() => "file".to_string(),
        None => out,
    };
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

/// The extension of a name whose stem has no letter or digit left once made safe (`日本語.png` → `png`), so the name
/// becomes `file.<ext>`; None for any other name (a dot file such as `.bashrc` has no stem, and keeps its name).
fn unnamed_extension(name: &str) -> Option<String> {
    let (stem, ext) = name.rsplit_once('.')?;
    let ext: String = ext.chars().map(plain).collect();
    let stem_left = stem.chars().map(plain).any(|c| c.is_ascii_alphanumeric());
    let named = !stem.trim_matches(['.', ' ']).is_empty();
    (named && !stem_left && !ext.is_empty() && ext.len() <= 16 && ext.chars().all(|c| c.is_ascii_alphanumeric()))
        .then_some(ext)
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

/// The media type, lower case (`image/jpg` is `image/jpeg`), made only of `[a-z0-9.+/-]` (it goes in a line, where a
/// `]`, `,`, `(` or line break would end or split it); `application/octet-stream` when the page sent none or another.
fn media_type(mime: &str) -> String {
    let mime = mime.trim().to_ascii_lowercase();
    let fit =
        !mime.is_empty() && mime.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || ".+/-".contains(c));
    match mime.as_str() {
        _ if !fit => "application/octet-stream".into(),
        "image/jpg" => "image/jpeg".into(),
        _ => mime,
    }
}

/// Could this attachment go to a model as an image (png, jpeg, gif or webp)?
pub fn is_image(attachment: &Attachment) -> bool {
    IMAGE_TYPES.contains(&media_type(&attachment.mime).as_str())
}

/// An attachment's base64 without a `data:` URL prefix.
fn body(data: &str) -> &str {
    match data.strip_prefix("data:") {
        Some(rest) => rest.split_once(',').map(|(_, body)| body).unwrap_or(rest),
        None => data,
    }
}

/// How many bytes a base64 text holds, counted before anything is decoded (whitespace and padding are not data).
fn decoded_len(body: &str) -> u64 {
    let chars = body.bytes().filter(|b| !b.is_ascii_whitespace() && *b != b'=').count() as u64;
    chars * 3 / 4
}

/// The bytes of an attachment's base64 (`body`), and that base64 without whitespace (what a model is sent).
fn decode(body: &str) -> Result<(Vec<u8>, String), String> {
    let clean: String = body.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(clean.as_bytes())
        .map_err(|_| "An attachment could not be read.".to_string())?;
    Ok((bytes, clean))
}

/// A date folder's name: `yyyy-mm-dd`.
fn is_date(day: &str) -> bool {
    let b = day.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
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

/// Saves a message's attachments in `root/<session>/<today>/` and says what each became; an image keeps its base64 for
/// the model only when the model can see images (`images_ok`). Nothing is written (or decoded) when one is over the
/// limits, nothing is written when one cannot be read, and what was written goes away when a later one fails (the
/// message is then refused as a whole). `Err` is a sentence for the person.
pub fn save(
    root: &Path,
    session: &str,
    today: &str,
    list: &[Attachment],
    images_ok: bool,
) -> Result<Vec<Saved>, String> {
    if list.is_empty() {
        return Ok(Vec::new());
    }
    if list.len() > MAX_FILES {
        return Err(format!("At most {MAX_FILES} attachments can go in one message."));
    }
    if !aria_core::sessions::valid_id(session) || !is_date(today) {
        return Err("The attachments could not be saved.".into());
    }
    let too_big = |attachment: &Attachment| format!("{} is larger than 20 MB.", line_name(&attachment.name));
    const TOO_MUCH: &str = "The attachments of one message can be 50 MB together at most.";
    // The sizes the base64 implies, before any of it is decoded.
    let mut total = 0u64;
    for attachment in list {
        let size = decoded_len(body(&attachment.data));
        if size > MAX_EACH {
            return Err(too_big(attachment));
        }
        total += size;
        if total > MAX_TOTAL {
            return Err(TOO_MUCH.into());
        }
    }
    let mut decoded = Vec::with_capacity(list.len());
    let mut total = 0u64;
    for attachment in list {
        let (bytes, clean) = decode(body(&attachment.data))?;
        let size = bytes.len() as u64;
        if size > MAX_EACH {
            return Err(too_big(attachment));
        }
        total += size;
        if total > MAX_TOTAL {
            return Err(TOO_MUCH.into());
        }
        let image = (images_ok && is_image(attachment)).then_some(clean);
        decoded.push((bytes, image));
    }
    let failed = |_| "The attachments could not be saved.".to_string();
    let dir = root.join(session).join(today);
    for level in [root.to_path_buf(), root.join(session), dir.clone()] {
        private_dir(&level).map_err(failed)?;
    }
    let mut saved: Vec<Saved> = Vec::with_capacity(list.len());
    for (attachment, (bytes, image)) in list.iter().zip(decoded) {
        match write_one(&dir, attachment, &bytes, image) {
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
/// `image` is the base64 a model is sent, when it is.
fn write_one(dir: &Path, attachment: &Attachment, bytes: &[u8], image: Option<String>) -> std::io::Result<Saved> {
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
    let image = image.map(|data| (mime.clone(), data));
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

/// The message Pi gets and its `images`: the text, then a line per attachment. An image saved for the model (`image`)
/// goes as an image and a line saying where it is saved; every other file, and an image for a model that cannot see
/// images, is a line with its name, type, size and path for the model to read with its tools.
pub fn message(text: &str, saved: &[Saved]) -> (String, Vec<Value>) {
    let mut lines = Vec::with_capacity(saved.len());
    let mut images = Vec::new();
    for one in saved {
        let path = one.path.display();
        match &one.image {
            Some((mime, data)) => {
                lines.push(format!("[attached image: {} at {path}]", one.name));
                images.push(json!({"type": "image", "data": data, "mimeType": mime}));
            }
            None => lines.push(format!("[attached: {} ({}, {}) at {path}]", one.name, one.mime, human_size(one.size))),
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
        assert_eq!(safe_name("日本語.png"), "file.png");
        assert_eq!(safe_name("_.tar"), "file.tar");
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
        let saved =
            save(&root, "s1", "2026-10-21", &[attachment("../../etc/passwd", "text/plain", b"hi")], false).unwrap();
        let path = &saved[0].path;
        assert_eq!(path, &root.join("s1/2026-10-21/1-etc_passwd"));
        assert_eq!(std::fs::read(path).unwrap(), b"hi");
        assert_eq!(mode(path), 0o600);
        for folder in [root.clone(), root.join("s1"), root.join("s1/2026-10-21")] {
            assert_eq!(mode(&folder), 0o700, "{}", folder.display());
        }
        // The next one takes the next number, even with the same name.
        let again = save(&root, "s1", "2026-10-21", &[attachment("passwd", "text/plain", b"x")], false).unwrap();
        assert_eq!(again[0].path, root.join("s1/2026-10-21/2-passwd"));
        // A session id that is not one never becomes a folder elsewhere.
        assert!(save(&root, "../evil", "2026-10-21", &[attachment("a", "text/plain", b"x")], false).is_err());
    }

    #[test]
    fn limits_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        let eleven: Vec<_> = (0..11).map(|i| attachment(&format!("{i}.txt"), "text/plain", b"x")).collect();
        assert!(save(&root, "s1", "2026-10-21", &eleven, false).is_err());
        let ten: Vec<_> = (0..10).map(|i| attachment(&format!("{i}.txt"), "text/plain", b"x")).collect();
        assert_eq!(save(&root, "s1", "2026-10-21", &ten, false).unwrap().len(), 10);
        let big = vec![0u8; 21 * 1024 * 1024];
        assert!(save(&root, "s2", "2026-10-21", &[attachment("big.bin", "", &big)], false).is_err());
        let part = vec![0u8; 17 * 1024 * 1024];
        let three: Vec<_> = (0..3).map(|i| attachment(&format!("{i}.bin"), "", &part)).collect();
        assert!(save(&root, "s3", "2026-10-21", &three, false).is_err(), "51 MB together");
        assert!(!root.join("s2").exists() && !root.join("s3").exists(), "nothing is written for a refused message");
        let bad = Attachment { name: "a".into(), mime: "".into(), data: "not base64!".into() };
        assert!(save(&root, "s4", "2026-10-21", &[bad], false).is_err());
    }

    fn png(dir: &Path, images_ok: bool) -> Vec<Saved> {
        let list = [attachment("shot.png", "image/png", b"PNGDATA")];
        save(&dir.join("attachments"), "s1", "2026-10-21", &list, images_ok).unwrap()
    }

    #[test]
    fn an_image_for_a_text_model_goes_as_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let saved = png(dir.path(), false);
        assert_eq!(saved[0].image, None, "no image data is kept for a model that cannot see it");
        let (text, images) = message("look", &saved);
        assert!(images.is_empty());
        assert_eq!(text, format!("look\n\n[attached: shot.png (image/png, 7 B) at {}]", saved[0].path.display()));
    }

    #[test]
    fn an_image_for_a_vision_model_goes_as_an_image_and_a_path_line() {
        let dir = tempfile::tempdir().unwrap();
        let saved = png(dir.path(), true);
        let (text, images) = message("look", &saved);
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
        let saved = save(&root, "s1", "2026-10-21", &list, true).unwrap();
        let (text, images) = message("", &saved);
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
        assert_eq!(message("just text", &[]), ("just text".to_string(), vec![]));
        assert_eq!(human_size(1_363_149), "1.3 MB");
    }

    #[test]
    fn an_odd_media_type_is_octet_stream() {
        for odd in ["text/plain]", "a,b", "text/x(y)", "text/plain\n[attached: x"] {
            assert_eq!(media_type(odd), "application/octet-stream", "{odd:?}");
        }
        assert_eq!(media_type(" Image/JPG "), "image/jpeg");
        assert_eq!(media_type("application/vnd.ms-excel"), "application/vnd.ms-excel");
        assert_eq!(media_type("image/svg+xml"), "image/svg+xml");
        let dir = tempfile::tempdir().unwrap();
        let saved =
            save(&dir.path().join("a"), "s1", "2026-10-21", &[attachment("x.txt", "text/plain]\n(", b"x")], true);
        let (text, _) = message("", &saved.unwrap());
        assert!(text.starts_with("[attached: x.txt (application/octet-stream, 1 B) at "), "{text}");
        assert_eq!(text.lines().count(), 1);
    }

    #[test]
    fn sizes_are_refused_before_anything_is_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        // Not base64 at all, but long enough to stand for 21 MB: the size refuses it, not the decoding.
        let huge = Attachment { name: "huge.bin".into(), mime: "".into(), data: "!".repeat(28 * 1024 * 1024) };
        assert_eq!(save(&root, "s1", "2026-10-21", &[huge], false), Err("huge.bin is larger than 20 MB.".into()));
        let part = "!".repeat(24 * 1024 * 1024); // 18 MB each
        let three: Vec<_> =
            (0..3).map(|i| Attachment { name: format!("{i}"), mime: "".into(), data: part.clone() }).collect();
        assert_eq!(
            save(&root, "s1", "2026-10-21", &three, false),
            Err("The attachments of one message can be 50 MB together at most.".into())
        );
        assert_eq!(decoded_len(&b64(&[1u8; 7])), 7);
        assert_eq!(decoded_len(&b64(&[1u8; 9])), 9);
        assert!(!root.exists());
    }

    #[test]
    fn only_a_session_id_and_a_date_become_folders() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("attachments");
        let one = [attachment("a.txt", "text/plain", b"x")];
        for (session, day) in
            [("s_1", "2026-10-21"), ("-s1", "2026-10-21"), ("s1", "2026-1-21"), ("s1", "today"), ("s1", "2026_10_21")]
        {
            assert!(save(&root, session, day, &one, false).is_err(), "{session} {day}");
        }
        assert!(save(&root, "a1b2-c3", "2026-10-21", &one, false).is_ok());
    }

    #[test]
    fn the_image_sent_is_the_base64_that_came() {
        let dir = tempfile::tempdir().unwrap();
        let data = format!("{}\n{}", &b64(b"PNGDATA!")[..4], &b64(b"PNGDATA!")[4..]);
        let list = [Attachment { name: "a.png".into(), mime: "image/png".into(), data }];
        let saved = save(&dir.path().join("a"), "s1", "2026-10-21", &list, true).unwrap();
        assert_eq!(saved[0].image, Some(("image/png".into(), b64(b"PNGDATA!"))));
    }

    #[test]
    fn today_is_a_date() {
        let today = today();
        assert_eq!(today.len(), 10);
        assert_eq!(&today[4..5], "-");
    }
}
