//! A tab's earlier history: the messages its compactions took out of Pi's context, read straight from Pi's session
//! files (JSONL), oldest segment first, a page at a time.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// How many messages a page holds.
pub const PAGE_SIZE: usize = 40;

/// The custom type of the message a rotation seeds a new segment with (the previous segment's summary).
pub const SEED_TYPE: &str = "aria-earlier";

/// Where a page starts: `index` in the earlier list of the tab's `segment`-th segment (oldest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EarlierCursor {
    pub segment: usize,
    pub index: usize,
}

/// A page of earlier messages, in order, as Pi wrote them. `cursor` points at its first message (none when the page is
/// empty); `more` says whether anything is before it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EarlierPage {
    pub messages: Vec<Value>,
    pub cursor: Option<EarlierCursor>,
    pub more: bool,
}

/// The `PAGE_SIZE` earlier messages just before `cursor` (no cursor: the end of the list). `segments` are the tab's Pi
/// session ids, oldest first, whose files are in `folder`; the cursor's `segment` indexes this list.
pub fn earlier_page(folder: &Path, segments: &[String], cursor: Option<EarlierCursor>) -> EarlierPage {
    let last = segments.len().saturating_sub(1);
    let lists: Vec<Vec<Value>> = segments
        .iter()
        .enumerate()
        .map(|(i, id)| segment_file(folder, id).map(|text| earlier_of(&text, i == last)).unwrap_or_default())
        .collect();
    let starts: Vec<usize> = lists
        .iter()
        .scan(0, |sum, list| {
            let start = *sum;
            *sum += list.len();
            Some(start)
        })
        .collect();
    let total: usize = lists.iter().map(Vec::len).sum();
    let end = match cursor {
        None => total,
        Some(c) if c.segment < lists.len() && c.index <= lists[c.segment].len() => starts[c.segment] + c.index,
        // A cursor that points nowhere (a forged one, or a chain that changed): nothing before it.
        Some(_) => return EarlierPage { messages: vec![], cursor: None, more: false },
    };
    let start = end.saturating_sub(PAGE_SIZE);
    let messages: Vec<Value> = lists.into_iter().flatten().skip(start).take(end - start).collect();
    let cursor = (!messages.is_empty()).then(|| {
        // The segment holding `start`: the last one starting at or before it (an empty one shares its start with the
        // segment after it, so the last such is the one with messages).
        let segment = (0..starts.len()).rev().find(|&i| starts[i] <= start).unwrap_or(0);
        EarlierCursor { segment, index: start - starts[segment] }
    });
    EarlierPage { messages, cursor, more: start > 0 }
}

/// The text of segment `id`'s file: `<time>_<id>.jsonl` whose header names `id` (the latest name when several match).
fn segment_file(folder: &Path, id: &str) -> Option<String> {
    let suffix = format!("_{id}.jsonl");
    let mut names: Vec<_> = std::fs::read_dir(folder)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(&suffix)))
        .collect();
    names.sort();
    names.into_iter().rev().find_map(|path| {
        let text = std::fs::read_to_string(&path).ok()?;
        let header: Value = serde_json::from_str(text.lines().next()?).ok()?;
        (header["type"] == "session" && header["id"] == id).then_some(text)
    })
}

/// One segment's earlier messages: the entries on its active path before its latest compaction's cut, as the messages
/// Pi's context makes of them (`message`, `custom_message`, `branch_summary`); with no compaction, nothing for the
/// current segment and the whole path for an older one. The seeds and system messages are left out.
fn earlier_of(text: &str, current: bool) -> Vec<Value> {
    let entries: Vec<Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|entry| entry["type"] != "session" && entry["id"].is_string())
        .collect();
    let by_id: std::collections::HashMap<&str, &Value> =
        entries.iter().filter_map(|entry| Some((entry["id"].as_str()?, entry))).collect();

    // The active path, leaf to root, then reversed.
    let mut path: Vec<&Value> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut next = entries.last();
    while let Some(entry) = next {
        let Some(id) = entry["id"].as_str() else { break };
        if !seen.insert(id) {
            break;
        }
        path.push(entry);
        next = entry["parentId"].as_str().and_then(|parent| by_id.get(parent).copied());
    }
    path.reverse();

    let cut = match path.iter().rposition(|entry| entry["type"] == "compaction") {
        Some(at) => {
            let kept = path[at]["firstKeptEntryId"].as_str();
            path[..at].iter().position(|entry| entry["id"].as_str() == kept).unwrap_or(at)
        }
        None if current => 0,
        None => path.len(),
    };
    path[..cut]
        .iter()
        .filter_map(|entry| as_message(entry))
        .filter(|message| message.is_object() && !is_seed(message) && message["role"] != "system")
        .collect()
}

/// An entry as the message Pi's context makes of it (`sessionEntryToContextMessages`): a `message` as written, a
/// `custom_message` as `createCustomMessage` shapes it, a `branch_summary` with a summary as `createBranchSummaryMessage`
/// does. Anything else (a compaction, a model change, an extension's own entry) is no message.
fn as_message(entry: &Value) -> Option<Value> {
    match entry["type"].as_str()? {
        "message" => Some(entry["message"].clone()),
        "custom_message" => {
            let content = match &entry["content"] {
                Value::Null => Value::Array(vec![]),
                content => content.clone(),
            };
            Some(without_nulls(serde_json::json!({
                "role": "custom",
                "customType": entry["customType"],
                "content": content,
                "display": entry["display"],
                "details": entry["details"],
                "timestamp": millis(&entry["timestamp"]),
            })))
        }
        "branch_summary" if entry["summary"].as_str().is_some_and(|s| !s.is_empty()) => {
            Some(without_nulls(serde_json::json!({
                "role": "branchSummary",
                "summary": entry["summary"],
                "fromId": entry["fromId"],
                "timestamp": millis(&entry["timestamp"]),
            })))
        }
        _ => None,
    }
}

/// A field the entry does not have is left out, as `JSON.stringify` drops an `undefined`.
fn without_nulls(mut message: Value) -> Value {
    if let Some(fields) = message.as_object_mut() {
        fields.retain(|_, value| !value.is_null());
    }
    message
}

/// An entry's ISO 8601 time (`2026-10-21T00:00:01.000Z`, UTC, as Pi writes it) in milliseconds since the epoch, as
/// `new Date(timestamp).getTime()`; null when it does not read as one.
fn millis(timestamp: &Value) -> Value {
    timestamp.as_str().and_then(iso_millis).map(Value::from).unwrap_or(Value::Null)
}

fn iso_millis(text: &str) -> Option<i64> {
    let (date, time) = text.strip_suffix('Z')?.split_once('T')?;
    let mut parts = date.splitn(3, '-').map(str::parse::<i64>);
    let (year, month, day) = (parts.next()?.ok()?, parts.next()?.ok()?, parts.next()?.ok()?);
    let (clock, fraction) = time.split_once('.').unwrap_or((time, "0"));
    let mut parts = clock.splitn(3, ':').map(str::parse::<i64>);
    let (hour, minute, second) = (parts.next()?.ok()?, parts.next()?.ok()?, parts.next()?.ok()?);
    let ms: i64 = format!("{fraction:0<3}").get(..3)?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    // Days since 1970-01-01 (Howard Hinnant's days_from_civil).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000 + ms)
}

fn is_seed(message: &Value) -> bool {
    message["role"] == "custom" && message["customType"] == SEED_TYPE
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header(id: &str) -> String {
        json!({"type": "session", "version": 3, "id": id, "timestamp": "2026-10-21T00:00:00.000Z", "cwd": "/work"})
            .to_string()
    }

    fn message(id: &str, parent: Option<&str>, text: &str) -> String {
        let role = if text.starts_with('a') { "assistant" } else { "user" };
        json!({"type": "message", "id": id, "parentId": parent, "timestamp": "2026-10-21T00:00:01.000Z",
               "message": {"role": role, "content": text, "timestamp": 1}})
        .to_string()
    }

    fn compaction(id: &str, parent: &str, first_kept: &str) -> String {
        json!({"type": "compaction", "id": id, "parentId": parent, "timestamp": "2026-10-21T00:00:02.000Z",
               "summary": "Earlier work.", "firstKeptEntryId": first_kept, "tokensBefore": 1000})
        .to_string()
    }

    fn seed(id: &str, parent: Option<&str>) -> String {
        json!({"type": "custom_message", "id": id, "parentId": parent, "timestamp": "2026-10-21T00:00:03.000Z",
               "customType": SEED_TYPE, "content": "Earlier work.", "display": true, "details": {"from": "s1"}})
        .to_string()
    }

    fn write(dir: &Path, segment: &str, lines: &[String]) {
        let body = std::iter::once(header(segment)).chain(lines.iter().cloned()).collect::<Vec<_>>().join("\n");
        std::fs::write(dir.join(format!("2026-10-21T00-00-00-000Z_{segment}.jsonl")), body + "\n").unwrap();
    }

    /// A linear run of messages `prefix0..prefix{n-1}`, the first one under `parent`; returns the lines and the last id.
    fn run(prefix: &str, n: usize, parent: Option<&str>) -> (Vec<String>, String) {
        let mut lines = Vec::new();
        let mut last = parent.map(str::to_string);
        for i in 0..n {
            let id = format!("{prefix}{i}");
            lines.push(message(&id, last.as_deref(), &format!("u {prefix}{i}")));
            last = Some(id);
        }
        (lines, last.unwrap_or_default())
    }

    fn texts(page: &EarlierPage) -> Vec<String> {
        page.messages.iter().map(|m| m["content"].as_str().unwrap_or_default().to_string()).collect()
    }

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn an_abandoned_branch_is_not_shown() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s1",
            &[
                message("m1", None, "u one"),
                message("m2", Some("m1"), "a two"),
                message("m3", Some("m2"), "u abandoned"),
                message("m4", Some("m2"), "u four"),
                message("m5", Some("m4"), "a five"),
            ],
        );
        write(dir.path(), "s2", &[message("n1", None, "u now")]);
        let page = earlier_page(dir.path(), &ids(&["s1", "s2"]), None);
        assert_eq!(texts(&page), ["u one", "a two", "u four", "a five"]);
        assert_eq!(page.cursor, Some(EarlierCursor { segment: 0, index: 0 }));
        assert!(!page.more);
        assert_eq!(
            page.messages[1],
            json!({"role": "assistant", "content": "a two", "timestamp": 1}),
            "as Pi wrote it"
        );
    }

    #[test]
    fn the_latest_compaction_cuts_the_current_segment() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s1",
            &[
                message("m1", None, "u one"),
                message("m2", Some("m1"), "a two"),
                message("m3", Some("m2"), "u three"),
                compaction("c1", "m3", "m2"),
                message("m4", Some("c1"), "a four"),
                message("m5", Some("m4"), "u five"),
                compaction("c2", "m5", "m4"),
                message("m6", Some("c2"), "a six"),
            ],
        );
        let page = earlier_page(dir.path(), &ids(&["s1"]), None);
        assert_eq!(texts(&page), ["u one", "a two", "u three"], "the cut is c2's firstKeptEntryId, m4");
    }

    #[test]
    fn a_compaction_that_keeps_nothing_cuts_at_itself() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s1",
            &[message("m1", None, "u one"), message("m2", Some("m1"), "a two"), compaction("c1", "m2", "c1")],
        );
        assert_eq!(texts(&earlier_page(dir.path(), &ids(&["s1"]), None)), ["u one", "a two"]);
    }

    #[test]
    fn a_current_segment_without_a_compaction_has_nothing_earlier() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "s1", &[message("m1", None, "u one"), message("m2", Some("m1"), "a two")]);
        let page = earlier_page(dir.path(), &ids(&["s1"]), None);
        assert_eq!(page, EarlierPage { messages: vec![], cursor: None, more: false });
    }

    #[test]
    fn two_segments_are_joined_oldest_first_and_the_seed_is_never_repeated() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s1",
            &[
                message("m1", None, "u one"),
                message("m2", Some("m1"), "a two"),
                message("m3", Some("m2"), "u three"),
                compaction("c1", "m3", "m3"),
                message("m4", Some("c1"), "a four"),
            ],
        );
        // The rotation's new segment: the seed, then the messages the compaction kept, then the conversation goes on.
        let seed_as_message = json!({"type": "message", "id": "k0", "parentId": "e0", "timestamp": "t",
            "message": {"role": "custom", "customType": SEED_TYPE, "content": "Earlier work.", "display": true}})
        .to_string();
        write(
            dir.path(),
            "s2",
            &[
                seed("e0", None),
                seed_as_message,
                message("k1", Some("k0"), "u three"),
                message("k2", Some("k1"), "a four"),
                message("k3", Some("k2"), "u five"),
                compaction("c2", "k3", "k3"),
                message("k4", Some("c2"), "a six"),
            ],
        );
        let page = earlier_page(dir.path(), &ids(&["s1", "s2"]), None);
        assert_eq!(texts(&page), ["u one", "a two", "u three", "a four"]);
        assert_eq!(page.cursor, Some(EarlierCursor { segment: 0, index: 0 }));
    }

    #[test]
    fn pages_hold_forty_messages_back_from_the_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let (old, _) = run("p", 50, None);
        write(dir.path(), "s1", &old);
        let (mut now, last) = run("q", 45, None);
        now.push(compaction("c1", &last, "q40"));
        write(dir.path(), "s2", &now);
        let segments = ids(&["s1", "s2"]);

        let first = earlier_page(dir.path(), &segments, None);
        assert_eq!(first.messages.len(), 40);
        assert_eq!(texts(&first)[0], "u q0");
        assert_eq!(texts(&first)[39], "u q39");
        assert_eq!(first.cursor, Some(EarlierCursor { segment: 1, index: 0 }));
        assert!(first.more);

        let second = earlier_page(dir.path(), &segments, first.cursor);
        assert_eq!(second.messages.len(), 40);
        assert_eq!(texts(&second)[0], "u p10");
        assert_eq!(texts(&second)[39], "u p49");
        assert_eq!(second.cursor, Some(EarlierCursor { segment: 0, index: 10 }));
        assert!(second.more);

        let third = earlier_page(dir.path(), &segments, second.cursor);
        assert_eq!(texts(&third), (0..10).map(|i| format!("u p{i}")).collect::<Vec<_>>());
        assert_eq!(third.cursor, Some(EarlierCursor { segment: 0, index: 0 }));
        assert!(!third.more);

        let past = earlier_page(dir.path(), &segments, third.cursor);
        assert_eq!(past, EarlierPage { messages: vec![], cursor: None, more: false });
    }

    #[test]
    fn a_missing_segment_file_is_skipped_and_bad_lines_too() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "s2",
            &["{ not json".to_string(), message("m1", None, "u one"), "".into(), message("m2", Some("m1"), "a two")],
        );
        // A file whose name matches but whose header names another session is not the segment's.
        std::fs::write(dir.path().join("2026-10-20T00-00-00-000Z_s3.jsonl"), header("other") + "\n").unwrap();
        let page = earlier_page(dir.path(), &ids(&["gone", "s2", "s3"]), None);
        assert_eq!(texts(&page), ["u one", "a two"]);
        assert_eq!(page.cursor, Some(EarlierCursor { segment: 1, index: 0 }));
        assert!(earlier_page(&dir.path().join("nowhere"), &ids(&["s2"]), None).messages.is_empty());
    }

    #[test]
    fn custom_messages_and_branch_summaries_read_as_pis_context_shapes_them_and_system_messages_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let background = json!({"type": "custom_message", "id": "b1", "parentId": "m1",
            "timestamp": "2026-10-21T00:00:01.500Z", "customType": "aria-background", "content": "the job ended",
            "display": true, "details": {"job": 2}})
        .to_string();
        let bare = json!({"type": "custom_message", "id": "b2", "parentId": "b1",
            "timestamp": "2026-10-21T00:00:02.000Z", "customType": "note", "display": false})
        .to_string();
        let branch = json!({"type": "branch_summary", "id": "r1", "parentId": "b2",
            "timestamp": "2026-10-21T00:00:03.000Z", "fromId": "x9", "summary": "Tried another way."})
        .to_string();
        let empty_branch = json!({"type": "branch_summary", "id": "r2", "parentId": "r1",
            "timestamp": "2026-10-21T00:00:03.000Z", "fromId": "x9", "summary": ""})
        .to_string();
        let system = json!({"type": "message", "id": "y1", "parentId": "r2", "timestamp": "2026-10-21T00:00:04.000Z",
            "message": {"role": "system", "content": "", "timestamp": 4}})
        .to_string();
        let model = json!({"type": "model_change", "id": "z1", "parentId": "y1", "timestamp": "t",
            "provider": "p", "modelId": "m"})
        .to_string();
        write(
            dir.path(),
            "s1",
            &[
                message("m1", None, "u one"),
                background,
                bare,
                seed("e1", Some("b2")).replace("\"parentId\":null", "\"parentId\":\"b2\""),
                branch,
                empty_branch,
                system,
                model,
                message("m2", Some("z1"), "a two"),
            ],
        );
        write(dir.path(), "s2", &[message("n1", None, "u now")]);
        let page = earlier_page(dir.path(), &ids(&["s1", "s2"]), None);
        assert_eq!(
            page.messages,
            vec![
                json!({"role": "user", "content": "u one", "timestamp": 1}),
                json!({"role": "custom", "customType": "aria-background", "content": "the job ended", "display": true,
                       "details": {"job": 2}, "timestamp": 1_792_540_801_500_i64}),
                json!({"role": "custom", "customType": "note", "content": [], "display": false,
                       "timestamp": 1_792_540_802_000_i64}),
                json!({"role": "branchSummary", "summary": "Tried another way.", "fromId": "x9",
                       "timestamp": 1_792_540_803_000_i64}),
                json!({"role": "assistant", "content": "a two", "timestamp": 1}),
            ]
        );
    }

    #[test]
    fn iso_times_read_as_javascript_dates_do() {
        assert_eq!(iso_millis("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(iso_millis("2000-02-29T12:34:56.789Z"), Some(951_827_696_789));
        assert_eq!(iso_millis("2026-10-21T00:00:01Z"), Some(1_792_540_801_000));
        assert_eq!(iso_millis("2026-10-21T00:00:01.5Z"), Some(1_792_540_801_500));
        assert_eq!(iso_millis("yesterday"), None);
        assert_eq!(iso_millis("2026-13-01T00:00:00.000Z"), None);
    }

    #[test]
    fn a_cursor_that_points_nowhere_gives_an_empty_page() {
        let dir = tempfile::tempdir().unwrap();
        let (old, _) = run("p", 5, None);
        write(dir.path(), "s1", &old);
        write(dir.path(), "s2", &[message("n1", None, "u now")]);
        let segments = ids(&["s1", "s2"]);
        let empty = EarlierPage { messages: vec![], cursor: None, more: false };
        assert_eq!(earlier_page(dir.path(), &segments, Some(EarlierCursor { segment: 7, index: 0 })), empty);
        assert_eq!(earlier_page(dir.path(), &segments, Some(EarlierCursor { segment: 0, index: 6 })), empty);
        // The end of a segment is a place: everything before it.
        assert_eq!(earlier_page(dir.path(), &segments, Some(EarlierCursor { segment: 0, index: 5 })).messages.len(), 5);
    }

    #[test]
    fn the_cursor_and_the_page_serialize_like_the_bridge() {
        let cursor = EarlierCursor { segment: 1, index: 7 };
        assert_eq!(serde_json::to_value(cursor).unwrap(), json!({"segment": 1, "index": 7}));
        assert_eq!(serde_json::from_value::<EarlierCursor>(json!({"segment": 1, "index": 7})).unwrap(), cursor);
        let page = EarlierPage { messages: vec![json!({"role": "user"})], cursor: Some(cursor), more: true };
        assert_eq!(
            serde_json::to_value(page).unwrap(),
            json!({"messages": [{"role": "user"}], "cursor": {"segment": 1, "index": 7}, "more": true})
        );
    }
}
