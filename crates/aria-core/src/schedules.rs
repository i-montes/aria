//! Scheduled tasks (docs/superpowers/specs/2026-10-21-schedules-design.md).
//!
//! A task is a prompt, a local time on some days (or one date), and fixed permissions. Whether it is due is one pure rule
//! over the task, the time now and a clock: the latest occurrence at or before now, on today's local date, later than its
//! last run and its creation. The dock asks once a second, so a time missed while the computer slept or Aria was off runs
//! once when it is back, and only today's.

use crate::json_file::{read_json, write_json_atomic};
use crate::paths::AriaPaths;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The days a task can run on, Monday first (`Local::weekday` indexes this).
pub const DAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
/// A run that starts more than this after its time says it ran late.
pub const LATE_MS: i64 = 120_000;
pub const MAX_TASKS: usize = 50;
const DAY_MS: i64 = 86_400_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    pub title: String,
    pub prompt: String,
    pub when: When,
    #[serde(default)]
    pub grants: Grants,
    pub enabled: bool,
    pub created: i64,
    pub last_run: Option<i64>,
    pub last_result: Option<RunResult>,
}

/// `at` is local `HH:MM`; `days` the days it runs on, unless `date` (`YYYY-MM-DD`) makes it a single run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct When {
    pub at: String,
    #[serde(default)]
    pub days: Vec<String>,
    #[serde(default)]
    pub date: Option<String>,
}

/// What a scheduled run may do without asking: each connector `read` (the default) or `allow`, and folders outside the
/// projects folder it may write.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Grants {
    pub connectors: BTreeMap<String, String>,
    pub writable: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunResult {
    pub at: i64,
    pub ok: bool,
    pub summary: String,
}

/// A local date and time; `weekday` 0 is Monday.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Local {
    pub y: i32,
    pub m: u32,
    pub d: u32,
    pub h: u32,
    pub min: u32,
    pub weekday: u32,
}

/// The computer's calendar: epoch milliseconds to local time and back.
pub trait Clock {
    fn local(&self, epoch_ms: i64) -> Local;
    fn epoch(&self, y: i32, m: u32, d: u32, h: u32, min: u32) -> i64;
}

/// The system's own time zone and daylight-saving rules (`localtime_r` and `mktime`).
pub struct SystemClock;

impl Clock for SystemClock {
    fn local(&self, epoch_ms: i64) -> Local {
        let seconds = epoch_ms.div_euclid(1000) as libc::time_t;
        // SAFETY: a zeroed `tm` is valid output space; localtime_r fills it from `seconds`.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::localtime_r(&seconds, &mut tm) };
        Local {
            y: tm.tm_year + 1900,
            m: (tm.tm_mon + 1) as u32,
            d: tm.tm_mday as u32,
            h: tm.tm_hour as u32,
            min: tm.tm_min as u32,
            weekday: ((tm.tm_wday + 6) % 7) as u32,
        }
    }

    fn epoch(&self, y: i32, m: u32, d: u32, h: u32, min: u32) -> i64 {
        // SAFETY: a zeroed `tm` with the fields set is a valid input; tm_isdst -1 lets mktime decide daylight saving.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        tm.tm_year = y - 1900;
        tm.tm_mon = m as i32 - 1;
        tm.tm_mday = d as i32;
        tm.tm_hour = h as i32;
        tm.tm_min = min as i32;
        tm.tm_isdst = -1;
        (unsafe { libc::mktime(&mut tm) }) as i64 * 1000
    }
}

fn parse_at(at: &str) -> Option<(u32, u32)> {
    let (h, m) = at.split_once(':')?;
    if h.len() != 2 || m.len() != 2 {
        return None;
    }
    let (h, m): (u32, u32) = (h.parse().ok()?, m.parse().ok()?);
    (h < 24 && m < 60).then_some((h, m))
}

fn parse_date(date: &str) -> Option<(i32, u32, u32)> {
    let mut parts = date.split('-');
    let (y, m, d) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() || y.len() != 4 || m.len() != 2 || d.len() != 2 {
        return None;
    }
    let (y, m, d): (i32, u32, u32) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    ((1..=12).contains(&m) && (1..=31).contains(&d)).then_some((y, m, d))
}

/// The local day `offset` days from the one `now` falls on (noon keeps daylight-saving days whole).
fn day(now: i64, offset: i64, clock: &dyn Clock) -> Local {
    let today = clock.local(now);
    clock.local(clock.epoch(today.y, today.m, today.d, 12, 0) + offset * DAY_MS)
}

/// When the task's time comes on `day`, if it runs that day.
fn on(task: &Task, day: Local, clock: &dyn Clock) -> Option<i64> {
    let (h, min) = parse_at(&task.when.at)?;
    let runs = match &task.when.date {
        Some(date) => parse_date(date) == Some((day.y, day.m, day.d)),
        None => task.when.days.iter().any(|d| DAYS.get(day.weekday as usize) == Some(&d.as_str())),
    };
    runs.then(|| clock.epoch(day.y, day.m, day.d, h, min))
}

/// The latest time the task should have run, at or before `now` (within the last week).
pub fn occurrence(task: &Task, now: i64, clock: &dyn Clock) -> Option<i64> {
    (0..=7).filter_map(|back| on(task, day(now, -back, clock), clock)).find(|&at| at <= now)
}

/// Whether the task runs now: enabled, its latest time is today's, and it has not run (nor been created) since.
pub fn due(task: &Task, now: i64, clock: &dyn Clock) -> bool {
    if !task.enabled {
        return false;
    }
    let Some(at) = occurrence(task, now, clock) else { return false };
    let (today, then) = (clock.local(now), clock.local(at));
    (today.y, today.m, today.d) == (then.y, then.m, then.d) && at > task.last_run.unwrap_or(i64::MIN).max(task.created)
}

pub fn late(occurrence: i64, now: i64) -> bool {
    now - occurrence > LATE_MS
}

/// The next time the task will run, for the panel; None when it is paused or will not run again.
pub fn next_run(task: &Task, now: i64, clock: &dyn Clock) -> Option<i64> {
    if !task.enabled {
        return None;
    }
    if due(task, now, clock) {
        return occurrence(task, now, clock);
    }
    (0..=8).filter_map(|ahead| on(task, day(now, ahead, clock), clock)).find(|&at| at > now)
}

/// A task as the dock will keep it: a time, days or a date, a prompt, and grants it can honour.
pub fn validate(task: &Task, servers: &[String]) -> Result<(), String> {
    if task.title.trim().is_empty() || task.prompt.trim().is_empty() {
        return Err("a task needs a title and a prompt".into());
    }
    if parse_at(&task.when.at).is_none() {
        return Err(format!("the time {:?} is not HH:MM (00:00 to 23:59)", task.when.at));
    }
    match &task.when.date {
        Some(date) if parse_date(date).is_none() => return Err(format!("the date {date:?} is not YYYY-MM-DD")),
        Some(_) => {}
        None if task.when.days.is_empty() => return Err("a task needs days or a date".into()),
        None => {}
    }
    if let Some(bad) = task.when.days.iter().find(|d| !DAYS.contains(&d.as_str())) {
        return Err(format!("{bad:?} is not a day (mon … sun)"));
    }
    for (name, level) in &task.grants.connectors {
        if !servers.contains(name) {
            return Err(format!("there is no connector {name:?}"));
        }
        if level != "read" && level != "allow" {
            return Err(format!("{name}: a scheduled task can only read or be allowed, never ask"));
        }
    }
    if let Some(bad) = task.grants.writable.iter().find(|p| !p.starts_with('/')) {
        return Err(format!("{bad:?} is not an absolute folder"));
    }
    Ok(())
}

/// `s-` and 12 lowercase letters or digits.
pub fn new_id() -> String {
    use rand::Rng;
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();
    let tail: String = (0..12).map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char).collect();
    format!("s-{tail}")
}

#[derive(Default, Serialize, Deserialize)]
struct File {
    tasks: Vec<Task>,
}

/// The tasks. A missing file is none; one that does not parse is moved aside (`schedules.json.broken`), never overwritten.
pub fn load(paths: &AriaPaths) -> Vec<Task> {
    let parsed = match read_json(&paths.schedules()) {
        Ok(None) => return Vec::new(),
        Ok(Some(value)) => serde_json::from_value::<File>(value).ok(),
        Err(_) => None,
    };
    match parsed {
        Some(file) => file.tasks,
        None => {
            let _ = std::fs::rename(paths.schedules(), paths.root.join("schedules.json.broken"));
            Vec::new()
        }
    }
}

pub fn save(paths: &AriaPaths, tasks: &[Task]) -> std::io::Result<()> {
    let value = serde_json::to_value(File { tasks: tasks.to_vec() }).map_err(std::io::Error::other)?;
    write_json_atomic(&paths.schedules(), &value, 0o600)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Local time is UTC shifted by `offset` hours: no daylight saving, so every case is exact.
    struct FakeClock {
        offset: i64,
    }

    const HOUR: i64 = 3_600_000;
    const DAY: i64 = 24 * HOUR;

    fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
        let y = if m <= 2 { y - 1 } else { y } as i64;
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let m = m as i64;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    impl Clock for FakeClock {
        fn local(&self, epoch_ms: i64) -> Local {
            let ms = epoch_ms + self.offset * HOUR;
            let days = ms.div_euclid(DAY);
            let rest = ms.rem_euclid(DAY);
            let z = days + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z - era * 146_097;
            let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
            let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
            let y = (yoe + era * 400 + if m <= 2 { 1 } else { 0 }) as i32;
            Local {
                y,
                m,
                d,
                h: (rest / HOUR) as u32,
                min: (rest % HOUR / 60_000) as u32,
                weekday: (days + 3).rem_euclid(7) as u32,
            }
        }
        fn epoch(&self, y: i32, m: u32, d: u32, h: u32, min: u32) -> i64 {
            days_from_civil(y, m, d) * DAY + h as i64 * HOUR + min as i64 * 60_000 - self.offset * HOUR
        }
    }

    const CLOCK: FakeClock = FakeClock { offset: -5 }; // Bogotá

    /// Local 2026-10-21 (a Wednesday) at `h:min`.
    fn wed(h: u32, min: u32) -> i64 {
        CLOCK.epoch(2026, 10, 21, h, min)
    }

    fn task(at: &str, days: &[&str]) -> Task {
        Task {
            id: "s-abcdefghijkl".into(),
            title: "Notion tasks".into(),
            prompt: "Give me my Notion tasks".into(),
            when: When { at: at.into(), days: days.iter().map(|d| d.to_string()).collect(), date: None },
            grants: Grants::default(),
            enabled: true,
            created: wed(0, 0) - 7 * DAY,
            last_run: None,
            last_result: None,
        }
    }

    const EVERY: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

    #[test]
    fn the_fake_clock_knows_its_weekdays() {
        assert_eq!(CLOCK.local(wed(7, 30)), Local { y: 2026, m: 10, d: 21, h: 7, min: 30, weekday: 2 });
    }

    #[test]
    fn every_day_runs_at_its_time() {
        let t = task("07:30", &EVERY);
        assert!(!due(&t, wed(7, 29), &CLOCK));
        assert!(due(&t, wed(7, 30), &CLOCK));
        assert_eq!(occurrence(&t, wed(9, 0), &CLOCK), Some(wed(7, 30)));
    }

    #[test]
    fn weekdays_skip_the_weekend() {
        let t = task("07:30", &["mon", "tue", "wed", "thu", "fri"]);
        let saturday = wed(8, 0) + 3 * DAY;
        assert!(!due(&t, saturday, &CLOCK));
        assert!(due(&t, wed(8, 0), &CLOCK));
    }

    #[test]
    fn a_one_off_runs_on_its_date_only() {
        let mut t = task("07:30", &[]);
        t.when.date = Some("2026-10-21".into());
        assert!(due(&t, wed(8, 0), &CLOCK));
        assert!(!due(&t, wed(8, 0) + DAY, &CLOCK), "not the next day");
        assert!(!due(&t, wed(8, 0) - DAY, &CLOCK), "not before");
    }

    #[test]
    fn created_after_todays_time_waits_for_tomorrow() {
        let mut t = task("07:30", &EVERY);
        t.created = wed(7, 40);
        assert!(!due(&t, wed(7, 41), &CLOCK));
        assert!(due(&t, wed(7, 30) + DAY, &CLOCK));
    }

    #[test]
    fn ran_already_today_is_not_due() {
        let mut t = task("07:30", &EVERY);
        t.last_run = Some(wed(7, 30));
        assert!(!due(&t, wed(9, 0), &CLOCK));
    }

    #[test]
    fn slept_across_the_time_runs_late_once() {
        let mut t = task("07:30", &EVERY);
        t.last_run = Some(wed(7, 30) - DAY);
        let woke = wed(8, 5);
        assert!(due(&t, woke, &CLOCK));
        assert!(late(occurrence(&t, woke, &CLOCK).unwrap(), woke));
        t.last_run = Some(woke);
        assert!(!due(&t, woke + 60_000, &CLOCK));
    }

    #[test]
    fn three_days_off_runs_only_today() {
        let mut t = task("07:30", &EVERY);
        t.last_run = Some(wed(7, 30) - 3 * DAY);
        assert!(due(&t, wed(9, 0), &CLOCK));
        t.last_run = Some(wed(9, 0));
        assert!(!due(&t, wed(9, 1), &CLOCK), "one run for the three days");
    }

    #[test]
    fn a_missed_day_does_not_run_the_next_morning_before_its_time() {
        let mut t = task("07:30", &EVERY);
        t.last_run = Some(wed(7, 30) - 2 * DAY);
        assert!(!due(&t, wed(6, 0), &CLOCK), "yesterday's is not today's");
    }

    #[test]
    fn disabled_never_runs() {
        let mut t = task("07:30", &EVERY);
        t.enabled = false;
        assert!(!due(&t, wed(7, 30), &CLOCK));
    }

    #[test]
    fn late_is_over_two_minutes() {
        assert!(!late(wed(7, 30), wed(7, 32)));
        assert!(late(wed(7, 30), wed(7, 32) + 1));
    }

    #[test]
    fn next_run_is_tomorrow_after_todays_run() {
        let mut t = task("07:30", &EVERY);
        assert_eq!(next_run(&t, wed(7, 0), &CLOCK), Some(wed(7, 30)));
        t.last_run = Some(wed(7, 30));
        assert_eq!(next_run(&t, wed(9, 0), &CLOCK), Some(wed(7, 30) + DAY));
        t.when.days = vec!["mon".into()];
        assert_eq!(next_run(&t, wed(9, 0), &CLOCK), Some(wed(7, 30) + 5 * DAY));
        t.enabled = false;
        assert_eq!(next_run(&t, wed(9, 0), &CLOCK), None);
    }

    #[test]
    fn validate_refuses_bad_times_days_connectors_and_paths() {
        let servers = vec!["notion".to_string()];
        let ok = task("07:30", &EVERY);
        assert!(validate(&ok, &servers).is_ok());
        for at in ["7:30", "24:00", "07:60", "0730", ""] {
            assert!(validate(&task(at, &EVERY), &servers).is_err(), "{at}");
        }
        assert!(validate(&task("07:30", &[]), &servers).is_err(), "no days and no date");
        assert!(validate(&task("07:30", &["monday"]), &servers).is_err());
        let mut one_off = task("07:30", &[]);
        one_off.when.date = Some("2026-13-01".into());
        assert!(validate(&one_off, &servers).is_err());
        one_off.when.date = Some("2026-10-30".into());
        assert!(validate(&one_off, &servers).is_ok());
        let mut grants = task("07:30", &EVERY);
        grants.grants.connectors.insert("notion".into(), "allow".into());
        assert!(validate(&grants, &servers).is_ok());
        grants.grants.connectors.insert("notion".into(), "ask".into());
        assert!(validate(&grants, &servers).is_err(), "only read or allow");
        grants.grants.connectors = [("gmail".to_string(), "read".to_string())].into();
        assert!(validate(&grants, &servers).is_err(), "unknown connector");
        let mut writes = task("07:30", &EVERY);
        writes.grants.writable = vec!["relative/dir".into()];
        assert!(validate(&writes, &servers).is_err());
        let mut empty = task("07:30", &EVERY);
        empty.prompt = "  ".into();
        assert!(validate(&empty, &servers).is_err());
    }

    #[test]
    fn new_ids_look_right() {
        let id = new_id();
        assert!(
            id.starts_with("s-")
                && id.len() == 14
                && id[2..].chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
            "{id}"
        );
        assert_ne!(new_id(), new_id());
    }

    #[test]
    fn store_round_trips_with_mode_600() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        assert!(load(&paths).is_empty());
        let t = task("07:30", &EVERY);
        save(&paths, std::slice::from_ref(&t)).unwrap();
        assert_eq!(load(&paths), vec![t]);
        assert_eq!(std::fs::metadata(paths.schedules()).unwrap().permissions().mode() & 0o777, 0o600);
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(paths.schedules()).unwrap()).unwrap();
        assert!(raw["tasks"][0]["lastRun"].is_null() && raw["tasks"][0]["when"]["at"] == "07:30", "{raw}");
    }

    #[test]
    fn a_broken_store_is_moved_aside() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AriaPaths::new(dir.path().to_path_buf());
        std::fs::write(paths.schedules(), "{ nope").unwrap();
        assert!(load(&paths).is_empty());
        assert_eq!(std::fs::read_to_string(dir.path().join("schedules.json.broken")).unwrap(), "{ nope");
    }

    #[test]
    fn system_clock_round_trips_a_local_time() {
        let at = SystemClock.epoch(2026, 10, 21, 7, 30);
        let local = SystemClock.local(at);
        assert_eq!((local.y, local.m, local.d, local.h, local.min, local.weekday), (2026, 10, 21, 7, 30, 2));
    }
}
