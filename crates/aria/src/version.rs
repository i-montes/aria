//! Versions of Aria as numbers: what `aria --version` and a hub's `hello` say, compared as major.minor.patch. Anything after
//! the third number (`-fake`, `+build`) is ignored, and a text that holds no three numbers compares to nothing, so a
//! notice about a difference is never made from something this cannot read.

use std::cmp::Ordering;

/// `0.21.0` out of `0.21.0`, `v0.21.0`, `aria 0.21.0` or `aria 0.21.0-rc1`: the last word, without a leading `v`, up to the
/// first character that is neither a digit nor a dot.
pub fn parse(text: &str) -> Option<(u64, u64, u64)> {
    let word = text.split_whitespace().last()?.trim_start_matches('v');
    let end = word.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(word.len());
    let mut parts = word[..end].split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    parts.next().is_none().then_some((major, minor, patch))
}

/// The version as it is written to the owner: `0.21.0`.
pub fn show(version: (u64, u64, u64)) -> String {
    format!("{}.{}.{}", version.0, version.1, version.2)
}

/// How the hub's version compares with this Aria's (`Less`: the hub is older); `None` when either cannot be read.
pub fn compare(hub: &str, mine: &str) -> Option<Ordering> {
    Some(parse(hub)?.cmp(&parse(mine)?))
}

/// This Aria's version.
pub fn mine() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_is_read_from_the_forms_aria_prints() {
        for text in ["0.21.0", "v0.21.0", "aria 0.21.0", "aria 0.21.0\n", "aria 0.21.0-rc1", "0.21.0+build5"] {
            assert_eq!(parse(text), Some((0, 21, 0)), "{text:?}");
        }
        assert_eq!(parse("aria 0.0.0-fake"), Some((0, 0, 0)));
        assert_eq!(show((1, 2, 30)), "1.2.30");
    }

    #[test]
    fn what_is_not_three_numbers_is_not_a_version() {
        for text in ["", "aria", "0.21", "0.21.0.1", "a.b.c", "x.21.0", "0..1", "aria unknown"] {
            assert_eq!(parse(text), None, "{text:?}");
        }
    }

    #[test]
    fn versions_compare_as_numbers_not_as_text() {
        assert_eq!(compare("0.21.0", "0.21.0"), Some(Ordering::Equal));
        assert_eq!(compare("aria 0.21.0", "0.22.0"), Some(Ordering::Less));
        assert_eq!(compare("0.23.0", "0.22.0"), Some(Ordering::Greater));
        assert_eq!(compare("0.9.0", "0.10.0"), Some(Ordering::Less), "9 is below 10, which text would get wrong");
        assert_eq!(compare("1.0.0", "0.99.99"), Some(Ordering::Greater));
        assert_eq!(compare("0.21.0-a", "0.21.0-b"), Some(Ordering::Equal), "only the three numbers count");
        assert_eq!(compare("oops", "0.21.0"), None);
        assert_eq!(compare("0.21.0", ""), None);
    }
}
