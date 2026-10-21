//! Reading lines from a child or a peer without trusting it: a line is read up to a limit, and whatever passes the
//! limit is skipped a bounded chunk at a time, so one that never ends cannot grow the dock's memory.

use std::io::{BufRead, Read};

/// Reads one line, at most `limit` bytes of it, into `bytes`; the rest of a longer line is skipped, a bounded chunk at a
/// time. False at the end of the input.
pub(crate) fn read_line(reader: &mut impl BufRead, limit: u64, bytes: &mut Vec<u8>) -> bool {
    match reader.by_ref().take(limit).read_until(b'\n', bytes) {
        Ok(0) | Err(_) => return false,
        Ok(_) => {}
    }
    if bytes.last() != Some(&b'\n') && bytes.len() as u64 >= limit {
        loop {
            let mut rest = Vec::new();
            match reader.by_ref().take(limit).read_until(b'\n', &mut rest) {
                Ok(0) | Err(_) => break,
                Ok(_) if rest.last() == Some(&b'\n') => break,
                Ok(_) => {}
            }
        }
    }
    true
}

/// One whole line of at most `max` bytes (with its newline): `None` at the end of the input, `Some(None)` for a longer
/// line, which is skipped.
pub(crate) fn read_bounded_line(reader: &mut impl BufRead, max: u64) -> Option<Option<Vec<u8>>> {
    let mut bytes = Vec::new();
    if !read_line(reader, max, &mut bytes) {
        return None;
    }
    let too_long = bytes.last() != Some(&b'\n') && bytes.len() as u64 >= max;
    Some((!too_long).then_some(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn a_line_past_the_limit_is_skipped_whole() {
        let mut input = vec![b'x'; 3 * 1024 * 1024];
        input.extend_from_slice(b"\n{\"ok\":1}\nlast");
        let mut reader = BufReader::new(&input[..]);
        assert_eq!(read_bounded_line(&mut reader, 1024 * 1024), Some(None));
        assert_eq!(read_bounded_line(&mut reader, 1024 * 1024), Some(Some(b"{\"ok\":1}\n".to_vec())));
        assert_eq!(read_bounded_line(&mut reader, 1024 * 1024), Some(Some(b"last".to_vec())));
        assert_eq!(read_bounded_line(&mut reader, 1024 * 1024), None);
    }

    #[test]
    fn a_long_line_keeps_its_start() {
        let mut reader = BufReader::new(&b"abcdefgh\nnext\n"[..]);
        let mut bytes = Vec::new();
        assert!(read_line(&mut reader, 3, &mut bytes));
        assert_eq!(bytes, b"abc");
        bytes.clear();
        assert!(read_line(&mut reader, 10, &mut bytes));
        assert_eq!(bytes, b"next\n");
    }
}
