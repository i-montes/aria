//! Pi's RPC output is strict JSONL: one JSON object per record, ended by LF. Records are split on
//! LF only (a `U+2028` is legal inside a JSON string), an optional CR before the LF is dropped, and
//! a record is decoded only once it is complete so a multibyte character split across two reads
//! survives.

#[derive(Debug, Default)]
pub struct LineSplitter {
    buffer: Vec<u8>,
}

impl LineSplitter {
    pub fn new() -> LineSplitter {
        LineSplitter::default()
    }

    /// Feeds bytes and returns the records that are now complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut records = Vec::new();
        for &byte in bytes {
            if byte == b'\n' {
                if self.buffer.last() == Some(&b'\r') {
                    self.buffer.pop();
                }
                records.push(String::from_utf8_lossy(&self.buffer).into_owned());
                self.buffer.clear();
            } else {
                self.buffer.push(byte);
            }
        }
        records
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_lf_only() {
        let mut splitter = LineSplitter::new();
        assert_eq!(splitter.push(b"{\"a\":1}\n{\"b\":2}\n"), vec!["{\"a\":1}", "{\"b\":2}"]);
    }

    #[test]
    fn keeps_a_partial_record_until_it_completes() {
        let mut splitter = LineSplitter::new();
        assert!(splitter.push(b"{\"a\"").is_empty());
        assert_eq!(splitter.push(b":1}\n"), vec!["{\"a\":1}"]);
    }

    #[test]
    fn strips_a_carriage_return() {
        let mut splitter = LineSplitter::new();
        assert_eq!(splitter.push(b"x\r\n"), vec!["x"]);
    }

    #[test]
    fn does_not_split_on_unicode_separators() {
        let mut splitter = LineSplitter::new();
        let records = splitter.push("{\"t\":\"a\u{2028}b\u{2029}c\"}\n".as_bytes());
        assert_eq!(records, vec!["{\"t\":\"a\u{2028}b\u{2029}c\"}"]);
    }

    #[test]
    fn a_multibyte_char_split_across_reads_survives() {
        let mut splitter = LineSplitter::new();
        let bytes = "é\n".as_bytes();
        assert!(splitter.push(&bytes[..1]).is_empty());
        assert_eq!(splitter.push(&bytes[1..]), vec!["é"]);
    }
}
