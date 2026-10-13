//! Bounds-checked TZif v1–v4 reader (RFC 9636). Leap records are validated but
//! ignored, like Python zoneinfo: volas stores POSIX instants, not leap seconds.

use crate::error::{Result, VolasError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TimeType {
    pub offset: i32,
    pub dst: i32,
    pub abbr: String,
}

pub(super) struct File {
    pub transitions: Vec<i64>,
    pub indices: Vec<usize>,
    pub types: Vec<TimeType>,
    pub before: usize,
    pub footer: String,
}

fn invalid(why: &str) -> VolasError {
    VolasError::Value(format!("invalid TZif: {why}"))
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or_else(|| invalid("length overflow"))?;
        let out = self.data.get(self.at..end).ok_or_else(|| invalid("truncated data"))?;
        self.at = end;
        Ok(out)
    }

    fn integer(&mut self, width: usize) -> Result<i64> {
        let mut result = 0u64;
        for &b in self.take(width)? {
            result = (result << 8) | u64::from(b);
        }
        Ok(if width == 4 { result as i32 as i64 } else { result as i64 })
    }
}

struct Header {
    version: u8,
    ut: usize,
    std: usize,
    leaps: usize,
    transitions: usize,
    types: usize,
    chars: usize,
}

impl Header {
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let magic = r.take(20)?;
        if &magic[..4] != b"TZif" || !matches!(magic[4], 0 | b'2' | b'3' | b'4') {
            return Err(invalid("unsupported header"));
        }
        let mut counts = [0usize; 6];
        for count in &mut counts {
            *count = usize::try_from(r.integer(4)?).map_err(|_| invalid("negative count"))?;
        }
        let [ut, std, leaps, transitions, types, chars] = counts;
        if !(1..=256).contains(&types) || chars == 0
            || (ut != 0 && ut != types) || (std != 0 && std != types)
        {
            return Err(invalid("invalid header counts"));
        }
        Ok(Self { version: magic[4], ut, std, leaps, transitions, types, chars })
    }

    fn length(&self, width: usize) -> Result<usize> {
        [(self.transitions, width + 1), (self.types, 6), (self.chars, 1),
            (self.leaps, width + 4), (self.std, 1), (self.ut, 1)]
            .into_iter().try_fold(0usize, |sum, (n, size)| {
                n.checked_mul(size).and_then(|n| sum.checked_add(n)).ok_or_else(|| invalid("length overflow"))
            })
    }
}

impl File {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { data: bytes, at: 0 };
        let mut h = Header::read(&mut r)?;
        let width = if h.version == 0 { 4 } else {
            r.take(h.length(4)?)?;
            h = Header::read(&mut r)?;
            if h.version == 0 { return Err(invalid("missing 64-bit header")); }
            8
        };
        // Validate the complete block before allocating from untrusted counts.
        let block = r.take(h.length(width)?)?;
        let mut b = Reader { data: block, at: 0 };
        let mut transitions = Vec::with_capacity(h.transitions);
        for _ in 0..h.transitions {
            let at = b.integer(width)?;
            if transitions.last().is_some_and(|&prev| prev >= at) {
                return Err(invalid("transitions are not strictly increasing"));
            }
            transitions.push(at);
        }
        let indices: Vec<usize> = b.take(h.transitions)?.iter().map(|&v| v as usize).collect();
        if indices.iter().any(|&i| i >= h.types) {
            return Err(invalid("transition type out of range"));
        }
        let mut offsets = Vec::with_capacity(h.types);
        let mut is_dst = Vec::with_capacity(h.types);
        let mut names = Vec::with_capacity(h.types);
        for _ in 0..h.types {
            let offset = b.integer(4)? as i32;
            let flags = b.take(2)?;
            if offset == i32::MIN || flags[0] > 1 { return Err(invalid("invalid local time type")); }
            offsets.push(offset);
            is_dst.push(flags[0] == 1);
            names.push(flags[1] as usize);
        }
        let chars = b.take(h.chars)?;
        let dst = dst_offsets(&indices, &offsets, &is_dst)?;
        let mut types = Vec::with_capacity(h.types);
        for i in 0..h.types {
            let name = chars.get(names[i]..).ok_or_else(|| invalid("designation index out of range"))?;
            let end = name.iter().position(|&v| v == 0).ok_or_else(|| invalid("unterminated designation"))?;
            let abbr = std::str::from_utf8(&name[..end]).map_err(|_| invalid("non-UTF-8 designation"))?.to_string();
            types.push(TimeType { offset: offsets[i], dst: dst[i], abbr });
        }
        let mut previous = None;
        for i in 0..h.leaps {
            let at = b.integer(width)?;
            let correction = b.integer(4)?;
            if at < 0 || previous.is_some_and(|(t, c): (i64, i64)| {
                at <= t || !((correction - c).abs() == 1
                    || (h.version == b'4' && i + 1 == h.leaps && correction == c))
            }) || (previous.is_none() && h.version != b'4' && correction.abs() != 1) {
                return Err(invalid("invalid leap-second record"));
            }
            previous = Some((at, correction));
        }
        let standard = b.take(h.std)?;
        let universal = b.take(h.ut)?;
        if standard.iter().any(|&v| v > 1) || universal.iter().enumerate().any(|(i, &v)| {
            v > 1 || (v == 1 && standard.get(i) != Some(&1))
        }) { return Err(invalid("invalid transition indicators")); }
        let footer = if width == 8 {
            let rest = &r.data[r.at..];
            if rest.len() < 2 || rest[0] != b'\n' || rest[rest.len() - 1] != b'\n'
                || rest[1..rest.len() - 1].iter().any(|b| *b == 0 || *b == b'\n' || !b.is_ascii())
            { return Err(invalid("invalid POSIX footer framing")); }
            std::str::from_utf8(&rest[1..rest.len() - 1]).expect("ASCII footer").to_string()
        } else { String::new() };
        // CPython uses the first standard type before the first transition;
        // with only DST types, the first transition's type (or final type).
        let before = is_dst.iter().position(|&dst| !dst)
            .unwrap_or_else(|| indices.first().copied().unwrap_or(h.types - 1));
        Ok(Self { transitions, indices, types, before, footer })
    }
}

/// CPython's `_utcoff_to_dstoff`: prefer the preceding standard transition,
/// otherwise the next standard transition; unresolved DST types use one hour.
fn dst_offsets(indices: &[usize], offsets: &[i32], is_dst: &[bool]) -> Result<Vec<i32>> {
    let mut dst = vec![0; offsets.len()];
    for i in 1..indices.len() {
        let idx = indices[i];
        if !is_dst[idx] || dst[idx] != 0 { continue; }
        let prev = indices[i - 1];
        let mut difference = if is_dst[prev] { 0 } else { offsets[idx] as i64 - offsets[prev] as i64 };
        if difference == 0 && idx + 1 < offsets.len() {
            if let Some(&next) = indices.get(i + 1) {
                if !is_dst[next] { difference = offsets[idx] as i64 - offsets[next] as i64; }
            }
        }
        // Offsets obey the TZif signed-32 domain; reject a malformed derived
        // component instead of overflowing the owner's signed-32 DST value.
        dst[idx] = i32::try_from(difference).map_err(|_| invalid("DST offset outside signed-32 range"))?;
    }
    for (dst, &is_dst) in dst.iter_mut().zip(is_dst) {
        if *dst == 0 && is_dst { *dst = 3600; }
    }
    Ok(dst)
}
