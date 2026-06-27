//! One Billion Row Challenge — pure-safe, std-only Rust solution.
//!
//! Computes min / mean / max temperature per weather station from a file of
//! `station;temperature` lines (1,000,000,000 of them, ~13 GB), printed
//! alphabetically as `station;min;mean;max`.
//!
//! No external crates, no `unsafe`, no C FFI. Strategy:
//!   * Carve the file into one byte-range per CPU, snapped to line
//!     boundaries, and process the ranges in parallel with `std::thread`.
//!   * Each thread streams its range in big blocks via the safe
//!     `FileExt::read_at` (so we never hold the whole 13 GB in memory) and
//!     aggregates into a custom open-addressing hash table. Station names are
//!     copied once (on first sight) into a per-thread byte arena; temperatures
//!     are kept as fixed-point tenths (i16).
//!   * Merge the per-thread tables, sort by name, and print.
//!
//! Hot-loop tricks, all in safe stable Rust:
//!   * A single fused pass per line: the `;` is found and the value parsed in
//!     one go (the value's fixed format `[-]?\d{1,2}\.\d` reveals where the
//!     newline is), so we never scan the line twice.
//!   * SWAR: the `;` is located 8 bytes at a time with the classic
//!     "bytes-equal-to-c" word trick instead of byte-by-byte.
//!   * A multiply-based hash over 8-byte words, mixed into a bucket index with
//!     Fibonacci hashing.
//!
//! Because the input dwarfs the page cache on a small-RAM box, the run is
//! disk-IO-bound there; on a machine where the file fits in cache it is
//! CPU-bound and this hot loop is what matters.
//!
//! Usage: `brc [path]`  (path defaults to `measurements.txt`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::thread;

// ---------------------------------------------------------------------------
// Per-thread aggregation table: open addressing with a side arena for names.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Entry {
    // Name stored as a (offset, len) slice into the table's `keys` arena.
    // `key_len == 0` marks an empty slot (names have a minimum length of 1).
    key_off: u32,
    key_len: u32,
    hash: u64,
    min: i16,
    max: i16,
    count: u64,
    sum: i64,
}

const EMPTY: Entry = Entry {
    key_off: 0,
    key_len: 0,
    hash: 0,
    min: 0,
    max: 0,
    count: 0,
    sum: 0,
};

// Power-of-two capacity comfortably above the 10,000 unique-station limit,
// keeping the load factor low (≈0.15) so probe chains stay short.
const CAP: usize = 1 << 16;
const MASK: usize = CAP - 1;

// Golden-ratio constant for both word mixing and Fibonacci index hashing.
const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

struct Table {
    slots: Box<[Entry]>,
    keys: Vec<u8>,
}

impl Table {
    fn new() -> Self {
        Table {
            slots: vec![EMPTY; CAP].into_boxed_slice(),
            keys: Vec::with_capacity(64 * 1024),
        }
    }

    /// Record one measurement `val` (in tenths) for `name`, given its
    /// precomputed `hash`.
    #[inline]
    fn record(&mut self, name: &[u8], hash: u64, val: i16) {
        let mut idx = (hash.wrapping_mul(PHI) >> 48) as usize & MASK;
        loop {
            let e = self.slots[idx]; // Entry is Copy
            if e.key_len == 0 {
                let off = self.keys.len() as u32;
                self.keys.extend_from_slice(name);
                self.slots[idx] = Entry {
                    key_off: off,
                    key_len: name.len() as u32,
                    hash,
                    min: val,
                    max: val,
                    count: 1,
                    sum: val as i64,
                };
                return;
            }
            if e.hash == hash
                && e.key_len as usize == name.len()
                && &self.keys[e.key_off as usize..e.key_off as usize + name.len()] == name
            {
                let s = &mut self.slots[idx];
                if val < s.min {
                    s.min = val;
                }
                if val > s.max {
                    s.max = val;
                }
                s.sum += val as i64;
                s.count += 1;
                return;
            }
            idx = (idx + 1) & MASK;
        }
    }

    #[inline]
    fn name_of(&self, e: &Entry) -> &[u8] {
        &self.keys[e.key_off as usize..e.key_off as usize + e.key_len as usize]
    }
}

// ---------------------------------------------------------------------------
// Hashing (must agree between the fast SWAR path and the slow tail path).
// ---------------------------------------------------------------------------

// SWAR constants for locating a `;` (0x3B) byte inside a 64-bit word.
const SEMI: u64 = 0x3B3B_3B3B_3B3B_3B3B;
const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

/// Hash a name by folding its little-endian 8-byte words (final partial word
/// zero-padded). Kept byte-for-byte identical to the fused fast path so a
/// station hashes to the same slot whichever path sees it.
#[inline]
fn hash_name(name: &[u8]) -> u64 {
    let mut h = 0u64;
    let mut chunks = name.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ w).wrapping_mul(PHI);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut last = [0u8; 8];
        last[..rem.len()].copy_from_slice(rem);
        h = (h ^ u64::from_le_bytes(last)).wrapping_mul(PHI);
    }
    h
}

/// Branchless SWAR parse of a temperature from the 8-byte little-endian word
/// at the value's first byte. Returns `(tenths, bytes_consumed_incl_newline)`.
///
/// Adapted from the well-known merykitty/1BRC technique: digits are 0x30–0x39
/// (bit 0x10 set) while '.' is 0x2E (bit 0x10 clear), so `!word & 0x10101000`
/// isolates the decimal point's position; from there the 1–2 integer digits
/// and optional sign are combined with a fixed multiply.
#[inline]
fn parse_num(word: u64) -> (i32, usize) {
    let dot = (!word & 0x1010_1000).trailing_zeros();
    let signed = ((!word << 59) as i64) >> 63; // 0 (positive) or -1 (negative)
    let design_mask = !(signed as u64 & 0xFF);
    let digits = ((word & design_mask) << (28 - dot)) & 0x0000_000F_000F_0F00;
    let abs = ((digits.wrapping_mul(0x640a_0001) >> 32) & 0x3FF) as i64;
    let value = (abs ^ signed) - signed;
    (value as i32, (dot as usize >> 3) + 3)
}

/// Parse a temperature `[-]?\d{1,2}\.\d` into fixed-point tenths.
#[inline]
fn parse_temp(b: &[u8]) -> i16 {
    let mut i = 0;
    let neg = b[0] == b'-';
    if neg {
        i = 1;
    }
    let mut v = (b[i] - b'0') as i16;
    i += 1;
    if b[i] != b'.' {
        v = v * 10 + (b[i] - b'0') as i16;
        i += 1;
    }
    i += 1; // skip '.'
    v = v * 10 + (b[i] - b'0') as i16;
    if neg {
        -v
    } else {
        v
    }
}

/// Slow, fully bounds-checked line parser for the file's tail (the last few
/// bytes where the SWAR path lacks its 8-byte look-ahead). Uses `hash_name` so
/// its results match the fast path exactly.
#[inline]
fn process_line_slow(line: &[u8], table: &mut Table) {
    let semi = line.iter().position(|&b| b == b';').unwrap_or(line.len());
    let name = &line[..semi];
    let val = parse_temp(&line[semi + 1..]);
    table.record(name, hash_name(name), val);
}

// ---------------------------------------------------------------------------
// File range scanning.
// ---------------------------------------------------------------------------

const BLK: usize = 1 << 21; // 2 MiB read block (≪ CHUNK so the trailing
// re-read that completes a chunk's straddling last line stays cheap)
// Work-stealing unit. Many chunks per thread let fast cores grab more than
// slow ones, which matters a lot on heterogeneous CPUs (P-cores + E-cores).
const CHUNK: u64 = 1 << 24; // 16 MiB
// Upper bound on a single line: name ≤100 B + ';' + value ≤6 B + '\n', plus
// slack so SWAR 8-byte word reads never run past a buffered line.
const MAX_LINE: usize = 128;

/// Offset just past the first newline at or after `off` (or `flen` if none).
fn next_line_start(file: &File, mut off: u64, flen: u64) -> u64 {
    let mut tmp = [0u8; 512];
    loop {
        if off >= flen {
            return flen;
        }
        let n = file.read_at(&mut tmp, off).expect("read_at failed");
        if n == 0 {
            return flen;
        }
        if let Some(j) = tmp[..n].iter().position(|&b| b == b'\n') {
            return off + j as u64 + 1;
        }
        off += n as u64;
    }
}

/// Scan the line-aligned byte range `[begin, end)` of the file (process every
/// line that starts in this range), accumulating into `table`. The bounds must
/// already be snapped to line starts by the caller. `buf` is a caller-owned
/// scratch buffer of length `BLK + MAX_LINE`, reused across calls so it is
/// allocated/zeroed only once.
fn scan_range(file: &File, begin: u64, end: u64, buf: &mut [u8], table: &mut Table) {
    if begin >= end {
        return;
    }

    let mut p = 0usize; // parse cursor within `buf`
    let mut filled = 0usize; // valid bytes in `buf`
    let mut base = begin; // file offset of buf[p]
    let mut read_off = begin; // next file offset to read

    loop {
        // Fast path: parse whole lines while we have MAX_LINE bytes of
        // look-ahead (so the value's '\n' is buffered and SWAR can't overrun).
        while base < end && p + MAX_LINE <= filled {
            let start = p;

            // --- find ';' and hash the name, 8 bytes at a time ---
            let mut h = 0u64;
            let mut i = start;
            loop {
                let w = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
                let m = w ^ SEMI;
                let z = m.wrapping_sub(LO) & !m & HI;
                if z != 0 {
                    let nb = (z.trailing_zeros() >> 3) as usize; // bytes before ';'
                    if nb != 0 {
                        // Keep only the `nb` bytes before the ';'.
                        let mask = !0u64 >> ((8 - nb) * 8);
                        h = (h ^ (w & mask)).wrapping_mul(PHI);
                    }
                    i += nb;
                    break;
                }
                h = (h ^ w).wrapping_mul(PHI);
                i += 8;
            }
            let name = &buf[start..i];
            i += 1; // skip ';'

            // --- value: [-]?\d{1,2}\.\d then '\n', parsed branchlessly ---
            let word = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
            let (val, adv) = parse_num(word);
            i += adv;

            table.record(name, h, val as i16);
            base += (i - start) as u64;
            p = i;
        }

        if base >= end {
            break;
        }

        // Refill: shift the unprocessed leftover (< MAX_LINE bytes) to the
        // front, then read another block into the reused tail of the buffer.
        let left = filled - p;
        buf.copy_within(p..filled, 0);
        p = 0;
        filled = left;
        let nread = file
            .read_at(&mut buf[filled..filled + BLK], read_off)
            .expect("read_at failed");
        filled += nread;
        read_off += nread as u64;

        if nread == 0 {
            // EOF: parse the remaining tail (may include a final line with no
            // trailing newline) with the slow, fully-checked parser.
            let mut data = &buf[p..filled];
            while !data.is_empty() && base < end {
                match data.iter().position(|&b| b == b'\n') {
                    Some(j) => {
                        process_line_slow(&data[..j], table);
                        base += (j + 1) as u64;
                        data = &data[j + 1..];
                    }
                    None => {
                        process_line_slow(data, table);
                        break;
                    }
                }
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Output.
// ---------------------------------------------------------------------------

/// Append a tenths value as a decimal with one fractional digit
/// (e.g. -123 → "-12.3", 5 → "0.5").
fn fmt_tenths(buf: &mut Vec<u8>, tenths: i64) {
    let mut v = tenths;
    if v < 0 {
        buf.push(b'-');
        v = -v;
    }
    let whole = v / 10;
    let frac = (v % 10) as u8;
    let mut tmp = [0u8; 20];
    let mut k = tmp.len();
    let mut w = whole;
    loop {
        k -= 1;
        tmp[k] = b'0' + (w % 10) as u8;
        w /= 10;
        if w == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[k..]);
    buf.push(b'.');
    buf.push(b'0' + frac);
}

struct Stat {
    min: i16,
    max: i16,
    count: u64,
    sum: i64,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "measurements.txt".into());

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("brc: cannot open {path}: {e}");
            std::process::exit(1);
        }
    };
    let flen = file.metadata().expect("stat failed").len();
    if flen == 0 {
        return;
    }

    let nthreads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Process the file as many small chunks pulled from a shared atomic
    // counter, so faster cores naturally take a larger share of the work.
    let nchunks = flen.div_ceil(CHUNK) as usize;

    // Snap every chunk boundary to a line start exactly once, up front, instead
    // of re-snapping each interior boundary inside `scan_range` (where it would
    // be done twice — as one chunk's end and the next chunk's start — and again
    // by every thread that grabbed a neighbouring chunk). `boundaries[c]` is the
    // start offset of chunk `c`; `boundaries[nchunks]` is the file end.
    let file = &file;
    let mut boundaries = vec![0u64; nchunks + 1];
    for (i, b) in boundaries.iter_mut().enumerate().take(nchunks).skip(1) {
        *b = next_line_start(file, i as u64 * CHUNK, flen);
    }
    boundaries[nchunks] = flen;
    let boundaries = &boundaries;

    let next = std::sync::atomic::AtomicUsize::new(0);
    let next = &next;
    let tables: Vec<Table> = thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                s.spawn(move || {
                    let mut buf = vec![0u8; BLK + MAX_LINE];
                    let mut table = Table::new();
                    loop {
                        let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if c >= nchunks {
                            break;
                        }
                        scan_range(file, boundaries[c], boundaries[c + 1], &mut buf, &mut table);
                    }
                    table
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge per-thread tables by station name (≤10,000 entries total).
    let mut merged: HashMap<&[u8], Stat> = HashMap::with_capacity(16_384);
    for table in &tables {
        for e in table.slots.iter() {
            if e.key_len == 0 {
                continue;
            }
            merged
                .entry(table.name_of(e))
                .and_modify(|m| {
                    if e.min < m.min {
                        m.min = e.min;
                    }
                    if e.max > m.max {
                        m.max = e.max;
                    }
                    m.count += e.count;
                    m.sum += e.sum;
                })
                .or_insert(Stat {
                    min: e.min,
                    max: e.max,
                    count: e.count,
                    sum: e.sum,
                });
        }
    }

    let mut results: Vec<(&[u8], Stat)> = merged.into_iter().collect();
    results.sort_unstable_by(|a, b| a.0.cmp(b.0));

    // Build all output in one buffer, then write it in a single call.
    let mut out = Vec::with_capacity(results.len() * 32);
    for (name, m) in &results {
        out.extend_from_slice(name);
        out.push(b';');
        fmt_tenths(&mut out, m.min as i64);
        out.push(b';');
        // Mean rounded to one decimal, half-up toward +inf (matches the Java
        // reference's `Math.round`): round(sum / count) in tenths.
        let mean_tenths = (m.sum as f64 / m.count as f64 + 0.5).floor() as i64;
        fmt_tenths(&mut out, mean_tenths);
        out.push(b';');
        fmt_tenths(&mut out, m.max as i64);
        out.push(b'\n');
    }

    let stdout = io::stdout();
    stdout.lock().write_all(&out).expect("write failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_num` must agree with the simple `parse_temp` (and report the
    /// right byte length) for every valid temperature in -99.9..=99.9.
    #[test]
    fn parse_num_exhaustive() {
        for tenths in -999i32..=999 {
            let whole = (tenths.abs() / 10) as i64;
            let frac = (tenths.abs() % 10) as u8;
            let sign = if tenths < 0 { "-" } else { "" };
            let s = format!("{sign}{whole}.{frac}\n");
            let bytes = s.as_bytes();

            let mut word_bytes = [0u8; 8];
            let n = bytes.len().min(8);
            word_bytes[..n].copy_from_slice(&bytes[..n]);
            let word = u64::from_le_bytes(word_bytes);

            let (val, adv) = parse_num(word);
            assert_eq!(val, tenths, "value mismatch for {s:?}");
            assert_eq!(adv, bytes.len(), "length mismatch for {s:?}");
            assert_eq!(parse_temp(bytes) as i32, tenths, "parse_temp {s:?}");
        }
    }
}
