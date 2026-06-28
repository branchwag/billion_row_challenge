# 1BRC — One Billion Row Challenge, in pure-safe Rust

Compute the min / mean / max temperature per weather station from a
1,000,000,000-row (~13 GB) text file of `station;temperature` lines, printed in
the canonical 1BRC format — a single line, stations alphabetical:

```
{Abha=-31.9/18.0/67.3, Abidjan=-29.7/26.0/76.8, Abéché=-23.1/29.4/77.9, ...}
```

## Constraints honored

- **No external crates** — `std` only.
- **No `unsafe`, no C FFI** — 100% safe Rust.
- Single source file for the solution (`src/main.rs`).
- Handles any valid input: UTF-8 station names 1–100 bytes, ≤10,000 unique
  stations, temperatures in `-99.9..=99.9`, any distribution.

## How it works

- The file is split into many 16 MiB **chunks** handed out via a shared atomic
  counter, so threads on faster cores grab more work — important on
  heterogeneous CPUs (Intel P-core + E-core). Each thread streams its chunks in
  2 MiB blocks via the safe `FileExt::read_at`, so the 13 GB is never all held
  in memory.
- Each thread aggregates into a custom open-addressing hash table; station names
  are copied once into a per-thread arena, temperatures kept as integer tenths
  (`i16`).
- Hot-loop tricks, all safe stable Rust: one fused pass per line (find `;` and
  parse the value together), **SWAR** 8-byte-at-a-time `;` search, and a
  **branchless SWAR** temperature parse (the merykitty technique).
- Per-thread tables are merged, sorted by name, and printed in one write.

The mean is rounded half-up toward +∞ to match the original Java reference's
`Math.round`, computed in exact integer arithmetic
(`floor((2·sum + count) / (2·count))`) — the solution uses no floating point
at all.

## Build

```shell
cargo build --release
```

Produces two binaries: `brc` (the solution) and `gen` (the data generator).

## Generate a measurements file

The real challenge file is generated on demand and never distributed. `gen`
reproduces the canonical distribution from the original `gunnarmorling/1brc`
generator (the 413 real weather stations, Gaussian temperatures). It writes to
stdout:

```shell
# Full billion rows (~13 GB — make sure you have the disk space!)
./target/release/gen 1000000000 > measurements.txt

# A smaller file for quick tests
./target/release/gen 1000000 > measurements.txt
```

Optional second arg limits how many of the 413 stations are used:
`gen <num_rows> [max_cities]`.

## Run

```shell
./target/release/brc measurements.txt
# (defaults to ./measurements.txt if no path is given)
```

## Test

An exhaustive unit test checks the branchless parser against every valid
temperature:

```shell
cargo test --release
```

## Benchmarks (Intel i3-1215U, 2 P-cores + 4 E-cores, 7.4 GB RAM)

| Scenario | Result |
|---|---|
| **Full 1B rows, 13 GB file on NVMe** | **~10 s** (disk-bound, ~1.37 GB/s; file ≫ RAM) |
| Warm 100M-row file in page cache | ~0.49 s (~204M rows/s) |

The full 1B file (13 GB) cannot fit in this machine's 7.4 GB page cache, so each
run is bounded by NVMe read bandwidth (~1.37 GB/s here). The warm number is the
one that scales on the published-leaderboard hardware (≫13 GB RAM, file cached,
many fast cores): at ~204M rows/s of pure parsing this code reaches the low-
seconds range there.

> A true 1–2 s result for a real 1B-row run requires the file to live entirely
> in RAM (page cache) — i.e. a box with ≫13 GB RAM and many fast cores, which is
> what the published 1BRC leaderboard numbers assume. On a small-RAM machine the
> file can't be cached, so every run is bounded by NVMe read bandwidth.

### I/O vs CPU split

Running the full 13 GB file a second time back-to-back (no reboot) stays flat —
there's no warm-up speedup. The file is far larger than the usable page cache
(~5 GB free of 7.4 GB total), so the cache thrashes: by the time a run finishes
streaming all 13 GB, its early bytes have already been evicted, and the next run
re-reads essentially the whole file from disk. The effective rate (~13.8 GB / ~10 s
≈ 1.37 GB/s) sits right at the NVMe's bandwidth ceiling.

Decomposing the work with the warm number:

- **Pure parsing**: ~0.5 s for 100M warm rows → **~5 s** of actual CPU for 1B.
- **Pure I/O**: ~13.8 GB at ~1.37 GB/s → **~10 s**.

The ~5 s of parsing is spread across 6 cores and **fully overlapped** with reading
(threads parse their buffers while others wait on the disk), so it fits entirely
underneath the I/O time and never extends the wall clock. Net: roughly half the
raw work is CPU, but ~100% of the *elapsed* time is gated by the SSD. On this
machine no hot-loop optimization moves the 1B number — only faster reads (more
RAM to cache the file, or a faster drive) would. The ~200M rows/s parse rate only
becomes the bottleneck where the file already fits in RAM.
