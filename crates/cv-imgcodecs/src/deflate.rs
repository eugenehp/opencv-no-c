//! DEFLATE (RFC 1951) and zlib (RFC 1950) in pure Rust.
//!
//! Upstream OpenCV links libpng, which links zlib. Since this port takes no C
//! dependency, the codec is implemented here. Decompression is complete —
//! stored, fixed-Huffman and dynamic-Huffman blocks — because real PNG files in
//! the wild use all three.
//!
//! Compression uses **stored (uncompressed) blocks** only. The output is a
//! valid DEFLATE stream that any decoder accepts; it is simply larger than a
//! matching-and-Huffman compressor would produce. Encoding a smaller stream is
//! an optimisation, not a correctness requirement, and is kept out of scope so
//! the decoder — where correctness actually matters — stays reviewable.

use cv_core::{cv_error, Code, Result};

/// Largest output we will produce from one stream, as a guard against a
/// malformed or hostile input expanding without bound.
const MAX_OUTPUT: usize = 1 << 30;

/// Reads bits least-significant-first, as DEFLATE requires.
struct BitReader<'a> {
    data: &'a [u8],
    /// Byte position of the next unread byte.
    pos: usize,
    /// Bits buffered from bytes already consumed.
    bit_buf: u64,
    /// How many bits in `bit_buf` are valid.
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, bit_buf: 0, bit_count: 0 }
    }

    /// Ensure at least `n` bits are buffered.
    fn fill(&mut self, n: u32) -> Result<()> {
        while self.bit_count < n {
            if self.pos >= self.data.len() {
                return cv_error!(Code::StsError, "truncated DEFLATE stream");
            }
            self.bit_buf |= (self.data[self.pos] as u64) << self.bit_count;
            self.pos += 1;
            self.bit_count += 8;
        }
        Ok(())
    }

    /// Consume `n` bits (n <= 32).
    fn bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        let v = (self.bit_buf & ((1u64 << n) - 1)) as u32;
        self.bit_buf >>= n;
        self.bit_count -= n;
        Ok(v)
    }

    /// Discard buffered bits back to a byte boundary.
    fn align(&mut self) {
        let drop = self.bit_count % 8;
        self.bit_buf >>= drop;
        self.bit_count -= drop;
    }

    /// Take `n` whole bytes, which must already be byte-aligned.
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        // Any bytes still sitting in the bit buffer have to be rewound first.
        let buffered = (self.bit_count / 8) as usize;
        let start = self.pos - buffered;
        self.bit_buf = 0;
        self.bit_count = 0;
        self.pos = start + n;
        self.data
            .get(start..start + n)
            .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated stored block".to_string()))
    }
}

/// A canonical Huffman decoding table built from per-symbol code lengths.
struct Huffman {
    /// `counts[l]` is how many symbols use length `l`.
    counts: [u16; 16],
    /// Symbols ordered by (length, symbol).
    symbols: Vec<u16>,
}

impl Huffman {
    /// Build from code lengths, where length 0 means "unused".
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l as usize > 15 {
                return cv_error!(Code::StsError, "Huffman code length {l} exceeds 15");
            }
            counts[l as usize] += 1;
        }
        counts[0] = 0;

        // Reject over-subscribed tables; an incomplete one is legal only when
        // it holds at most a single symbol.
        let mut left = 1i32;
        for &count in counts.iter().skip(1) {
            left = (left << 1) - count as i32;
            if left < 0 {
                return cv_error!(Code::StsError, "over-subscribed Huffman table");
            }
        }

        let mut offsets = [0u16; 16];
        for l in 1..15 {
            offsets[l + 1] = offsets[l] + counts[l];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Huffman { counts, symbols })
    }

    /// Decode one symbol, walking lengths shortest-first.
    fn decode(&self, r: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= r.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        cv_error!(Code::StsError, "invalid Huffman code in DEFLATE stream")
    }
}

/// Base lengths for length codes 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits for length codes 257..=285.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distances for distance codes 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits for distance codes 0..=29.
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The permuted order in which code-length code lengths are stored.
const CLCL_ORDER: [usize; 19] =
    [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// The fixed literal/length table from RFC 1951 §3.2.6.
fn fixed_literal_table() -> Result<Huffman> {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Huffman::new(&lengths)
}

/// The fixed distance table: 30 codes of 5 bits (plus two unused).
fn fixed_distance_table() -> Result<Huffman> {
    Huffman::new(&[5u8; 32])
}

/// Decompress a raw DEFLATE stream (no zlib wrapper).
pub fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut r = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();

    loop {
        let final_block = r.bits(1)? == 1;
        let btype = r.bits(2)?;

        match btype {
            0 => {
                r.align();
                let header = r.bytes(4)?;
                let len = u16::from_le_bytes([header[0], header[1]]) as usize;
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if (len as u16) != !nlen {
                    return cv_error!(Code::StsError, "stored block length check failed");
                }
                out.extend_from_slice(r.bytes(len)?);
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 {
                    (fixed_literal_table()?, fixed_distance_table()?)
                } else {
                    read_dynamic_tables(&mut r)?
                };
                inflate_block(&mut r, &lit, &dist, &mut out)?;
            }
            _ => return cv_error!(Code::StsError, "reserved DEFLATE block type 3"),
        }

        if out.len() > MAX_OUTPUT {
            return cv_error!(Code::StsError, "DEFLATE output exceeds {MAX_OUTPUT} bytes");
        }
        if final_block {
            break;
        }
    }
    Ok(out)
}

/// Read the dynamic Huffman tables that precede a BTYPE=2 block.
fn read_dynamic_tables(r: &mut BitReader<'_>) -> Result<(Huffman, Huffman)> {
    let hlit = r.bits(5)? as usize + 257;
    let hdist = r.bits(5)? as usize + 1;
    let hclen = r.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return cv_error!(Code::StsError, "invalid dynamic table sizes {hlit}/{hdist}");
    }

    // The code-length alphabet's own lengths, stored in a fixed permuted order.
    let mut clcl = [0u8; 19];
    for &idx in CLCL_ORDER.iter().take(hclen) {
        clcl[idx] = r.bits(3)? as u8;
    }
    let cl_table = Huffman::new(&clcl)?;

    // Literal/length and distance lengths share one run-length coded stream.
    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0usize;
    while i < total {
        let sym = cl_table.decode(r)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Repeat the previous length 3..=6 times.
                if i == 0 {
                    return cv_error!(Code::StsError, "repeat code at start of length stream");
                }
                let prev = lengths[i - 1];
                let n = 3 + r.bits(2)? as usize;
                if i + n > total {
                    return cv_error!(Code::StsError, "length repeat overruns the table");
                }
                lengths[i..i + n].fill(prev);
                i += n;
            }
            17 => {
                let n = 3 + r.bits(3)? as usize;
                if i + n > total {
                    return cv_error!(Code::StsError, "zero run overruns the table");
                }
                i += n;
            }
            18 => {
                let n = 11 + r.bits(7)? as usize;
                if i + n > total {
                    return cv_error!(Code::StsError, "zero run overruns the table");
                }
                i += n;
            }
            _ => return cv_error!(Code::StsError, "invalid code-length symbol {sym}"),
        }
    }

    Ok((Huffman::new(&lengths[..hlit])?, Huffman::new(&lengths[hlit..])?))
}

/// Decode symbols until end-of-block, appending to `out`.
fn inflate_block(
    r: &mut BitReader<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
) -> Result<()> {
    loop {
        let sym = lit.decode(r)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let i = sym as usize - 257;
                let length = LENGTH_BASE[i] as usize + r.bits(LENGTH_EXTRA[i] as u32)? as usize;

                let dsym = dist.decode(r)? as usize;
                if dsym >= 30 {
                    return cv_error!(Code::StsError, "invalid distance code {dsym}");
                }
                let distance =
                    DIST_BASE[dsym] as usize + r.bits(DIST_EXTRA[dsym] as u32)? as usize;
                if distance > out.len() {
                    return cv_error!(
                        Code::StsError,
                        "back-reference distance {distance} exceeds {} bytes produced",
                        out.len()
                    );
                }
                // Copy byte by byte: overlapping runs are legal and common.
                let start = out.len() - distance;
                for k in 0..length {
                    let b = out[start + k];
                    out.push(b);
                }
                if out.len() > MAX_OUTPUT {
                    return cv_error!(Code::StsError, "DEFLATE output exceeds {MAX_OUTPUT} bytes");
                }
            }
            _ => return cv_error!(Code::StsError, "invalid literal/length symbol {sym}"),
        }
    }
}

/// Compress with stored (uncompressed) DEFLATE blocks.
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    // A stored block's length field is 16 bits, so long inputs are split.
    const MAX_BLOCK: usize = 65535;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX_BLOCK * 5 + 5);

    if data.is_empty() {
        // A single empty final block.
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        return out;
    }
    let mut offset = 0usize;
    while offset < data.len() {
        let n = (data.len() - offset).min(MAX_BLOCK);
        let is_final = offset + n >= data.len();
        out.push(if is_final { 1 } else { 0 }); // BFINAL, BTYPE=0, byte-aligned
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&data[offset..offset + n]);
        offset += n;
    }
    out
}

/// Wrap a DEFLATE stream in the zlib container (RFC 1950).
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // CMF/FLG: deflate, 32K window, no preset dictionary, check bits chosen so
    // the 16-bit header is a multiple of 31.
    out.push(0x78);
    out.push(0x01);
    out.extend_from_slice(&deflate_stored(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Unwrap and decompress a zlib stream, verifying the Adler-32 checksum.
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return cv_error!(Code::StsError, "zlib stream is too short");
    }
    let cmf = data[0];
    let flg = data[1];
    if cmf & 0x0F != 8 {
        return cv_error!(Code::StsError, "unsupported zlib compression method {}", cmf & 0x0F);
    }
    if (u16::from_be_bytes([cmf, flg])) % 31 != 0 {
        return cv_error!(Code::StsError, "corrupt zlib header");
    }
    if flg & 0x20 != 0 {
        return cv_error!(Code::StsError, "zlib preset dictionaries are not supported");
    }

    let body = &data[2..data.len() - 4];
    let out = inflate(body)?;

    let expected = u32::from_be_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let actual = adler32(&out);
    if expected != actual {
        return cv_error!(
            Code::StsError,
            "zlib checksum mismatch: expected {expected:#010x}, got {actual:#010x}"
        );
    }
    Ok(out)
}

/// Adler-32 (RFC 1950 §9).
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    // Chunked so the accumulators cannot overflow before the modulo.
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// CRC-32 (IEEE 802.3), as PNG chunks use.
pub fn crc32(data: &[u8]) -> u32 {
    let table = crc_table();
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Lazily built CRC-32 lookup table.
fn crc_table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, slot) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_matches_the_specification_example() {
        // RFC 1950 gives Adler-32("Wikipedia") = 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn crc32_matches_known_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn stored_blocks_roundtrip() {
        for data in [
            b"".to_vec(),
            b"a".to_vec(),
            b"hello world".to_vec(),
            (0..1000u32).map(|i| (i % 251) as u8).collect(),
        ] {
            let c = deflate_stored(&data);
            assert_eq!(inflate(&c).unwrap(), data, "len {}", data.len());
        }
    }

    #[test]
    fn stored_blocks_split_past_65535_bytes() {
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 253) as u8).collect();
        let c = deflate_stored(&data);
        assert_eq!(inflate(&c).unwrap(), data);
    }

    #[test]
    fn zlib_roundtrip_verifies_the_checksum() {
        let data = b"the quick brown fox jumps over the lazy dog".repeat(20);
        let c = zlib_compress(&data);
        assert_eq!(zlib_decompress(&c).unwrap(), data);

        // Corrupting the trailer must be detected.
        let mut bad = c.clone();
        let n = bad.len();
        bad[n - 1] ^= 0xFF;
        assert!(zlib_decompress(&bad).is_err());
    }

    /// Decode a hex string into bytes.
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap()).collect()
    }

    /// These streams come from zlib itself (`compressobj(9, DEFLATED, -15)`),
    /// so they exercise paths this crate's own encoder never produces: the
    /// stored-block writer emits neither Huffman codes nor back-references.
    #[test]
    fn decodes_a_fixed_huffman_stream_with_backreferences() {
        // zlib -9 raw deflate of b"hello hello hello": BTYPE=1, and the tail
        // is an overlapping copy (distance 6, length 11).
        let compressed = hex("cb48cdc9c957c8409000");
        assert_eq!((compressed[0] >> 1) & 3, 1, "expected a fixed-Huffman block");
        assert_eq!(inflate(&compressed).unwrap(), b"hello hello hello");
    }

    #[test]
    fn decodes_a_dynamic_huffman_stream() {
        // zlib -9 raw deflate of the 30-byte source below, which is skewed
        // enough to make a custom table worthwhile.
        let compressed = hex("05c1810100000402b05b2954ff1f60638c8dd2298d49d93a10be6ae401");
        assert_eq!((compressed[0] >> 1) & 3, 2, "expected a dynamic-Huffman block");
        assert_eq!(inflate(&compressed).unwrap(), b"djicfjhjbjaheiddhiihgcdcigabcj");
    }

    #[test]
    fn overlapping_backreference_repeats_correctly() {
        // "abababababab" compresses to a 2-byte literal seed plus a copy whose
        // distance (2) is shorter than its length (10) — the overlapping case
        // that a naive block memcpy gets wrong.
        let compressed = hex("4b4c4a844300");
        assert_eq!(inflate(&compressed).unwrap(), b"abababababab");
    }

    #[test]
    fn truncated_streams_error_rather_than_panic() {
        let data = b"some data to compress".repeat(10);
        let c = zlib_compress(&data);
        for cut in [1, 3, 6, c.len() / 2, c.len() - 1] {
            assert!(zlib_decompress(&c[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn corrupt_stored_length_check_is_caught() {
        let mut c = deflate_stored(b"hello");
        c[3] ^= 0xFF; // break the NLEN complement
        assert!(inflate(&c).is_err());
    }

    #[test]
    fn reserved_block_type_is_rejected() {
        // BFINAL=1, BTYPE=3 -> 0b111
        assert!(inflate(&[0x07]).is_err());
    }

    #[test]
    fn bad_zlib_headers_are_rejected() {
        assert!(zlib_decompress(&[0x78, 0x02, 0, 0, 0, 0]).is_err()); // not %31
        assert!(zlib_decompress(&[0x79, 0x01, 0, 0, 0, 0]).is_err()); // method != 8
        assert!(zlib_decompress(&[0x78]).is_err()); // too short
    }

    #[test]
    fn oversubscribed_huffman_table_is_rejected() {
        // Three symbols all claiming a 1-bit code cannot form a prefix code.
        assert!(Huffman::new(&[1, 1, 1]).is_err());
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        // The decoder runs on untrusted file content, so every malformed input
        // must come back as an error rather than an out-of-bounds read or an
        // unbounded allocation. A deterministic xorshift keeps this
        // reproducible without a dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [1usize, 2, 3, 5, 8, 13, 32, 64, 200] {
            for _ in 0..200 {
                let buf: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
                // Only the return value matters; a panic fails the test.
                let _ = inflate(&buf);
                let _ = zlib_decompress(&buf);
            }
        }
    }

    #[test]
    fn every_prefix_of_a_valid_stream_is_handled() {
        // Truncation at any byte must error rather than panic or hang.
        let data = b"the quick brown fox".repeat(30);
        let full = zlib_compress(&data);
        for cut in 0..full.len() {
            let r = zlib_decompress(&full[..cut]);
            assert!(r.is_err(), "prefix of {cut} bytes decoded as complete");
        }
        assert_eq!(zlib_decompress(&full).unwrap(), data);
    }
}
