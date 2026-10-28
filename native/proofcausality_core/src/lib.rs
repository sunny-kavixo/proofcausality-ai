//! High-performance native acceleration kernels for ProofCausality AI.
//!
//! Provides zero-allocation C-ABI primitives for:
//! - Hardware-vectorized 64-bit chunk bitwise popcount for mask IoU & Dice.
//! - Batch IoU evaluation across multiple spatial masks.
//! - Cache-friendly single-pass 2D Laplacian variance convolution for blur detection.

use std::slice;

/// Compute Intersection over Union (IoU) between two contiguous uint8/bool byte arrays.
///
/// Uses 64-bit word chunks and hardware `POPCNT` instruction for zero-allocation streaming.
#[no_mangle]
pub extern "C" fn rust_mask_iou(
    ptr_a: *const u8,
    ptr_b: *const u8,
    len: usize,
) -> f64 {
    if ptr_a.is_null() || ptr_b.is_null() || len == 0 {
        return 1.0;
    }

    let slice_a = unsafe { slice::from_raw_parts(ptr_a, len) };
    let slice_b = unsafe { slice::from_raw_parts(ptr_b, len) };

    let mut intersection: u64 = 0;
    let mut union_count: u64 = 0;

    let chunks_a = slice_a.chunks_exact(8);
    let chunks_b = slice_b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        let a = u64::from_ne_bytes(ca.try_into().unwrap());
        let b = u64::from_ne_bytes(cb.try_into().unwrap());

        // For boolean masks where true is 1 and false is 0,
        // bit 0 of each byte is set. Bitwise AND & OR directly yield active bytes:
        let inter_bytes = a & b;
        let union_bytes = a | b;

        intersection += inter_bytes.count_ones() as u64;
        union_count += union_bytes.count_ones() as u64;
    }

    for (ba, bb) in rem_a.iter().zip(rem_b.iter()) {
        let val_a = (*ba != 0) as u64;
        let val_b = (*bb != 0) as u64;
        intersection += val_a & val_b;
        union_count += val_a | val_b;
    }

    if union_count == 0 {
        1.0
    } else {
        (intersection as f64) / (union_count as f64)
    }
}

/// Compute Sørensen-Dice coefficient: 2 * |A ∩ B| / (|A| + |B|).
#[no_mangle]
pub extern "C" fn rust_mask_dice(
    ptr_a: *const u8,
    ptr_b: *const u8,
    len: usize,
) -> f64 {
    if ptr_a.is_null() || ptr_b.is_null() || len == 0 {
        return 1.0;
    }

    let slice_a = unsafe { slice::from_raw_parts(ptr_a, len) };
    let slice_b = unsafe { slice::from_raw_parts(ptr_b, len) };

    let mut intersection: u64 = 0;
    let mut count_a: u64 = 0;
    let mut count_b: u64 = 0;

    let chunks_a = slice_a.chunks_exact(8);
    let chunks_b = slice_b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        let a = u64::from_ne_bytes(ca.try_into().unwrap());
        let b = u64::from_ne_bytes(cb.try_into().unwrap());

        intersection += (a & b).count_ones() as u64;
        count_a += a.count_ones() as u64;
        count_b += b.count_ones() as u64;
    }

    for (ba, bb) in rem_a.iter().zip(rem_b.iter()) {
        let val_a = (*ba != 0) as u64;
        let val_b = (*bb != 0) as u64;
        intersection += val_a & val_b;
        count_a += val_a;
        count_b += val_b;
    }

    let total = count_a + count_b;
    if total == 0 {
        1.0
    } else {
        (2.0 * (intersection as f64)) / (total as f64)
    }
}

/// Compute batch IoU for multiple mask pairs using 64-bit word chunks.
#[no_mangle]
pub extern "C" fn rust_batch_mask_iou(
    ptr_a: *const u8,
    ptr_b: *const u8,
    num_masks: usize,
    mask_len: usize,
    out_ptr: *mut f64,
) -> i32 {
    if ptr_a.is_null() || ptr_b.is_null() || out_ptr.is_null() || num_masks == 0 || mask_len == 0 {
        return -1;
    }

    let total_len = num_masks * mask_len;
    let slice_a = unsafe { slice::from_raw_parts(ptr_a, total_len) };
    let slice_b = unsafe { slice::from_raw_parts(ptr_b, total_len) };
    let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, num_masks) };

    for i in 0..num_masks {
        let offset = i * mask_len;
        let sub_a = &slice_a[offset..offset + mask_len];
        let sub_b = &slice_b[offset..offset + mask_len];

        let mut inter: u64 = 0;
        let mut union_count: u64 = 0;

        let chunks_a = sub_a.chunks_exact(8);
        let chunks_b = sub_b.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let a = u64::from_ne_bytes(ca.try_into().unwrap());
            let b = u64::from_ne_bytes(cb.try_into().unwrap());
            inter += (a & b).count_ones() as u64;
            union_count += (a | b).count_ones() as u64;
        }

        for (ba, bb) in rem_a.iter().zip(rem_b.iter()) {
            let va = (*ba != 0) as u64;
            let vb = (*bb != 0) as u64;
            inter += va & vb;
            union_count += va | vb;
        }

        out_slice[i] = if union_count == 0 {
            1.0
        } else {
            (inter as f64) / (union_count as f64)
        };
    }

    0
}

/// Fast 2D Laplacian variance convolution on float64 grayscale image.
///
/// Convolution kernel:
///  [ 0,  1,  0]
///  [ 1, -4,  1]
///  [ 0,  1,  0]
#[no_mangle]
pub extern "C" fn rust_laplacian_variance(
    gray_ptr: *const f64,
    width: usize,
    height: usize,
) -> f64 {
    if gray_ptr.is_null() || width < 3 || height < 3 {
        return 0.0;
    }

    let total = width * height;
    let img = unsafe { slice::from_raw_parts(gray_ptr, total) };

    let mut sum_lap: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let inner_count = ((width - 2) * (height - 2)) as f64;

    for y in 1..(height - 1) {
        let row_curr = y * width;
        let row_prev = (y - 1) * width;
        let row_next = (y + 1) * width;

        for x in 1..(width - 1) {
            let top = img[row_prev + x];
            let bottom = img[row_next + x];
            let left = img[row_curr + x - 1];
            let right = img[row_curr + x + 1];
            let center = img[row_curr + x];

            let lap = top + bottom + left + right - 4.0 * center;
            sum_lap += lap;
            sum_sq += lap * lap;
        }
    }

    let mean = sum_lap / inner_count;
    let variance = (sum_sq / inner_count) - (mean * mean);
    if variance < 0.0 {
        0.0
    } else {
        variance
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── rust_mask_iou ─────────────────────────────────────────────────────────

    #[test]
    fn iou_identical_masks_returns_one() {
        let a = vec![1u8; 100];
        let b = a.clone();
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0, got {result}");
    }

    #[test]
    fn iou_disjoint_masks_returns_zero() {
        let a = [1u8, 0, 0, 0];
        let b = [0u8, 1, 0, 0];
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 0.0).abs() < 1e-9, "expected 0.0, got {result}");
    }

    #[test]
    fn iou_empty_masks_returns_one() {
        // Both masks all-zero → union = 0 → returns 1.0 (empty-equals-empty convention)
        let a = vec![0u8; 64];
        let b = vec![0u8; 64];
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0 for empty masks, got {result}");
    }

    #[test]
    fn iou_half_overlap_is_one_third() {
        // A = [1,1,0,0], B = [0,1,1,0] → I=1, U=3 → IoU = 1/3
        let a = [1u8, 1, 0, 0];
        let b = [0u8, 1, 1, 0];
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        let expected = 1.0 / 3.0;
        assert!(
            (result - expected).abs() < 1e-9,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn iou_null_ptr_a_returns_one() {
        let b = vec![1u8; 4];
        let result = rust_mask_iou(std::ptr::null(), b.as_ptr(), b.len());
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0 for null ptr, got {result}");
    }

    #[test]
    fn iou_zero_length_returns_one() {
        let a = vec![1u8; 4];
        let b = vec![1u8; 4];
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), 0);
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0 for len=0, got {result}");
    }

    #[test]
    fn iou_non_chunk_aligned_length() {
        // Length 9 — not a multiple of 8 — exercises remainder path
        let a: Vec<u8> = (0..9).map(|i| if i < 5 { 1 } else { 0 }).collect();
        let b: Vec<u8> = (0..9).map(|i| if i < 3 { 1 } else { 0 }).collect();
        // I=3, A=5, B=3, U=5 → IoU = 3/5
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        assert!(
            (result - 3.0 / 5.0).abs() < 1e-9,
            "expected 0.6, got {result}"
        );
    }

    // ── rust_mask_dice ────────────────────────────────────────────────────────

    #[test]
    fn dice_identical_masks_returns_one() {
        let a = vec![1u8; 64];
        let b = a.clone();
        let result = rust_mask_dice(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0, got {result}");
    }

    #[test]
    fn dice_disjoint_masks_returns_zero() {
        let a = [1u8, 0, 0, 0];
        let b = [0u8, 0, 0, 1];
        let result = rust_mask_dice(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 0.0).abs() < 1e-9, "expected 0.0, got {result}");
    }

    #[test]
    fn dice_empty_masks_returns_one() {
        let a = vec![0u8; 64];
        let b = vec![0u8; 64];
        let result = rust_mask_dice(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 1.0).abs() < 1e-9, "expected 1.0, got {result}");
    }

    #[test]
    fn dice_half_overlap_correct() {
        // A=[1,1,0,0], B=[0,1,1,0] → I=1, |A|=2, |B|=2 → Dice = 2*1/(2+2) = 0.5
        let a = [1u8, 1, 0, 0];
        let b = [0u8, 1, 1, 0];
        let result = rust_mask_dice(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 0.5).abs() < 1e-9, "expected 0.5, got {result}");
    }

    // ── rust_batch_mask_iou ───────────────────────────────────────────────────

    #[test]
    fn batch_iou_single_identical_pair() {
        let a = vec![1u8; 16];
        let b = vec![1u8; 16];
        let mut out = vec![0.0f64; 1];
        let ret = rust_batch_mask_iou(a.as_ptr(), b.as_ptr(), 1, 16, out.as_mut_ptr());
        assert_eq!(ret, 0, "expected 0 return code");
        assert!((out[0] - 1.0).abs() < 1e-9, "expected 1.0, got {}", out[0]);
    }

    #[test]
    fn batch_iou_multiple_pairs_correct() {
        // Pair 0: identical → 1.0
        // Pair 1: disjoint → 0.0
        let mask_len = 4;
        let a: Vec<u8> = [1u8, 1, 1, 1, 1, 0, 0, 0].to_vec();
        let b: Vec<u8> = [1u8, 1, 1, 1, 0, 1, 1, 1].to_vec();
        let mut out = vec![0.0f64; 2];
        let ret = rust_batch_mask_iou(a.as_ptr(), b.as_ptr(), 2, mask_len, out.as_mut_ptr());
        assert_eq!(ret, 0);
        assert!((out[0] - 1.0).abs() < 1e-9, "pair0 expected 1.0, got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-9, "pair1 expected 0.0, got {}", out[1]);
    }

    #[test]
    fn batch_iou_null_ptr_returns_error() {
        let b = vec![1u8; 4];
        let mut out = vec![0.0f64; 1];
        let ret = rust_batch_mask_iou(std::ptr::null(), b.as_ptr(), 1, 4, out.as_mut_ptr());
        assert_eq!(ret, -1, "expected -1 for null ptr");
    }

    // ── rust_laplacian_variance ───────────────────────────────────────────────

    #[test]
    fn laplacian_constant_image_returns_zero() {
        // Flat image → Laplacian = 0 everywhere → variance = 0
        let img = vec![128.0f64; 9 * 9];
        let result = rust_laplacian_variance(img.as_ptr(), 9, 9);
        assert!(result.abs() < 1e-9, "expected 0.0, got {result}");
    }

    #[test]
    fn laplacian_small_image_below_minimum_returns_zero() {
        // width < 3 → skip computation, return 0.0
        let img = vec![1.0f64; 4];
        let result = rust_laplacian_variance(img.as_ptr(), 2, 2);
        assert!((result - 0.0).abs() < 1e-9, "expected 0.0, got {result}");
    }

    #[test]
    fn laplacian_checkerboard_returns_positive_variance() {
        // Checkerboard 5×5 has high edge content → positive variance
        let w = 5usize;
        let h = 5usize;
        let img: Vec<f64> = (0..h)
            .flat_map(|y| (0..w).map(move |x| if (x + y) % 2 == 0 { 255.0 } else { 0.0 }))
            .collect();
        let result = rust_laplacian_variance(img.as_ptr(), w, h);
        assert!(result > 0.0, "expected positive variance, got {result}");
    }

    #[test]
    fn laplacian_null_ptr_returns_zero() {
        let result = rust_laplacian_variance(std::ptr::null(), 5, 5);
        assert!((result - 0.0).abs() < 1e-9, "expected 0.0 for null ptr, got {result}");
    }

    // ── Python parity: IoU matches Python numpy reference ─────────────────────

    #[test]
    fn iou_parity_with_manual_numpy_calculation() {
        // For mask A = 8 ones followed by 8 zeros, mask B = 4 ones followed by 12 zeros
        // I=4, U=8, IoU=0.5
        let mut a = vec![0u8; 16];
        let mut b = vec![0u8; 16];
        for i in 0..8 { a[i] = 1; }
        for i in 0..4 { b[i] = 1; }
        let result = rust_mask_iou(a.as_ptr(), b.as_ptr(), a.len());
        assert!((result - 0.5).abs() < 1e-9, "expected 0.5, got {result}");
    }
}
