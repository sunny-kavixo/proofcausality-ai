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
