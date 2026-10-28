"""Optional high-performance native acceleration bridge for ProofCausality AI.

Loads the compiled Rust library (`proofcausality_core`) via C-ABI pointer-passing.
If the native library is absent (e.g. non-Rust runtime), seamlessly falls back
to the optimized pure-NumPy implementation with identical mathematical semantics.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Any
import numpy as np

# Search paths for compiled dynamic library
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_NATIVE_RELEASE_DIR = os.path.join(_PROJECT_ROOT, "native", "proofcausality_core", "target", "release")

if platform.system() == "Darwin":
    _LIB_NAME = "libproofcausality_core.dylib"
elif platform.system() == "Windows":
    _LIB_NAME = "proofcausality_core.dll"
else:
    _LIB_NAME = "libproofcausality_core.so"

_CANDIDATE_PATHS = [
    os.path.join(_NATIVE_RELEASE_DIR, _LIB_NAME),
    os.path.join(_CURRENT_DIR, _LIB_NAME),
]

_RUST_LIB: ctypes.CDLL | None = None
_RUST_AVAILABLE: bool = False
_LOAD_ERROR: str | None = None

for path in _CANDIDATE_PATHS:
    if os.path.exists(path):
        try:
            _RUST_LIB = ctypes.CDLL(path)
            # Configure function signatures
            _RUST_LIB.rust_mask_iou.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            _RUST_LIB.rust_mask_iou.restype = ctypes.c_double

            _RUST_LIB.rust_mask_dice.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            _RUST_LIB.rust_mask_dice.restype = ctypes.c_double

            _RUST_LIB.rust_batch_mask_iou.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
            ]
            _RUST_LIB.rust_batch_mask_iou.restype = ctypes.c_int

            _RUST_LIB.rust_laplacian_variance.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            _RUST_LIB.rust_laplacian_variance.restype = ctypes.c_double

            _RUST_AVAILABLE = True
            break
        except Exception as e:
            _LOAD_ERROR = str(e)


def is_rust_accelerator_available() -> bool:
    """Return True if the compiled native Rust acceleration library is loaded."""
    return _RUST_AVAILABLE


def get_accelerator_info() -> dict[str, Any]:
    """Return runtime metadata regarding the active computational acceleration engine."""
    return {
        "rust_accelerator_active": _RUST_AVAILABLE,
        "backend_engine": "Rust Native cdylib (SIMD/zero-copy)" if _RUST_AVAILABLE else "Pure NumPy Fallback",
        "library_path": _CANDIDATE_PATHS[0] if _RUST_AVAILABLE else None,
        "load_error": _LOAD_ERROR,
    }


def compute_mask_iou_fast(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) using Rust if available, else NumPy."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch in mask IoU: {mask_a.shape} vs {mask_b.shape}")

    if _RUST_AVAILABLE and _RUST_LIB is not None:
        c_a = np.ascontiguousarray(mask_a, dtype=np.uint8)
        c_b = np.ascontiguousarray(mask_b, dtype=np.uint8)
        ptr_a = c_a.ctypes.data_as(ctypes.c_void_p)
        ptr_b = c_b.ctypes.data_as(ctypes.c_void_p)
        return round(float(_RUST_LIB.rust_mask_iou(ptr_a, ptr_b, c_a.size)), 4)

    # NumPy Fallback
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)
    intersection = np.count_nonzero(a_bool & b_bool)
    union = np.count_nonzero(a_bool | b_bool)
    if union == 0:
        return 1.0
    return round(float(intersection / union), 4)


def compute_mask_dice_fast(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Sørensen-Dice coefficient using Rust if available, else NumPy."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch in mask Dice: {mask_a.shape} vs {mask_b.shape}")

    if _RUST_AVAILABLE and _RUST_LIB is not None:
        c_a = np.ascontiguousarray(mask_a, dtype=np.uint8)
        c_b = np.ascontiguousarray(mask_b, dtype=np.uint8)
        ptr_a = c_a.ctypes.data_as(ctypes.c_void_p)
        ptr_b = c_b.ctypes.data_as(ctypes.c_void_p)
        return round(float(_RUST_LIB.rust_mask_dice(ptr_a, ptr_b, c_a.size)), 4)

    # NumPy Fallback
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)
    intersection = np.count_nonzero(a_bool & b_bool)
    total = np.count_nonzero(a_bool) + np.count_nonzero(b_bool)
    if total == 0:
        return 1.0
    return round(float(2.0 * intersection / total), 4)


def compute_batch_mask_iou_fast(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Compute batch IoU across N mask pairs (shape: N x H x W or N x L)."""
    if masks_a.shape != masks_b.shape:
        raise ValueError(f"Batch mask shape mismatch: {masks_a.shape} vs {masks_b.shape}")

    num_masks = masks_a.shape[0]
    if num_masks == 0:
        return np.empty(0, dtype=np.float64)

    mask_len = masks_a[0].size

    if _RUST_AVAILABLE and _RUST_LIB is not None:
        c_a = np.ascontiguousarray(masks_a, dtype=np.uint8)
        c_b = np.ascontiguousarray(masks_b, dtype=np.uint8)
        out = np.empty(num_masks, dtype=np.float64)

        ptr_a = c_a.ctypes.data_as(ctypes.c_void_p)
        ptr_b = c_b.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)

        ret = _RUST_LIB.rust_batch_mask_iou(ptr_a, ptr_b, num_masks, mask_len, out_ptr)
        if ret == 0:
            return np.round(out, 4)

    # NumPy Fallback
    flat_a = masks_a.reshape(num_masks, -1).astype(bool)
    flat_b = masks_b.reshape(num_masks, -1).astype(bool)
    inter = np.count_nonzero(flat_a & flat_b, axis=1)
    union = np.count_nonzero(flat_a | flat_b, axis=1)

    res = np.empty(num_masks, dtype=np.float64)
    for i in range(num_masks):
        res[i] = 1.0 if union[i] == 0 else float(inter[i] / union[i])
    return np.round(res, 4)


def compute_laplacian_variance_fast(gray_img: np.ndarray) -> float:
    """Compute 2D Laplacian variance using Rust if available, else NumPy."""
    h, w = gray_img.shape
    if h < 3 or w < 3:
        return 0.0

    if _RUST_AVAILABLE and _RUST_LIB is not None:
        c_gray = np.ascontiguousarray(gray_img, dtype=np.float64)
        ptr = c_gray.ctypes.data_as(ctypes.c_void_p)
        return round(float(_RUST_LIB.rust_laplacian_variance(ptr, w, h)), 2)

    # NumPy Fallback
    lap = (
        gray_img[:-2, 1:-1]
        + gray_img[2:, 1:-1]
        + gray_img[1:-1, :-2]
        + gray_img[1:-1, 2:]
        - 4.0 * gray_img[1:-1, 1:-1]
    )
    return round(float(np.var(lap)), 2)
