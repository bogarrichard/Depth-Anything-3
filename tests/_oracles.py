"""Independent reference implementations used to judge the package.

Nothing in this module may import ``depth_anything_3``. That is the whole
point: a test that checks the code against itself (a round-trip, or two
copies of the same function compared to each other) still passes when the
implementation is self-consistently wrong. The helpers here are written from
the textbook definitions, or delegate to scipy, so a disagreement means the
package changed behaviour -- not that the test drifted with it.

Conventions used throughout, matching the package's docstrings:
  * quaternions are **scalar-last** (x, y, z, w), the same order scipy uses;
  * ``extrinsics`` are world-to-camera, ``pose``/``c2w`` is its inverse;
  * pixel coordinates are (u, v) with u along the image width.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def quat_to_mat_oracle(quat_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-last quaternions -> (..., 3, 3) rotation matrices."""
    flat = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(flat).as_matrix().reshape(*np.shape(quat_xyzw)[:-1], 3, 3)


def mat_to_quat_oracle(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 4) scalar-last quaternions."""
    flat = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    quat = Rotation.from_matrix(flat).as_quat()
    return quat.reshape(*np.shape(matrix)[:-2], 4)


def quat_multiply_oracle(p_xyzw: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-last quaternions, written out longhand.

    ``(w1 + v1)(w2 + v2) = w1 w2 - v1.v2  +  w1 v2 + w2 v1 + v1 x v2``
    """
    p = np.asarray(p_xyzw, dtype=np.float64)
    q = np.asarray(q_xyzw, dtype=np.float64)
    pv, pw = p[..., :3], p[..., 3:]
    qv, qw = q[..., :3], q[..., 3:]
    w = pw * qw - (pv * qv).sum(-1, keepdims=True)
    v = pw * qv + qw * pv + np.cross(pv, qv)
    return np.concatenate([v, w], axis=-1)


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    """Right-handed rotation about a single canonical axis."""
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def nearest_rotation_oracle(m: np.ndarray) -> np.ndarray:
    """Special orthogonal Procrustes: the SO(3) matrix closest to ``m``.

    ``argmax_R trace(R^T m)`` over SO(3), i.e. the Frobenius-nearest rotation.
    """
    u, _, vh = np.linalg.svd(np.asarray(m, dtype=np.float64))
    d = np.sign(np.linalg.det(u @ vh))
    correction = np.zeros(u.shape[:-2] + (3, 3))
    correction[..., 0, 0] = correction[..., 1, 1] = 1.0
    correction[..., 2, 2] = d
    return u @ correction @ vh


# ---------------------------------------------------------------------------
# Rigid and similarity transforms
# ---------------------------------------------------------------------------
def se3_inverse_oracle(a: np.ndarray) -> np.ndarray:
    """Inverse of a rigid (..., 4, 4) transform, by the block formula."""
    a = np.asarray(a, dtype=np.float64)
    r = a[..., :3, :3]
    t = a[..., :3, 3]
    out = np.zeros_like(a)
    out[..., :3, :3] = np.swapaxes(r, -1, -2)
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", np.swapaxes(r, -1, -2), t)
    out[..., 3, 3] = 1.0
    return out


def umeyama_sim3_oracle(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama (1991) similarity fit: find (r, t, s) with ``dst ~ s r src + t``.

    Written from the paper rather than reused from the package, so it can
    referee ``utils/pose_align.py``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    sigma = dst_c.T @ src_c / n
    u, d, vh = np.linalg.svd(sigma)

    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        s_mat[2, 2] = -1.0

    rot = u @ s_mat @ vh
    var_src = (src_c**2).sum() / n
    scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    trans = mu_dst - scale * rot @ mu_src
    return rot, trans, scale


# ---------------------------------------------------------------------------
# Pinhole camera
# ---------------------------------------------------------------------------
def pinhole_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_oracle(points_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space points through K. Returns ((..., 2) pixels, (...) depth)."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    hom = np.einsum("ij,...j->...i", np.asarray(k, dtype=np.float64), points_cam)
    depth = points_cam[..., 2]
    return hom[..., :2] / hom[..., 2:3], depth


def unproject_oracle(pixels: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Back-project (..., 2) pixels at the given z-depth into camera space."""
    pixels = np.asarray(pixels, dtype=np.float64)
    hom = np.concatenate([pixels, np.ones_like(pixels[..., :1])], axis=-1)
    rays = np.einsum("ij,...j->...i", np.linalg.inv(np.asarray(k, dtype=np.float64)), hom)
    return rays * np.asarray(depth, dtype=np.float64)[..., None]


# ---------------------------------------------------------------------------
# Image preprocessing (PIL + cv2 + longhand normalisation)
# ---------------------------------------------------------------------------
def preprocess_image_oracle(
    img: np.ndarray, process_res: int, method: str, patch: int = 14
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Reference for ``InputProcessor`` on one HxWx3 uint8 RGB image.

    Returns ((3, H', W') float32 ImageNet-normalised, (scale_x, scale_y, crop_x, crop_y)
    where intrinsics map as fx,cx *= scale_x; fy,cy *= scale_y; then cx -= crop_x;
    cy -= crop_y).  Longhand from the documented behaviour: boundary resize to
    ``process_res`` (cv2 INTER_CUBIC up, INTER_AREA down), then either round to the nearest
    multiple of ``patch`` by a second resize ("*resize") or floor via centre crop ("*crop").
    """
    import cv2

    h0, w0 = img.shape[:2]
    side = max(w0, h0) if method.startswith("upper") else min(w0, h0)
    out = img
    if side != process_res:
        s = process_res / side
        nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        out = cv2.resize(out, (nw, nh), interpolation=interp)
    h1, w1 = out.shape[:2]
    sx, sy = w1 / w0, h1 / h0
    crop_x = crop_y = 0.0
    if method.endswith("resize"):

        def near(x):
            lo = (x // patch) * patch
            return lo + patch if (lo + patch - x) <= (x - lo) else lo

        nw, nh = max(1, near(w1)), max(1, near(h1))
        if (nw, nh) != (w1, h1):
            interp = cv2.INTER_CUBIC if (nw > w1 or nh > h1) else cv2.INTER_AREA
            out = cv2.resize(out, (nw, nh), interpolation=interp)
        sx *= nw / w1
        sy *= nh / h1
    else:
        nw, nh = (w1 // patch) * patch, (h1 // patch) * patch
        left, top = (w1 - nw) // 2, (h1 - nh) // 2
        out = out[top : top + nh, left : left + nw]
        crop_x, crop_y = left, top
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (out.astype(np.float32) / 255.0 - mean) / std
    return norm.transpose(2, 0, 1), (sx, sy, crop_x, crop_y)
