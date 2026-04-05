#!/usr/bin/env python3
# """
# Real-Time Face Reenactment via SVD-Based Landmark Mapping

# Mathematical pipeline
# ─────────────────────
# 1.  Detect k facial landmarks on a source portrait  →  x_s ∈ R^{2k}
# 2.  Calibration: detect target neutral landmarks    →  y_0 ∈ R^{2k}
#     Build overdetermined system  M a ≈ b   (M ∈ R^{2k×6}, a ∈ R^6)
#     Solve via SVD:  â = V Σ⁺ Uᵀ b
# 3.  Per frame: detect target landmarks y_t,
#     transfer expression:  x_new = x_s + A·(y_t − y_0),
#     warp source image from x_s → x_new, display result.

# Usage
# ─────
#     python reenact.py --source portrait.jpg
#     python reenact.py --test                    # synthetic SVD verification
# """

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ─── MediaPipe → 68-point landmark mapping ────────────────────────────────────
# Indices into MediaPipe's 468 face-mesh that approximate the iBUG 68-point set.

LANDMARKS_68 = [
    # jaw (0-16)
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400,
    # right eyebrow (17-21)
    70, 63, 105, 66, 107,
    # left eyebrow (22-26)
    336, 296, 334, 293, 300,
    # nose bridge (27-30)
    168, 6, 197, 195,
    # nose tip (31-35)
    5, 4, 1, 2, 98,
    # right eye (36-41)
    33, 160, 158, 133, 153, 144,
    # left eye (42-47)
    362, 385, 387, 263, 373, 380,
    # outer lip (48-59)
    61, 39, 37, 0, 267, 269, 291,
    321, 314, 17, 84, 91,
    # inner lip (60-67)
    78, 82, 13, 312, 308,
    317, 14, 87,
]

K = len(LANDMARKS_68)  # 68


# ═══════════════════════════════════════════════════════════════════════════════
#  Landmark detection
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkDetector:
    """Detect 68 facial landmarks using the MediaPipe Tasks FaceLandmarker."""

    def __init__(self, static_image: bool = False):
        if not os.path.isfile(_MODEL_PATH):
            sys.exit(
                f"Model not found at {_MODEL_PATH}\n"
                "Download it:\n"
                "  curl -L -o face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/"
                "face_landmarker.task"
            )
        mode = RunningMode.IMAGE if static_image else RunningMode.VIDEO
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=_MODEL_PATH,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._static = static_image
        self._frame_ts = 0

    def detect(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """Return (k, 2) float64 array of landmark positions, or None."""
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            self._frame_ts += 33
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        pts = np.array(
            [(face[i].x * w, face[i].y * h) for i in LANDMARKS_68],
            dtype=np.float64,
        )
        return pts

    def close(self):
        self._landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  SVD-based expression transfer  (core linear algebra)
# ═══════════════════════════════════════════════════════════════════════════════

class SVDExpressionTransfer:
    """
    Estimate an affine transformation  A·y + t ≈ x  from target to source
    landmark space using least squares solved via the SVD of the design matrix.

    System layout (per landmark j = 1…k):

        [ y_jx  y_jy  1   0     0    0 ] [ a11 ]     [ x_jx ]
        [  0     0    0  y_jx  y_jy  1 ] [ a12 ]  ≈  [ x_jy ]
                                          [ t_x ]
                                          [ a21 ]
                                          [ a22 ]
                                          [ t_y ]

    Stacked:  M (2k × 6)  ·  a (6 × 1)  ≈  b (2k × 1)
    Solution: â = M⁺ b = V Σ⁺ Uᵀ b
    """

    def __init__(self):
        self.A: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.x_source: np.ndarray | None = None
        self.y_neutral: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.condition_number: float = 0.0

    # ── calibration ──────────────────────────────────────────────────────────

    def calibrate(self, x_source: np.ndarray, y_neutral: np.ndarray) -> dict:
        """
        Solve for affine parameters (A, t) mapping target neutral landmarks
        to source landmarks.

        Parameters
        ----------
        x_source  : (k, 2)  source portrait landmarks.
        y_neutral : (k, 2)  target (webcam) neutral-face landmarks.

        Returns
        -------
        dict with SVD diagnostics.
        """
        self.x_source = x_source.copy()
        self.y_neutral = y_neutral.copy()
        k = x_source.shape[0]

        M = np.zeros((2 * k, 6), dtype=np.float64)
        b = np.zeros(2 * k, dtype=np.float64)

        for j in range(k):
            yx, yy = y_neutral[j]
            M[2 * j, 0:3] = [yx, yy, 1.0]
            M[2 * j + 1, 3:6] = [yx, yy, 1.0]
            b[2 * j] = x_source[j, 0]
            b[2 * j + 1] = x_source[j, 1]

        # SVD of the design matrix
        U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
        self.singular_values = sigma.copy()
        eps = 1e-12
        self.condition_number = float(
            sigma[0] / sigma[-1] if sigma[-1] > eps else np.inf
        )

        # Pseudoinverse solution: â = V Σ⁺ Uᵀ b
        sigma_inv = np.where(sigma > eps, 1.0 / sigma, 0.0)
        a_hat = (Vt.T * sigma_inv) @ (U.T @ b)

        self.A = np.array([[a_hat[0], a_hat[1]],
                           [a_hat[3], a_hat[4]]])
        self.t = np.array([a_hat[2], a_hat[5]])

        return self.diagnostics()

    # ── per-frame transfer ───────────────────────────────────────────────────

    def transfer(self, y_current: np.ndarray) -> np.ndarray:
        """
        Transfer expression from target to source space.

            δ  = y_current − y_neutral          (expression delta in target space)
            x_new = x_source + A · δ            (mapped into source space)

        Equivalent to  x_new = A · y_current + t  (full affine applied to current).
        """
        delta = y_current - self.y_neutral
        mapped = (self.A @ delta.T).T
        return self.x_source + mapped

    # ── diagnostics ──────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "singular_values": self.singular_values,
            "condition_number": self.condition_number,
            "effective_rank": int(np.sum(self.singular_values > 1e-10)),
            "A": self.A,
            "t": self.t,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Piecewise-affine image warping (Delaunay triangulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _boundary_points(shape: tuple) -> np.ndarray:
    """Eight points on the image boundary to anchor the triangulation."""
    h, w = shape[:2]
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2],                      [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
    ], dtype=np.float64)


def _delaunay_indices(points: np.ndarray, shape: tuple) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangulation and return index triples."""
    h, w = shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))

    clamped = points.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)

    for pt in clamped:
        subdiv.insert((float(pt[0]), float(pt[1])))

    triangles = []
    for tri in subdiv.getTriangleList():
        idx = []
        for i in range(0, 6, 2):
            px, py = tri[i], tri[i + 1]
            if not (0 <= px < w and 0 <= py < h):
                break
            dists = np.sum((clamped - [px, py]) ** 2, axis=1)
            idx.append(int(np.argmin(dists)))
        else:
            if len(set(idx)) == 3:
                triangles.append(tuple(idx))
    return triangles


def _warp_triangle(
    src: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    dst: np.ndarray,
):
    """Warp one triangle region from src into dst."""
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    src_rect = src_tri - [r1[0], r1[1]]
    dst_rect = dst_tri - [r2[0], r2[1]]

    crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if crop.size == 0:
        return

    mat = cv2.getAffineTransform(np.float32(src_rect), np.float32(dst_rect))
    warped = cv2.warpAffine(
        crop, mat, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)

    roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if roi.shape[:2] != warped.shape[:2]:
        return
    mask3 = mask[:, :, np.newaxis] / 255.0
    roi[:] = (roi * (1 - mask3) + warped * mask3).astype(np.uint8)


def warp_face(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    dst_img: np.ndarray,
):
    """
    Piecewise-affine warp: map texture from src_img (at src_pts)
    into dst_img (at dst_pts) using Delaunay triangulation.
    """
    boundary = _boundary_points(src_img.shape)
    all_src = np.vstack([src_pts, boundary])
    all_dst = np.vstack([dst_pts, boundary])

    triangles = _delaunay_indices(all_src, src_img.shape)

    for i0, i1, i2 in triangles:
        s_tri = np.float32([all_src[i0], all_src[i1], all_src[i2]])
        d_tri = np.float32([all_dst[i0], all_dst[i1], all_dst[i2]])
        _warp_triangle(src_img, s_tri, d_tri, dst_img)


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic self-test  (verifies SVD solver on known affine transform)
# ═══════════════════════════════════════════════════════════════════════════════

def run_synthetic_test():
    """
    Generate synthetic landmarks, apply a known affine transform,
    solve via SVD, and check reconstruction error.
    """
    np.random.seed(42)
    k = 68

    # Ground-truth affine parameters
    A_true = np.array([[0.95, -0.10],
                       [0.08,  1.05]])
    t_true = np.array([12.0, -8.0])

    # Random "target neutral" landmarks
    y_neutral = np.random.randn(k, 2) * 50 + np.array([200, 200])

    # "Source" landmarks generated by the known affine transform
    x_source = (A_true @ y_neutral.T).T + t_true

    solver = SVDExpressionTransfer()
    diag = solver.calibrate(x_source, y_neutral)

    print("═══ Synthetic SVD self-test ═══")
    print(f"  Singular values : {diag['singular_values']}")
    print(f"  Condition number: {diag['condition_number']:.4f}")
    print(f"  Effective rank  : {diag['effective_rank']}")
    print(f"  True  A:\n{A_true}")
    print(f"  Found A:\n{diag['A']}")
    print(f"  True  t: {t_true}")
    print(f"  Found t: {diag['t']}")

    err_A = np.linalg.norm(diag["A"] - A_true)
    err_t = np.linalg.norm(diag["t"] - t_true)
    print(f"  ‖A_err‖ = {err_A:.2e}")
    print(f"  ‖t_err‖ = {err_t:.2e}")

    # Verify expression transfer on a perturbed frame
    delta = np.random.randn(k, 2) * 5
    y_expr = y_neutral + delta
    x_pred = solver.transfer(y_expr)
    x_true = x_source + (A_true @ delta.T).T
    transfer_err = np.mean(np.linalg.norm(x_pred - x_true, axis=1))
    print(f"  Mean transfer error: {transfer_err:.2e}")

    ok = err_A < 1e-8 and err_t < 1e-8 and transfer_err < 1e-8
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class ReenactmentPipeline:
    """
    Complete face-reenactment loop.

    Stages:
      1. Load source portrait → detect landmarks x_s.
      2. Calibration: press [c] to capture neutral face → solve Ma≈b via SVD.
      3. Real-time: press [r] to toggle reenactment display.
    """

    def __init__(self, source_path: str):
        self.source_img = cv2.imread(source_path)
        if self.source_img is None:
            sys.exit(f"Error: cannot load '{source_path}'")

        det = LandmarkDetector(static_image=True)
        self.src_lm = det.detect(self.source_img)
        det.close()
        if self.src_lm is None:
            sys.exit("Error: no face detected in the source image.")

        self.detector = LandmarkDetector(static_image=False)
        self.solver = SVDExpressionTransfer()
        self.calibrated = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Error: cannot open webcam.")

        reenacting = False

        print("──── Face Reenactment ────")
        print("  [c]  Calibrate (neutral face)")
        print("  [d]  Print SVD diagnostics")
        print("  [r]  Toggle reenactment")
        print("  [q]  Quit")
        print("──────────────────────────")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h_f = frame.shape[0]

            target_lm = self.detector.detect(frame)

            # ── reenactment output ────────────────────────────────────────
            portrait = self.source_img.copy()
            if target_lm is not None and self.calibrated and reenacting:
                predicted = self.solver.transfer(target_lm)

                h_s, w_s = self.source_img.shape[:2]
                predicted[:, 0] = np.clip(predicted[:, 0], 1, w_s - 2)
                predicted[:, 1] = np.clip(predicted[:, 1], 1, h_s - 2)

                canvas = self.source_img.copy()
                try:
                    warp_face(self.source_img, self.src_lm, predicted, canvas)
                except (cv2.error, ValueError, IndexError):
                    pass
                portrait = canvas

            # ── display ───────────────────────────────────────────────────
            display_cam = frame.copy()
            if target_lm is not None:
                for px, py in target_lm.astype(int):
                    cv2.circle(display_cam, (px, py), 2, (0, 255, 0), -1)

            # Resize portrait to match camera height
            scale = h_f / portrait.shape[0]
            portrait_resized = cv2.resize(
                portrait, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

            combined = np.hstack([display_cam, portrait_resized])

            label = (
                "REENACTING" if (self.calibrated and reenacting)
                else "CALIBRATED - press [r]" if self.calibrated
                else "Press [c] with neutral face"
            )
            cv2.putText(
                combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
            )

            cv2.imshow("Face Reenactment (SVD)", combined)

            # ── keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and target_lm is not None:
                diag = self.solver.calibrate(self.src_lm, target_lm)
                self.calibrated = True
                print("\nCalibration complete:")
                print(f"  sigma  = {diag['singular_values']}")
                print(f"  kappa  = {diag['condition_number']:.2f}")
                print(f"  rank   = {diag['effective_rank']}")
            elif key == ord("d") and self.calibrated:
                diag = self.solver.diagnostics()
                print("\nSVD diagnostics:")
                print(f"  sigma  = {diag['singular_values']}")
                print(f"  kappa  = {diag['condition_number']:.4f}")
                print(f"  rank   = {diag['effective_rank']}")
                print(f"  A =\n{diag['A']}")
                print(f"  t = {diag['t']}")
            elif key == ord("r") and self.calibrated:
                reenacting = not reenacting

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Real-time face reenactment via SVD-based landmark mapping.",
    )
    ap.add_argument("--source", type=str, help="Path to source portrait image.")
    ap.add_argument(
        "--test", action="store_true",
        help="Run synthetic self-test (no webcam needed).",
    )
    args = ap.parse_args()

    if args.test:
        ok = run_synthetic_test()
        sys.exit(0 if ok else 1)

    if not args.source:
        ap.error("--source is required (or use --test for the synthetic check)")

    pipeline = ReenactmentPipeline(args.source)
    pipeline.run()


if __name__ == "__main__":
    main()
