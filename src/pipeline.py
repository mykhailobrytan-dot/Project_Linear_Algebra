import sys
import cv2
import numpy as np
from .landmark_detector import LandmarkDetector
from .svd_solver import SVDExpressionTransfer
from .face_warp import build_triangulation, warp_face


class ReenactmentPipeline:
    _WORK_HEIGHT = 720

    def __init__(self, source_path: str):
        raw = cv2.imread(source_path)
        if raw is None:
            sys.exit(f"Error: cannot load '{source_path}'")
        h, w = raw.shape[:2]
        if h > self._WORK_HEIGHT:
            s = self._WORK_HEIGHT / h
            raw = cv2.resize(raw, (int(w * s), self._WORK_HEIGHT), interpolation=cv2.INTER_AREA)
        self.source_img = raw

        det = LandmarkDetector(static_image=True)
        self.src_lm = det.detect(self.source_img)
        det.close()
        if self.src_lm is None:
            sys.exit("Error: no face detected in the source image.")

        self.detector = LandmarkDetector(static_image=False)
        self.solver = SVDExpressionTransfer()
        self._triangles: np.ndarray | None = None

    def _calibrate(self, target_lm: np.ndarray) -> dict:
        diag = self.solver.calibrate(self.src_lm, target_lm)
        self._triangles = build_triangulation(self.src_lm, self.source_img.shape)
        return diag

    def _reenact_frame(self, target_lm: np.ndarray) -> np.ndarray:
        predicted = self.solver.transfer(target_lm)
        h_s, w_s = self.source_img.shape[:2]
        predicted[:, 0] = np.clip(predicted[:, 0], 1, w_s - 2)
        predicted[:, 1] = np.clip(predicted[:, 1], 1, h_s - 2)
        canvas = self.source_img.copy()
        try:
            warp_face(self.source_img, self.src_lm, predicted, self._triangles, canvas)
        except Exception as e:
            print(f"[warp error] {e}")
        return canvas

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Error: cannot open webcam.")

        reenacting = False

        print("──── Face Reenactment ────")
        print("  [c]  Calibrate (show neutral face)")
        print("  [d]  Print SVD diagnostics")
        print("  [r]  Toggle reenactment")
        print("  [q]  Quit")
        print("──────────────────────────")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            target_lm = self.detector.detect(frame)

            if (target_lm is not None
                    and self.solver.is_calibrated
                    and reenacting
                    and self._triangles is not None):
                portrait = self._reenact_frame(target_lm)
            else:
                portrait = self.source_img.copy()

            display_cam = frame.copy()
            if target_lm is not None:
                for px, py in target_lm.astype(int):
                    cv2.circle(display_cam, (px, py), 2, (0, 255, 0), -1)

            h_f = frame.shape[0]
            scale = h_f / portrait.shape[0]
            portrait_resized = cv2.resize(
                portrait, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            combined = np.hstack([display_cam, portrait_resized])

            if self.solver.is_calibrated and reenacting:
                label = "REENACTING"
            elif self.solver.is_calibrated:
                label = "CALIBRATED — press [r]"
            else:
                label = "Press [c] with neutral face"
            cv2.putText(combined, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.imshow("Face Reenactment (SVD)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and target_lm is not None:
                diag = self._calibrate(target_lm)
                print("\nCalibration complete:")
                print(f"  sigma = {diag['singular_values']}")
                print(f"  kappa = {diag['condition_number']:.2f}")
                print(f"  rank  = {diag['effective_rank']}")
            elif key == ord("d") and self.solver.is_calibrated:
                diag = self.solver.diagnostics()
                print("\nSVD diagnostics:")
                print(f"  sigma = {diag['singular_values']}")
                print(f"  kappa = {diag['condition_number']:.4f}")
                print(f"  rank  = {diag['effective_rank']}")
                print(f"  A =\n{diag['A']}")
                print(f"  t = {diag['t']}")
            elif key == ord("r") and self.solver.is_calibrated:
                reenacting = not reenacting

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
