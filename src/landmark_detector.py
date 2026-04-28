import sys
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from .config import MODEL_PATH, LANDMARKS_68


class LandmarkDetector:
    def __init__(self, static_image: bool = False):
        import os
        if not os.path.isfile(MODEL_PATH):
            sys.exit(
                f"Model not found at {MODEL_PATH}\n"
                "Download it with:\n"
                "  curl -L -o face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/"
                "face_landmarker.task"
            )
        mode = RunningMode.IMAGE if static_image else RunningMode.VIDEO
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
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
