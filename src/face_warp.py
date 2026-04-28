import cv2
import numpy as np
from scipy.spatial import Delaunay


def boundary_points(shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    return np.array([
        [0,       0      ],
        [w // 2,  0      ],
        [w - 1,   0      ],
        [0,       h // 2 ],
        [w - 1,   h // 2 ],
        [0,       h - 1  ],
        [w // 2,  h - 1  ],
        [w - 1,   h - 1  ],
    ], dtype=np.float64)


def build_triangulation(src_pts: np.ndarray, shape: tuple) -> np.ndarray:
    boundary = boundary_points(shape)
    all_pts = np.vstack([src_pts, boundary])
    h, w = shape[:2]
    clamped = all_pts.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)
    tri = Delaunay(clamped)
    return tri.simplices


def warp_face(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    triangles: np.ndarray,
    dst_img: np.ndarray,
) -> None:
    boundary = boundary_points(src_img.shape)
    all_src = np.vstack([src_pts, boundary])
    all_dst = np.vstack([dst_pts, boundary])

    for i0, i1, i2 in triangles:
        s_tri = np.float32([all_src[i0], all_src[i1], all_src[i2]])
        d_tri = np.float32([all_dst[i0], all_dst[i1], all_dst[i2]])
        _warp_triangle(src_img, s_tri, d_tri, dst_img)


def _warp_triangle(
    src: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    dst: np.ndarray,
) -> None:
    h, w = dst.shape[:2]

    r1 = cv2.boundingRect(src_tri)
    r2 = cv2.boundingRect(dst_tri)

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    src_rect = np.float32(src_tri - [r1[0], r1[1]])
    dst_rect = np.float32(dst_tri - [r2[0], r2[1]])

    crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if crop.size == 0:
        return

    mat    = cv2.getAffineTransform(src_rect, dst_rect)
    warped = cv2.warpAffine(
        crop, mat, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)

    x0 = max(r2[0], 0);  y0 = max(r2[1], 0)
    x1 = min(r2[0] + r2[2], w);  y1 = min(r2[1] + r2[3], h)
    if x0 >= x1 or y0 >= y1:
        return

    ox, oy   = x0 - r2[0], y0 - r2[1]
    dh, dw_  = y1 - y0,    x1 - x0

    roi         = dst   [y0:y1, x0:x1]
    warped_crop = warped[oy:oy + dh, ox:ox + dw_]
    mask_crop   = mask  [oy:oy + dh, ox:ox + dw_]

    if roi.shape[:2] != warped_crop.shape[:2]:
        return

    alpha  = mask_crop[:, :, np.newaxis] / 255.0
    roi[:] = np.clip(roi * (1.0 - alpha) + warped_crop * alpha, 0, 255).astype(np.uint8)
