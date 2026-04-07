# Real-Time Face Reenactment via SVD-Based Landmark Mapping

Transfer facial expressions from a webcam feed to a static portrait in real time
using least squares solved via the singular value decomposition (SVD).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Interactive reenactment

```bash
python reenact.py --source portrait.jpg
```

1. A window opens showing the webcam (left) and the source portrait (right).
2. Keep a **neutral face** and press **`c`** to calibrate.
3. Press **`r`** to toggle reenactment — the portrait follows your expressions.
4. Press **`d`** to print SVD diagnostics (singular values, condition number, rank).
5. Press **`q`** to quit.

### Synthetic self-test (no webcam needed)

```bash
python reenact.py --test
```

Generates synthetic landmarks, applies a known affine transform, recovers
parameters via SVD, and prints reconstruction error.

## Project structure

| File | Description |
|------|-------------|
| `reenact.py` | Full pipeline: landmark detection, SVD solver, Delaunay warping |
| `requirements.txt` | Python dependencies |
| `interim_report_2.tex` | Second interim report (LaTeX source) |

## How it works

1. **Landmark detection** — MediaPipe Face Mesh extracts 68 key facial points.
2. **Calibration** — An overdetermined affine system `M a ≈ b` (`M ∈ R^{136×6}`)
   maps target-neutral landmarks to source landmarks. The SVD of `M` yields the
   minimum-norm least-squares solution `â = V Σ⁺ Uᵀ b`.
3. **Expression transfer** — For each new frame the expression delta
   `δ = y_t − y_0` is mapped through the learned matrix `A` and added to the
   source landmarks.
4. **Warping** — Delaunay triangulation + piecewise-affine transforms move the
   source portrait pixels to match the predicted landmarks.


## Team's Video Presentations

 - Viktor Syrotiuk - [Watch the video](https://youtu.be/DZFlOIkjiE4)
 - Mykhailo Brytan - [Watch the video](https://www.youtube.com/watch?v=7KGt8O7A21s)
 - Yulian Volodymyr Zaiats - [Watch the video](https://www.youtube.com/watch?v=3lpWnrP8UYY&t=1s)

## Authors

Viktor Syrotiuk, Mykhailo Brytan, Yulian Volodymyr Zaiats
