

import argparse
import sys
import numpy as np


def run_synthetic_test() -> bool:
    from src.svd_solver import SVDExpressionTransfer

    np.random.seed(42)
    k = 68

    A_true = np.array([[0.95, -0.10], [0.08, 1.05]])
    t_true = np.array([12.0, -8.0])

    y_neutral = np.random.randn(k, 2) * 50 + np.array([200.0, 200.0])
    x_source  = (A_true @ y_neutral.T).T + t_true

    solver = SVDExpressionTransfer()
    diag   = solver.calibrate(x_source, y_neutral)

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

    delta  = np.random.randn(k, 2) * 5
    y_expr = y_neutral + delta
    x_pred = solver.transfer(y_expr)
    x_true = x_source + (A_true @ delta.T).T
    terr   = np.mean(np.linalg.norm(x_pred - x_true, axis=1))
    print(f"  Mean transfer error: {terr:.2e}")

    ok = err_A < 1e-8 and err_t < 1e-8 and terr < 1e-8
    print(f"\n  Result: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Real-time face reenactment via SVD-based landmark mapping.",
    )
    ap.add_argument("--source", type=str, help="Path to the source portrait image.")
    ap.add_argument("--test", action="store_true", help="Run synthetic SVD self-test.")
    args = ap.parse_args()

    if args.test:
        ok = run_synthetic_test()
        sys.exit(0 if ok else 1)

    if not args.source:
        ap.error("--source <image> is required (or use --test).")

    from src.pipeline import ReenactmentPipeline
    pipeline = ReenactmentPipeline(args.source)
    pipeline.run()


if __name__ == "__main__":
    main()
