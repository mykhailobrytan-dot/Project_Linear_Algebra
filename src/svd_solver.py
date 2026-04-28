import numpy as np


def _power_svd(M: np.ndarray, max_iter: int = 1000, tol: float = 1e-14):
    rng = np.random.default_rng(0)
    m, n = M.shape
    r = min(m, n)

    MtM = M.T @ M
    A_rem = MtM.copy()

    us, sigmas, vs = [], [], []

    for k in range(r):
        v = rng.standard_normal(n)
        for vp in vs:
            v -= (v @ vp) * vp
        if np.linalg.norm(v) < tol:
            break
        v /= np.linalg.norm(v)

        for i in range(max_iter):
            v_new = A_rem @ v
            nrm = np.linalg.norm(v_new)
            if nrm < tol:
                break
            v_new /= nrm
            for vp in vs:
                v_new -= (v_new @ vp) * vp
            nrm2 = np.linalg.norm(v_new)
            if nrm2 < tol:
                break
            v_new /= nrm2
            converged = min(np.linalg.norm(v_new - v), np.linalg.norm(v_new + v)) < tol
            v = v_new
            if converged:
                break

        lam = float(v @ MtM @ v)
        if lam < tol:
            break

        sigma = np.sqrt(lam)
        u = M @ v / sigma

        sigmas.append(sigma)
        vs.append(v.copy())
        us.append(u)

        A_rem -= lam * np.outer(v, v)

    if not sigmas:
        return np.zeros((m, r)), np.zeros(r), np.zeros((r, n))

    return np.column_stack(us), np.array(sigmas), np.vstack(vs)


class SVDExpressionTransfer:
    def __init__(self):
        self.A: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.x_source: np.ndarray | None = None
        self.y_neutral: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.condition_number: float = 0.0

    def calibrate(self, x_source: np.ndarray, y_neutral: np.ndarray) -> dict:
        self.x_source = x_source.copy()
        self.y_neutral = y_neutral.copy()
        k = x_source.shape[0]

        M = np.zeros((2 * k, 6), dtype=np.float64)
        b = np.zeros(2 * k, dtype=np.float64)

        for j in range(k):
            yx, yy = y_neutral[j]
            M[2 * j,     0:3] = [yx, yy, 1.0]
            M[2 * j + 1, 3:6] = [yx, yy, 1.0]
            b[2 * j]     = x_source[j, 0]
            b[2 * j + 1] = x_source[j, 1]

        U, sigma, Vt = _power_svd(M)
        self.singular_values = sigma.copy()

        eps = 1e-12
        self.condition_number = float(sigma[0] / sigma[-1] if sigma[-1] > eps else np.inf)

        sigma_inv = np.where(sigma > eps, 1.0 / sigma, 0.0)
        a_hat = (Vt.T * sigma_inv) @ (U.T @ b)

        self.A = np.array([[a_hat[0], a_hat[1]],
                           [a_hat[3], a_hat[4]]])
        self.t = np.array([a_hat[2], a_hat[5]])

        return self.diagnostics()

    def transfer(self, y_current: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("Call calibrate() before transfer().")
        delta = y_current - self.y_neutral
        return self.x_source + (self.A @ delta.T).T

    def diagnostics(self) -> dict:
        return {
            "singular_values":  self.singular_values,
            "condition_number": self.condition_number,
            "effective_rank":   int(np.sum(self.singular_values > 1e-10)),
            "A": self.A,
            "t": self.t,
        }

    @property
    def is_calibrated(self) -> bool:
        return self.A is not None
