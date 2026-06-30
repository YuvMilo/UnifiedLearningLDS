"""Reproduces Table 1: a multi-seed comparison of online predictors.

We build an explicit linear dynamical system (A, B, C) with small instability
complexity, roll a bounded controlling input through it, and train four feature
maps (FIR, AR, SF, UF) with the same parameter budget using one shared VAW
online learner. We average over several seeds and report the normalized MSE.

Run:

    python main.py
"""

from __future__ import annotations

import numpy as np
from scipy import fft
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, eigsh


# ---------------------------------------------------------------------------
# Configuration (all predictors share the same parameter budget).
# ---------------------------------------------------------------------------
SEEDS = (0, 1, 2, 3, 4)   # averaged over these seeds
T = 200_000               # sequence length
STABLE_D = 300            # number of stable real modes
BUDGET = 16               # learned scalar parameters per predictor

# Unified predictor split: k output lags + a short input window + spectral
# filters, adding up to BUDGET.
UF_AR_ORDER = 3           # output lags (= instability complexity k)
UF_FIR_LEN = 7            # input lags (finite-memory window)
UF_H = 6                  # spectral filters
assert UF_AR_ORDER + UF_FIR_LEN + UF_H == BUDGET

# Stable-block design: its AR residual matches a late-weighted combination of
# the first STABLE_TARGET_RANK spectral filters.
STABLE_TARGET_RANK = 6
LATE_WEIGHT = 50.0
STABLE_SCALE = 25.0
MAX_ABS = 0.99999

# Hard modes (counted by the instability complexity k = 3).
GAMMA = 1.3               # one unstable real mode
COMPLEX_RADIUS = 0.98     # one slow complex pair
COMPLEX_ANGLE = 1.57
COMPLEX_SCALE = 1.0
UNSTABLE_SCALE = 1.0

# Fast-decaying complex modes (do not raise the instability complexity).
FAST_COUNT = 100
FAST_MIN_RADIUS = 0.0
FAST_MAX_RADIUS = 0.25
FAST_SCALE = 0.1

# VAW learner.
VAW_LAMBDA = 0.1
EVAL_LAST = 10_000
VAW_BURN_IN = T - EVAL_LAST

ORDER = ("FIR", "AR", "SF", "UF")


# ---------------------------------------------------------------------------
# Spectral filters: eigenvectors of the moment matrix Z with
#   Z[i, j] = integral_{-1}^{1} alpha^{i+j} d alpha.
# Z is applied through its Hankel structure, so the T x T matrix is never
# formed; the filters have full length T.
# ---------------------------------------------------------------------------
def moment_sequence(length: int) -> np.ndarray:
    """m_k = integral_{-1}^1 alpha^k d alpha for k = 0, ..., length-1."""
    k = np.arange(length, dtype=np.float64)
    m = np.zeros(length, dtype=np.float64)
    even = (k % 2) == 0
    m[even] = 2.0 / (k[even] + 1.0)
    return m


def spectral_filters(length: int, num_filters: int) -> np.ndarray:
    """Return the top ``num_filters`` spectral filters as rows, shape (h, T)."""
    moments = moment_sequence(2 * length - 1)
    conv_len = len(moments) + length - 1
    fft_len = fft.next_fast_len(conv_len)
    moments_fft = fft.rfft(moments, n=fft_len)

    def apply_Z(vector: np.ndarray) -> np.ndarray:
        # (Z v)_i = sum_j m_{i+j} v_j, evaluated as a convolution.
        rev_fft = fft.rfft(vector[::-1], n=fft_len)
        conv = fft.irfft(moments_fft * rev_fft, n=fft_len)
        return conv[length - 1 : 2 * length - 1].copy()

    operator = LinearOperator((length, length), matvec=apply_Z, rmatvec=apply_Z, dtype=np.float64)
    eigenvalues, eigenvectors = eigsh(operator, k=num_filters, which="LA", tol=1e-10)
    order = np.argsort(eigenvalues)[::-1]
    filters = eigenvectors[:, order].T  # (h, T)

    # Deterministic sign: largest-magnitude entry positive.
    for i in range(filters.shape[0]):
        pivot = int(np.argmax(np.abs(filters[i])))
        if filters[i, pivot] < 0.0:
            filters[i] *= -1.0
    return filters


# ---------------------------------------------------------------------------
# System construction.
# ---------------------------------------------------------------------------
def hard_ar_coefficients(gamma: float, radius: float, angle: float) -> np.ndarray:
    """AR(3) coefficients whose roots are gamma and radius * exp(+- i angle)."""
    pair_sum = 2.0 * radius * np.cos(angle)
    pair_prod = radius ** 2
    return np.array(
        [gamma + pair_sum, -(pair_prod + gamma * pair_sum), gamma * pair_prod],
        dtype=np.float64,
    )


def log_stable_nodes(d: int, max_abs: float) -> np.ndarray:
    """Stable real eigenvalues on a logarithmic grid clustered near +-1.

    The distance to the unit circle, 1 - |lambda|, is geometrically spaced
    between 1 - max_abs and 1; half the nodes are positive and half negative.
    """
    half = d // 2
    gaps = np.logspace(np.log10(1.0 - max_abs), 0.0, half)  # 1 - max_abs ... 1
    mags = 1.0 - gaps                                        # max_abs ... 0
    nodes = np.concatenate([mags, -mags])
    if nodes.size < d:  # odd d: add one extra node near +1
        nodes = np.concatenate([nodes, [max_abs]])
    return nodes


def fit_stable_readout(diagonal: np.ndarray, target_residual: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Fit a readout C so the AR residual of the stable impulse matches the target.

    The stable block has eigenvalues ``diagonal``, input B = 1, and readout C.
    Its impulse response is h_t = sum_i C_i * diagonal_i^t; for t >= k the AR
    residual equals sum_i (C_i * factor_i) * diagonal_i^{t-k}, where factor_i is
    the AR characteristic polynomial at diagonal_i. We solve a least-squares
    problem in the tail powers and divide by factor_i.
    """
    k = len(alphas)
    length = len(target_residual)
    tail_exponents = np.arange(length - k)
    tail_powers = diagonal[None, :] ** tail_exponents[:, None]  # (T-k, d)

    factors = diagonal ** k
    for lag, alpha in enumerate(alphas, start=1):
        factors -= alpha * diagonal ** (k - lag)

    target_tail = target_residual[k:]
    beta, *_ = np.linalg.lstsq(tail_powers, target_tail, rcond=None)
    return beta / factors


def complex_block(radius: float, angle: float) -> np.ndarray:
    """Real 2x2 block realizing eigenvalues radius * exp(+- i angle)."""
    c = radius * np.cos(angle)
    s = radius * np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def complex_pair_peak(length: int, radius: float, angle: float) -> float:
    """Max absolute impulse of the (1,0)-readout slow complex block."""
    block = complex_block(radius, angle)
    powers = np.eye(2, dtype=np.float64)
    peak = 0.0
    for _ in range(length):
        peak = max(peak, abs(float(powers[0, 0])))
        powers = block @ powers
    return peak


def block_diagonal(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble a dense block-diagonal matrix."""
    total = sum(b.shape[0] for b in blocks)
    A = np.zeros((total, total), dtype=np.float64)
    offset = 0
    for b in blocks:
        size = b.shape[0]
        A[offset : offset + size, offset : offset + size] = b
        offset += size
    return A


# ---------------------------------------------------------------------------
# Input and rollout. The unstable mode is kept bounded by exact feedback: with
# desired state z_t in {-1, +1} and x_t = gamma x_{t-1} + u_t, choosing
# u_t = z_t - gamma z_{t-1} forces x_t = z_t, so the trajectory stays bounded.
# ---------------------------------------------------------------------------
def controlling_input(length: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    u = np.zeros(length, dtype=np.float64)
    state = 0.0
    for t in range(length):
        desired = float(rng.choice(np.array([-1.0, 1.0])))
        u[t] = desired - gamma * state
        state = desired
    return u


def rollout(A: np.ndarray, B: np.ndarray, C: np.ndarray, u: np.ndarray) -> np.ndarray:
    """State-space rollout: x_t = A x_{t-1} + B u_t, y_t = C x_t."""
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    b = B[:, 0]
    c = C[0]
    y = np.zeros(len(u), dtype=np.float64)
    for t in range(len(u)):
        x = A @ x + b * u[t]
        y[t] = c @ x
    return y


# ---------------------------------------------------------------------------
# Feature maps. Every map returns a (T, BUDGET) matrix.
# ---------------------------------------------------------------------------
def input_lag_features(u: np.ndarray, length: int) -> np.ndarray:
    """Columns are u_t, u_{t-1}, ..., u_{t-length+1} (zero padded)."""
    T_local = len(u)
    F = np.zeros((T_local, length), dtype=np.float64)
    for lag in range(length):
        if lag == 0:
            F[:, 0] = u
        else:
            F[lag:, lag] = u[:-lag]
    return F


def output_lag_features(y: np.ndarray, order: int) -> np.ndarray:
    """Columns are y_{t-1}, ..., y_{t-order} (zero padded)."""
    T_local = len(y)
    F = np.zeros((T_local, order), dtype=np.float64)
    for lag in range(1, order + 1):
        F[lag:, lag - 1] = y[:-lag]
    return F


def spectral_features(u: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Causal convolution of u with each full-length filter (zero padded)."""
    T_local = len(u)
    h = filters.shape[0]
    F = np.zeros((T_local, h), dtype=np.float64)
    for s in range(h):
        F[:, s] = fftconvolve(u, filters[s], mode="full")[:T_local]
    return F


def feature_maps(u: np.ndarray, y: np.ndarray, filters: np.ndarray) -> dict[str, np.ndarray]:
    fir = input_lag_features(u, BUDGET)
    ar = np.concatenate(
        [output_lag_features(y, BUDGET // 2), input_lag_features(u, BUDGET // 2)], axis=1
    )
    sf = spectral_features(u, filters[:BUDGET])
    uf = np.concatenate(
        [
            output_lag_features(y, UF_AR_ORDER),
            input_lag_features(u, UF_FIR_LEN),
            spectral_features(u, filters[:UF_H]),
        ],
        axis=1,
    )
    return {"FIR": fir, "AR": ar, "SF": sf, "UF": uf}


# ---------------------------------------------------------------------------
# Vovk-Azoury-Warmuth online learner
# ---------------------------------------------------------------------------
def vaw_normalized_mse(
    features: np.ndarray, labels: np.ndarray, regularization: float, burn_in: int
) -> float:
    """VAW forecaster; returns final-window normalized MSE.

    At step t: A_t = A_{t-1} + a_t a_t^T, prediction = a_t^T A_t^{-1} v_{t-1},
    then observe b_t and update v_t = v_{t-1} + b_t a_t. A_t^{-1} is maintained
    with the Sherman-Morrison rank-one update. The zero predictor scores 1.
    """
    X = features
    n, d = X.shape
    A_inv = np.eye(d, dtype=np.float64) / regularization
    v = np.zeros(d, dtype=np.float64)
    predictions = np.zeros(n, dtype=np.float64)

    for t in range(n):
        a = X[t]
        A_inv_a = A_inv @ a
        leverage = float(a @ A_inv_a)
        A_t_inv = A_inv - np.outer(A_inv_a, A_inv_a) / (1.0 + leverage)
        predictions[t] = float(a @ (A_t_inv @ v))
        v = v + labels[t] * a
        A_inv = A_t_inv

    eval_labels = labels[burn_in:]
    residual = predictions[burn_in:] - eval_labels
    mse = float(np.mean(residual ** 2))
    denom = max(1e-12, float(np.mean(eval_labels ** 2)))
    return mse / denom


# ---------------------------------------------------------------------------
# Build the system for one seed.
# ---------------------------------------------------------------------------
def build_system(rng: np.random.Generator, filters: np.ndarray):
    alphas = hard_ar_coefficients(GAMMA, COMPLEX_RADIUS, COMPLEX_ANGLE)

    # Target for the stable AR residual: a late-weighted signed combination of
    # the first STABLE_TARGET_RANK spectral filters.
    signs = rng.choice(np.array([-1.0, 1.0]), size=STABLE_TARGET_RANK)
    weights = np.linspace(1.0, LATE_WEIGHT, STABLE_TARGET_RANK)
    target = (signs * weights) @ filters[:STABLE_TARGET_RANK]
    target = target / max(1e-12, float(np.max(np.abs(target))))
    target = STABLE_SCALE * target

    # Stable eigenvalues on a logarithmic grid, with a readout fit to the target.
    stable_diagonal = log_stable_nodes(STABLE_D, MAX_ABS)
    stable_readout = fit_stable_readout(stable_diagonal, target, alphas)

    # Slow complex pair, scaled to a fixed peak impulse.
    peak = complex_pair_peak(T, COMPLEX_RADIUS, COMPLEX_ANGLE)
    c_pair = np.array([[COMPLEX_SCALE / max(1e-12, peak), 0.0]], dtype=np.float64)

    # Fast-decaying complex modes.
    fast_blocks = []
    fast_C = []
    for _ in range(FAST_COUNT):
        radius = rng.uniform(FAST_MIN_RADIUS, FAST_MAX_RADIUS)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        fast_blocks.append(complex_block(radius, angle))
        fast_C.append(FAST_SCALE * rng.normal(size=(1, 2)))

    # Assemble A, B, C.
    blocks = [np.diag(stable_diagonal), complex_block(COMPLEX_RADIUS, COMPLEX_ANGLE)]
    blocks.append(np.array([[GAMMA]], dtype=np.float64))
    blocks.extend(fast_blocks)
    A = block_diagonal(blocks)

    B_parts = [np.ones((STABLE_D, 1)), np.array([[1.0], [0.0]]), np.array([[1.0]])]
    B_parts += [np.array([[1.0], [0.0]]) for _ in range(FAST_COUNT)]
    B = np.vstack(B_parts)

    C_parts = [stable_readout[None, :], c_pair, np.array([[UNSTABLE_SCALE]])]
    C_parts += fast_C
    C = np.hstack(C_parts)
    return A, B, C


def evaluate_seed(seed: int, filters: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A, B, C = build_system(rng, filters)
    u = controlling_input(T, GAMMA, rng)
    y = rollout(A, B, C, u)
    maps = feature_maps(u, y, filters)
    out = {}
    for name in ORDER:
        features = maps[name]
        assert features.shape[1] == BUDGET, (name, features.shape)
        out[name] = vaw_normalized_mse(features, y, VAW_LAMBDA, VAW_BURN_IN)
    return out


def main() -> None:
    # Spectral filters do not depend on the seed; compute them once.
    filters = spectral_filters(T, BUDGET)

    per_seed = {name: [] for name in ORDER}
    for seed in SEEDS:
        result = evaluate_seed(seed, filters)
        for name in ORDER:
            per_seed[name].append(result[name])

    print(
        f"VAW results over {len(SEEDS)} seeds "
        f"(T={T}, final_window={EVAL_LAST}, lambda={VAW_LAMBDA}, "
        f"fast radius in [{FAST_MIN_RADIUS}, {FAST_MAX_RADIUS}])"
    )
    print(f"{'Predictor':<10}{'Parameters':>12}{'Normalized MSE (min--max)':>36}")
    for name in ORDER:
        values = np.array(per_seed[name], dtype=np.float64)
        min_value = float(values.min())
        max_value = float(values.max())
        print(f"{name:<10}{BUDGET:>12}{min_value:>18.3e} -- {max_value:.3e}")


if __name__ == "__main__":
    main()
