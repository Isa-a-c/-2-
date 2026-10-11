"""불연속점을 찾아 구간을 나누는 Gauss–Legendre 구적입니다.

사다리꼴 적분은 도약(jump)이 있는 경계값에서 O(h) 오차를 남깁니다. 여기서는 표본
차분으로 도약 후보를 고르고 이분법으로 위치를 좁힌 뒤, 도약점에서 구간을 나누어
각 조각에 Gauss–Legendre 구적을 적용합니다. 조각 안에서 매끄러운 함수는 빠르게
수렴하며 도약점 자체의 함수값은 적분에 사용하지 않습니다.
"""
import numpy as np


def find_jumps(f, a, b, samples=4097, max_jumps=64, iterations=60):
    """[a, b]에서 f의 도약 위치를 반환합니다. 가파른 연속 구간은 제외합니다."""
    x = np.linspace(a, b, samples)
    y = np.asarray(f(x), dtype=float)
    differences = np.abs(np.diff(y))
    scale = max(float(np.max(np.abs(y))), 1e-300)
    typical = float(np.median(differences))
    candidates = np.flatnonzero(differences > max(20 * typical, 1e-9 * scale))
    if len(candidates) == 0:
        return np.array([])
    candidates = candidates[np.argsort(differences[candidates])[::-1][:max_jumps]]
    left, right = x[candidates].copy(), x[candidates + 1].copy()
    f_left, f_right = y[candidates].copy(), y[candidates + 1].copy()
    for _ in range(iterations):
        middle = (left + right) / 2
        f_middle = np.asarray(f(middle), dtype=float)
        # 변화가 더 큰 쪽 반구간을 남깁니다.
        go_left = np.abs(f_middle - f_left) >= np.abs(f_right - f_middle)
        right = np.where(go_left, middle, right)
        f_right = np.where(go_left, f_middle, f_right)
        left = np.where(go_left, left, middle)
        f_left = np.where(go_left, f_left, f_middle)
    # 연속 함수라면 폭이 0에 가까워질 때 값의 차이도 사라집니다.
    jumps = (left + right)[np.abs(f_right - f_left) > 1e-7 * scale] / 2
    return np.unique(np.round(jumps, 13))


def piecewise_gauss(f, a, b, total=512, samples=4097):
    """도약점에서 나눈 Gauss–Legendre 노드와 가중치를 반환합니다."""
    cuts = np.r_[a, find_jumps(f, a, b, samples), b]
    cuts = cuts[(cuts >= a) & (cuts <= b)]
    cuts = np.unique(cuts)
    pieces = len(cuts) - 1
    order = max(16, total // pieces)
    base, base_weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = [], []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        half = (upper - lower) / 2
        nodes.append(lower + half * (base + 1))
        weights.append(half * base_weights)
    return np.concatenate(nodes), np.concatenate(weights), cuts[1:-1]


def gauss_legendre(a, b, order):
    base, base_weights = np.polynomial.legendre.leggauss(order)
    half = (b - a) / 2
    return a + half * (base + 1), half * base_weights
