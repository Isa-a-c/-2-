"""UI 비의존 Fourier 수치 계산. 기존 v0.5.3 알고리즘을 보존합니다."""
import numpy as np
from .expr import parse_expression, evaluate, compiled, symbols_for
X = symbols_for(('x',))[0]
numpy_function = compiled

def sample_function(expression, x):
    return evaluate(expression,(x,))

def calculate_fourier_coefficients(x, y, L, max_N):
    """[-L,L] 표본에 사다리꼴 적분을 적용한다. an[0]은 a_1이다."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L은 유한한 양수여야 합니다.")
    if not isinstance(max_N, (int, np.integer)) or max_N < 1:
        raise ValueError("max_N은 양의 정수여야 합니다.")
    if x.ndim != 1 or x.size < 2 or x.shape != y.shape:
        raise ValueError("x와 y는 길이가 같은 1차원 표본이어야 합니다.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("표본은 모두 유한해야 합니다.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x는 오름차순이어야 합니다.")
    if not np.isclose(x[0], -L) or not np.isclose(x[-1], L):
        raise ValueError("표본 구간은 [-L, L]이어야 합니다.")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a0 = float(trapezoid_weights(x) @ y / (2 * L))
        cosine_coefficients, sine_coefficients = integrate_harmonics(x,y,L,1,max_N)
    if not np.all(np.isfinite(np.r_[a0, cosine_coefficients, sine_coefficients])):
        raise ValueError("계수가 유한하지 않습니다. 함수의 크기를 줄여 주세요.")
    return a0, cosine_coefficients, sine_coefficients

def trapezoid_weights(x):
    """증가하는 격자의 사다리꼴 가중치 w입니다. ∫f dx ≈ w·f (불균일 격자 허용)."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError('계산 결과가 유한하지 않습니다. 입력 크기와 구간을 확인하십시오.')
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError('적분 격자는 증가하는 1차원 배열이어야 합니다.')
    weights = np.empty_like(x)
    weights[0] = (x[1] - x[0]) / 2
    weights[-1] = (x[-1] - x[-2]) / 2
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    return weights


def harmonic_projections(x, weighted, scale, orders, waves=('cos', 'sin'), block=4_000_000):
    """scale·Σ w·f·cos/sin(nπx/scale…)의 공통 계산입니다. 큰 행렬은 차수 블록으로 나눕니다.

    weighted는 w·f, orders는 정수 차수 배열, 각도는 n·π·x/scale입니다.
    """
    orders = np.asarray(orders)
    base = np.pi * np.asarray(x, dtype=float) / scale
    results = {wave: np.empty(len(orders)) for wave in waves}
    step = max(1, block // max(len(base), 1))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for start in range(0, len(orders), step):
            angle = np.outer(orders[start:start+step], base)
            for wave in waves:
                function = np.cos if wave == 'cos' else np.sin
                results[wave][start:start+step] = function(angle) @ weighted
    return results


def integrate_harmonics(x, y, L, first, last):
    """검증된 [-L,L] 표본에서 aₙ, bₙ (n=first…last)만 적분하는 내부 함수 (캐시 증분 계산용)."""
    weighted = trapezoid_weights(x) * np.asarray(y, dtype=float)
    results = harmonic_projections(x, weighted, L, np.arange(first, last+1))
    an, bn = results['cos'] / L, results['sin'] / L
    if not np.all(np.isfinite(an)) or not np.all(np.isfinite(bn)):
        raise ValueError('계수가 유한하지 않습니다. 함수의 크기를 줄이십시오.')
    return an, bn


def half_range_coefficients(x, values, length, count, cosine):
    """[0, L] 표본의 반구간 전개 계수입니다(11.2절 식 6*·6**).

    사인: bₙ=(2/L)∫f sin(nπx/L)dx, 코사인: A₀=(1/L)∫f dx, Aₙ=(2/L)∫f cos(nπx/L)dx.
    기확장·우확장한 [−L, L]의 Euler 공식과 같은 값을 배열을 늘리지 않고 계산합니다.
    반환: (A₀ 또는 0, 계수 배열).
    """
    weighted = trapezoid_weights(x) * np.asarray(values, dtype=float)
    wave = 'cos' if cosine else 'sin'
    coefficients = 2 / length * harmonic_projections(x, weighted, length, np.arange(1, count+1), (wave,))[wave]
    constant = float(weighted.sum() / length) if cosine else 0.
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(constant):
        raise ValueError('계수가 유한하지 않습니다. 함수의 크기를 줄이십시오.')
    return constant, coefficients


def calculate_fourier_sum(x, L, a0, an, bn, N):
    """a0 + Σ(an cos(nπx/L) + bn sin(nπx/L))를 계산한다."""
    return calculate_fourier_sums(x, L, a0, an, bn, [N])[N]

def calculate_fourier_sums(x, L, a0, an, bn, orders):
    """한 번 누적하여 여러 차수의 부분합을 만든다. 필요한 차수만 복사한다."""
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L은 유한한 양수여야 합니다.")
    orders = set(orders)
    if any(not isinstance(n, (int, np.integer)) or not 0 <= n <= min(len(an), len(bn)) for n in orders):
        raise ValueError("N에 필요한 Fourier 계수가 부족합니다.")
    if not orders:
        return {}
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, a0, dtype=float)
    snapshots = {0: result.copy()} if 0 in orders else {}
    with np.errstate(over="raise", invalid="raise"):
        base_angle = np.pi * (x / L)
        for n in range(1, max(orders) + 1):
            angle = n * base_angle
            result += an[n - 1] * np.cos(angle) + bn[n - 1] * np.sin(angle)
            if n in orders:
                snapshots[n] = result.copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("부분합에서 비유한 값이 발생했습니다.")
    return snapshots

class FourierSession:
    """마지막 함수/구간/표본의 계수를 보관하는 작은 데스크톱 전용 캐시."""

    def __init__(self):
        self.key = None
        self.data = None

    def prepare(self, expression_text, L, num_points, max_N):
        key = (expression_text.strip(), L, num_points)
        if key == self.key and len(self.data[4]) >= max_N:
            expression, x, y, a0, an, bn = self.data
            return expression, x, y, a0, an[:max_N], bn[:max_N]
        expression = parse_expression(key[0])
        if key == self.key:
            x, y = self.data[1:3]
            a0, previous_an, previous_bn = self.data[3:]
            extra_an, extra_bn = integrate_harmonics(x,y,L,len(previous_an)+1,max_N)
            an = np.concatenate((previous_an,extra_an))
            bn = np.concatenate((previous_bn,extra_bn))
        else:
            x = np.linspace(-L, L, num_points)
            y = sample_function(expression, x)
            a0, an, bn = calculate_fourier_coefficients(x, y, L, max_N)
        # 계산에 성공한 결과만 교체한다. 잘못된 입력은 기존 캐시를 오염시키지 않는다.
        self.key = key
        self.data = (expression, x, y, a0, an, bn)
        return self.data

def calculate_errors(y, approximation):
    """같은 표본에서의 오차: 연속 구간 전체의 최대 오차와는 구별된다."""
    with np.errstate(over="raise", invalid="raise"):
        error = y - approximation
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        maximum_error = float(np.max(np.abs(error)))
    return error, {"MSE": mse, "RMSE": rmse, "Maximum Absolute Error": maximum_error}
