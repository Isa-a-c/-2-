"""교재 11장의 UI 독립 수치 계산입니다. 적분은 유한 표본의 근사입니다."""
import numpy as np
from scipy.special import eval_legendre, jn_zeros, jv
# 사다리꼴 가중치와 반구간 계수는 core.fourier의 공통 구현을 씁니다(chapter12도 여기서 가져옵니다).
from .fourier import trapezoid_weights, half_range_coefficients


# ── 공통 적분과 유한성 검사 ──────────────────────────────────────
def checked(values):
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError('계산 결과가 유한하지 않습니다. 입력 크기와 구간을 확인하십시오.')
    return array


def project_basis(x, values, basis, weight=None):
    """직교기저의 계수, 정규화 Gram 행렬과 누적 부분합을 반환합니다."""
    values, basis = checked(values), checked(basis)
    weights = trapezoid_weights(x)
    if weight is not None:
        weights = weights * checked(weight)
    if values.shape != np.shape(x) or basis.ndim != 2 or basis.shape[1] != len(x):
        raise ValueError('함수와 기저의 표본 크기가 일치하지 않습니다.')
    norms = checked(np.sum(basis * basis * weights, axis=1))
    if np.any(norms <= 0):
        raise ValueError('기저의 제곱노름은 양수여야 합니다.')
    coefficients = checked((basis @ (weights * values)) / norms)
    gram = checked((basis * weights) @ basis.T / np.sqrt(np.outer(norms, norms)))
    sums = checked(np.cumsum(coefficients[:, None] * basis, axis=0))
    return coefficients, gram, sums, norms


def half_range(x, values, length, order, cosine):
    """반구간 계수와 차수별 누적 부분합입니다."""
    constant, coefficients = half_range_coefficients(x, values, length, order, cosine)
    angles = np.arange(1, order + 1)[:, None] * np.pi * x / length
    waves = np.cos(angles) if cosine else np.sin(angles)
    return constant, checked(coefficients), checked(constant + np.cumsum(coefficients[:, None] * waves, axis=0))


def orthogonal_basis(kind, x, count, length=1., bessel_order=0):
    """Legendre: [-1,1], Bessel: [0,L], 고정 차수 ν의 양의 영점 사용."""
    if kind == 'Legendre':
        indices = np.arange(count)
        basis = np.array([eval_legendre(int(n), x) for n in indices])
        return basis, np.ones_like(x), indices * (indices + 1)
    roots = jn_zeros(bessel_order, count)
    basis = jv(bessel_order, roots[:, None] * x / length)
    return basis, x, (roots / length) ** 2


def sturm_basis(x, length, count, boundary, p=1., q=0., r=1., robin=(1., 0., 1., 0.)):
    """교재 부호 (p y')' + (q + λr)y=0, 상수 p,r>0의 고유함수와 고유값 λ=(pκ²−q)/r.

    boundary: DD·NN·DN, 'Periodic'(y(0)=y(L), y'(0)=y'(L), 식 7), 'Robin'(식 2:
    k₁y(0)+k₂y'(0)=0, l₁y(L)+l₂y'(L)=0, robin=(k₁,k₂,l₁,l₂)).
    """
    if boundary == 'DD':
        frequencies = np.arange(1, count + 1) * np.pi / length
        basis = np.sin(frequencies[:, None] * x)
    elif boundary == 'NN':
        frequencies = np.arange(count) * np.pi / length
        basis = np.cos(frequencies[:, None] * x)
    elif boundary == 'DN':
        frequencies = (np.arange(count) + .5) * np.pi / length
        basis = np.sin(frequencies[:, None] * x)
    elif boundary == 'Periodic':
        # 상수 모드 하나와 각 n≥1의 cos·sin 두 모드(중복 고유값)입니다.
        orders = (np.arange(count) + 1) // 2
        frequencies = 2 * np.pi * orders / length
        basis = np.where((np.arange(count) % 2 == 1)[:, None], np.cos(frequencies[:, None] * x),
                         np.sin(frequencies[:, None] * x))
        basis[0] = 1.
    elif boundary == 'Robin':
        squared, basis = robin_modes(x, length, count, *robin)
        return basis, (p * squared - q) / r
    else:
        raise ValueError('지원하는 경계조건은 DD, NN, DN, Periodic, Robin입니다.')
    return basis, (p * frequencies ** 2 - q) / r


def robin_modes(x, length, count, k1, k2, l1, l2):
    """y''+κ²y=0의 Robin 문제에서 κ² 오름차순 고유값과 고유함수를 구합니다.

    y(0)조건을 만족하는 해는 y=k₂κ cos κx − k₁ sin κx (κ²>0), y=k₂ − k₁x (κ=0),
    y=k₂μ cosh μx − k₁ sinh μx (κ²=−μ²<0)이고, 오른쪽 조건의 값이 0이 되는 κ를 찾습니다.
    """
    from scipy.optimize import brentq
    if (k1 == 0 and k2 == 0) or (l1 == 0 and l2 == 0):
        raise ValueError('각 끝의 경계조건 계수가 모두 0일 수는 없습니다.')

    def trig(kappa):
        return k1 * (l1 * np.sin(kappa * length) + l2 * kappa * np.cos(kappa * length)) \
            - k2 * kappa * (l1 * np.cos(kappa * length) - l2 * kappa * np.sin(kappa * length))

    def hyperbolic(mu):
        # cosh(μL)로 나누어 넘침을 피합니다. 근의 위치는 바뀌지 않습니다.
        t = np.tanh(mu * length)
        return k1 * (l1 * t + l2 * mu) - k2 * mu * (l1 + l2 * mu * t)

    def roots(function, upper, samples):
        grid = np.linspace(1e-9, upper, samples)
        values = function(grid)
        found = []
        for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if fa == 0:
                found.append(a)
            elif fa * fb < 0:
                found.append(brentq(function, a, b, xtol=1e-14))
        return found

    squared = [-(mu ** 2) for mu in roots(hyperbolic, 50 / length, 20000)]
    if abs(k1 * (l1 * length + l2) - k2 * l1) < 1e-12:
        squared.append(0.)
    upper = (count + 2) * np.pi / length
    trig_roots = roots(trig, upper, 400 * (count + 2))
    while len(squared) + len(trig_roots) < count:
        upper *= 2
        trig_roots = roots(trig, upper, 400 * int(upper * length / np.pi + 2))
    squared = sorted(squared + [kappa ** 2 for kappa in trig_roots])[:count]
    modes = []
    for value in squared:
        if value > 0:
            kappa = np.sqrt(value)
            modes.append(k2 * kappa * np.cos(kappa * x) - k1 * np.sin(kappa * x))
        elif value < 0:
            mu = np.sqrt(-value)
            # 큰 μ에서도 유한하도록 e^(μL)로 나눈 형태를 씁니다.
            scale = np.exp(-mu * length)
            modes.append(scale * (k2 * mu * np.cosh(mu * x) - k1 * np.sinh(mu * x)))
        else:
            modes.append(k2 - k1 * x)
    basis = np.array(modes)
    basis /= np.max(np.abs(basis), axis=1, keepdims=True)
    return np.array(squared), basis


# ── 주기 외력의 Fourier 성분과 운동방정식 ────────────────────────
def homogeneous_motion(time, mass, damping, stiffness, y0, v0):
    """m y″+c y′+k y=0, y(0)=y0, y′(0)=v0의 닫힌 해와 속도입니다.

    σ=−c/(2m), q=√(σ²−k/m)(복소수 허용)이면 y=y0·C+(v0−σy0)·S이고
    C=½(e^((σ+q)t)+e^((σ−q)t)), S=(e^((σ+q)t)−e^((σ−q)t))/(2q)입니다.
    부족감쇠(q 허수)·과감쇠(q 실수)·임계감쇠(q→0, S=t·e^(σt))를 한 식으로 다루며,
    지수 형태로 계산하여 큰 t에서도 cosh·sinh가 넘치지 않습니다.
    """
    sigma = -damping / (2 * mass)
    q = np.sqrt(complex(sigma * sigma - stiffness / mass))
    t = np.asarray(time, dtype=float)
    if abs(q) * max(float(t.max()), 1e-300) < 1e-8:
        base = np.exp(sigma * t)
        c_part, s_part = base, t * base
        c_rate, s_rate = sigma * base, base * (1 + sigma * t)
    else:
        plus, minus = np.exp((sigma + q) * t), np.exp((sigma - q) * t)
        c_part, s_part = (plus + minus) / 2, (plus - minus) / (2 * q)
        c_rate = ((sigma + q) * plus + (sigma - q) * minus) / 2
        s_rate = ((sigma + q) * plus - (sigma - q) * minus) / (2 * q)
    weight = v0 - sigma * y0
    return np.real(y0 * c_part + weight * s_part), np.real(y0 * c_rate + weight * s_rate)


def forced_response(time, length, a0, an, bn, mass, damping, stiffness, y0, v0):
    """m y″+c y′+k y = a₀+Σ(aₙcos ωₙt+bₙsin ωₙt), ωₙ=nπ/L의 초기값 문제를 닫힌 해로 풉니다.

    특수해: 상수 a₀/k와 성분별 Re[(aₙ−ibₙ)/(k−mωₙ²+icωₙ)·e^(iωₙt)].
    무감쇠 공진 성분(분모 0)은 t/(2mω)·(a sin ωt − b cos ωt)입니다.
    동차해는 초기조건에서 특수해의 초기값을 뺀 값으로 맞춥니다. 수치 시간 적분을 쓰지 않습니다.
    """
    time = np.asarray(time, dtype=float)
    frequencies = np.arange(1, len(an) + 1) * np.pi / length
    denominator = stiffness - mass * frequencies ** 2 + 1j * damping * frequencies
    amplitudes = np.hypot(an, bn)
    active = amplitudes > 1e-9 * max(1., amplitudes.max())
    resonant = (np.abs(denominator) <= 1e-10 * max(stiffness, 1.)) & active
    regular = active & ~resonant
    response = np.zeros_like(denominator)
    response[regular] = (an[regular] - 1j * bn[regular]) / denominator[regular]
    phase = np.exp(1j * np.outer(frequencies, time))
    periodic = a0 / stiffness + np.real(response @ phase)
    periodic_rate = np.real((1j * frequencies * response) @ phase)
    position, velocity = periodic.copy(), periodic_rate.copy()
    for index in np.flatnonzero(resonant):
        omega, a, b = frequencies[index], an[index], bn[index]
        sine, cosine = np.sin(omega * time), np.cos(omega * time)
        position += time / (2 * mass * omega) * (a * sine - b * cosine)
        velocity += (a * sine - b * cosine) / (2 * mass * omega) + time / (2 * mass) * (a * cosine + b * sine)
    free_position, free_velocity = homogeneous_motion(time, mass, damping, stiffness,
                                                      y0 - position[0], v0 - velocity[0])
    state = checked(np.vstack((position + free_position, velocity + free_velocity)))
    particular = None if np.any(resonant) else checked(periodic)
    return state, particular, resonant, frequencies


# ── 연속 변환: 블록 처리로 큰 외적의 메모리 사용을 제한합니다 ───────
def integral_transform(grid, values, output, kernel, factor=1., block_size=64):
    values = checked(values)
    output = checked(np.asarray(output))
    if values.shape != np.shape(grid):
        raise ValueError('변환 입력 표본의 크기가 일치하지 않습니다.')
    weighted = values * trapezoid_weights(grid)
    result = np.empty(len(output), dtype=complex if kernel in ('forward', 'inverse') else float)
    for start in range(0, len(output), block_size):
        angle = output[start:start + block_size, None] * grid
        if kernel == 'forward':
            matrix = np.exp(-1j * angle)
        elif kernel == 'inverse':
            matrix = np.exp(1j * angle)
        elif kernel == 'cosine':
            matrix = np.cos(angle)
        elif kernel == 'sine':
            matrix = np.sin(angle)
        else:
            raise ValueError('지원하지 않는 변환 커널입니다.')
        result[start:start + block_size] = factor * (matrix @ weighted)
    return checked(result)
