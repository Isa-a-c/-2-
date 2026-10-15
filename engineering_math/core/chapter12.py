"""12장 PDE의 해석 표현과 수치 구적입니다. UI 상태를 참조하지 않습니다."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import fftconvolve
from scipy.special import erf, erfc, jn_zeros, jv, eval_legendre
from .chapter11 import checked, trapezoid_weights
from .quadrature import piecewise_gauss, gauss_legendre, find_jumps


# ── 무한 현: 이동과 초기 속도의 누적 적분 ─────────────────────────
def dalembert(displacement, velocity, x, time, speed, quadrature_points):
    left = x[None, :] - speed * time[:, None]
    right = x[None, :] + speed * time[:, None]
    grid = np.linspace(left.min(), right.max(), quadrature_points)
    primitive = cumulative_trapezoid(velocity(grid), grid, initial=0.)
    integral = np.interp(right, grid, primitive) - np.interp(left, grid, primitive)
    frames = .5 * (displacement(left) + displacement(right)) + integral / (2 * speed)
    return checked(frames), grid


# ── 무한 막대: Fourier 해와 동등한 열핵 합성곱 ────────────────────
def heat_kernel_weights(step, width, reach):
    """열핵 K=exp(−d²/w²)/(w√π), w=2√(αt)를 격자 함수에 적용하는 가중치입니다.

    셀 상수 모델 ∫_cell K = ½[erf(…)−erf(…)]의 오차는 (h²/24)∫f″K, 선형 보간 모델
    (hat 함수와 K의 정확한 적분)의 오차는 (h²/12)∫f″K입니다. 2·셀 − hat 조합은 O(h²)
    항이 상쇄되어 매끄러운 초기값에서 O(h⁴)이고(Richardson 외삽), 도약이 있으면 O(h)입니다.
    hat 가중치는 K의 2차 원시함수 H(z)=|z|/2+R(|z|)의 2차 차분입니다. |z|/2 부분은
    d=0에서 1이 되고, 작은 R만 수치로 차분하여 상쇄 오차를 피합니다.
    """
    d = np.arange(-reach, reach + 1) * step
    cell = .5 * (erf((d + step / 2) / width) - erf((d - step / 2) / width))

    def remainder(z):
        z = np.abs(z)
        return -z / 2 * erfc(z / width) + width / (2 * np.sqrt(np.pi)) * np.exp(-(z / width) ** 2)

    hat = (remainder(d + step) - 2 * remainder(d) + remainder(d - step)) / step
    hat[reach] += 1.
    kernel = 2 * cell - hat
    # 해석적 합은 1−O(erfc(8))입니다. 반올림 누적을 정규화로 제거하여 상수를 보존합니다.
    return kernel / kernel.sum()


def infinite_heat(initial, x, time, alpha, points):
    """균일 격자에서 열핵을 정확히 적분한 가중치로 합성곱합니다.

    격자 간격은 표시 간격의 정수분의 1이므로 표시점이 격자점과 일치하여 보간이 필요 없습니다.
    격자는 표시 구간 양쪽에 8·2√(αt_max)만큼 여유를 두어 꼬리의 영향을 포함합니다.
    초기값의 도약 Jₖ(위치 cₖ)은 계단 함수로 분리하여 정확한 해 (Jₖ/2)·erfc((cₖ−x)/2√(αt))로
    더합니다. 남은 연속 부분만 수치 합성곱하므로 교재 12.7 예제 1의 사각 펄스도 정확합니다.
    남은 부분에 꺾임(도함수의 도약)이 있으면 그 근처 오차는 O(h²)입니다.
    """
    spread = 2 * np.sqrt(alpha * float(np.max(time)))
    margin = 8 * spread
    display_step = x[1] - x[0]
    target = (x[-1] - x[0] + 2 * margin) / (points - 1)
    refine = max(1, int(np.ceil(display_step / target)))
    step = display_step / refine
    pad = int(np.ceil(margin / step))
    grid = x[0] + (np.arange(-pad, (len(x) - 1) * refine + pad + 1)) * step
    jumps = find_jumps(initial, grid[0], grid[-1], samples=min(len(grid), 65537))
    tiny = 1e-9 * max(1., grid[-1] - grid[0])
    sizes = checked(initial(jumps + tiny) - initial(jumps - tiny)) if len(jumps) else np.array([])

    def steps(points):
        # 도약점 자체는 좌우 평균(½)으로 둡니다.
        return sum(size * np.where(points > c, 1., np.where(points < c, 0., .5)) for c, size in zip(jumps, sizes))

    values = checked(initial(grid) - steps(grid))
    selected = pad + np.arange(len(x)) * refine
    frames = np.empty((len(time), len(x)))
    for i, t in enumerate(time):
        if t == 0:
            frames[i] = initial(x)
            continue
        width = 2 * np.sqrt(alpha * t)
        reach = min(len(grid) - 1, max(1, int(np.ceil(8 * width / step))))
        smoothed = fftconvolve(values, heat_kernel_weights(step, width, reach), mode='same')
        frames[i] = smoothed[selected] + sum(size / 2 * erfc((c - x) / width) for c, size in zip(jumps, sizes))
    return checked(frames)


def fourier_heat(initial, x, time, alpha, points=2048):
    """Fourier 변환 해 u(x,t) = (1/√2π)∫ f̂(w) exp(−αw²t) e^(iwx) dw (교재 12.7 예제 2, 식 14).

    1) f̂(w) = (1/√2π)∫ f(v) e^(−iwv) dv를 도약점에서 나눈 Gauss–Legendre 구적으로 계산합니다.
       적분 구간은 표시 구간 양쪽에 max(8·2√(αt_max), 표시 폭)만큼 넓힌 구간입니다.
    2) f가 실수이므로 역변환은 (2/√2π)·Re∫₀^∞ f̂(w)exp(−αw²t)e^(iwx)dw이고, exp(−αw²t)가 e^(−36)이 되는
       w까지 구간별 Gauss–Legendre로 적분합니다. t=0은 역변환 없이 초기값을 그대로 씁니다.
    반환: (프레임, w 노드, |f̂(w)|, 구간 끝의 |f|/max|f|).
    """
    spread = 2 * np.sqrt(alpha * float(np.max(time)))
    span = x[-1] - x[0]
    margin = max(8 * spread, span)
    lower, upper = x[0] - margin, x[-1] + margin
    v, v_weights, _ = piecewise_gauss(initial, lower, upper, total=points)
    values = checked(initial(v))
    positive = time[time > 0]
    frames = np.empty((len(time), len(x)))
    if not len(positive):
        frames[:] = initial(x)
        return checked(frames), np.zeros(1), np.zeros(1), 0.
    reach = min(np.sqrt(36 / (alpha * float(positive.min()))), 4000.)
    # f̂(w)는 w 방향으로 대략 2π/(적분 구간 길이) 간격으로 변하므로 그 간격마다 16점을 둡니다.
    pieces = int(min(1250, max(8, np.ceil(reach * (upper - lower) / (2 * np.pi)))))
    base, base_weights = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0, reach, pieces + 1)
    half = (edges[1:] - edges[:-1]) / 2
    w = (edges[:-1, None] + half[:, None] * (base + 1)).ravel()
    w_weights = (half[:, None] * base_weights).ravel()
    transform = np.exp(-1j * np.outer(w, v)) @ (v_weights * values) / np.sqrt(2 * np.pi)
    waves = np.exp(1j * np.outer(w, x))
    for i, t in enumerate(time):
        if t == 0:
            frames[i] = initial(x)
        else:
            frames[i] = 2 / np.sqrt(2 * np.pi) * np.real((w_weights * transform * np.exp(-alpha * w**2 * t)) @ waves)
    edge = float(max(abs(initial(np.array([lower]))[0]), abs(initial(np.array([upper]))[0]))
                 / max(float(np.max(np.abs(values))), 1e-300))
    return checked(frames), w, np.abs(transform), edge


# ── 공통 모드 시간 진화와 에너지 ─────────────────────────────────
def modal_motion(time, omega, displacement, velocity, norms):
    phase = np.outer(time, omega)
    amplitudes = np.cos(phase) * displacement + np.sin(phase) * velocity / omega
    rates = -np.sin(phase) * omega * displacement + np.cos(phase) * velocity
    energy = .5 * np.sum(norms * (rates ** 2 + (amplitudes * omega) ** 2), axis=1)
    return checked(amplitudes), checked(energy)


def rectangle_membrane(x, y, initial, velocity, length, height, speed, count, time):
    indices = np.arange(1, count + 1)
    bx = np.sin(np.outer(indices * np.pi / length, x))
    by = np.sin(np.outer(indices * np.pi / height, y))
    weights = np.outer(trapezoid_weights(x), trapezoid_weights(y))
    norm = length * height / 4
    a = bx @ (initial * weights) @ by.T / norm
    v = bx @ (velocity * weights) @ by.T / norm
    omega = speed * np.sqrt((indices[:, None] * np.pi / length) ** 2
                            + (indices[None, :] * np.pi / height) ** 2)
    amplitudes, energy = modal_motion(time, omega.ravel(), a.ravel(), v.ravel(), norm)
    frames = np.einsum('mi,tmn,nj->tij', bx, amplitudes.reshape(len(time), count, count), by, optimize=True)
    frames[:, [0, -1], :] = 0.
    frames[:, :, [0, -1]] = 0.
    return checked(frames), energy, a, v, omega


def disk_membrane(r, theta, initial, velocity, radius, speed, radial_count, angular_count, time,
                  radial_order=None):
    """initial·velocity는 (r, theta) 함수입니다.

    계수는 표시 격자와 분리된 Gauss–Legendre 방사 노드(가중치 r)와 주기 θ 격자로
    적분합니다. θ 방향 균일 합은 매끄러운 주기 함수에서 지수적으로 수렴합니다.
    """
    radial_order = radial_order or max(2 * len(r), 8 * radial_count)
    nodes, node_weights = gauss_legendre(0., radius, radial_order)
    rr, tt = np.meshgrid(nodes, theta, indexing='ij')
    weights = (node_weights * nodes)[:, None] * (2 * np.pi / len(theta))
    f_values, g_values = checked(initial(rr, tt)), checked(velocity(rr, tt))
    basis, norms, omega, labels, a, v = [], [], [], [], [], []
    for m in range(angular_count + 1):
        roots = jn_zeros(m, radial_count)
        for n, root in enumerate(roots, 1):
            radial = jv(m, root * r / radius)
            radial_nodes = jv(m, root * nodes / radius)
            radial_norm = radius ** 2 / 2 * jv(m + 1, root) ** 2
            for label, trig in [('cos', np.cos), ('sin', np.sin)]:
                if m == 0 and label == 'sin':
                    continue
                angular = trig(m * theta)
                norm = radial_norm * (2 * np.pi if m == 0 else np.pi)
                projector = np.outer(radial_nodes, angular) * weights
                a.append(np.sum(projector * f_values) / norm)
                v.append(np.sum(projector * g_values) / norm)
                basis.append(np.outer(radial, angular))
                norms.append(norm)
                omega.append(speed * root / radius)
                labels.append((m, n, label))
    basis, norms, omega = np.array(basis), np.array(norms), np.array(omega)
    a, v = checked(a), checked(v)
    amplitudes, energy = modal_motion(time, omega, a, v, norms)
    frames = (amplitudes @ basis.reshape(len(basis), -1)).reshape(len(time), len(r), len(theta))
    frames[:, -1, :] = 0.
    return checked(frames), energy, a, v, omega, labels


# ── 직사각형 정상 상태: 네 변 Dirichlet 문제의 sinh 급수 중첩 ───────
def sinh_ratio(k, s, total):
    """sinh(k·s)/sinh(k·total)을 넘침 없이 계산합니다 (0 ≤ s ≤ total)."""
    k, s = np.asarray(k, dtype=float)[:, None], np.asarray(s, dtype=float)[None, :]
    return np.exp(k * (s - total)) * (-np.expm1(-2 * k * s)) / (-np.expm1(-2 * k * total))


def sine_side_coefficients(boundary, length, count, quadrature_points):
    nodes, weights, _ = piecewise_gauss(boundary, 0., length, quadrature_points)
    values = checked(boundary(nodes))
    modes = np.arange(1, count + 1)
    return checked(2 / length * (np.sin(np.outer(modes, nodes) * np.pi / length) * values) @ weights)


def rectangle_laplace(sides, x, y, length, height, count, quadrature_points=1024):
    """u_xx+u_yy=0, 네 변의 값을 각각 주는 문제를 네 개의 한 변 문제로 나누어 더합니다.

    윗변 u(x,b)=f(x)만 주어진 경우가 교재 12.6 식 (17)·(18)입니다:
    u=ΣAₙ* sin(nπx/a) sinh(nπy/a), Aₙ*=(2/(a sinh(nπb/a)))∫f sin(nπx/a)dx.
    sides는 'top','bottom'(x의 함수), 'left','right'(y의 함수) 중 0이 아닌 변만 담습니다.
    """
    modes = np.arange(1, count + 1)
    kx, ky = modes * np.pi / length, modes * np.pi / height
    field = np.zeros((len(x), len(y)))
    coefficients = {}
    for side, boundary in sides.items():
        horizontal = side in ('top', 'bottom')
        c = sine_side_coefficients(boundary, length if horizontal else height, count, quadrature_points)
        coefficients[side] = c
        if horizontal:
            profile = sinh_ratio(kx, y if side == 'top' else height - y, height)
            field += (np.sin(np.outer(kx, x)) * c[:, None]).T @ profile
        else:
            profile = sinh_ratio(ky, x if side == 'right' else length - x, length)
            field += profile.T @ (np.sin(np.outer(ky, y)) * c[:, None])
    return checked(field), coefficients


# ── 내부 Dirichlet 퍼텐셜: 원판(원통의 z 독립 해), 축대칭 구 ────────
def disk_potential(boundary, radius_grid, theta, radius, count, quadrature_points=1024):
    """경계 계수는 도약점에서 나눈 Gauss–Legendre 구적으로 계산합니다."""
    nodes, weights, _ = piecewise_gauss(boundary, 0., 2 * np.pi, quadrature_points)
    values = checked(boundary(nodes))
    constant = float(weights @ values / (2 * np.pi))
    coefficients = []
    field = np.full((len(radius_grid), len(theta)), constant)
    for n in range(1, count + 1):
        cosine = float(weights @ (values * np.cos(n * nodes)) / np.pi)
        sine = float(weights @ (values * np.sin(n * nodes)) / np.pi)
        field += np.outer((radius_grid / radius) ** n, cosine * np.cos(n * theta) + sine * np.sin(n * theta))
        coefficients.append((n, cosine, sine))
    return checked(field), constant, coefficients


def legendre_coefficients(boundary, count, quadrature_points):
    """h(μ)의 Fourier–Legendre 계수 cₙ=(2n+1)/2∫h Pₙ dμ, n=0…count−1."""
    mu, weights, _ = piecewise_gauss(boundary, -1., 1., quadrature_points)
    values = checked(boundary(mu))
    return np.array([(2 * n + 1) / 2 * weights @ (values * eval_legendre(n, mu)) for n in range(count)])


def sphere_potential(boundary, radius_grid, theta, radius, count, quadrature_points, exterior=False):
    """내부 u=Σcₙ(r/R)ⁿPₙ, 외부 u=Σcₙ(R/r)ⁿ⁺¹Pₙ (교재 12.11 식 17·20)."""
    coefficients = legendre_coefficients(boundary, count, quadrature_points)
    field = np.zeros((len(radius_grid), len(theta)))
    for n, coefficient in enumerate(coefficients):
        radial = (radius / radius_grid) ** (n + 1) if exterior else (radius_grid / radius) ** n
        field += coefficient * np.outer(radial, eval_legendre(n, np.cos(theta)))
    return checked(field), coefficients
