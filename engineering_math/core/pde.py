"""열·파동방정식에서 공유하는 사인 급수와 격자 비교입니다."""
import numpy as np
from .fourier import calculate_fourier_coefficients
from .expr import evaluate


def evolution_animation(x, time, initial, frames, ylabel='u'):
    """열·파동·열핵·진행파에서 공유하는 1차원 재생 명세입니다."""
    return dict(x=x, frames=frames, reference=initial,
                labels=[f't = {t:.5g}' for t in time], xlabel='x', ylabel=ylabel)


def sine_coefficients(x, values, length, count):
    """[0, ell] 함수를 기확장하여 (2/ell) 사인 적분을 계산합니다."""
    odd_x = np.r_[-x[:0:-1], x]
    odd_y = np.r_[-values[:0:-1], values]
    odd_y[len(x)-1] = 0.
    return calculate_fourier_coefficients(odd_x, odd_y, length, count)[2]


def cosine_coefficients(x, values, length, count):
    """[0, ell] 함수를 우확장하여 A₀=(1/ell)∫f, Aₙ=(2/ell)∫f cos(nπx/ell)를 계산합니다."""
    even_x = np.r_[-x[:0:-1], x]
    even_y = np.r_[values[:0:-1], values]
    a0, an, _ = calculate_fourier_coefficients(even_x, even_y, length, count)
    return a0, an


def sine_basis(x, length, count):
    return np.sin(np.arange(1, count+1)[:, None]*np.pi*x/length)


def heat_accuracy(expression, params, coefficients):
    """적분 표본 수와 차수의 영향을 각각 고정된 평가 격자에서 비교합니다.

    격자 간 변화는 참해에 대한 오차 상한이 아닙니다. t=0 재구성 오차는
    초기 입력과 비교하며, 최종 시간의 차이는 두 근사 해끼리 비교합니다.
    params['mode']가 '양 끝 단열'이면 코사인 급수, 아니면 고정 온도의 사인 급수입니다.
    """
    length, count, points = params['length'], params['N'], params['num_points']
    insulated = params.get('mode') == '양 끝 단열'
    higher_count = min(2*count, 200, points-2)
    evaluation_x = np.linspace(0, length, 2049)

    def steady(x):
        return params['left']+(params['right']-params['left'])*x/length

    def integrate(sample_count, order):
        grid = np.linspace(0, length, sample_count)
        values = evaluate(expression, (grid, length), ('x', 'ell'))
        if insulated:
            return np.r_[cosine_coefficients(grid, values, length, order)]
        return sine_coefficients(grid, values-steady(grid), length, order)

    refined = integrate(2*points-1, count)
    higher = integrate(points, higher_count)
    initial = evaluate(expression, (evaluation_x, length), ('x', 'ell'))
    if insulated:
        grid = np.linspace(0, length, points)
        coefficients = np.r_[cosine_coefficients(grid, evaluate(expression, (grid, length), ('x', 'ell')), length, count)]

    def snapshots(values):
        times = np.array([0., params['end_time']])[:, None]
        if insulated:
            wave_numbers = np.arange(len(values))*np.pi/length
            basis = np.cos(wave_numbers[:, None]*evaluation_x)
            return np.exp(-params['alpha']*times*wave_numbers**2) @ (values[:, None]*basis)
        wave_numbers = np.arange(1, len(values)+1)*np.pi/length
        basis = sine_basis(evaluation_x, length, len(values))
        frames = np.exp(-params['alpha']*times*wave_numbers**2) @ (values[:, None]*basis)
        frames += steady(evaluation_x)
        frames[:, 0], frames[:, -1] = params['left'], params['right']
        return frames

    baseline = snapshots(coefficients)
    rows = []
    for label, sample_count, values in [('기준', points, coefficients), ('표본 수 증가', 2*points-1, refined), ('차수 증가', points, higher)]:
        frames = snapshots(values)
        rows.append([label, sample_count, len(values) - insulated,
                     float(np.sqrt(np.mean((frames[0]-initial)**2))),
                     float(np.max(np.abs(frames[0]-baseline[0]))),
                     float(np.max(np.abs(frames[1]-baseline[1])))])
    return dict(x=evaluation_x, initial=initial, reconstruction=baseline[0],
                rows=rows, coefficient_change=float(np.max(np.abs(refined-coefficients))))


def evolution_figures(x, time, initial, frames, ylabel, title):
    """두 PDE에서 같은 형태의 곡선과 시공간 히트맵을 생성합니다."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 5), layout='constrained')
    axes = figure.subplots()
    axes.plot(x, initial, 'k--', label='Initial input')
    for index in np.unique(np.linspace(0, len(time)-1, 5, dtype=int)):
        axes.plot(x, frames[index], label=f't={time[index]:.4g}')
    axes.set(xlabel='x', ylabel=ylabel, title=title)
    axes.grid(alpha=.3)
    axes.legend()
    heatmap = Figure(figsize=(8, 5), layout='constrained')
    axes = heatmap.subplots()
    mesh = axes.pcolormesh(x, time, frames, shading='auto')
    axes.set(xlabel='x', ylabel='t', title=title)
    heatmap.colorbar(mesh, ax=axes, label=ylabel)
    return figure, heatmap
