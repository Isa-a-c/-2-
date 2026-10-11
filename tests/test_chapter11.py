"""교재 11장의 해석식·정규화·경계조건·저장 회귀 검증입니다."""
import numpy as np
import pytest
from engineering_math.core.chapter11 import half_range, project_basis, sturm_basis, orthogonal_basis, integral_transform
from engineering_math.topics import get_topic, get_topics
from engineering_math.topics.chapter11_transforms import PAIRS, ROOT_TWO_OVER_PI
from engineering_math.core.project import save_settings, load_settings, export_results


def test_half_range_x_coefficients():
    length = np.pi
    x = np.linspace(0, length, 12001)
    n = np.arange(1, 11)
    a0, bn, sums = half_range(x, x, length, 10, False)
    assert a0 == 0
    # 사다리꼴 오차는 O(n Δx²)이며 n=10에서 약 1.15e-7입니다.
    np.testing.assert_allclose(bn, 2 * length * (-1.) ** (n + 1) / (n * np.pi), rtol=0, atol=1.3e-7)
    fine = half_range(np.linspace(0, length, 24001), np.linspace(0, length, 24001), length, 10, False)[1]
    exact = 2 * length * (-1.) ** (n + 1) / (n * np.pi)
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(bn - exact)) / 3.9
    a0, an, sums = half_range(x, x, length, 10, True)
    assert a0 == pytest.approx(length / 2)
    np.testing.assert_allclose(an, 2 * length * ((-1.) ** n - 1) / (n * np.pi) ** 2, atol=1e-8)


@pytest.mark.parametrize('boundary', ['DD', 'NN', 'DN'])
def test_sturm_boundary_and_weighted_orthogonality(boundary):
    length = 2.
    x = np.linspace(0, length, 5001)
    basis, eigenvalues = sturm_basis(x, length, 8, boundary, p=2., q=3., r=4.)
    _, gram, _, _ = project_basis(x, x, basis, np.full_like(x, 4.))
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-12)
    start = {'DD': 1., 'NN': 0., 'DN': .5}[boundary]
    assert eigenvalues[0] == pytest.approx((2 * (start * np.pi / length) ** 2 - 3) / 4)
    if boundary != 'NN':
        np.testing.assert_allclose(basis[:, 0], 0, atol=1e-12)
    if boundary == 'DD':
        np.testing.assert_allclose(basis[:, -1], 0, atol=1e-12)
    else:
        derivative = np.gradient(basis, x, axis=1, edge_order=2)
        np.testing.assert_allclose(derivative[:, -1], 0, atol=1e-5)


def test_legendre_polynomial_expansion():
    result = get_topic('orthogonal_series').compute({'function': 'x^2', 'N': 4, 'num_points': 20001})
    np.testing.assert_allclose(result.data['coefficients'], [1/3, 0, 2/3, 0], atol=2e-8)
    assert result.metrics['가중 적분 제곱오차'] < 1e-14


def test_bessel_single_mode_and_weight():
    x = np.linspace(0, 1, 10001)
    basis, weight, eigenvalues = orthogonal_basis('Bessel', x, 5)
    coefficients, gram, sums, _ = project_basis(x, basis[1], basis, weight)
    np.testing.assert_allclose(coefficients, [0, 1, 0, 0, 0], atol=5e-7)
    np.testing.assert_allclose(gram, np.eye(5), atol=5e-7)
    np.testing.assert_allclose(basis[:, -1], 0, atol=1e-14)
    assert np.all(np.diff(eigenvalues) > 0)


def test_minimum_squared_error_perturbation():
    result = get_topic('trig_approximation').compute({'function': 'x^2', 'perturbation': -.7})
    increase = result.metrics['변형 후 제곱오차'] - result.metrics['적분 제곱오차']
    assert increase == pytest.approx(np.pi * .7 ** 2, abs=1e-10)


def test_undamped_resonance_initial_value_solution():
    result = get_topic('fourier_forced').compute({'function': 'sin(x)', 'N': 3,
             'mass': 1., 'stiffness': 1., 'damping': 0., 'end_time': 8.})
    time = result.data['x']
    exact = (np.sin(time) - time * np.cos(time)) / 2
    np.testing.assert_allclose(result.data['state'][0], exact, atol=2e-8)
    assert result.data['particular'] is None
    assert result.metrics['공진 성분 수'] == 1


def test_constant_force_initial_conditions():
    result = get_topic('fourier_forced').compute({'function': '2', 'N': 2,
             'mass': 1., 'stiffness': 4., 'damping': 0., 'end_time': 4.})
    time = result.data['x']
    np.testing.assert_allclose(result.data['state'][0], .5 * (1 - np.cos(2 * time)), atol=1e-8)
    np.testing.assert_allclose(result.data['particular'], .5, atol=1e-12)


def test_continuous_gaussian_and_fft_normalization():
    result = get_topic('continuous_fourier').compute({})
    data = result.data
    np.testing.assert_allclose(data['spectrum'], np.exp(-data['omega'] ** 2 / 2), atol=1e-12)
    np.testing.assert_allclose(data['fft_spectrum'], np.exp(-data['fft_omega'] ** 2 / 2), atol=1e-12)
    assert result.metrics['복원 RMSE'] < 1e-12


def test_translation_phase():
    result = get_topic('continuous_fourier').compute({'function': 'exp(-(x-1)^2/2)'})
    omega = result.data['omega']
    np.testing.assert_allclose(result.data['spectrum'], np.exp(-omega ** 2 / 2 - 1j * omega), atol=2e-12)


@pytest.mark.parametrize('mode', ['코사인', '사인'])
def test_half_line_transform_of_exponential(mode):
    result = get_topic('sine_cosine_transform').compute({'mode': mode, 'function': 'exp(-x)',
              'extent': 16., 'num_points': 8193})
    omega = result.data['omega']
    exact = ROOT_TWO_OVER_PI * (1 if mode == '코사인' else omega) / (1 + omega ** 2)
    np.testing.assert_allclose(result.data['spectrum'], exact, atol=4e-6)


def test_fourier_integral_gaussian():
    result = get_topic('fourier_integral').compute({})
    data = result.data
    np.testing.assert_allclose(data['spectrum'].real, ROOT_TWO_OVER_PI * np.exp(-data['omega'] ** 2 / 2), atol=1e-12)
    np.testing.assert_allclose(data['spectrum'].imag, 0, atol=1e-14)
    assert result.metrics['복원 RMSE'] < 1e-12


def independent_transform(pair, omega, a, b, c, n):
    """SciPy 적응 구적으로 공식과 독립적인 변환값을 계산합니다.

    무한구간은 QAWF(weight='cos'/'sin')로 계산하여 느리게 감소하거나 조건부 수렴하는 원함수도 다룹니다.
    """
    from scipy.integrate import quad
    from engineering_math.core.expr import parameter_expression, compiled
    params = f'a={a}; b={b}; c={c}; n={n}'
    real_f = compiled(parameter_expression(pair.function, params))
    imag_f = compiled(parameter_expression(pair.imaginary, params)) if pair.imaginary else None

    def integral(f, wave, left, right):
        g = lambda x: float(f(x))
        weight = np.cos if wave == 'cos' else np.sin
        if np.isinf(right):
            # [0,1]은 끝점을 평가하지 않는 적응 구적(원점의 적분 가능한 특이점 허용), [1,∞)는 QAWF입니다.
            head = quad(lambda x: g(x)*weight(omega*x), left, 1., limit=400)[0]
            return head + quad(g, 1., np.inf, weight=wave, wvar=omega, limlst=200)[0]
        cuts = [value for value in (-b, b, c, 2*b, a) if left < value < right]
        return quad(lambda x: g(x)*weight(omega*x), left, right, points=cuts or None, limit=400)[0]

    if pair.key in ('표 I-8', '표 I-9'):
        # cos(ax²)·sin(ax²)는 감소하지 않는 처프이므로 영점 사이 합을 외삽하는 mpmath.quadosc를 씁니다.
        import mpmath
        chirp = mpmath.cos if pair.key == '표 I-8' else mpmath.sin
        value = mpmath.quadosc(lambda x: chirp(a*x**2)*mpmath.cos(omega*x), [0, mpmath.inf],
                               zeros=lambda k: mpmath.sqrt(k*mpmath.pi/a))
        return np.sqrt(2/np.pi)*float(value)
    support = {'표 III-1': (-b, b), '표 III-2': (b, c), '표 III-4': (0, 2*b), '표 III-5': (0, np.inf),
               '표 III-3': (-np.inf, np.inf), '표 III-9': (-np.inf, np.inf), '표 III-10': (-np.inf, np.inf),
               '표 III-6': (b, c), '표 III-7': (-b, b), '표 III-8': (b, c),
               '표 I-1': (0, a), '표 I-7': (0, a), '표 II-1': (0, a), '표 II-10': (0, a)}
    left, right = support.get(pair.key, (0, np.inf))
    if pair.kind != 'complex':
        return np.sqrt(2/np.pi)*integral(real_f, 'cos' if pair.kind == 'cosine' else 'sin', left, right)

    def transform(f):
        if np.isinf(right) and left == 0:
            return integral(f, 'cos', 0, np.inf) - 1j*integral(f, 'sin', 0, np.inf)
        if np.isinf(right):
            # 짝함수(III-3·9·10)는 2∫₀^∞ f cos입니다.
            return 2*integral(f, 'cos', 0, np.inf)
        return integral(f, 'cos', left, right) - 1j*integral(f, 'sin', left, right)

    value = transform(real_f) + (1j*transform(imag_f) if imag_f else 0)
    return value/np.sqrt(2*np.pi)


@pytest.mark.parametrize('pair', list(PAIRS.values()), ids=lambda item: item.key)
def test_formula_table_against_independent_quadrature(pair):
    a = .6 if pair.needs == '0<a<1' else 1.3
    b, c, n = .4, 1.7, 2
    for omega in [.3, 1.1, 2.7]:
        if pair.key in ('표 I-10', '표 I-12', '표 II-11', '표 III-10') and abs(omega - a) < .05:
            continue
        exact = complex(np.asarray(pair.exact(np.array([omega]), a, b, c, n))[0])
        value = independent_transform(pair, omega, a, b, c, n)
        assert exact == pytest.approx(value, abs=5e-7), (pair.key, omega)
    assert np.all(np.isfinite(np.asarray(pair.exact(np.array([0.]), a, b, c, n)))) or pair.key in ('표 I-2', '표 II-2', '표 II-3', '표 II-4')


def test_transform_table_has_all_34_entries_and_migrates_old_names(tmp_path):
    import json
    assert len(PAIRS) == 34
    assert [pair.key for pair in PAIRS.values()].count('표 III-4') == 1
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'version': 2, 'topic': 'transform_table', 'params': {'pair': '복소 · 사각 펄스'}}), encoding='utf-8')
    assert load_settings(path)['params']['pair'].startswith('표 III-1')


@pytest.mark.parametrize('pair', list(PAIRS.values()), ids=lambda item: item.key)
def test_transform_table_screen_runs_every_entry(pair):
    # 꼬리가 긴 함수(xⁿe^(−ax), 1/(x²+a²))도 검산하도록 공간 절단을 넓힙니다.
    # 도약점 a, b, c는 격자 간격(80/8192)의 정수배로 두어 사다리꼴 적분이 대칭이 되게 합니다.
    params = {'pair': pair.label, 'a': .6 if pair.needs == '0<a<1' else 1.25, 'b': .625, 'c': 2.5,
              'extent': 40., 'num_points': 8193}
    result = get_topic('transform_table').compute(params)
    assert result.tables['공식 표'][1][0][0] == '표 I-1'
    if not pair.unsupported:
        # 절단 적분과 공식의 차이: 느리게 감소하는 1/(x²+a²)만 꼬리 절단이 큽니다.
        limit = .03 if pair.key == '표 III-3' else 2e-3
        assert result.metrics['공식·적분 최대 차이'] < limit, pair.key


from engineering_math.topics.catalog import EXPERIMENTS
# 교재 11.2 이후의 11장 주제를 절 번호로 고릅니다(목록 위치에 의존하지 않음).
NEW_IDS = list(dict.fromkeys(item.topic for item in EXPERIMENTS
                             if item.lesson.startswith('11.') and item.topic != 'fourier'))


@pytest.mark.parametrize('topic_id', NEW_IDS)
def test_topic_figures_settings_and_export(topic_id, tmp_path):
    import matplotlib.pyplot as plt
    from engineering_math.topics.lessons import lesson_for
    topic = get_topic(topic_id)
    result = topic.compute({})
    path = tmp_path / '설정.json'
    save_settings(path, topic_id, result.params)
    assert load_settings(path)['params'] == result.params
    export_results(tmp_path / '결과.zip', result)
    assert '11.' in lesson_for(topic_id)
    for figure in topic.figures(result).values():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(figure).draw()
        plt.close(figure)


@pytest.mark.parametrize('topic_id', NEW_IDS[:-1])
def test_nonfinite_function_input_rejected(topic_id):
    params = {'function': '1/0'}
    if topic_id == 'dft':
        params['mode'] = '함수 표본 x_k = 2πk/N'
    with pytest.raises(ValueError):
        get_topic(topic_id).compute(params)


def test_invalid_kernel_and_grid():
    with pytest.raises(ValueError):
        integral_transform(np.array([1, 0]), np.ones(2), np.array([0]), 'cosine')
    with pytest.raises(ValueError):
        integral_transform(np.array([0, 1]), np.ones(2), np.array([0]), 'unknown')


def test_dft_matches_numpy_and_handles_non_power_of_two():
    topic = get_topic('dft')
    result = topic.compute({'data': '1, 2, 3, 4, 5, 6'})
    np.testing.assert_allclose(result.data['spectrum'], np.fft.fft([1, 2, 3, 4, 5, 6]), atol=1e-12)
    assert 'FFT 분해 단계' not in result.tables
    sampled = topic.compute({'mode': '함수 표본 x_k = 2πk/N', 'function': 'cos(7*x)', 'N': 8})
    # cos 7x를 8개 표본으로 보면 cos x와 구별되지 않습니다(aliasing).
    np.testing.assert_allclose(np.abs(sampled.data['spectrum'])[[1, 7]], 4, atol=1e-12)
    with pytest.raises(ValueError):
        topic.compute({'data': '1'})


def test_convolution_theorem_pulses_give_triangle():
    result = get_topic('continuous_fourier').compute({'mode': '합성곱 정리', 'num_points': 8001,
        'function': 'Heaviside(x+1,1/2)-Heaviside(x-1,1/2)', 'second': 'Heaviside(x+1,1/2)-Heaviside(x-1,1/2)'})
    d = result.data
    np.testing.assert_allclose(d['convolution_display'], np.maximum(2-np.abs(d['x']), 0), atol=3e-3)
    with pytest.raises(ValueError, match='홀수'):
        get_topic('continuous_fourier').compute({'mode': '합성곱 정리', 'num_points': 2000})


def test_robin_reduces_to_dirichlet_neumann():
    x = np.linspace(0, 2., 4001)
    basis, eigenvalues = sturm_basis(x, 2., 5, 'Robin', robin=(1., 0., 0., 1.))
    np.testing.assert_allclose(eigenvalues, ((np.arange(5)+.5)*np.pi/2)**2, rtol=1e-10)


def test_robin_eigenvalues_satisfy_tangent_equation_and_orthogonality():
    """y(0)=0, y(L)+h y′(L)=0이면 tan κL = −hκ (11.5 연습문제형)."""
    length, h = 1., .5
    result = get_topic('sturm_liouville').compute({'mode': 'Robin', 'k1': 1., 'k2': 0., 'l1': 1., 'l2': h,
                                                  'L': '1', 'N': 6, 'num_points': 20001})
    kappa = np.sqrt([row[1] for row in result.tables['전개 계수'][1]])
    np.testing.assert_allclose(np.tan(kappa*length), -h*kappa, atol=1e-8)
    np.testing.assert_allclose(result.data['gram'], np.eye(6), atol=1e-6)


def test_robin_negative_eigenvalue_with_hyperbolic_mode():
    """y(0)+y′(0)=0, y(L)=0: λ<0인 쌍곡선 모드 하나와 양의 고유값들 (가중 직교)."""
    result = get_topic('sturm_liouville').compute({'mode': 'Robin', 'k1': 1., 'k2': 1., 'l1': 1., 'l2': 0.,
                                                  'L': '3', 'N': 5, 'num_points': 20001})
    values = [row[1] for row in result.tables['전개 계수'][1]]
    assert values[0] < 0 < values[1]
    mu = np.sqrt(-values[0])
    assert np.tanh(3*mu) == pytest.approx(mu, abs=1e-9)
    np.testing.assert_allclose(result.data['gram'], np.eye(5), atol=1e-6)


def test_periodic_and_invalid_robin():
    x = np.linspace(0, 1, 101)
    _, eigenvalues = sturm_basis(x, 1., 5, 'Periodic')
    np.testing.assert_allclose(eigenvalues, (2*np.pi*np.array([0, 1, 1, 2, 2]))**2)
    with pytest.raises(ValueError):
        sturm_basis(x, 1., 3, 'Robin', robin=(0., 0., 1., 0.))


@pytest.mark.parametrize('function', ['exp(-x^2/2)', 'exp(-x)', 'x*exp(-x)'])
def test_sine_cosine_derivative_rule(function):
    """교재 11.8 정리 1: F_c{f′}=wF_s{f}−√(2/π)f(0), F_s{f′}=−wF_c{f}."""
    result = get_topic('sine_cosine_transform').compute({'function': function, 'extent': 40., 'num_points': 8193})
    # 높은 ω에서 진동 커널의 사다리꼴 오차 (ωh)²/12 ≈ 3e-4 수준을 허용합니다.
    assert result.metrics['정리 1 (4a)·(4b) 최대 차이'] < 5e-4


def test_derivative_rule_skipped_for_jump():
    result = get_topic('sine_cosine_transform').compute({'function': 'Heaviside(1-x,1/2)'})
    assert '정리 1 (4a)·(4b) 최대 차이' not in result.metrics
    assert any('도약' in notice for notice in result.notices)


@pytest.mark.parametrize('mass,damping,stiffness', [
    (1., .2, 4.),      # 부족감쇠
    (1., 4., 4.),      # 임계감쇠 c²=4mk
    (1., 6., 4.),      # 과감쇠
    (1., 0., 4.),      # 무감쇠, 외력 성분 n=2가 공진 (ω=2)
    (1., 0., 2.),      # 무감쇠 비공진
    (.5, 40., 1.),     # 강한 과감쇠: 큰 t에서 지수 넘침이 없어야 합니다
])
def test_forced_response_closed_form_matches_numeric_ode(mass, damping, stiffness):
    from scipy.integrate import solve_ivp
    from engineering_math.core.chapter11 import forced_response
    a0, an, bn = .3, np.array([1., .5, -.2]), np.array([-.4, .8, .1])
    length, y0, v0 = np.pi, .7, -1.2
    time = np.linspace(0, 30, 601)
    state, particular, resonant, frequencies = forced_response(time, length, a0, an, bn, mass, damping, stiffness, y0, v0)

    def rhs(t, s):
        force = a0 + an @ np.cos(frequencies*t) + bn @ np.sin(frequencies*t)
        return [s[1], (force - damping*s[1] - stiffness*s[0])/mass]

    reference = solve_ivp(rhs, (0, 30), [y0, v0], t_eval=time, method='DOP853', rtol=1e-11, atol=1e-12, max_step=.01)
    np.testing.assert_allclose(state, reference.y, atol=2e-7)
    assert resonant.any() == (damping == 0 and stiffness == 4.)
    assert (particular is None) == resonant.any()


def test_forced_response_long_time_is_instant():
    """최대 설정(N=60, 100초)도 수치 적분 없이 즉시 계산됩니다."""
    import time as clock
    start = clock.perf_counter()
    result = get_topic('fourier_forced').compute({'function': 'sign(sin(x))', 'N': 60, 'num_points': 20001, 'end_time': 100.})
    assert clock.perf_counter() - start < .5
    assert np.all(np.isfinite(result.data['state']))
