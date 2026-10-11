"""교재 본문 예제 프리셋을 실행하여 교재에 실린 값과 비교합니다.

기댓값은 Kreyszig 『공업수학 10판 하』 본문의 식·표에서 옮겼습니다(쪽수는 프리셋 이름 참조).
허용 오차는 각 수치 방법의 정밀도(사다리꼴 적분의 O(h²) 등)와 교재 표의 반올림 자릿수에 맞춥니다.
"""
import numpy as np
import pytest
from scipy.special import erf, jn_zeros, jv
from engineering_math.topics import get_topic
from engineering_math.topics.textbook import EXAMPLES, UNSUPPORTED, examples_for

PI = np.pi
n = np.arange(1, 11)


def run(key, title_part=''):
    matches = [item for item in EXAMPLES if item.key == key and title_part in item.title]
    assert len(matches) == 1, f'{key} {title_part}: 프리셋이 하나여야 합니다.'
    example = matches[0]
    return get_topic(example.topic).compute(dict(example.params))


@pytest.mark.parametrize('example', EXAMPLES, ids=lambda item: f'{item.key}-{item.title}')
def test_every_preset_runs_and_is_listed(example):
    topic = get_topic(example.topic)
    assert example.label in topic.examples
    result = topic.compute(dict(example.params))
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_keys_are_unique_per_label_and_unsupported_is_disjoint():
    labels = [item.label for item in EXAMPLES]
    assert len(labels) == len(set(labels))
    assert not set(UNSUPPORTED) & {item.key for item in EXAMPLES}
    assert examples_for('ode') == {}


# ── 11장 ────────────────────────────────────────────────────────
def test_11_1_1_square_wave():
    d = run('11.1-1').data
    np.testing.assert_allclose(d['bn'][:5], 4/(n[:5]*PI)*(n[:5] % 2), atol=1e-6)
    np.testing.assert_allclose(d['an'][:5], 0, atol=1e-12)


def test_11_1_2_partial_sums_vanish_at_jump():
    d = run('11.1-2').data
    zero = np.argmin(np.abs(d['x']))
    assert all(abs(values[zero]) < 1e-12 for values in d['sums'].values())


def test_11_2_1_pulse_cosine_series():
    result = run('11.2-1')
    assert result.metrics['a₀'] == pytest.approx(.5, abs=1e-6)
    np.testing.assert_allclose(result.data['an'][:6], 2/(n[:6]*PI)*np.sin(n[:6]*PI/2), atol=1e-6)
    np.testing.assert_allclose(result.data['bn'][:6], 0, atol=1e-12)


def test_11_2_2_scaled_square_wave():
    d = run('11.2-2').data
    np.testing.assert_allclose(d['bn'][:5], 4/(n[:5]*PI)*(n[:5] % 2), atol=5e-6)


def test_11_2_3_half_wave_rectifier():
    result = run('11.2-3')
    d = result.data
    assert result.metrics['a₀'] == pytest.approx(1/PI, abs=1e-7)
    assert d['bn'][0] == pytest.approx(.5, abs=1e-7)
    np.testing.assert_allclose(d['an'][[1, 3, 5]], [-2/(1*3*PI), -2/(3*5*PI), -2/(5*7*PI)], atol=1e-7)
    np.testing.assert_allclose(d['an'][[0, 2, 4]], 0, atol=1e-7)


def test_11_2_5_sawtooth():
    result = run('11.2-5')
    assert result.metrics['a₀'] == pytest.approx(PI, abs=1e-6)
    np.testing.assert_allclose(result.data['bn'][:4], [2, -1, 2/3, -1/2], atol=1e-4)


def test_11_2_6_triangle_half_range():
    sine = run('11.2-6', '반구간 사인').data['coefficients']
    np.testing.assert_allclose(sine[:6], 8/(n[:6]**2*PI**2)*np.sin(n[:6]*PI/2), atol=1e-8)
    cosine = run('11.2-6', '반구간 코사인').data
    assert cosine['a0'] == pytest.approx(.5, abs=1e-8)
    expected = 4/(n[:6]**2*PI**2)*(2*np.cos(n[:6]*PI/2) - np.cos(n[:6]*PI) - 1)
    np.testing.assert_allclose(cosine['coefficients'][:6], expected, atol=1e-8)
    assert cosine['coefficients'][1] == pytest.approx(-16/(4*PI**2))


def test_11_3_1_force_coefficients():
    result = run('11.3-1')
    rows = result.tables['외력 성분'][1]
    np.testing.assert_allclose([row[2] for row in rows], 4/(n[:9]**2*PI)*(n[:9] % 2), atol=1e-7)
    np.testing.assert_allclose([row[3] for row in rows], 0, atol=1e-10)


def test_11_3_1_response_amplitudes_and_dominant_harmonic():
    """교재 24쪽: C₁=0.0531, C₃=0.0088, C₅=0.2037, C₇=0.0011, C₉=0.0003."""
    result = run('11.3-1')
    amplitudes = np.array([row[5] for row in result.tables['외력 성분'][1]])
    np.testing.assert_allclose(amplitudes[[0, 2, 4, 6, 8]], [.0531, .0088, .2037, .0011, .0003], atol=6e-5)
    np.testing.assert_allclose(amplitudes[[1, 3, 5, 7]], 0, atol=1e-8)
    assert result.metrics['지배 성분 n'] == 5


def test_11_4_1_square_error_table():
    assert run('11.4-1').metrics['적분 제곱오차'] == pytest.approx(.6129, abs=1e-4)


def test_11_5_1_string_eigenvalues():
    rows = run('11.5-1').tables['전개 계수'][1]
    np.testing.assert_allclose([row[1] for row in rows], n[:6]**2)
    gram = run('11.5-1').data['gram']
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-6)


def test_11_5_2_orthonormal_sines_norm_root_pi():
    """교재 31~32쪽: 길이 2π 구간에서 ‖sin mx‖² = π (cos도 같음), 상수는 2π."""
    result = run('11.5-2')
    rows = result.tables['전개 계수'][1]
    np.testing.assert_allclose([row[3] for row in rows], [2*PI] + [PI]*6, rtol=1e-6)
    np.testing.assert_allclose([row[1] for row in rows], [0, 1, 1, 4, 4, 9, 9])
    np.testing.assert_allclose(result.data['gram'], np.eye(7), atol=1e-6)


def test_11_5_4_legendre_orthogonality():
    np.testing.assert_allclose(run('11.5-4').data['gram'], np.eye(6), atol=1e-6)


def test_11_6_1_fourier_legendre():
    c = run('11.6-1').data['coefficients']
    np.testing.assert_allclose(c[[1, 3, 5, 7, 9]], [.95493, -1.15824, .21929, -.01664, .00068], atol=6e-6)
    np.testing.assert_allclose(c[[0, 2, 4]], 0, atol=1e-10)


def test_11_6_3_fourier_bessel():
    c = run('11.6-3').data['coefficients']
    roots = jn_zeros(0, 4)
    np.testing.assert_allclose(c[:4], 4*jv(2, roots)/(roots**2*jv(1, roots)**2), atol=1e-6)
    np.testing.assert_allclose(c[:4], [1.1081, -.1398, .0455, -.0210], atol=1e-4)


def test_11_7_1_period_to_infinity():
    """교재 42쪽: aₙ = 2 sin(ωₙ)/(L ωₙ), ωₙ=nπ/L. 2L=4, 8, 16에서 반파장마다 1, 3, 7개의 진폭."""
    result = run('11.7-1')
    for period, omega, an, bn in result.data['period_series']:
        half = period/2
        np.testing.assert_allclose(an, 2*np.sin(omega)/(half*omega), atol=1e-10)
        np.testing.assert_allclose(bn, 0, atol=1e-12)
        first_lobe = np.sum(omega < PI)
        assert first_lobe == {4: 1, 8: 3, 16: 7}[period]
    assert result.metrics['급수 계수와 (π/L)·적분 계수 최대 차이'] < 1e-5


def test_11_7_2_single_pulse_integral():
    d = run('11.7-2').data
    w = d['omega'][1:]
    np.testing.assert_allclose(d['spectrum'].real[1:], 2*np.sin(w)/(PI*w), atol=1e-5)
    np.testing.assert_allclose(d['spectrum'].imag, 0, atol=1e-10)


def test_11_7_3_laplace_integral():
    d = run('11.7-3').data
    w = d['omega']
    np.testing.assert_allclose(d['spectrum'], np.sqrt(2/PI)/(1+w**2), atol=5e-6)


def test_11_8_1_rectangle_transforms():
    cosine = run('11.8-1', '코사인 변환').data
    w = cosine['omega'][1:]
    np.testing.assert_allclose(cosine['spectrum'][1:], np.sqrt(2/PI)*np.sin(w)/w, atol=3e-5)
    sine = run('11.8-1', '펄스의 사인').data
    np.testing.assert_allclose(sine['spectrum'][1:], np.sqrt(2/PI)*(1-np.cos(w))/w, atol=3e-5)


@pytest.mark.parametrize('key,a', [('11.8-2', 1.), ('11.8-3', 2.)])
def test_11_8_exponential_cosine_transform(key, a):
    d = run(key).data
    np.testing.assert_allclose(d['spectrum'], np.sqrt(2/PI)*a/(a**2+d['omega']**2), atol=1e-5)


def test_11_9_1_pulse_transform():
    d = run('11.9-1').data
    w = d['omega']
    exact = np.sqrt(2/PI)*np.sinc(w/PI)
    np.testing.assert_allclose(d['spectrum'], exact, atol=1e-5)


def test_11_9_2_one_sided_exponential():
    d = run('11.9-2').data
    # 높은 ω에서 진동 커널의 사다리꼴 오차 (ωh)²/12가 커집니다.
    np.testing.assert_allclose(d['spectrum'], 1/(np.sqrt(2*PI)*(1+1j*d['omega'])), atol=1e-4)


def test_11_9_3_derivative_formula():
    d = run('11.9-3').data
    w = d['omega']
    np.testing.assert_allclose(d['spectrum'], -1j*w/(2*np.sqrt(2))*np.exp(-w**2/4), atol=1e-8)


def test_11_9_4_dft_of_four_samples():
    result = run('11.9-4')
    np.testing.assert_allclose(result.data['spectrum'], [14, -4+8j, -6, -4-8j], atol=1e-12)
    assert result.metrics['역변환 최대 오차'] < 1e-12
    rows = result.tables['Fourier 행렬 F_N'][1]
    assert rows[1] == ['1', '-1i', '-1', '1i']


def test_11_9_5_fft_even_odd_split():
    rows = run('11.9-5').tables['FFT 분해 단계'][1]
    top = rows[-1]
    assert top[0] == 4 and top[3] == '[4, -4]' and top[4] == '[10, -8]'
    assert top[5] == '[14, -4+8i, -6, -4-8i]'


def test_11_9_convolution_theorem_gaussians():
    result = run('11.9', '합성곱')
    d = result.data
    np.testing.assert_allclose(d['convolution_display'], np.sqrt(PI)*np.exp(-d['x']**2/4), atol=1e-8)
    np.testing.assert_allclose(d['theorem'].real, np.sqrt(PI)*np.exp(-d['x']**2/4), atol=1e-6)


# ── 12장 ────────────────────────────────────────────────────────
@pytest.mark.parametrize('title', ['x²−y²', 'eˣcos y', 'sin x cosh y'])
def test_12_1_1_harmonic_functions(title):
    result = run('12.1-1', title)
    assert '기호적으로 0: True' in result.data['symbolic']


def test_12_1_2_solve_like_ode():
    """교재 75쪽: u(x,y) = A(y)eˣ + B(y)e⁻ˣ."""
    rows = dict(run('12.1-2').tables['풀이'][1])
    assert rows['일반해 u'] == 'A(y)*exp(-x) + B(y)*exp(x)'
    assert rows['대입 검산 잔차'] == '0'


def test_12_1_3_substitution_p_equals_u_x():
    """교재 75쪽: u = f(x)e⁻ʸ + g(y), f=∫c dx."""
    rows = dict(run('12.1-3').tables['풀이'][1])
    assert rows['p = u_x'] == 'c(x)*exp(-y)'
    assert rows['일반해 u'] == 'g(y) + exp(-y)*Integral(c(x), x)'
    assert rows['대입 검산 잔차'] == '0'


def test_12_3_1_triangle_string_coefficients_and_dalembert():
    result = run('12.3-1')
    rows = result.tables['모드 계수'][1]
    k = .1
    np.testing.assert_allclose(rows[:10, 1], 8*k/(n**2*PI**2)*np.sin(n*PI/2), atol=1e-7)
    # 식 (17): u = ½[f*(x−ct)+f*(x+ct)], f*는 주기 2의 기주기 확장입니다.
    d = result.data
    def odd_extension(s):
        s = (s + 1) % 2 - 1
        return np.sign(s)*np.where(np.abs(s) < .5, 2*k*np.abs(s), 2*k*(1-np.abs(s)))
    index = np.argmin(np.abs(d['time'] - .2))
    t = d['time'][index]
    exact = .5*(odd_extension(d['x']-t) + odd_extension(d['x']+t))
    np.testing.assert_allclose(d['displacement'][index], exact, atol=2e-3)


def test_12_4_1_wave_characteristics():
    """교재 90쪽: y=ct로 두면 u_xx−u_yy=0, 특성 y+x=상수, y−x=상수."""
    result = run('12.4-1')
    rows = dict(result.tables['특성과 표준형'][1])
    assert dict(result.tables['분류'][1])['형태'] == '쌍곡선형'
    assert {rows['특성 Φ = 상수'], rows['특성 Ψ = 상수']} == {'x + y', '-x + y'}
    assert rows['표준형'].startswith('u_vw')


@pytest.mark.parametrize('key,seconds', [('12.6-1', 388), ('12.6-2', 43)])
def test_12_6_copper_bar_half_time(key, seconds):
    d = run(key).data
    peak = d['temperature'].max(axis=1)
    half = d['time'][np.argmax(peak <= 50)]
    assert half == pytest.approx(seconds, abs=1.5)


def test_12_6_3_triangle_initial_temperature():
    rows = run('12.6-3').tables['사인 계수'][1]
    np.testing.assert_allclose(rows[:10, 1], 4*PI/(n**2*PI**2)*np.sin(n*PI/2), atol=1e-6)


def test_12_6_4_insulated_bar_cosine_series():
    result = run('12.6-4')
    rows = result.tables['코사인 계수'][1]
    assert rows[0, 1] == pytest.approx(PI/2, abs=1e-6)
    np.testing.assert_allclose(rows[1:11, 1], 2/PI*(np.cos(n*PI) - 1)/n**2, atol=1e-6)
    # 열이 빠져나가지 않으므로 평균 온도가 보존됩니다.
    d = result.data
    mean = np.trapezoid(d['temperature'], x=d['x'], axis=1)/PI
    np.testing.assert_allclose(mean, PI/2, atol=1e-6)


def test_12_6_5_insulated_triangle_tends_to_quarter_length():
    result = run('12.6-5')
    rows = result.tables['코사인 계수'][1]
    assert rows[0, 1] == pytest.approx(PI/4, abs=1e-6)
    expected = 2*PI/(n**2*PI**2)*(2*np.cos(n*PI/2) - np.cos(n*PI) - 1)
    np.testing.assert_allclose(rows[1:11, 1], expected, atol=1e-6)
    assert result.metrics['정상 상태 (초기 평균)'] == pytest.approx(PI/4, abs=1e-6)
    np.testing.assert_allclose(result.data['temperature'][-1], PI/4, atol=2e-2)


def test_12_6_rectangle_dirichlet_plate():
    """정사각형 판의 윗변 25°C, 나머지 0°C: 대칭성으로 중심 온도는 25/4입니다."""
    result = run('12.6')
    assert result.metrics['중심 온도'] == pytest.approx(25/4, abs=1e-10)
    rows = result.tables['변별 사인 계수'][1]
    np.testing.assert_allclose([row[2] for row in rows[:6]], 2*25*(1-np.cos(n[:6]*PI))/(n[:6]*PI), atol=1e-9)


def test_12_7_1_infinite_bar_erf_solution():
    d = run('12.7-1').data
    t = d['time'][1:, None]
    exact = 50*(erf((1-d['x'])/(2*np.sqrt(t))) + erf((1+d['x'])/(2*np.sqrt(t))))
    np.testing.assert_allclose(d['frames'][1:], exact, atol=1e-9)


def test_12_7_4_semi_infinite_bar_sine_transform():
    """f=1 (0<x<1), u(0,t)=0의 해: 기확장 [−1,1]의 열핵 합성곱 = erf(x/s) − ½erf((x−1)/s) − ½erf((x+1)/s)."""
    d = run('12.7-4').data
    s = 2*np.sqrt(d['time'][1:, None])
    x = d['x']
    exact = erf(x/s) - .5*erf((x-1)/s) - .5*erf((x+1)/s)
    np.testing.assert_allclose(d['frames'][1:], exact, atol=1e-10)
    np.testing.assert_allclose(d['frames'][1:, 0], 0, atol=0)


def test_12_9_1_degenerate_eigenvalues():
    result = run('12.9-1')
    d = result.data
    assert d['omega'][0, 1] == pytest.approx(PI*np.sqrt(5))
    assert d['omega'][1, 0] == pytest.approx(d['omega'][0, 1])
    groups = {row[1] for row in result.tables['중복 고유값'][1]}
    assert '(1,2), (2,1)' in groups and '(1,3), (3,1)' in groups and '(2,3), (3,2)' in groups


def test_12_9_1_sixty_five_has_four_modes():
    """교재 116쪽: λ₁₈=λ₈₁=λ₄₇=λ₇₄=cπ√65 (65=1²+8²=4²+7²)."""
    from engineering_math.topics.chapter12 import degenerate_groups
    m, k = np.meshgrid(np.arange(1, 9), np.arange(1, 9), indexing='ij')
    groups = dict((round(value, 9), pairs) for value, pairs in degenerate_groups(PI*np.sqrt(m**2 + k**2)))
    assert groups[round(PI*np.sqrt(65), 9)] == [(1, 8), (4, 7), (7, 4), (8, 1)]


def test_12_9_2_rectangular_membrane():
    rows = run('12.9-2').tables['모드 계수'][1]
    for m, k, value, *_ in rows:
        expected = .426050/(m**3*k**3) if m % 2 and k % 2 else 0.
        assert value == pytest.approx(expected, abs=2e-6)


def test_12_10_1_drum():
    result = run('12.10-1')
    rows = result.tables['모드 계수'][1]
    roots = jn_zeros(0, 10)
    np.testing.assert_allclose([row[3] for row in rows], 8/(roots**3*jv(1, roots)), rtol=1e-9)
    np.testing.assert_allclose([row[3] for row in rows][:5], [1.10801, -.13978, .04548, -.02099, .01164], atol=2e-5)
    assert rows[0][5] == pytest.approx(4.8097, abs=1e-4)


@pytest.mark.parametrize('title', ['내부', '외부'])
def test_12_11_1_spherical_capacitor(title):
    coefficients = [row[1] for row in run('12.11-1', title).tables['계수'][1]]
    np.testing.assert_allclose(coefficients[:4], [55, 165/2, 0, -385/8], atol=1e-9)


def test_12_11_2_cos_two_phi():
    coefficients = [row[1] for row in run('12.11-2').tables['계수'][1]]
    np.testing.assert_allclose(coefficients, [-1/3, 0, 4/3, 0, 0], atol=1e-12)


def test_12_12_1_semi_infinite_string():
    d = run('12.12-1').data
    delay = d['time'][:, None] - d['x'][None, :]
    exact = np.where((delay >= 0) & (delay < 2*PI), np.sin(np.clip(delay, 0, None)), 0.)
    np.testing.assert_allclose(d['frames'], exact, atol=1e-12)
