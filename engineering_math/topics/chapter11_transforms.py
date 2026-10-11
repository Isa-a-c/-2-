"""11.7~11.10의 연속 적분·변환 및 공식 검산 실험입니다."""
import numpy as np
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate, real_constant, compiled
from engineering_math.core.chapter11 import integral_transform, checked
from .chapter11_series import StaticTopic, line_figure

ROOT_TWO_PI = np.sqrt(2 * np.pi)
ROOT_TWO_OVER_PI = np.sqrt(2 / np.pi)
TRANSFORM_INPUTS = (
    InputSpec('function', '함수 f(x)', 'text', 'exp(-x^2/2)'),
    InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
    InputSpec('extent', '공간 적분 끝점 X', 'float', 8., .1, 100),
    InputSpec('cutoff', '주파수 절단 W (rad/unit)', 'float', 12., .1, 100),
    InputSpec('num_points', '공간 적분 표본 수', 'int', 2001, 501, 8193),
    InputSpec('frequency_points', '주파수 적분 표본 수', 'int', 601, 101, 1601),
)


def transform_data(p, kind):
    """무한 적분을 유한 구간으로 절단한 수치 실험입니다. 존재성 판정은 아닙니다."""
    extent, cutoff = p['extent'], p['cutoff']
    full_line = kind in ('complex', 'integral')
    x = np.linspace(-extent if full_line else 0., extent, p['num_points'])
    expression = parameter_expression(p['function'], p['parameters'])
    values = evaluate(expression, (x,))
    omega = np.linspace(-cutoff if kind == 'complex' else 0., cutoff, p['frequency_points'])
    display_x = np.linspace(x[0], x[-1], 1001)
    reference = evaluate(expression, (display_x,))
    if kind == 'integral':
        cosine = integral_transform(x, values, omega, 'cosine', 1 / np.pi)
        sine = integral_transform(x, values, omega, 'sine', 1 / np.pi)
        restored = (integral_transform(omega, cosine, display_x, 'cosine')
                    + integral_transform(omega, sine, display_x, 'sine'))
        curves = {'A(omega)': cosine, 'B(omega)': sine}
        spectrum = cosine - 1j * sine
    else:
        kernel = 'forward' if kind == 'complex' else kind
        factor = 1 / ROOT_TWO_PI if kind == 'complex' else ROOT_TWO_OVER_PI
        spectrum = integral_transform(x, values, omega, kernel, factor)
        restored = integral_transform(omega, spectrum, display_x, 'inverse' if kind == 'complex' else kind, factor)
        curves = {'Real': spectrum.real, 'Imaginary': spectrum.imag, 'Magnitude': np.abs(spectrum)}
    checked(restored)
    notices = ['공간과 주파수를 유한 구간으로 절단한 사다리꼴 적분입니다. 무한구간 변환의 존재나 정확한 역변환을 보장하지 않습니다.',
               'X·W와 각각의 표본 수를 따로 늘려 절단 오차와 적분 오차를 확인하십시오. 불연속점은 좌우 극한의 평균으로 복원됩니다.']
    if cutoff * (x[1] - x[0]) > np.pi / 4 or extent * (omega[1] - omega[0]) > np.pi / 4:
        notices.append('진동 커널의 표본이 성깁니다. 공간 또는 주파수 표본 수를 늘리십시오.')
    edges = np.abs(values[[0, -1]]) if full_line else np.abs(values[-1:])
    if np.max(edges) > .01 * max(1e-15, np.max(np.abs(values))):
        notices.append('공간 절단 경계에서 함수가 충분히 작지 않습니다. 꼬리 적분의 영향이 클 수 있습니다.')
    tables = {'변환 표본': (['omega', 'real', 'imaginary', 'magnitude'], np.column_stack((omega, spectrum.real, spectrum.imag, np.abs(spectrum)))),
              '복원 표본': (['x', 'f', 'restored real', 'restored imaginary'], np.column_stack((display_x, reference, restored.real, restored.imag)))}
    metrics = {'복원 RMSE': float(np.sqrt(np.mean(np.abs(reference - restored) ** 2))),
               '복원 허수 최대값': float(np.max(np.abs(restored.imag)))}
    checked(list(metrics.values()))
    if kind == 'integral':
        tables['Fourier 적분 계수'] = (['omega', 'A', 'B'], np.column_stack((omega, cosine, sine)))
    data = dict(x=display_x, curves={'f(x)': reference, 'Finite-band reconstruction': restored.real},
                omega=omega, spectrum=spectrum, spectrum_curves=curves, restored=restored,
                integration_x=x, integration_y=values)
    return metrics, tables, data, notices


LIMIT_MODE = '주기 2L→∞ 스펙트럼'


def parse_periods(text):
    try:
        periods = sorted({float(item) for item in text.replace(';', ',').split(',') if item.strip()})
    except ValueError as exc:
        raise ValueError('주기 2L은 쉼표로 구분한 양수로 입력하십시오.') from exc
    if not 1 <= len(periods) <= 4 or any(not .1 <= period <= 200 for period in periods):
        raise ValueError('주기 2L은 0.1~200 범위의 값 1~4개로 입력하십시오.')
    return periods


class IntegralTopic(StaticTopic):
    id = 'fourier_integral'
    title = '11.7 · Fourier 적분'
    description = 'A(ω)·B(ω)를 적분하여 비주기 함수를 복원합니다. 주기를 늘리며 Fourier 급수가 적분으로 바뀌는 과정도 볼 수 있습니다.'
    inputs = TRANSFORM_INPUTS + (
        InputSpec('mode', '실험', 'choice', '적분 표현과 복원', choices=('적분 표현과 복원', LIMIT_MODE)),
        InputSpec('periods', '비교할 주기 2L (쉼표 구분)', 'text', '4, 8, 16', visible_in=(LIMIT_MODE,)),
    )
    examples = {'Gaussian': {}, '사각 펄스': {'function': 'Piecewise((1, abs(x)<1), (0, True))'},
                '비대칭 펄스': {'function': 'Piecewise((1, And(x>0,x<2)), (0, True))'}}
    kind = 'integral'

    def compute(self, params):
        p = validate_params(self.inputs, params)
        metrics, tables, data, notices = transform_data(p, self.kind)
        if p.get('mode') == LIMIT_MODE:
            self.add_period_limit(p, metrics, tables, data, notices)
        return Result(self.id, p, metrics, tables, data, notices)

    @staticmethod
    def add_period_limit(p, metrics, tables, data, notices):
        """주기 2L의 f_L에 대한 aₙ, bₙ을 ωₙ=nπ/L에서 구하고 (π/L)A(ωₙ), (π/L)B(ωₙ)와 비교합니다."""
        from engineering_math.core.quadrature import piecewise_gauss
        expression = parameter_expression(p['function'], p['parameters'])
        f = lambda values: evaluate(expression, (values,))
        series, rows, deviation = [], [], 0.
        for period in parse_periods(p['periods']):
            half = period / 2
            count = int(np.floor(p['cutoff'] * half / np.pi))
            if count < 1:
                raise ValueError(f'주기 {period:g}에서 ωₙ=nπ/L ≤ W인 항이 없습니다. W를 늘리십시오.')
            count = min(count, 2000)
            nodes, weights, _ = piecewise_gauss(f, -half, half, 4096)
            values = f(nodes)
            omega = np.arange(1, count + 1) * np.pi / half
            an = np.cos(np.outer(omega, nodes)) @ (weights * values) / half
            bn = np.sin(np.outer(omega, nodes)) @ (weights * values) / half
            envelope_a = integral_transform(data['integration_x'], data['integration_y'], omega, 'cosine', 1 / np.pi) * np.pi / half
            envelope_b = integral_transform(data['integration_x'], data['integration_y'], omega, 'sine', 1 / np.pi) * np.pi / half
            deviation = max(deviation, float(np.max(np.abs(np.r_[an - envelope_a, bn - envelope_b]))))
            series.append((period, omega, an, bn))
            rows += [[period, n + 1, omega[n], an[n], bn[n], envelope_a[n], envelope_b[n]] for n in range(count)]
        data['period_series'] = series
        tables['주기별 Fourier 계수'] = (['2L', 'n', 'omega_n = nπ/L', 'a_n', 'b_n', '(π/L)A(omega_n)', '(π/L)B(omega_n)'], rows)
        metrics['급수 계수와 (π/L)·적분 계수 최대 차이'] = deviation
        notices.insert(0, '교재 11.7 예제 1(42쪽): 주기 2L의 함수 f_L의 계수는 ωₙ=nπ/L에 놓이며, f가 [−L,L] 밖에서 0이면 aₙ=(π/L)A(ωₙ)입니다. '
                          'L을 두 배로 할 때마다 간격 Δω=π/L은 절반이 되고 진폭도 절반이 되어, L→∞에서 급수가 적분 (5)로 바뀝니다.')
        if deviation > 1e-6:
            notices.append('함수가 가장 짧은 반주기 밖에서 0이 아니면 f_L이 f와 달라 (π/L)A(ωₙ)와 차이가 납니다.')

    def figures(self, result):
        figures = super().figures(result)
        figures['변환 스펙트럼'] = line_figure(result.data['omega'], result.data['spectrum_curves'], 'Continuous spectrum', 'omega (rad/unit)')
        if 'period_series' in result.data:
            from matplotlib.figure import Figure
            series = result.data['period_series']
            figure = Figure(figsize=(8, 2.4 * len(series) + .6), layout='constrained')
            axes = figure.subplots(len(series), 1, sharex=True, squeeze=False)[:, 0]
            omega = result.data['omega']
            amplitude = result.data['spectrum'].real
            for axis, (period, w, an, bn) in zip(axes, series):
                axis.stem(w, an, basefmt='k-')
                axis.plot(omega, amplitude * 2 * np.pi / period, 'r--', lw=1, label='(π/L)A(ω)')
                axis.set(ylabel='a_n', title=f'2L = {period:g}: spacing Δω = π/L = {2 * np.pi / period:.4g}')
                axis.legend(fontsize=8)
                axis.grid(alpha=.3)
            axes[-1].set_xlabel('omega_n = nπ/L')
            figures['주기 2L→∞'] = figure
        return figures


class SineCosineTopic(IntegralTopic):
    id = 'sine_cosine_transform'
    title = '11.8 · Fourier 사인·코사인 변환'
    description = '양의 반축 함수의 변환·복원을 비교합니다. 정변환과 역변환에 모두 √(2/π)를 사용합니다.'
    inputs = TRANSFORM_INPUTS + (InputSpec('mode', '변환 종류', 'choice', '코사인', choices=('코사인', '사인')),)
    examples = {'코사인 Gaussian': {}, '사인 지수감쇠': {'mode': '사인', 'function': 'exp(-x)'},
                '사인 x Gaussian': {'mode': '사인', 'function': 'x*exp(-x^2/2)'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        metrics, tables, data, notices = transform_data(p, 'cosine' if p['mode'] == '코사인' else 'sine')
        self.check_derivative_rule(p, metrics, tables, data, notices)
        return Result(self.id, p, metrics, tables, data, notices)

    @staticmethod
    def check_derivative_rule(p, metrics, tables, data, notices):
        """정리 1(52쪽): F_c{f′}=w·F_s{f}−√(2/π)f(0), F_s{f′}=−w·F_c{f}를 같은 격자의 적분으로 확인합니다."""
        import sympy as sp
        expression = parameter_expression(p['function'], p['parameters'])
        derivative = sp.diff(expression, sp.Symbol('x', real=True))
        x, omega = data['integration_x'], data['omega']
        if derivative.has(sp.DiracDelta):
            notices.append('정리 1 검산 생략: f가 연속이 아니어서(도약) f′에 델타 함수가 생깁니다. 정리 1은 연속인 f를 가정합니다.')
            return
        try:
            slope = evaluate(derivative, (x,))
        except ValueError:
            notices.append('정리 1 검산 생략: f′을 적분 구간에서 유한하게 평가할 수 없습니다.')
            return
        values = data['integration_y']
        cosine = integral_transform(x, values, omega, 'cosine', ROOT_TWO_OVER_PI)
        sine = integral_transform(x, values, omega, 'sine', ROOT_TWO_OVER_PI)
        cosine_slope = integral_transform(x, slope, omega, 'cosine', ROOT_TWO_OVER_PI)
        sine_slope = integral_transform(x, slope, omega, 'sine', ROOT_TWO_OVER_PI)
        rule_a = omega * sine - ROOT_TWO_OVER_PI * values[0]
        rule_b = -omega * cosine
        metrics['정리 1 (4a)·(4b) 최대 차이'] = float(max(np.max(np.abs(cosine_slope - rule_a)), np.max(np.abs(sine_slope - rule_b))))
        tables['정리 1 검산'] = (['omega', 'F_c{f′}', 'w·F_s{f} − √(2/π)f(0)', 'F_s{f′}', '−w·F_c{f}'],
                               np.column_stack((omega, cosine_slope, rule_a, sine_slope, rule_b)))
        notices.append('정리 1(52쪽): 미분이 변환 공간에서 w를 곱하는 대수 연산으로 바뀝니다. "정리 1 검산" 표에서 양변을 비교하십시오. '
                       'f가 X에서 충분히 작아야 부분적분의 경계항이 사라집니다.')


class ContinuousTransformTopic(IntegralTopic):
    id = 'continuous_fourier'
    title = '11.9 · 연속 Fourier 변환·FFT 비교'
    description = '실수 입력의 복소 스펙트럼과 역적분을 계산합니다. FFT에는 표본 간격과 원점 위상을 보정합니다.'
    kind = 'complex'
    MODES = ('변환과 역변환', '합성곱 정리')
    inputs = TRANSFORM_INPUTS + (
        InputSpec('mode', '실험', 'choice', MODES[0], choices=MODES),
        InputSpec('second', '두 번째 함수 g(x)', 'text', 'exp(-x^2/2)', visible_in=(MODES[1],)),
    )
    examples = {'Gaussian 자기 변환': {}, '이동 Gaussian': {'function': 'exp(-(x-2)^2/2)'},
                '양측 지수': {'function': 'exp(-abs(x))'},
                '합성곱 정리: 사각 펄스 두 개': {'mode': MODES[1], 'function': 'Heaviside(x+1,1/2)-Heaviside(x-1,1/2)',
                                        'second': 'Heaviside(x+1,1/2)-Heaviside(x-1,1/2)', 'num_points': 8001}}

    def compute(self, params):
        result = super().compute(params)
        p, d = result.params, result.data
        if p['mode'] == self.MODES[1]:
            self.add_convolution(result)
        count = p['num_points']
        x = np.linspace(-p['extent'], p['extent'], count, endpoint=False)
        values = evaluate(parameter_expression(p['function'], p['parameters']), (x,))
        dx = x[1] - x[0]
        omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(count, d=dx))
        spectrum = np.fft.fftshift(np.fft.fft(values)) * dx / ROOT_TWO_PI * np.exp(-1j * omega * x[0])
        selected = np.flatnonzero(np.abs(omega) <= p['cutoff'])
        # 동등 주파수의 비교 표본은 최대 401개로 제한합니다.
        selected = selected[::max(1, int(np.ceil(len(selected) / 401)))]
        omega, spectrum = omega[selected], spectrum[selected]
        direct = integral_transform(d['integration_x'], d['integration_y'], omega, 'forward', 1 / ROOT_TWO_PI)
        d.update(fft_omega=omega, fft_spectrum=spectrum, direct_at_bins=direct)
        result.metrics['동일 ω에서 FFT·적분 최대 차이'] = float(np.max(np.abs(spectrum - direct)))
        result.tables['FFT 비교'] = (['omega', 'FFT real', 'FFT imag', 'integral real', 'integral imag'],
                                    np.column_stack((omega, spectrum.real, spectrum.imag, direct.real, direct.imag)))
        result.notices.append('FFT는 끝점 중복 없는 격자의 주기적 표본을 사용합니다. 적분과의 차이는 절단·구적 차이를 포함합니다. 데이터 벡터의 DFT·FFT 분해는 "11.9 · DFT·FFT" 화면에서 볼 수 있습니다.')
        return result

    @staticmethod
    def add_convolution(result):
        """정리 4: F(f*g)=√(2π)F(f)F(g), 식 (13): (f*g)(x)=∫f̂(w)ĝ(w)e^(iwx)dw."""
        p, d = result.params, result.data
        x, f = d['integration_x'], d['integration_y']
        if len(x) % 2 == 0:
            raise ValueError('합성곱 정리 실험은 원점이 격자점이 되도록 공간 표본 수를 홀수로 입력하십시오.')
        g = evaluate(parameter_expression(p['second'], p['parameters']), (x,))
        # 같은 균일 격자에서 사다리꼴 가중치를 곱한 이산 합성곱이며 'same'은 [−X, X] 구간을 돌려줍니다.
        from scipy.signal import fftconvolve
        from engineering_math.core.chapter11 import trapezoid_weights
        direct = fftconvolve(f * trapezoid_weights(x), g, mode='same')
        omega = d['omega']
        f_hat = d['spectrum']
        g_hat = integral_transform(x, g, omega, 'forward', 1 / ROOT_TWO_PI)
        via_theorem = integral_transform(omega, f_hat * g_hat, d['x'], 'inverse')
        direct_display = np.interp(d['x'], x, direct)
        checked(via_theorem)
        d.update(convolution_x=x, second=g, convolution=direct, theorem=via_theorem,
                 convolution_display=direct_display, second_spectrum=g_hat)
        result.metrics['합성곱 직접·정리 최대 차이'] = float(np.max(np.abs(direct_display - via_theorem)))
        result.tables['합성곱 정리'] = (['x', '(f*g) direct', '∫f̂ĝe^(iwx)dw real', 'imag'],
                                        np.column_stack((d['x'], direct_display, via_theorem.real, via_theorem.imag)))
        result.notices.insert(0, '정리 4(60쪽): F(f*g)=√(2π)F(f)F(g). 직접 합성곱 ∫f(p)g(x−p)dp와 식 (13)의 역적분을 비교합니다. '
                                 '차이에는 공간·주파수 절단(X, W)과 구적 오차가 포함됩니다.')

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['FFT·연속 적분 비교'] = line_figure(d['fft_omega'], {'FFT magnitude': np.abs(d['fft_spectrum']),
                          'Quadrature magnitude': np.abs(d['direct_at_bins'])}, 'Same angular-frequency bins', 'omega')
        if 'theorem' in d:
            figures['합성곱 정리'] = line_figure(d['x'], {
                'f(x)': d['curves']['f(x)'], 'g(x)': np.interp(d['x'], d['convolution_x'], d['second']),
                '(f*g)(x) direct': d['convolution_display'],
                '∫ f̂ ĝ e^(iwx) dw': d['theorem'].real}, 'Convolution theorem')
        return figures


# ── 11.9 후반: 이산 Fourier 변환, Fourier 행렬, FFT 분해 ──────────────
def parse_data(text):
    items = [item.strip() for item in text.replace(';', ',').split(',') if item.strip()]
    if not 2 <= len(items) <= 64:
        raise ValueError('데이터는 쉼표로 구분한 2~64개의 실수로 입력하십시오.')
    return np.array([real_constant(item) for item in items])


def fft_steps(values):
    """식 (22)의 짝·홀 분해를 재귀적으로 기록합니다. N은 2의 거듭제곱이어야 합니다."""
    rows = []

    def transform(vector, label):
        size = len(vector)
        if size == 1:
            return vector.astype(complex)
        even = transform(vector[0::2], label + '·ev')
        odd = transform(vector[1::2], label + '·od')
        twiddle = np.exp(-2j * np.pi * np.arange(size // 2) / size) * odd
        result = np.r_[even + twiddle, even - twiddle]
        rows.append([size, label, format_vector(vector), format_vector(even), format_vector(odd), format_vector(result)])
        return result

    return transform(np.asarray(values, dtype=complex), 'f'), rows


def format_complex(value, digits=6):
    value = complex(value)
    real = 0. if abs(value.real) < 1e-12 else value.real
    imag = 0. if abs(value.imag) < 1e-12 else value.imag
    if imag == 0:
        return f'{real:.{digits}g}'
    if real == 0:
        return f'{imag:.{digits}g}i'
    return f'{real:.{digits}g}{imag:+.{digits}g}i'


def format_vector(values):
    return '[' + ', '.join(format_complex(value) for value in values) + ']'


class DiscreteFourierTopic:
    id = 'dft'
    title = '11.9 · 이산 Fourier 변환과 FFT'
    description = 'N개 표본의 DFT f̂=F_N f를 Fourier 행렬로 계산하고 FFT의 짝·홀 분해와 비교합니다.'
    supports_animation = False
    MODES = ('데이터 직접 입력', '함수 표본 x_k = 2πk/N')
    inputs = (InputSpec('mode', '입력 방식', 'choice', MODES[0], choices=MODES),
              InputSpec('data', '표본 값 f_0, f_1, … (쉼표 구분)', 'text', '0, 1, 4, 9', visible_in=(MODES[0],)),
              InputSpec('function', '주기 2π 함수 f(x)', 'text', 'x^2', visible_in=(MODES[1],)),
              InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', '', visible_in=(MODES[1],)),
              InputSpec('N', '표본 수 N', 'int', 8, 2, 64, visible_in=(MODES[1],)))
    examples = {'사각파 8표본': {'data': '1,1,1,1,-1,-1,-1,-1'},
                '함수 표본 cos 3x (N=16)': {'mode': MODES[1], 'function': 'cos(3*x)', 'N': 16},
                'Aliasing: cos 7x를 N=8로 표본화': {'mode': MODES[1], 'function': 'cos(7*x)', 'N': 8}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        if p['mode'] == self.MODES[0]:
            values = parse_data(p['data'])
        else:
            grid = 2 * np.pi * np.arange(p['N']) / p['N']
            values = evaluate(parameter_expression(p['function'], p['parameters']), (grid,))
        size = len(values)
        indices = np.arange(size)
        matrix = np.exp(-2j * np.pi * np.outer(indices, indices) / size)
        spectrum = matrix @ values
        restored = np.conj(matrix) @ spectrum / size
        fast = np.fft.fft(values)
        power_of_two = size & (size - 1) == 0
        notices = ['식 (18): f̂ₙ = Σₖ fₖ e^(−inxₖ), xₖ=2πk/N. Fourier 행렬은 e_nk = wⁿᵏ, w = e^(−2πi/N)입니다.',
                   '식 (21): F_N⁻¹ = (1/N)·conj(F_N)로 원래 신호를 복원합니다. cₙ=f̂ₙ/N은 복소 삼각 보간 다항식의 계수입니다.',
                   f'직접 계산은 약 N²={size ** 2}번, FFT는 약 N log₂N={size * np.log2(size):.0f}번의 곱셈이 필요합니다.']
        tables = {'스펙트럼': (['n', 'f̂ real', 'f̂ imag', '|f̂|', 'c_n = f̂/N'],
                              [[n, spectrum[n].real, spectrum[n].imag, abs(spectrum[n]), format_complex(spectrum[n] / size)] for n in indices]),
                  '역변환 검산': (['k', 'f_k', 'restored'], [[k, values[k], restored[k].real] for k in indices])}
        if size <= 16:
            tables['Fourier 행렬 F_N'] = ([str(k) for k in indices], [[format_complex(v, 4) for v in row] for row in matrix])
        if power_of_two:
            _, steps = fft_steps(values)
            tables['FFT 분해 단계'] = (['크기', '부분 신호', '입력', 'f̂_ev', 'f̂_od', '결합 결과 (식 22a·b)'], steps)
        else:
            notices.append('N이 2의 거듭제곱이 아니므로 짝·홀 분해 표를 생략합니다. 행렬 계산과 NumPy FFT는 그대로 비교합니다.')
        if p['mode'] == self.MODES[1]:
            notices.append(f'aliasing을 피하려면 신호에 포함된 정수 진동수가 N/2={size / 2:g}보다 충분히 작아야 합니다(교재 63쪽).')
        metrics = {'행렬·FFT 최대 차이': float(np.max(np.abs(spectrum - fast))),
                   '역변환 최대 오차': float(np.max(np.abs(restored - values)))}
        return Result(self.id, p, metrics, tables, dict(values=values, spectrum=spectrum, matrix=matrix, size=size), notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        d = result.data
        size = d['size']
        figure = Figure(figsize=(8, 6), layout='constrained')
        axes = figure.subplots(2, 1)
        grid = 2 * np.pi * np.arange(size) / size
        # 실수 데이터를 매끄럽게 잇기 위해 −N/2…N/2 대칭 진동수로 보간합니다.
        harmonics = np.fft.fftfreq(size, 1 / size)
        fine = np.linspace(0, 2 * np.pi, 400)
        interpolant = (np.exp(1j * np.outer(fine, harmonics)) @ (d['spectrum'] / size)).real
        axes[0].plot(fine, interpolant, label='Trigonometric interpolation')
        axes[0].plot(grid, d['values'], 'ko', label='Samples f_k')
        axes[0].set(xlabel='x', ylabel='f', title='Samples and trigonometric interpolation (symmetric frequencies)')
        axes[0].legend()
        axes[1].stem(np.arange(size), np.abs(d['spectrum']))
        axes[1].set(xlabel='n', ylabel='|f̂_n|', title='DFT magnitude spectrum')
        for axis in axes:
            axis.grid(alpha=.3)
        return {'표본과 스펙트럼': figure}


# ── 11.10: 교재 표 I·II·III의 34개 공식을 데이터로 관리하고 수치 적분과 검산합니다 ──
from dataclasses import dataclass
from math import factorial
from scipy.special import gamma as gamma_function


def sinc(t):
    """sin(t)/t, t=0에서 1 (NumPy의 정규화 sinc를 변환)."""
    return np.sinc(np.asarray(t) / np.pi)


def step(t):
    """단위 계단 u(t), t=0에서 ½."""
    return np.where(t > 0, 1., np.where(t < 0, 0., .5))


def power_term(omega, a, n, part):
    value = (a + 1j * omega) ** (n + 1)
    return ROOT_TWO_OVER_PI * factorial(n) / (a * a + omega * omega) ** (n + 1) * (value.real if part == 'real' else value.imag)


def positive(omega, values):
    """ω>0에서만 정의되는 공식은 ω≤0을 NaN으로 둡니다(그래프에서 비움)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(omega > 0, values, np.nan)


@dataclass(frozen=True)
class TransformPair:
    key: str             # '표 I-3'처럼 교재 표와 번호
    kind: str            # cosine · sine · complex
    function: str        # 원함수 f(x)의 실수부 (매개변수 a, b, c, n)
    display: str         # 표에 보이는 원함수
    formula: str         # 교재의 변환 공식
    exact: object        # (ω, a, b, c, n) → 공식 값
    condition: str = ''
    unsupported: str = ''   # 절단 수치 적분으로 검산하지 않는 이유 (없으면 검산)
    imaginary: str = ''     # 복소수 원함수의 허수부
    needs: str = ''         # 'b>0', 'c>b' 등 매개변수 조건

    @property
    def label(self):
        return f'{self.key} · {self.display}'


R2P, S2P = ROOT_TWO_OVER_PI, np.sqrt(np.pi / 2)
TABLE = (
    # 표 I. Fourier 코사인 변환 (66쪽)
    TransformPair('표 I-1', 'cosine', 'Heaviside(a-x,1/2)', '1 (0<x<a)', '√(2/π)·sin(aw)/w',
                  lambda w, a, b, c, n: R2P * a * sinc(a * w), 'a>0'),
    TransformPair('표 I-2', 'cosine', 'x^(a-1)', 'x^(a−1) (0<a<1)', '√(2/π)·Γ(a)/w^a·cos(aπ/2)',
                  lambda w, a, b, c, n: positive(w, R2P * gamma_function(a) / np.abs(w) ** a * np.cos(a * np.pi / 2)),
                  '0<a<1', '원점 특이점과 느린 감소 때문에 절단 적분으로 검산하지 않습니다.', needs='0<a<1'),
    TransformPair('표 I-3', 'cosine', 'exp(-a*x)', 'e^(−ax)', '√(2/π)·a/(a²+w²)',
                  lambda w, a, b, c, n: R2P * a / (a * a + w * w), 'a>0'),
    TransformPair('표 I-4', 'cosine', 'exp(-x^2/2)', 'e^(−x²/2)', 'e^(−w²/2)', lambda w, a, b, c, n: np.exp(-w * w / 2)),
    TransformPair('표 I-5', 'cosine', 'exp(-a*x^2)', 'e^(−ax²)', 'e^(−w²/(4a))/√(2a)',
                  lambda w, a, b, c, n: np.exp(-w * w / (4 * a)) / np.sqrt(2 * a), 'a>0'),
    TransformPair('표 I-6', 'cosine', 'x^n*exp(-a*x)', 'xⁿe^(−ax)', '√(2/π)·n!/(a²+w²)ⁿ⁺¹·Re(a+iw)ⁿ⁺¹',
                  lambda w, a, b, c, n: power_term(w, a, n, 'real'), 'a>0, n=0,1,2,…'),
    TransformPair('표 I-7', 'cosine', 'cos(x)*Heaviside(a-x,1/2)', 'cos x (0<x<a)',
                  '(1/√(2π))·[sin a(1−w)/(1−w) + sin a(1+w)/(1+w)]',
                  lambda w, a, b, c, n: a * (sinc(a * (1 - w)) + sinc(a * (1 + w))) / ROOT_TWO_PI, 'a>0'),
    TransformPair('표 I-8', 'cosine', 'cos(a*x^2)', 'cos(ax²)', '(1/√(2a))·cos(w²/(4a) − π/4)',
                  lambda w, a, b, c, n: np.cos(w * w / (4 * a) - np.pi / 4) / np.sqrt(2 * a), 'a>0',
                  '절대적분 가능하지 않은 진동 함수라 절단 적분이 수렴하지 않습니다.'),
    TransformPair('표 I-9', 'cosine', 'sin(a*x^2)', 'sin(ax²)', '(1/√(2a))·cos(w²/(4a) + π/4)',
                  lambda w, a, b, c, n: np.cos(w * w / (4 * a) + np.pi / 4) / np.sqrt(2 * a), 'a>0',
                  '절대적분 가능하지 않은 진동 함수라 절단 적분이 수렴하지 않습니다.'),
    TransformPair('표 I-10', 'cosine', 'a*sinc(a*x)', 'sin(ax)/x', '√(π/2)·(1 − u(w−a))',
                  lambda w, a, b, c, n: S2P * (1 - step(w - a)), 'a>0',
                  '조건부 수렴 적분이라 절단 적분의 수렴이 매우 느립니다.'),
    TransformPair('표 I-11', 'cosine', 'exp(-x)*sinc(x)', 'e^(−x)·sin x / x', '(1/√(2π))·arctan(2/w²)',
                  lambda w, a, b, c, n: np.arctan2(2., w * w) / ROOT_TWO_PI),
    TransformPair('표 I-12', 'cosine', 'besselj(0,a*x)', 'J₀(ax)', '√(2/π)·(1 − u(w−a))/√(a²−w²)',
                  lambda w, a, b, c, n: np.where(np.abs(w) < a, R2P / np.sqrt(np.abs(a * a - w * w)) , 0.), 'a>0',
                  'J₀(ax)는 1/√x로 느리게 감소하여 절단 적분이 수렴하지 않습니다.'),
    # 표 II. Fourier 사인 변환 (67쪽)
    TransformPair('표 II-1', 'sine', 'Heaviside(a-x,1/2)', '1 (0<x<a)', '√(2/π)·(1 − cos aw)/w',
                  lambda w, a, b, c, n: R2P * a * np.sin(a * w / 2) * sinc(a * w / 2), 'a>0'),
    TransformPair('표 II-2', 'sine', '1/sqrt(x)', '1/√x', '1/√w',
                  lambda w, a, b, c, n: positive(w, 1 / np.sqrt(np.abs(w))), '',
                  '원점 특이점과 느린 감소 때문에 절단 적분으로 검산하지 않습니다.'),
    TransformPair('표 II-3', 'sine', 'x^(-3/2)', '1/x^(3/2)', '2√w',
                  lambda w, a, b, c, n: positive(w, 2 * np.sqrt(np.abs(w))), '',
                  '원점 특이점 때문에 절단 적분으로 검산하지 않습니다.'),
    TransformPair('표 II-4', 'sine', 'x^(a-1)', 'x^(a−1) (0<a<1)', '√(2/π)·Γ(a)/w^a·sin(aπ/2)',
                  lambda w, a, b, c, n: positive(w, R2P * gamma_function(a) / np.abs(w) ** a * np.sin(a * np.pi / 2)),
                  '0<a<1', '원점 특이점과 느린 감소 때문에 절단 적분으로 검산하지 않습니다.', needs='0<a<1'),
    TransformPair('표 II-5', 'sine', 'exp(-a*x)', 'e^(−ax)', '√(2/π)·w/(a²+w²)',
                  lambda w, a, b, c, n: R2P * w / (a * a + w * w), 'a>0'),
    TransformPair('표 II-6', 'sine', 'exp(-a*x)/x', 'e^(−ax)/x', '√(2/π)·arctan(w/a)',
                  lambda w, a, b, c, n: R2P * np.arctan(w / a), 'a>0',
                  '원점에서 함수가 발산하므로 표본 적분으로 검산하지 않습니다(피적분함수는 유한).'),
    TransformPair('표 II-7', 'sine', 'x^n*exp(-a*x)', 'xⁿe^(−ax)', '√(2/π)·n!/(a²+w²)ⁿ⁺¹·Im(a+iw)ⁿ⁺¹',
                  lambda w, a, b, c, n: power_term(w, a, n, 'imag'), 'a>0, n=0,1,2,…'),
    TransformPair('표 II-8', 'sine', 'x*exp(-x^2/2)', 'x·e^(−x²/2)', 'w·e^(−w²/2)', lambda w, a, b, c, n: w * np.exp(-w * w / 2)),
    TransformPair('표 II-9', 'sine', 'x*exp(-a*x^2)', 'x·e^(−ax²)', 'w/(2a)^(3/2)·e^(−w²/(4a))',
                  lambda w, a, b, c, n: w / (2 * a) ** 1.5 * np.exp(-w * w / (4 * a)), 'a>0'),
    TransformPair('표 II-10', 'sine', 'sin(x)*Heaviside(a-x,1/2)', 'sin x (0<x<a)',
                  '(1/√(2π))·[sin a(1−w)/(1−w) − sin a(1+w)/(1+w)]',
                  lambda w, a, b, c, n: a * (sinc(a * (1 - w)) - sinc(a * (1 + w))) / ROOT_TWO_PI, 'a>0'),
    TransformPair('표 II-11', 'sine', 'cos(a*x)/x', 'cos(ax)/x', '√(π/2)·u(w−a)',
                  lambda w, a, b, c, n: S2P * step(w - a), 'a>0',
                  '원점에서 발산하고 조건부 수렴하므로 절단 적분으로 검산하지 않습니다.'),
    TransformPair('표 II-12', 'sine', 'atan(2*a/x)', 'arctan(2a/x)', '√(2π)·sinh(aw)/w·e^(−aw)',
                  # sinh(aw)e^(−aw)/w = (1−e^(−2aw))/(2w), w→0에서 a입니다.
                  lambda w, a, b, c, n: ROOT_TWO_PI * np.where(np.abs(w) > 1e-12,
                      -np.expm1(-2 * a * np.abs(w)) / (2 * np.where(np.abs(w) > 1e-12, np.abs(w), 1.)), a) * np.sign(w + (w == 0)),
                  'a>0 (교재 표기 보정: sin(aw)→sinh(aw))',
                  '1/x로 느리게 감소하여 절단 적분의 수렴이 매우 느립니다.'),
    # 표 III. Fourier 변환 (68쪽)
    TransformPair('표 III-1', 'complex', 'Heaviside(b-x,1/2)*Heaviside(x+b,1/2)', '1 (−b<x<b)', '√(2/π)·sin(bw)/w',
                  lambda w, a, b, c, n: R2P * b * sinc(b * w), 'b>0', needs='b>0'),
    TransformPair('표 III-2', 'complex', 'Heaviside(x-b,1/2)*Heaviside(c-x,1/2)', '1 (b<x<c)',
                  '(e^(−ibw) − e^(−icw))/(iw√(2π))',
                  lambda w, a, b, c, n: (c - b) * sinc(w * (c - b) / 2) * np.exp(-1j * w * (b + c) / 2) / ROOT_TWO_PI,
                  'b<c', needs='c>b'),
    TransformPair('표 III-3', 'complex', '1/(x^2+a^2)', '1/(x²+a²)', '√(π/2)·e^(−a|w|)/a',
                  lambda w, a, b, c, n: S2P * np.exp(-a * np.abs(w)) / a, 'a>0'),
    TransformPair('표 III-4', 'complex', 'Piecewise((x, And(x>0, x<b)), (2*b-x, And(x>=b, x<2*b)), (0, True))',
                  '삼각형: x (0<x<b), 2b−x (b<x<2b)', '(−1 + 2e^(−ibw) − e^(−2ibw))/(√(2π)·w²)',
                  lambda w, a, b, c, n: b * b * sinc(b * w / 2) ** 2 * np.exp(-1j * b * w) / ROOT_TWO_PI,
                  'b>0 (교재 표기 보정: 2x−b→2b−x, 2e^(ibw)→2e^(−ibw))', needs='b>0'),
    TransformPair('표 III-5', 'complex', 'Heaviside(x,1/2)*exp(-a*x)', 'e^(−ax) (x>0)', '1/(√(2π)(a+iw))',
                  lambda w, a, b, c, n: 1 / (ROOT_TWO_PI * (a + 1j * w)), 'a>0'),
    TransformPair('표 III-6', 'complex', 'Heaviside(x-b,1/2)*Heaviside(c-x,1/2)*exp(a*x)', 'e^(ax) (b<x<c)',
                  '(e^((a−iw)c) − e^((a−iw)b))/(√(2π)(a−iw))',
                  lambda w, a, b, c, n: (np.exp((a - 1j * w) * c) - np.exp((a - 1j * w) * b)) / (ROOT_TWO_PI * (a - 1j * w)),
                  'b<c', needs='c>b'),
    TransformPair('표 III-7', 'complex', 'cos(a*x)*Heaviside(b-abs(x),1/2)', 'e^(iax) (−b<x<b)', '√(2/π)·sin b(w−a)/(w−a)',
                  lambda w, a, b, c, n: R2P * b * sinc(b * (w - a)), 'b>0',
                  imaginary='sin(a*x)*Heaviside(b-abs(x),1/2)', needs='b>0'),
    TransformPair('표 III-8', 'complex', 'cos(a*x)*Heaviside(x-b,1/2)*Heaviside(c-x,1/2)', 'e^(iax) (b<x<c)',
                  '(i/√(2π))·(e^(ib(a−w)) − e^(ic(a−w)))/(a−w)',
                  lambda w, a, b, c, n: (c - b) * sinc((c - b) * (a - w) / 2) * np.exp(1j * (a - w) * (b + c) / 2) / ROOT_TWO_PI,
                  'b<c', imaginary='sin(a*x)*Heaviside(x-b,1/2)*Heaviside(c-x,1/2)', needs='c>b'),
    TransformPair('표 III-9', 'complex', 'exp(-a*x^2)', 'e^(−ax²)', 'e^(−w²/(4a))/√(2a)',
                  lambda w, a, b, c, n: np.exp(-w * w / (4 * a)) / np.sqrt(2 * a), 'a>0'),
    TransformPair('표 III-10', 'complex', 'a*sinc(a*x)', 'sin(ax)/x', '√(π/2) (|w|<a), 0 (|w|>a)',
                  lambda w, a, b, c, n: S2P * (1 - step(np.abs(w) - a)), 'a>0',
                  '조건부 수렴 적분이라 절단 적분의 수렴이 매우 느립니다.'),
)
PAIRS = {pair.label: pair for pair in TABLE}
DEFAULT_PAIR = TABLE[-2].label
# 이전 버전(11개 공식)의 선택 이름을 새 표 항목으로 연결합니다.
LEGACY_PAIRS = {
    '코사인 · 지수': '표 I-3', '코사인 · Gaussian': '표 I-5', '코사인 · 사각 펄스': '표 I-1',
    '사인 · 지수': '표 II-5', '사인 · x Gaussian': '표 II-9', '사인 · 사각 펄스': '표 II-1',
    '복소 · Gaussian': '표 III-9', '복소 · 사각 펄스': '표 III-1', '복소 · 유리함수': '표 III-3',
    '복소 · 단측 지수': '표 III-5', '복소 · 이동 펄스': '표 III-2',
}


def pair_by_key(key):
    return next(pair for pair in TABLE if pair.key == key)


class TransformTableTopic(IntegralTopic):
    id = 'transform_table'
    title = '11.10 · 변환공식 표와 수치 검산'
    description = '교재 표 I(코사인)·II(사인)·III(Fourier 변환)의 34개 공식을 선택하여 매개변수를 바꾸고 수치 적분과 비교합니다.'
    inputs = (
        InputSpec('pair', '변환 공식', 'choice', DEFAULT_PAIR, choices=tuple(PAIRS)),
        InputSpec('a', '매개변수 a', 'float', 1., .05, 20),
        InputSpec('b', '매개변수 b', 'float', 1., -20, 20),
        InputSpec('c', '매개변수 c', 'float', 2., -20, 20),
        InputSpec('n', '정수 n (표 I-6·II-7)', 'int', 2, 0, 8),
    ) + TRANSFORM_INPUTS[2:]
    examples = {'Gaussian 검산': {}, '사각 펄스와 sinc': {'pair': pair_by_key('표 III-1').label},
                '이동 펄스의 위상': {'pair': pair_by_key('표 III-2').label, 'b': .5, 'c': 2.5},
                '사인 지수': {'pair': pair_by_key('표 II-5').label},
                '삼각형 펄스 (표 III-4 보정)': {'pair': pair_by_key('표 III-4').label},
                '복소 원함수 e^(iax)': {'pair': pair_by_key('표 III-7').label, 'a': 2.},
                'J₀(ax): 공식만 표시': {'pair': pair_by_key('표 I-12').label}}

    @staticmethod
    def migrate(params):
        """이전 설정의 공식 이름을 새 이름으로 바꿉니다."""
        params = dict(params)
        if params.get('pair') in LEGACY_PAIRS:
            params['pair'] = pair_by_key(LEGACY_PAIRS[params['pair']]).label
        return params

    def compute(self, params):
        # 이전 버전의 공식 이름으로 직접 호출해도(노트북 등) 같은 공식을 계산합니다.
        p = validate_params(self.inputs, self.migrate(params))
        pair = PAIRS[p['pair']]
        a, b, c, n = p['a'], p['b'], p['c'], p['n']
        if pair.needs == 'b>0' and b <= 0:
            raise ValueError('이 공식은 b>0이어야 합니다.')
        if pair.needs == 'c>b' and c <= b:
            raise ValueError('이 공식은 b<c이어야 합니다.')
        if pair.needs == '0<a<1' and not 0 < a < 1:
            raise ValueError('이 공식은 0<a<1이어야 합니다.')
        parameters = f'a={a}; b={b}; c={c}; n={n}'
        calculation = {key: p[key] for key in ('extent', 'cutoff', 'num_points', 'frequency_points')}
        notices = [f'선택 공식: {pair.formula}. 조건: {pair.condition or "없음"}. 교재 11.10절 66~68쪽 표 I~III.']
        if pair.unsupported:
            omega = np.linspace(-p['cutoff'] if pair.kind == 'complex' else 0., p['cutoff'], p['frequency_points'])
            exact = np.asarray(pair.exact(omega, a, b, c, n), dtype=complex)
            start = -p['extent'] if pair.kind == 'complex' else 1e-3 * p['extent']
            x = np.linspace(start, p['extent'], 1001)
            with np.errstate(all='ignore'):
                values = np.asarray(compiled(parameter_expression(pair.function, parameters))(x), dtype=float)
            values = np.where(np.isfinite(values), values, np.nan)
            data = dict(x=x, curves={'f(x)': values}, omega=omega, spectrum=exact, exact=exact,
                        spectrum_curves={'Formula real': exact.real, 'Formula imaginary': exact.imag})
            metrics = {'공식 최댓값 (유한한 ω)': float(np.nanmax(np.abs(exact)))}
            tables = {'공식 값': (['omega', 'real', 'imaginary'], np.column_stack((omega, exact.real, exact.imag)))}
            notices.append(f'수치 검산 생략: {pair.unsupported} 공식 곡선만 표시합니다. '
                           '독립 검산은 tests/test_chapter11.py의 무한구간 진동 적분(QAWF)으로 수행합니다.')
        else:
            calculation.update(function=pair.function, parameters=parameters)
            metrics, tables, data, more = transform_data(calculation, pair.kind)
            notices += more
            if pair.imaginary:
                # 복소수 원함수는 선형성으로 F(f)=F(Re f)+i·F(Im f)를 계산합니다.
                calculation.update(function=pair.imaginary)
                _, _, imaginary, _ = transform_data(calculation, pair.kind)
                data['spectrum'] = data['spectrum'] + 1j * imaginary['spectrum']
                notices.append('원함수가 복소수이므로 실수부·허수부를 따로 변환하여 F(f)=F(Re f)+i·F(Im f)로 합칩니다.')
            exact = np.asarray(pair.exact(data['omega'], a, b, c, n), dtype=complex)
            data['spectrum_curves'] = {'Quadrature real': data['spectrum'].real, 'Formula real': exact.real,
                                       'Quadrature imaginary': data['spectrum'].imag, 'Formula imaginary': exact.imag}
            data['exact'] = exact
            metrics['공식·적분 최대 차이'] = float(np.max(np.abs(data['spectrum'] - exact)))
            tables['공식 검산'] = (['omega', 'exact real', 'exact imaginary', 'absolute error'],
                                  np.column_stack((data['omega'], exact.real, exact.imag, np.abs(data['spectrum'] - exact))))
            notices.append('공식과 절단 적분의 차이에는 공간·주파수 절단과 구적 오차가 포함됩니다. 버그와 같은 뜻이 아닙니다.')
        if pair.key == '표 II-12':
            notices.append('교재 67쪽 표 II-12는 √(2π)·sin(aw)/w·e^(−aw)로 인쇄되어 있지만, ∫₀^∞arctan(b/x)sin(wx)dx=π(1−e^(−bw))/(2w)에서 '
                           '√(2π)·sinh(aw)/w·e^(−aw)가 맞습니다. 독립 수치 적분도 sinh 공식과 일치합니다.')
        if pair.key == '표 III-4':
            notices.append('교재 68쪽 표 III-4는 원함수를 "2x−b (b<x<2b)", 변환의 가운데 항을 "2e^(ibw)"로 인쇄했지만, '
                           '두 식은 서로 맞지 않습니다. 삼각형 펄스 2b−x와 2e^(−ibw)로 보정한 쌍이 수치 적분과 일치합니다.')
        tables['공식 표'] = (['번호', '종류', 'f(x)', '변환 공식', '조건', '수치 검산'],
                           [[item.key, item.kind, item.display, item.formula, item.condition,
                             '생략' if item.unsupported else '가능'] for item in TABLE])
        return Result(self.id, p, metrics, tables, data, notices)
