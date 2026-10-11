"""푸리에 주제: 입력 명세, 계산, 시각화를 분리합니다."""
import numpy as np
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import real_constant, parameter_expression
from engineering_math.core.fourier import FourierSession, calculate_fourier_sums, calculate_errors
from engineering_math.core.plotting import make_comparison_figure, make_error_figure, make_spectrum_figure
from engineering_math.core.analysis import prepare_analysis, make_analysis_figure

MODES = ('기본 비교', 'Gibbs 확대', '진폭·위상', 'DFT·FFT·급수 비교', 'Aliasing 실험', '적분 안정성', 'N별 오차 곡선')


def parse_orders(text):
    if not text.strip():
        return []
    try:
        orders = sorted(set(int(item.strip()) for item in text.split(',')))
    except ValueError as exc:
        raise ValueError('비교 차수는 쉼표로 구분한 정수로 입력하십시오.') from exc
    if any(not 1 <= order <= 100 for order in orders):
        raise ValueError('비교 차수는 1~100이어야 합니다.')
    return orders


class FourierTopic:
    id = 'fourier'
    title = '11.1 · Fourier 급수와 표본화 실험'
    description = '주기 함수의 사인·코사인 분해, 부분합의 오차 및 표본화에 따른 변화를 살펴보십시오.'
    inputs = (
        InputSpec('function', '함수 f(x)', 'text', 'x'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('L', '반주기 L', 'constant', 'pi'),
        InputSpec('N', '부분합 차수 N', 'int', 10, 1, 100),
        InputSpec('num_points', '적분 표본 수', 'int', 5000, 500, 20000),
        InputSpec('comparison_N', '비교 N (쉼표 구분)', 'orders', '1,3,5,10'),
        InputSpec('mode', '추가 분석', 'choice', MODES[0], choices=MODES),
        InputSpec('M', 'DFT / 실험 표본 수 M', 'int', 64, 16, 256, visible_in=(MODES[3], MODES[4])),
        InputSpec('center', 'Gibbs 확대 중심', 'float', 0., -100000, 100000, visible_in=(MODES[1],)),
        InputSpec('width', 'Gibbs 확대 반폭', 'float', .5, .000001, 100000, visible_in=(MODES[1],)),
        InputSpec('orders', 'Gibbs 비교 N', 'orders', '1,10,50', visible_in=(MODES[1],)),
        InputSpec('frequency', '실험 주파수 (Hz)', 'float', 9., 0, 10000, visible_in=(MODES[4],)),
        InputSpec('rate', '샘플링 주파수 (Hz)', 'float', 12., .001, 10000, visible_in=(MODES[4],)),
    )
    examples = {'기함수 x': {'function': 'x'}, '우함수 x²': {'function': 'x**2'},
                '상수 함수': {'function': '3'}, '사인 함수': {'function': 'sin(x)'},
                '매개변수 조합': {'function': 'A*exp(-x^2)*cos(w*x)', 'parameters': 'A=2; w=3'}}

    def __init__(self):
        self.session = FourierSession()

    def restore_cache(self, snapshot):
        if snapshot is not None:
            self.session.key, self.session.data = snapshot

    def snapshot_cache(self):
        return self.session.key, self.session.data

    def compute(self, params):
        params = validate_params(self.inputs, params)
        expression_text = str(parameter_expression(params['function'], params['parameters']))
        length = real_constant(params['L'])
        if length <= 0 or not np.isfinite(2*length):
            raise ValueError('L과 주기 2L은 유한한 양수여야 합니다.')
        orders = parse_orders(params['comparison_N'])
        current = params['N']
        required = max([current, *orders])
        if params['mode'] == MODES[1]:
            required = max([required, *parse_orders(params['orders'])])
        same_input = self.session.key == (expression_text, length, params['num_points'])
        previous_count = len(self.session.data[4]) if same_input else 0
        expression, x, y, a0, an, bn = self.session.prepare(
            expression_text, length, params['num_points'], required)
        sums = calculate_fourier_sums(x, length, a0, an, bn, [current, *orders])
        error, metrics = calculate_errors(y, sums[current])
        metrics['a₀'] = a0
        data = dict(x=x, y=y, sums=sums, error=error, an=an, bn=bn,
                    original=(expression, x, y, length, current, a0, an, bn),
                    cache_reused=min(previous_count, required), cache_integrated=max(0, required-previous_count))
        tables = {
            '계수': (['n', 'a_n', 'b_n', '|a_n|', '|b_n|'],
                     [[n, an[n-1], bn[n-1], abs(an[n-1]), abs(bn[n-1])] for n in range(1, len(an)+1)]),
            '수렴 오차': (['N', 'MSE', 'RMSE'],
                        [[n, *list(calculate_errors(y, sums[n])[1].values())[:2]] for n in orders]),
            '표본': (['x', 'f(x)', 'S_N(x)', 'error'], np.column_stack((x, y, sums[current], error))),
        }
        if params['mode'] != MODES[0]:
            options = dict(M=params['M'], frequency=params['frequency'], sampling_rate=params['rate'],
                           center=params['center'], width=params['width'],
                           orders=parse_orders(params['orders']) if params['mode'] == MODES[1] else None)
            data['options'] = options
            data['prepared'] = prepare_analysis(params['mode'], data['original'], **options)
            if 'transform' in data['prepared']:
                grid, values, direct, fast, coefficients = data['prepared']['transform']
                bins = np.fft.fftfreq(params['M'], d=2*length/params['M'])
                harmonics = np.fft.fftfreq(params['M'])*params['M']
                corrected = fast/params['M']*np.exp(1j*np.pi*harmonics)
                magnitude = np.abs(corrected)
                phase = np.angle(corrected)
                phase[magnitude <= 1e-10*max(1., float(magnitude.max()))] = np.nan
                tables['DFT'] = (['frequency', 'DFT real', 'DFT imag', 'FFT real', 'FFT imag', 'c_x real', 'c_x imag', 'magnitude', 'phase rad'],
                                 np.column_stack((bins, direct.real, direct.imag, fast.real, fast.imag, corrected.real, corrected.imag, magnitude, phase)))
                tables['DFT 표본'] = (['x', 'f(x)'], np.column_stack((grid, values)))
        return Result(self.id, params, metrics, tables, data,
                      [f'f(x) = {expression} · 구간 [-{length:g}, {length:g}] · 주기 {2*length:g} · N={current}',
                       f"계수 재사용 {data['cache_reused']}개 · 새로 적분한 차수 {data['cache_integrated']}개입니다.",
                       '오차는 입력 표본에서 계산합니다. 불연속점의 값은 급수의 극한과 다를 수 있습니다.'])

    def figures(self, result):
        data, params = result.data, result.params
        figures = {
            '현재 부분합': make_comparison_figure(data['x'], data['y'], {params['N']: data['sums'][params['N']]}, 'Fourier partial sum'),
            '여러 차수 비교': make_comparison_figure(data['x'], data['y'], {n: data['sums'][n] for n in parse_orders(params['comparison_N'])}, 'Convergence comparison'),
            '오차': make_error_figure(data['x'], data['error']),
            '계수 스펙트럼': make_spectrum_figure(data['an'][:params['N']], data['bn'][:params['N']]),
        }
        if 'prepared' in data:
            figure, note = make_analysis_figure(params['mode'], data['original'], prepared=data['prepared'], **data['options'])
            figures[params['mode']] = figure
            figure.suptitle('Additional analysis', fontsize=10)
            if note not in result.notices:
                result.notices.append(note)
        return figures

    def animation(self, result):
        expression, x, y, length, current, a0, an, bn = result.data['original']
        sums = calculate_fourier_sums(x, length, a0, an, bn, range(current+1))
        return dict(x=x, frames=np.array(list(sums.values())), labels=[f'N = {n}' for n in range(current+1)],
                    reference=y, xlabel='x', ylabel='Function value')
