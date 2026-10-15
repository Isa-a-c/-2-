"""11.2~11.6: 반구간, 강제진동, 최소제곱, 고유함수 전개 화면입니다."""
import numpy as np
from matplotlib.figure import Figure
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate, real_constant
from engineering_math.core.fourier import calculate_fourier_coefficients, calculate_fourier_sum
from engineering_math.core.chapter11 import half_range, project_basis, orthogonal_basis, sturm_basis, forced_response, checked


FUNCTION_INPUTS = (
    InputSpec('function', '함수 f(x)', 'text', 'x'),
    InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
)
SERIES_INPUTS = (
    InputSpec('L', '길이 L', 'constant', 'pi'),
    InputSpec('N', '차수 / 항 개수 N', 'int', 10, 1, 60),
    InputSpec('num_points', '적분 표본 수', 'int', 2001, 501, 20001),
)


def positive_length(text):
    length = real_constant(text)
    if not 1e-5 <= length <= 10000:
        raise ValueError('L은 0.00001~10000 범위의 양수여야 합니다.')
    return length


def sample(p, start, end):
    x = np.linspace(start, end, p['num_points'])
    expression = parameter_expression(p['function'], p['parameters'])
    return x, evaluate(expression, (x,))


def line_figure(x, curves, title, xlabel='x', ylabel='Value'):
    figure = Figure(figsize=(8, 5), layout='constrained')
    axis = figure.subplots()
    for label, values in curves.items():
        axis.plot(x, values, label=label)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=.3)
    axis.legend()
    return figure


class StaticTopic:
    supports_animation = False

    def figures(self, result):
        d = result.data
        return {'함수 비교': line_figure(d['x'], d['curves'], self.title.split(' · ')[0])}


# ── 11.2: 반구간 전개 ───────────────────────────────────────────
class HalfRangeTopic(StaticTopic):
    id = 'half_range'
    title = '11.2 · 반구간 사인·코사인 전개'
    description = '[0,L]에서 입력한 함수를 홀수 또는 짝수로 확장하고 2L 주기 급수와 비교합니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '확장 방식', 'choice', '사인 · 홀수 확장', choices=('사인 · 홀수 확장', '코사인 · 짝수 확장')),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, 0, length)
        cosine = p['mode'].startswith('코사인')
        a0, coefficients, sums = half_range(x, y, length, p['N'], cosine)
        extended_x = np.linspace(-2 * length, 2 * length, 2001)
        folded = (extended_x + length) % (2 * length) - length
        expression = parameter_expression(p['function'], p['parameters'])
        extended_y = evaluate(expression, (np.abs(folded),))
        if not cosine:
            extended_y *= np.sign(folded)
        zeros = np.zeros_like(coefficients)
        extended_sum = calculate_fourier_sum(extended_x, length, a0,
                                             coefficients if cosine else zeros,
                                             zeros if cosine else coefficients, p['N'])
        error = checked(y - sums[-1])
        checked(error ** 2)
        tables = {'계수': (['n', 'a_n' if cosine else 'b_n'], list(enumerate(coefficients, 1))),
                  '수렴': (['N', 'RMSE'], [[n + 1, float(np.sqrt(np.mean((y - row) ** 2)))] for n, row in enumerate(sums)]),
                  '표본': (['x', 'f(x)', 'S_N(x)'], np.column_stack((x, y, sums[-1])))}
        data = dict(x=x, curves={'f(x)': y, 'S_N(x)': sums[-1]}, extended_x=extended_x,
                    extended_y=extended_y, extended_sum=extended_sum, coefficients=coefficients, a0=a0)
        return Result(self.id, p, {'RMSE': float(np.sqrt(np.mean(error ** 2))), 'a₀': a0}, tables, data,
                      ['교재 11.2, 12쪽. 사인 전개는 양 끝에서 0입니다. 불연속점에서는 좌우 극한의 평균에 수렴합니다.',
                       '확장 그래프의 경계 표본값과 원함수의 끝점 값은 다를 수 있습니다.'])

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['주기 확장'] = line_figure(d['extended_x'], {'Periodic extension': d['extended_y'], 'Partial sum': d['extended_sum']}, 'Even / odd periodic extension')
        return figures


# ── 11.3: Fourier 외력과 진동 응답 ───────────────────────────────
class ForcedTopic(StaticTopic):
    id = 'fourier_forced'
    title = '11.3 · Fourier 외력과 강제진동'
    description = 'f(x)의 x를 시간으로 해석합니다. 한 주기 [-L,L]의 외력을 급수화하여 m y″+c y′+k y=F_N(t)를 풉니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mass', '질량 m', 'float', 1., .01, 100),
        InputSpec('damping', '감쇠 c', 'float', .2, 0, 100),
        InputSpec('stiffness', '강성 k', 'float', 4., .01, 100),
        InputSpec('y0', '초기 변위', 'float', 0., -100, 100),
        InputSpec('v0', '초기 속도', 'float', 0., -100, 100),
        InputSpec('end_time', '마지막 시간', 'float', 20., .1, 100),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, -length, length)
        a0, an, bn = calculate_fourier_coefficients(x, y, length, p['N'])
        time = np.linspace(0, p['end_time'], 2001)
        state, particular, resonant, frequencies = forced_response(time, length, a0, an, bn,
            p['mass'], p['damping'], p['stiffness'], p['y0'], p['v0'])
        force = calculate_fourier_sum(time, length, a0, an, bn, p['N'])
        curves = {'Initial-value response': state[0]}
        if particular is not None:
            curves['Periodic particular solution'] = particular
        # 교재 식 (5)·(6): 성분별 정상상태 응답 yₙ=Aₙcos ωₙt+Bₙsin ωₙt의 진폭 Cₙ=√(Aₙ²+Bₙ²).
        denominator = p['stiffness'] - p['mass'] * frequencies ** 2 + 1j * p['damping'] * frequencies
        forcing = np.hypot(an, bn)
        with np.errstate(divide='ignore', invalid='ignore'):
            response = np.where(resonant, np.inf, forcing / np.abs(denominator))
        response[(forcing == 0) & ~resonant] = 0.
        natural = np.sqrt(p['stiffness'] / p['mass'])
        tables = {'외력 성분': (['n', 'omega', 'a_n', 'b_n', 'input amplitude', 'response amplitude C_n', 'gain 1/|D_n|', 'resonant'],
                  [[i + 1, frequencies[i], an[i], bn[i], forcing[i], response[i], 1 / abs(denominator[i]) if not resonant[i] else np.inf,
                    bool(resonant[i])] for i in range(p['N'])]),
                  '응답': (['t', 'F_N', 'y', 'velocity'], np.column_stack((time, force, state.T)))}
        notices = ['교재 11.3, 22쪽. 원래 외력 대신 유한 Fourier 부분합에 대한 초기값 문제를 계산합니다.',
                   '무감쇠계의 주기 특수해는 초기값 응답의 장시간 극한을 뜻하지 않습니다.']
        metrics = {'최대 |y|': float(np.max(np.abs(state[0]))), '공진 성분 수': int(resonant.sum()),
                   '고유 각진동수 √(k/m)': float(natural)}
        if np.any(resonant):
            notices.append('활성 외력 성분에 무감쇠 공진이 있습니다. 유계 주기 특수해를 표시하지 않으며 시간 응답의 증가를 관찰하십시오.')
        elif np.any(response > 0):
            order = np.argsort(response)[::-1]
            dominant = order[0]
            metrics['지배 성분 n'] = int(dominant + 1)
            metrics['지배 성분 진폭 C_n'] = float(response[dominant])
            text = f'정상상태 응답에서 n={dominant + 1} 성분(ω={frequencies[dominant]:.4g})이 가장 큽니다: C={response[dominant]:.4g}.'
            if len(order) > 1 and response[order[1]] > 0:
                text += f' 다음은 n={order[1] + 1} (C={response[order[1]]:.4g}), 비율 {response[order[1]] / response[dominant]:.3g}입니다.'
            text += f' 고유 각진동수 {natural:.4g}에 가까운 외력 성분이 작은 입력 진폭으로도 응답을 지배합니다(교재 예제 1의 C₅).'
            notices.insert(1, text)
        return Result(self.id, p, metrics, tables,
                      dict(x=time, curves=curves, force=force, state=state, resonant=resonant, particular=particular,
                           frequencies=frequencies, input_amplitude=forcing, response_amplitude=response, natural=natural), notices)

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['외력'] = line_figure(d['x'], {'F_N(t)': d['force']}, 'Truncated periodic force', 't')
        figure = Figure(figsize=(8, 5), layout='constrained')
        axes = figure.subplots(2, 1, sharex=True)
        omega = d['frequencies']
        width = .35 * (omega[1] - omega[0]) if len(omega) > 1 else .3
        axes[0].bar(omega, d['input_amplitude'], width=width, color='tab:gray')
        axes[0].set(ylabel='Input amplitude', title='Force harmonics and steady-state response amplitudes')
        finite = np.where(np.isfinite(d['response_amplitude']), d['response_amplitude'], np.nan)
        axes[1].bar(omega, finite, width=width, color='tab:red')
        axes[1].axvline(d['natural'], color='k', ls='--', lw=1, label='natural frequency √(k/m)')
        axes[1].set(xlabel='omega_n', ylabel='Response amplitude C_n')
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=.3)
        figures['응답 진폭 스펙트럼'] = figure
        return figures


# ── 11.4: 직교투영의 최소제곱 성질 ────────────────────────────────
class ApproximationTopic(StaticTopic):
    id = 'trig_approximation'
    title = '11.4 · 삼각다항식과 최소제곱 근사'
    description = 'Fourier 계수의 직교투영과 계수를 변형한 근사의 적분 제곱오차를 비교합니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('perturbation', 'a₁에 더할 값 δ', 'float', .5, -10, 10),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, -length, length)
        a0, an, bn = calculate_fourier_coefficients(x, y, length, p['N'])
        approximation = calculate_fourier_sum(x, length, a0, an, bn, p['N'])
        changed = approximation + p['perturbation'] * np.cos(np.pi * x / length)
        energy = float(np.trapezoid(y ** 2, x=x))
        captured = 2 * length * a0 ** 2 + length * np.cumsum(an ** 2 + bn ** 2)
        error = float(np.trapezoid((y - approximation) ** 2, x=x))
        changed_error = float(np.trapezoid((y - changed) ** 2, x=x))
        checked([energy, error, changed_error, *captured])
        tables = {'오차와 에너지': (['N', 'captured energy', 'energy - captured'],
                    [[i + 1, captured[i], energy - captured[i]] for i in range(p['N'])]),
                  '표본': (['x', 'f', 'S_N', 'perturbed'], np.column_stack((x, y, approximation, changed)))}
        return Result(self.id, p, {'적분 제곱오차': error, '변형 후 제곱오차': changed_error,
                      '예측 오차 증가 Lδ²': length * p['perturbation'] ** 2}, tables,
                      dict(x=x, curves={'f(x)': y, 'Projection': approximation, 'Perturbed a1': changed}),
                      ['교재 11.4, 25쪽. 같은 차수의 삼각다항식 중 Fourier 투영이 연속 L² 오차를 최소화합니다.',
                       '표는 사다리꼴 적분 근사입니다. 에너지 차이는 반올림으로 작은 음수가 될 수 있습니다.'])


# ── 11.5~11.6: 고유함수와 가중 직교투영 ──────────────────────────
class OrthogonalTopic(StaticTopic):
    id = 'orthogonal_series'
    title = '11.6 · Legendre·Bessel 직교급수'
    description = 'Legendre는 [-1,1], Bessel은 [0,L]에서 계산합니다. N은 포함할 기저의 개수입니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '직교기저', 'choice', 'Legendre', choices=('Legendre', 'Bessel')),
        InputSpec('nu', 'Bessel 차수 ν', 'int', 0, 0, 5, visible_in=('Bessel',)),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        if p['mode'] == 'Legendre':
            x, y = sample(p, -1, 1)
        else:
            x, y = sample(p, 0, length)
        basis, weight, eigenvalues = orthogonal_basis(p['mode'], x, p['N'], length, p['nu'])
        return self.projection_result(p, x, y, basis, weight, eigenvalues,
            'Legendre: r=1, λ=n(n+1), n=0부터 시작합니다. Bessel: r=x, λ=(jν,n/L)², n=1부터 시작합니다.')

    def projection_result(self, p, x, y, basis, weight, eigenvalues, note):
        coefficients, gram, sums, norms = project_basis(x, y, basis, weight)
        errors = [float(np.trapezoid(weight * (y - row) ** 2, x=x)) for row in sums]
        checked(errors)
        gram_error = float(np.max(np.abs(gram - np.eye(len(gram)))))
        tables = {'전개 계수': (['term', 'lambda', 'coefficient', 'norm squared'],
                  [[i + 1, eigenvalues[i], coefficients[i], norms[i]] for i in range(p['N'])]),
                  '수렴': (['terms', 'weighted squared error'], list(enumerate(errors, 1))),
                  '정규화 Gram': ([str(i + 1) for i in range(p['N'])], gram),
                  '표본': (['x', 'f', 'sum'], np.column_stack((x, y, sums[-1])))}
        return Result(self.id, p, {'가중 적분 제곱오차': errors[-1], 'Gram 최대 편차': gram_error}, tables,
                      dict(x=x, curves={'f(x)': y, 'Projection': sums[-1]}, basis=basis, gram=gram,
                           coefficients=coefficients, eigenvalues=eigenvalues),
                      [note, 'Gram 행렬은 정규화된 기저의 가중 내적입니다. 단위행렬과의 차이로 적분 해상도를 점검하십시오.',
                       '표본 적분으로 계산합니다. 표본 수를 늘려 계수와 오차의 안정성을 확인하십시오.'])

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['기저 함수'] = line_figure(d['x'], {f'Term {i + 1}': row for i, row in enumerate(d['basis'][:6])}, 'First six basis functions')
        figure = Figure(figsize=(6, 5), layout='constrained')
        axis = figure.subplots()
        mesh = axis.imshow(d['gram'], origin='lower', vmin=-1, vmax=1, cmap='coolwarm')
        axis.set(title='Normalized weighted Gram matrix', xlabel='Basis index (zero based)', ylabel='Basis index (zero based)')
        figure.colorbar(mesh, ax=axis)
        figures['직교성'] = figure
        return figures


class SturmTopic(OrthogonalTopic):
    id = 'sturm_liouville'
    title = '11.5 · Sturm–Liouville 고유함수'
    description = '상수계수 (p y′)′+(q+λr)y=0의 DD·NN·DN·주기·일반(Robin) 경계조건을 비교합니다. 임의 가변계수 해석기는 아닙니다.'
    BOUNDARIES = ('DD', 'NN', 'DN', 'Periodic', 'Robin')
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '경계조건', 'choice', 'DD', choices=BOUNDARIES),
        InputSpec('p', '계수 p > 0', 'float', 1., .01, 100),
        InputSpec('q', '계수 q', 'float', 0., -100, 100),
        InputSpec('r', '가중치 r > 0', 'float', 1., .01, 100),
        InputSpec('k1', 'k₁ (k₁y(0)+k₂y′(0)=0)', 'float', 1., -100, 100, visible_in=('Robin',)),
        InputSpec('k2', 'k₂', 'float', 0., -100, 100, visible_in=('Robin',)),
        InputSpec('l1', 'l₁ (l₁y(L)+l₂y′(L)=0)', 'float', 1., -100, 100, visible_in=('Robin',)),
        InputSpec('l2', 'l₂', 'float', 1., -100, 100, visible_in=('Robin',)),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, 0, length)
        robin = (p['k1'], p['k2'], p['l1'], p['l2'])
        basis, eigenvalues = sturm_basis(x, length, p['N'], p['mode'], p['p'], p['q'], p['r'], robin)
        notes = {'Periodic': '주기 경계 y(0)=y(L), y′(0)=y′(L)(식 7): 상수 모드 하나와 n≥1마다 cos·sin 두 모드가 같은 고유값을 가집니다.',
                 'Robin': f'일반 경계(식 2): {robin[0]:g}y(0)+{robin[1]:g}y′(0)=0, {robin[2]:g}y(L)+{robin[3]:g}y′(L)=0. '
                          '고유값은 특성 행렬식의 근이며 κ²<0(쌍곡선 해)인 음의 고유값도 포함합니다. 고유함수는 최댓값 1로 맞춥니다.'}
        result = self.projection_result(p, x, y, basis, np.full_like(x, p['r']), eigenvalues,
            '교재 11.5, 29쪽의 부호: λ=(pκ²−q)/r. D는 y=0, N은 y′=0이며 DN은 왼쪽 D·오른쪽 N입니다. NN은 상수 모드를 포함합니다.')
        if p['mode'] in notes:
            result.notices.insert(0, notes[p['mode']])
        return result
