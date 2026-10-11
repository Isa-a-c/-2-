"""1차원 유한 막대 열방정식의 변수분리 해입니다.

고정 온도 경계는 정상 상태를 뺀 나머지를 사인 급수로, 양 끝 단열 경계(교재 12.6 예제 4·5)는
고유값 0을 포함하는 코사인 급수로 전개합니다.
"""
import numpy as np
from engineering_math.core.tables import GridRows
from engineering_math.core.expr import parameter_expression, evaluate
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.pde import (sine_coefficients, cosine_coefficients, heat_accuracy,
                                       evolution_figures, evolution_animation)

FIXED, INSULATED = '양 끝 고정 온도', '양 끝 단열'


def heat_balance(coefficients, wave_numbers, alpha, length, insulated, moment, steady):
    """교재 12.5: 열흐름 q=−K u_x (단위 ρσ=1이면 K=α). 미소부분의 순유입 [q(x)−q(x+Δx)]/Δx는 u_t와 같습니다.

    발산정리로 유도한 3차원 식 u_t=c²∇²u의 1차원 형태입니다. x₀는 |u_t|가 가장 큰 내부 점입니다.
    """
    amplitude = coefficients*np.exp(-alpha*wave_numbers**2*moment)
    gradient = (steady[-1] - steady[0])/length

    def derivative(s):
        s = np.atleast_1d(s)
        if insulated:
            return -np.sin(np.outer(s, wave_numbers)) @ (amplitude*wave_numbers)
        return np.cos(np.outer(s, wave_numbers)) @ (amplitude*wave_numbers) + gradient

    def rate(s):
        s = np.atleast_1d(s)
        trig = np.cos if insulated else np.sin
        return trig(np.outer(s, wave_numbers)) @ (-alpha*wave_numbers**2*amplitude)

    grid = np.linspace(.1*length, .9*length, 801)
    start = float(grid[np.argmax(np.abs(rate(grid)))])
    target = float(rate(start)[0])
    rows = []
    for divisor in (8, 16, 32, 64, 128, 256):
        delta = length/divisor
        inflow = float((-alpha*derivative(start - delta/2)[0] + alpha*derivative(start + delta/2)[0])/delta)
        rows.append([delta, inflow, target, abs(inflow - target)])
    profile = np.linspace(0, length, 41)
    return dict(rows=rows, start=start, moment=moment, profile=profile, flux=-alpha*derivative(profile))


class HeatTopic:
    id = 'heat'
    title = '12.5–12.6 · 열전도 모델링과 급수 해'
    description = 'u_t = α u_xx를 급수로 계산합니다. 양 끝 온도 고정 또는 양 끝 단열(u_x=0)을 선택하십시오.'
    inputs = (
        InputSpec('mode', '경계조건', 'choice', FIXED, choices=(FIXED, INSULATED)),
        InputSpec('initial', '초기 온도 u(x,0)', 'text', 'sin(pi*x/ell)', help='x와 막대 길이 ell을 사용하십시오.'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('length', '막대 길이 ell', 'float', 1., .001, 1000),
        InputSpec('alpha', '열확산계수 α', 'float', .1, .000001, 1000),
        InputSpec('left', '왼쪽 경계 온도', 'float', 0., -100000, 100000, visible_in=(FIXED,)),
        InputSpec('right', '오른쪽 경계 온도', 'float', 0., -100000, 100000, visible_in=(FIXED,)),
        InputSpec('N', '급수 차수 N', 'int', 20, 1, 100),
        InputSpec('num_points', '공간 표본 수', 'int', 501, 201, 5001),
        InputSpec('time_points', '시간 표본 수', 'int', 101, 2, 501),
        InputSpec('end_time', '마지막 시간', 'float', 1., .000001, 10000),
    )
    examples = {'기본 사인 모드': {}, '두 사인 모드': {'initial': 'sin(pi*x/ell)+0.5*sin(3*pi*x/ell)'},
                '삼각형 초기 온도': {'initial': '1-abs(2*x/ell-1)'},
                '정상 상태': {'initial': '20+60*x/ell', 'left': 20., 'right': 80.},
                '단열 막대의 코사인 모드': {'mode': INSULATED, 'initial': '1+cos(pi*x/ell)'}}

    def compute(self, params):
        params = validate_params(self.inputs, params)
        length, count = params['length'], params['N']
        insulated = params['mode'] == INSULATED
        x = np.linspace(0, length, params['num_points'])
        time = np.linspace(0, params['end_time'], params['time_points'])
        expression = parameter_expression(params['initial'], params['parameters'], variables=('x', 'ell'))
        initial = evaluate(expression, (x, length), variables=('x', 'ell'))
        wave_numbers = np.arange(1, count+1)*np.pi/length
        decay = np.exp(-params['alpha']*time[:, None]*wave_numbers**2)
        if insulated:
            # 우확장의 코사인 계수: A₀=(1/ell)∫f, Aₙ=(2/ell)∫f cos(nπx/ell). A₀가 정상 상태입니다.
            mean, coefficients = cosine_coefficients(x, initial, length, count)
            steady = np.full_like(x, mean)
            basis = np.cos(wave_numbers[:, None]*x)
            temperature = steady + decay @ (coefficients[:, None]*basis)
        else:
            steady = params['left'] + (params['right']-params['left'])*x/length
            # 기확장의 사인 계수는 (2/ell) ∫[0,ell] residual*sin(nπx/ell) dx와 같습니다.
            coefficients = sine_coefficients(x, initial-steady, length, count)
            basis = np.sin(wave_numbers[:, None]*x)
            temperature = steady + decay @ (coefficients[:, None]*basis)
            temperature[:, 0], temperature[:, -1] = params['left'], params['right']
        if not np.all(np.isfinite(temperature)):
            raise ValueError('온도 계산에서 유한하지 않은 값이 발생했습니다.')
        energy = np.trapezoid((temperature-steady)**2, x=x, axis=1)
        accuracy = heat_accuracy(expression, params, coefficients)
        notices = ['길이·시간·열확산계수의 단위는 일관되게 입력하십시오. α의 차원은 길이²/시간입니다.',
                   '유한 차수의 변수분리 해입니다. 공간 표본은 계수 적분과 시각화에 사용됩니다.',
                   '정확도 비교는 고정된 2049개 평가점에서 수행합니다. 격자·차수 간 차이는 참해에 대한 오차 상한이 아닙니다.']
        if insulated:
            notices.insert(0, f'단열 경계 u_x(0,t)=u_x(ell,t)=0: 열이 빠져나가지 않으므로 온도는 초기 평균 {steady[0]:.6g}로 수렴합니다. 고유값 λ₀=0의 상수 모드가 정상 상태입니다.')
        elif not np.allclose(initial[[0, -1]], [params['left'], params['right']]):
            notices.append('초기 함수의 끝점이 경계조건과 다릅니다. 급수 해의 끝점은 지정한 경계 온도로 고정됩니다.')
        metrics = {'초기 재구성 RMSE': float(np.sqrt(np.mean((temperature[0]-initial)**2))),
                   '최종 최저 온도': float(temperature[-1].min()), '최종 최고 온도': float(temperature[-1].max()),
                   '격자 세분화 계수 변화': accuracy['coefficient_change']}
        if insulated:
            metrics['정상 상태 (초기 평균)'] = float(steady[0])
            columns, rows = ['n', 'A_n'], np.column_stack((np.arange(count+1), np.r_[steady[0], coefficients]))
            name = '코사인 계수'
        else:
            columns, rows, name = ['n', 'b_n'], np.column_stack((np.arange(1, count+1), coefficients)), '사인 계수'
        tables = {name: (columns, rows),
                  '잔차 에너지': (['t', 'integral (u-steady)^2 dx'], np.column_stack((time, energy))),
                  '온도': (['t', 'x', 'u'], GridRows((time, x), (temperature,)))}
        tables['정확도 비교'] = (['변경 항목', '적분 표본 수', 'N', '초기 재구성 RMSE (공통 격자)', '기준과 차이 t=0 (최대)', '기준과 차이 t=마지막 (최대)'], accuracy['rows'])
        modeling = heat_balance(coefficients, wave_numbers, params['alpha'], length, insulated,
                                time[max(1, len(time)//4)], steady)
        tables['모델링: 열의 균형'] = (['Δx', '[q(x−Δx/2) − q(x+Δx/2)]/Δx', 'u_t (점 x₀)', '차이'], modeling['rows'])
        return Result(self.id, params, metrics, tables,
                      dict(x=x, time=time, initial=initial, steady=steady, temperature=temperature, coefficients=coefficients,
                           accuracy=accuracy, modeling=modeling), notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        data = result.data
        title = 'Heat equation (insulated ends)' if result.params['mode'] == INSULATED else 'Heat equation'
        figure, heatmap = evolution_figures(data['x'], data['time'], data['initial'], data['temperature'],
                                           'Temperature', title)
        accuracy_figure = Figure(figsize=(8, 5), layout='constrained')
        axes = accuracy_figure.subplots(2, 1, sharex=True)
        accuracy = data['accuracy']
        axes[0].plot(accuracy['x'], accuracy['initial'], 'k--', label='Initial input')
        axes[0].plot(accuracy['x'], accuracy['reconstruction'], label='Series reconstruction at t=0')
        axes[0].set(ylabel='Temperature', title='Initial reconstruction (not time evolution)')
        axes[0].legend()
        axes[1].plot(accuracy['x'], accuracy['initial']-accuracy['reconstruction'])
        axes[1].set(xlabel='x', ylabel='Initial error')
        for axis in axes:
            axis.grid(alpha=.3)
        return {'시간별 온도': figure, '시공간 히트맵': heatmap, '초기 재구성': accuracy_figure,
                '모델링: 열흐름': self.flux_figure(result)}

    @staticmethod
    def flux_figure(result):
        """열은 온도가 높은 곳에서 낮은 곳으로 흐르며 q=−K u_x입니다(교재 12.5 가정 2)."""
        from matplotlib.figure import Figure
        data, model = result.data, result.data['modeling']
        index = int(np.argmin(np.abs(data['time'] - model['moment'])))
        figure = Figure(figsize=(8, 5), layout='constrained')
        axes = figure.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        axes[0].plot(data['x'], data['temperature'][index], color='tab:red')
        axes[0].axvline(model['start'], color='k', ls=':', lw=1)
        axes[0].set(ylabel='u', title=f"Temperature at t={data['time'][index]:.3g} and heat flux q = −K u_x (K=α, ρσ=1)")
        flux = model['flux']
        limit = max(float(np.max(np.abs(flux))), 1e-12)
        axes[1].quiver(model['profile'], np.zeros_like(flux), flux/limit, np.zeros_like(flux),
                       color=np.where(flux >= 0, 'tab:orange', 'tab:blue'), scale=12, width=.004)
        axes[1].axvline(model['start'], color='k', ls=':', lw=1)
        axes[1].set(xlabel='x', yticks=[], ylim=(-1, 1), title='Arrows: direction and relative size of heat flow along the rod')
        for axis in axes:
            axis.grid(alpha=.3)
        return figure

    def animation(self, result):
        data = result.data
        return evolution_animation(data['x'], data['time'], data['initial'], data['temperature'], 'Temperature')
