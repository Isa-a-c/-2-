"""고정된 양 끝을 갖는 현의 무감쇠 파동방정식을 계산합니다."""
import numpy as np
from engineering_math.core.tables import GridRows
from engineering_math.core.expr import parameter_expression, evaluate
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.pde import sine_coefficients, sine_basis, evolution_figures, evolution_animation


def force_balance(coefficients, velocities, wave_numbers, omega, speed, length, moment):
    """교재 12.2 그림 286: 미소부분 [x₀, x₀+Δx]에 걸리는 장력의 수직 성분과 Newton 제2법칙.

    수평 성분 T₁cos α = T₂cos β = T이므로 수직 합력/질량 = (T/ρ)(tan β − tan α)/Δx입니다.
    tan은 그 점의 기울기 u_x이며, Δx→0에서 c²u_xx = u_tt가 됩니다. x₀는 곡률이 가장 큰 내부 점입니다.
    """
    amplitude = coefficients*np.cos(omega*moment) + velocities/omega*np.sin(omega*moment)

    def slope(s):
        return np.cos(np.outer(np.atleast_1d(s), wave_numbers)) @ (amplitude*wave_numbers)

    grid = np.linspace(.1*length, .9*length, 801)
    curvature = -np.sin(np.outer(grid, wave_numbers)) @ (amplitude*wave_numbers**2)
    start = float(grid[np.argmax(np.abs(curvature))])
    acceleration = float(-np.sin(start*wave_numbers) @ (amplitude*omega**2))
    rows = []
    for divisor in (8, 16, 32, 64, 128, 256):
        delta = length/divisor
        # 중심차분으로 미소부분 [x₀−Δx/2, x₀+Δx/2]의 양 끝 기울기 차이를 씁니다.
        quotient = float(speed**2*(slope(start + delta/2)[0] - slope(start - delta/2)[0])/delta)
        rows.append([delta, quotient, acceleration, abs(quotient - acceleration)])
    return dict(rows=rows, start=start, moment=moment, amplitude=amplitude)


class WaveTopic:
    id = 'wave'
    title = '12.2–12.3 · 현의 모델링과 변수분리'
    description = 'u_tt = c² u_xx의 고정단 현을 계산합니다. 초기 변위와 초기 속도를 각각 입력하십시오.'
    inputs = (
        InputSpec('initial', '초기 변위 u(x,0)', 'text', 'sin(pi*x/l)'),
        InputSpec('velocity', '초기 속도 u_t(x,0)', 'text', '0'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('length', '현의 길이 l', 'float', 1., .001, 1000, help='수식에 l이라고 쓰면 이 값이 들어갑니다. 예: sin(pi*x/l)'),
        InputSpec('speed', '파동 속력 c', 'float', 1., .000001, 1000),
        InputSpec('N', '사인 급수 차수 N', 'int', 20, 1, 100),
        InputSpec('num_points', '공간 표본 수', 'int', 501, 201, 5001),
        InputSpec('time_points', '시간 표본 수', 'int', 201, 2, 501),
        InputSpec('end_time', '마지막 시간', 'float', 2., .000001, 10000),
        InputSpec('comparison_alpha', '비교 열확산계수 α', 'float', .1, .000001, 1000,
                  help='열·파동 비교 그림에만 사용하며 파동 해에는 영향을 주지 않습니다.'),
    )
    examples = {
        '기본 사인 모드': {},
        '두 사인 모드': {'initial': 'sin(pi*x/l)+0.5*sin(3*pi*x/l)'},
        '삼각형 초기 변위': {'initial': '1-abs(2*x/l-1)'},
        '초기 속도만 부여': {'initial': '0', 'velocity': 'sin(pi*x/l)'},
    }

    def compute(self, params):
        params = validate_params(self.inputs, params)
        length, count = params['length'], params['N']
        x = np.linspace(0, length, params['num_points'])
        time = np.linspace(0, params['end_time'], params['time_points'])
        displacement_expr = parameter_expression(params['initial'], params['parameters'], ('x', 'l'))
        velocity_expr = parameter_expression(params['velocity'], params['parameters'], ('x', 'l'))
        initial = evaluate(displacement_expr, (x, length), ('x', 'l'))
        initial_velocity = evaluate(velocity_expr, (x, length), ('x', 'l'))
        displacement_coefficients = sine_coefficients(x, initial, length, count)
        velocity_coefficients = sine_coefficients(x, initial_velocity, length, count)
        wave_numbers = np.arange(1, count+1)*np.pi/length
        omega = params['speed']*wave_numbers
        phase = time[:, None]*omega
        # u_n(t)=A_n cos(ω_n t)+(V_n/ω_n)sin(ω_n t)입니다.
        amplitudes = (np.cos(phase)*displacement_coefficients
                      + np.sin(phase)*(velocity_coefficients/omega))
        derivatives = (-np.sin(phase)*(omega*displacement_coefficients)
                       + np.cos(phase)*velocity_coefficients)
        basis = sine_basis(x, length, count)
        displacement = amplitudes @ basis
        displacement[:, [0, -1]] = 0.
        reconstructed_velocity = velocity_coefficients @ basis
        reconstructed_velocity[[0, -1]] = 0.
        # 직교성을 이용한 단위 선밀도 에너지: 1/2 ∫(u_t²+c²u_x²)dx입니다.
        energy = length/4*np.sum(derivatives**2+(amplitudes*omega)**2, axis=1)
        heat = np.exp(-params['comparison_alpha']*time[:, None]*wave_numbers**2) @ (displacement_coefficients[:, None]*basis)
        heat[:, [0, -1]] = 0.
        if not all(np.all(np.isfinite(values)) for values in (displacement, energy, heat)):
            raise ValueError('유한하지 않은 결과가 발생했습니다. 입력 크기를 줄이십시오.')
        notices = [
            '무감쇠·무외력·일정한 파동 속력과 고정단 u(0,t)=u(l,t)=0을 가정합니다.',
            '열 비교는 동일한 초기 모양과 0 경계조건을 사용합니다. 초기 속도는 파동에만 적용됩니다.',
            '시간 간격은 해의 적분 간격이 아니라 출력 간격입니다. 너무 크면 진동을 놓칠 수 있습니다.',
        ]
        if not np.allclose(initial[[0, -1]], 0) or not np.allclose(initial_velocity[[0, -1]], 0):
            notices.append('초기 변위 또는 속도의 끝점이 고정단 조건과 다릅니다. 급수 해의 끝점은 0으로 고정됩니다.')
        samples_per_period = 2*np.pi/(omega[-1]*(time[1]-time[0]))
        if samples_per_period < 10:
            notices.append(f'최고 차수 모드의 한 주기당 출력 표본이 {samples_per_period:.3g}개입니다. 시간 표본 수를 늘리거나 마지막 시간을 줄이십시오.')
        drift = float(np.max(np.abs(energy-energy[0])))
        metrics = {'초기 변위 RMSE': float(np.sqrt(np.mean((displacement[0]-initial)**2))),
                   '초기 속도 RMSE': float(np.sqrt(np.mean((reconstructed_velocity-initial_velocity)**2))),
                   '에너지 최대 변화': drift}
        modeling = force_balance(displacement_coefficients, velocity_coefficients, wave_numbers, omega,
                                 params['speed'], length, time[len(time)//3])
        tables = {
            '모드 계수': (['n', 'A_n', 'V_n', 'omega_n'],
                        np.column_stack((np.arange(1, count+1), displacement_coefficients, velocity_coefficients, omega))),
            '에너지': (['t', 'energy'], np.column_stack((time, energy))),
            '변위 및 열 비교': (['t', 'x', 'wave u', 'heat u'],
                              GridRows((time, x), (displacement, heat))),
            '모델링: 미소부분의 힘': (['Δx', 'c²·(tanβ − tanα)/Δx', 'u_tt (점 x₀)', '차이'], modeling['rows']),
        }
        return Result(self.id, params, metrics, tables,
                      dict(x=x, time=time, initial=initial, displacement=displacement, heat=heat,
                           energy=energy, velocity_initial=reconstructed_velocity, modeling=modeling,
                           displacement_coefficients=displacement_coefficients, velocity_coefficients=velocity_coefficients), notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        data = result.data
        curve, heatmap = evolution_figures(data['x'], data['time'], data['initial'], data['displacement'],
                                          'Displacement', 'Fixed-end wave equation')
        comparison = Figure(figsize=(8, 6), layout='constrained')
        axes = comparison.subplots(2, 1, sharex=True)
        limit = max(float(np.max(np.abs(data['heat']))), float(np.max(np.abs(data['displacement']))), 1e-9)
        for axis, values, title in zip(axes, (data['heat'], data['displacement']), ('Heat: decay', 'Wave: oscillation')):
            for index in np.unique(np.linspace(0, len(data['time'])-1, 5, dtype=int)):
                axis.plot(data['x'], values[index], label=f"t={data['time'][index]:.3g}")
            axis.set(title=title, ylabel='u', ylim=(-1.05*limit, 1.05*limit))
            axis.grid(alpha=.3)
            axis.legend(fontsize=8, ncols=3)
        axes[-1].set_xlabel('x')
        energy_figure = Figure(figsize=(8, 4), layout='constrained')
        axis = energy_figure.subplots()
        axis.plot(data['time'], data['energy'])
        axis.set(xlabel='t', ylabel='Energy', title='Modal energy (unit linear density)')
        axis.grid(alpha=.3)
        return {'시간별 변위': curve, '시공간 히트맵': heatmap, '열·파동 비교': comparison, '에너지': energy_figure,
                '정규진동': self.normal_modes(result), '진행파 분해': self.travelling_waves(result),
                '모델링: 장력과 힘': self.tension_diagram(result)}

    @staticmethod
    def tension_diagram(result):
        """교재 그림 286: 미소부분 양 끝의 장력 T₁, T₂는 접선 방향이고 수직 성분의 차가 가속도를 만듭니다."""
        from matplotlib.figure import Figure
        p, model = result.params, result.data['modeling']
        length = p['length']
        modes = np.arange(1, len(model['amplitude'])+1)*np.pi/length
        x = np.linspace(0, length, 600)
        shape = np.sin(np.outer(x, modes)) @ model['amplitude']
        slope = lambda s: float(np.cos(s*modes) @ (model['amplitude']*modes))
        height = lambda s: float(np.sin(s*modes) @ model['amplitude'])
        delta = length/8
        left, right = model['start'] - delta/2, model['start'] + delta/2
        figure = Figure(figsize=(8, 5), layout='constrained')
        axis = figure.subplots()
        axis.plot(x, shape, color='0.6', lw=1)
        segment = (x >= left) & (x <= right)
        axis.plot(x[segment], shape[segment], color='k', lw=2.5, label='element [x, x+Δx]')
        scale = .25*length
        ends = [shape.min(), shape.max()]
        for point, sign, name, color in ((left, -1, 'T₁ (at P)', 'tab:blue'), (right, 1, 'T₂ (at Q)', 'tab:red')):
            # 장력의 크기는 같고 방향만 접선을 따릅니다. 화살표는 데이터 좌표의 접선 방향입니다.
            direction = np.array([1., slope(point)])
            dx, dy = scale*sign*direction/np.hypot(*direction)
            ends.append(height(point) + dy)
            axis.annotate('', xy=(point + dx, height(point) + dy), xytext=(point, height(point)), annotation_clip=False,
                          arrowprops=dict(arrowstyle='->', color=color, lw=2))
            axis.plot([point + dx, point + dx], [height(point), height(point) + dy], ':', color=color)
            axis.text(point + dx, height(point) + dy, ' ' + name, color=color, fontsize=9)
        axis.set(xlabel='x', ylabel='u(x, t)',
                 title=f"String at t={model['moment']:.3g}: tension along the tangent; vertical parts T sinβ − T sinα ≈ ρΔx u_tt")
        # 화살표 끝이 현의 범위 밖이어도 보이도록 세로 범위를 넓힙니다.
        margin = .08*max(max(ends) - min(ends), 1e-12)
        axis.set_ylim(min(ends) - margin, max(ends) + margin)
        axis.legend(loc='lower right', fontsize=8)
        axis.grid(alpha=.3)
        return figure

    @staticmethod
    def normal_modes(result):
        """교재 그림 287: n차 정규진동 sin(nπx/L)은 n−1개의 마디점을 가집니다."""
        from matplotlib.figure import Figure
        length, speed = result.params['length'], result.params['speed']
        x = np.linspace(0, length, 400)
        figure = Figure(figsize=(9, 3.2), layout='constrained')
        axes = figure.subplots(1, 4, sharey=True)
        for n, axis in enumerate(axes, 1):
            for phase in np.linspace(0, np.pi, 5):
                axis.plot(x, np.cos(phase)*np.sin(n*np.pi*x/length), color='tab:blue', alpha=.25 + .15*np.cos(phase)**2)
            nodes = np.arange(1, n)*length/n
            axis.plot(nodes, np.zeros_like(nodes), 'ko')
            axis.axhline(0, color='k', lw=.5)
            axis.set(title=f'n={n}, f={speed*n/(2*length):.3g}', xlabel='x', xticks=[0, length])
        axes[0].set_ylabel('u_n / amplitude')
        figure.suptitle('Normal modes sin(nπx/L)cos(λₙt) and nodes (black)', fontsize=10)
        return figure

    @staticmethod
    def travelling_waves(result):
        """교재 그림 291: 초기 속도 0이면 u=½[f*(x−ct)+f*(x+ct)], f*는 f의 주기 2L 기확장입니다."""
        from matplotlib.figure import Figure
        data, p = result.data, result.params
        length, speed = p['length'], p['speed']
        coefficients = data['displacement_coefficients']
        modes = np.arange(1, len(coefficients)+1)*np.pi/length
        x = data['x']

        def extension(s):
            return np.sin(np.outer(s, modes)) @ coefficients

        times = np.linspace(0, length/speed, 6)
        figure = Figure(figsize=(8, 8), layout='constrained')
        axes = figure.subplots(len(times), 1, sharex=True, sharey=True)
        for axis, t in zip(axes, times):
            right, left = extension(x - speed*t)/2, extension(x + speed*t)/2
            axis.plot(x, right, '--', color='tab:red', lw=1, label='½f*(x−ct) →')
            axis.plot(x, left, '--', color='tab:blue', lw=1, label='← ½f*(x+ct)')
            axis.plot(x, right + left, 'k', lw=1.5, label='u(x,t)')
            axis.set_ylabel(f't={t:.3g}')
            axis.grid(alpha=.3)
        axes[0].legend(fontsize=8, ncols=3, loc='upper right')
        axes[-1].set_xlabel('x')
        title = 'Superposition of right- and left-travelling waves (t = 0 … L/c)'
        if np.any(np.abs(data['velocity_coefficients']) > 1e-12):
            title += '\n(initial velocity omitted: shows displacement part only)'
        figure.suptitle(title, fontsize=10)
        return figure

    def animation(self, result):
        data = result.data
        return evolution_animation(data['x'], data['time'], data['initial'], data['displacement'], 'Displacement')
