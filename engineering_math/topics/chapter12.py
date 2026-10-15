"""교재 12장의 입력 명세·결과·시각화입니다. 계산은 core에 분리합니다."""
import numpy as np
from engineering_math.core.tables import GridRows
import sympy as sp
from scipy.special import erfc
from matplotlib.figure import Figure
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate, symbols_for
from engineering_math.core.chapter11 import checked
from engineering_math.core.chapter12 import (dalembert, infinite_heat, rectangle_membrane,
    disk_membrane, disk_potential, sphere_potential, rectangle_laplace)
from engineering_math.core.pde import evolution_figures, evolution_animation

PARAMETERS = InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', '')
TIME_INPUTS = (
    InputSpec('end_time', '마지막 시간', 'float', 2., .001, 20),
    InputSpec('time_points', '시간 표본 수', 'int', 81, 11, 201),
)


def function(text, parameters='', variables=('x',)):
    expression = parameter_expression(text, parameters, variables)
    return lambda *values: evaluate(expression, values, variables)


def evolution_result(topic_id, p, x, time, frames, initial, notices, metrics=None):
    checked(frames)
    statistics = {'최대 |u|': float(np.max(np.abs(frames))),
                  '초기 재구성 RMSE': float(np.sqrt(np.mean((frames[0] - initial) ** 2)))}
    statistics.update(metrics or {})
    checked(list(statistics.values()))
    tables = {'시공간 표본': (['t', 'x', 'u'], GridRows((time, x), (frames,)))}
    return Result(topic_id, p, statistics, tables, dict(x=x, time=time, frames=frames, initial=initial), notices)


class EvolutionTopic:
    def figures(self, result):
        d = result.data
        curve, heatmap = evolution_figures(d['x'], d['time'], d['initial'], d['frames'], 'u', self.id)
        return {'시간별 함수': curve, '시공간 히트맵': heatmap}

    def animation(self, result):
        d = result.data
        return evolution_animation(d['x'], d['time'], d['initial'], d['frames'])


# ── 12.1: 후보 해를 방정식에 대입하여 잔차 확인 ────────────────────
ODE_MODE = '상미분방정식처럼 풀기'
CHECK_MODES = ('Laplace', 'Poisson', '열', '파동')
SUBSTITUTIONS = ('u 그대로', 'p = u_x로 치환', 'p = u_y로 치환')


def solve_like_ode(a, b, c, g, variable, substitution, x, y):
    """한 변수의 도함수만 있는 PDE a·w″+b·w′+c·w=g를 그 변수의 ODE로 풉니다(교재 12.1 예제 2·3).

    다른 변수는 매개변수이므로 적분상수 C1, C2는 그 변수의 임의 함수가 됩니다.
    substitution이 p=u_x(또는 u_y)이면 w=p로 풀고 u=∫p dx + g(y)(또는 ∫p dy + f(x))로 되돌립니다.
    반환: (u, 잔차, 설명 행 목록).
    """
    v = x if variable == 'x' else y
    other = y if v == x else x
    if a == 0 and b == 0:
        raise ValueError('a와 b가 모두 0이면 미분방정식이 아닙니다.')
    w = sp.Function('w')(v)
    equation = sp.Eq(a*w.diff(v, 2) + b*w.diff(v) + c*w, g)
    try:
        solution = sp.dsolve(equation, w)
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(f'SymPy가 이 상미분방정식을 풀지 못했습니다: {exc}') from exc
    if isinstance(solution, list):
        solution = solution[0]
    rhs = solution.rhs
    order = "w''" if a != 0 else "w'"
    rows = [['풀이 변수', f'{v} (다른 변수 {other}는 매개변수로 취급)'],
            ['상미분방정식', f"({a})·w'' + ({b})·w' + ({c})·w = {g}" if a != 0 else f"({b})·w' + ({c})·w = {g}"],
            ['최고계', order]]
    if substitution == SUBSTITUTIONS[0]:
        names = {'C1': 'A', 'C2': 'B'}
        u = rhs.subs({sp.Symbol(k): sp.Function(name)(other) for k, name in names.items()})
        residual = sp.simplify(a*u.diff(v, 2) + b*u.diff(v) + c*u - g)
        rows.append(['일반해 u', str(u)])
        rows.append(['임의 함수', f'A({other}), B({other})는 {other}의 임의 함수입니다.'])
        return u, residual, rows
    by = x if substitution == SUBSTITUTIONS[1] else y
    rest = y if by == x else x
    # p의 적분상수는 p를 구한 변수가 아닌 변수의 함수입니다.
    p_solution = rhs.subs({sp.Symbol('C1'): sp.Function('c')(other), sp.Symbol('C2'): sp.Function('d')(other)})
    residual = sp.simplify(a*p_solution.diff(v, 2) + b*p_solution.diff(v) + c*p_solution - g)
    tail = sp.Function('g' if by == x else 'f')(rest)
    u = sp.integrate(p_solution, by) + tail
    rows += [[f'p = u_{by}', str(p_solution)], ['일반해 u', str(u)],
             ['임의 함수', f'c({other}), {tail}는 임의 함수이며 ∫c d{by}도 임의 함수로 볼 수 있습니다.']]
    return u, residual, rows


class PdeBasicsTopic:
    id = 'pde_basics'
    title = '12.1 · PDE 기본개념과 해 검산'
    description = '후보 u(x,y,t)를 기호 미분하여 방정식의 잔차를 확인하거나, 한 변수의 도함수만 있는 PDE를 상미분방정식처럼 풉니다.'
    supports_animation = False
    inputs = (
        InputSpec('mode', '방정식', 'choice', 'Laplace', choices=CHECK_MODES + (ODE_MODE,)),
        InputSpec('candidate', '후보 해 u(x,y,t)', 'text', 'x^2-y^2', visible_in=CHECK_MODES),
        InputSpec('forcing', 'Poisson 우변 f(x,y,t)', 'text', '0', visible_in=('Poisson',)), PARAMETERS,
        InputSpec('alpha', '열확산계수 α', 'float', .1, .000001, 100, visible_in=('열',)),
        InputSpec('speed', '파동 속력 c', 'float', 1., .001, 100, visible_in=('파동',)),
        InputSpec('time', '검산 시간 t', 'float', .5, 0, 20, visible_in=CHECK_MODES),
        InputSpec('extent', '표시 구간 [-X,X]', 'float', 1., .01, 10),
        InputSpec('variable', '미분하는 변수', 'choice', 'x', choices=('x', 'y'), visible_in=(ODE_MODE,)),
        InputSpec('substitution', '치환', 'choice', SUBSTITUTIONS[0], choices=SUBSTITUTIONS, visible_in=(ODE_MODE,)),
        InputSpec('a', "a (w''의 계수, x·y 가능)", 'text', '1', visible_in=(ODE_MODE,)),
        InputSpec('b', "b (w'의 계수)", 'text', '0', visible_in=(ODE_MODE,)),
        InputSpec('c', 'c (w의 계수)', 'text', '-1', visible_in=(ODE_MODE,)),
        InputSpec('g', '우변 g(x,y)', 'text', '0', visible_in=(ODE_MODE,)),
    )
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x, y, t = symbols_for(('x', 'y', 't'))
        if p['mode'] == ODE_MODE:
            return self.solve(p, x, y)
        u = parameter_expression(p['candidate'], p['parameters'], ('x', 'y', 't'))
        if p['mode'] in ('Laplace', 'Poisson'):
            residual = sp.diff(u, x, 2) + sp.diff(u, y, 2)
            if p['mode'] == 'Poisson':
                residual -= parameter_expression(p['forcing'], p['parameters'], ('x', 'y', 't'))
        elif p['mode'] == '열':
            residual = sp.diff(u, t) - sp.Rational(str(p['alpha'])) * sp.diff(u, x, 2)
        else:
            residual = sp.diff(u, t, 2) - sp.Rational(str(p['speed'])) ** 2 * sp.diff(u, x, 2)
        residual = sp.simplify(residual)
        grid = np.linspace(-p['extent'], p['extent'], 81)
        xx, yy = np.meshgrid(grid, grid, indexing='ij')
        values = evaluate(u, (xx, yy, p['time']), ('x', 'y', 't'))
        errors = evaluate(residual, (xx, yy, p['time']), ('x', 'y', 't'))
        text = f'후보 u = {u}\n\n방정식 잔차 = {residual}\n\n기호적으로 0: {residual == 0}'
        return Result(self.id, p, {'표본 잔차 최대값': float(np.max(np.abs(errors)))},
            {'기호 검산': (['항목', '값'], [['u', str(u)], ['잔차', str(residual)], ['기호적으로 0', str(residual == 0)]])},
            dict(x=grid, y=grid, field=values, residual=errors, symbolic=text),
            ['표본 잔차가 작다는 사실만으로 전체 영역의 해임을 증명할 수 없습니다.',
             '기호 잔차가 0이어도 특이점의 정의역과 초기·경계조건을 별도로 확인하십시오. 분포를 포함한 후보는 수치 검산을 지원하지 않습니다.'])

    def solve(self, p, x, y):
        a, b, c, g = (parameter_expression(p[key], p['parameters'], ('x', 'y')) for key in 'abcg')
        u, residual, rows = solve_like_ode(a, b, c, g, p['variable'], p['substitution'], x, y)
        rows.append(['대입 검산 잔차', str(residual)])
        # 대표 그림: 모든 임의 함수를 1로 둡니다. 미정 적분이 남으면 그림을 생략합니다.
        sample = u.replace(lambda e: isinstance(e, sp.core.function.AppliedUndef), lambda e: sp.Integer(1)).doit()
        data = dict(symbolic='\n'.join(f'{k}: {v}' for k, v in rows))
        notices = ['교재 12.1 예제 2·3(75쪽): 한 변수의 도함수만 있으면 다른 변수를 매개변수로 보고 상미분방정식처럼 풉니다. '
                   '이때 적분 "상수"는 다른 변수의 임의 함수입니다.',
                   '그림은 모든 임의 함수를 1로 둔 대표 해입니다.']
        grid = np.linspace(-p['extent'], p['extent'], 81)
        if not sample.free_symbols - {x, y} and not sample.has(sp.Integral):
            xx, yy = np.meshgrid(grid, grid, indexing='ij')
            try:
                field = evaluate(sample, (xx, yy), ('x', 'y')) if sample.free_symbols else np.full(xx.shape, float(sample))
                data.update(x=grid, y=grid, field=field)
            except ValueError:
                notices.append('대표 해가 표시 구간에서 유한하지 않아 그림을 생략합니다.')
        return Result(self.id, p, {'대입 검산 잔차가 0': float(residual == 0)},
                      {'풀이': (['항목', '값'], rows)}, data, notices)

    def figures(self, result):
        d = result.data
        if 'residual' in d:
            return {'후보와 잔차': field_figure(d['x'], d['y'], [d['field'], d['residual']], ['Candidate', 'PDE residual'])}
        if 'field' in d:
            return {'대표 해': field_figure(d['x'], d['y'], [d['field']], ['Representative solution (arbitrary functions = 1)'])}
        return {}


# ── 12.4 후반: 2계 준선형 PDE의 분류, 특성, 표준형 ───────────────────
TYPE_NAMES = {-1: '쌍곡선형', 0: '포물선형', 1: '타원형'}


def characteristic_directions(A, B, C, root):
    """A dy²−2B dx dy+C dx²=0을 만족하는 두 방향 (dx, dy)를 반환합니다 (AC−B²<0인 점).

    |A|≥|C|이면 (A, B±√·), 아니면 (B∓√·, C)를 씁니다. 두 표현의 기울기는 모두 (B±√·)/A이므로
    표현을 바꿔도 같은 족이 유지됩니다((B+s)(B−s)=AC).
    """
    A, B, C, root = np.broadcast_arrays(A, B, C, root)
    use_a = np.abs(A) >= np.abs(C)
    directions = []
    for sign in (1., -1.):
        dx = np.where(use_a, A, B - sign*root)
        dy = np.where(use_a, B + sign*root, C)
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1.
        directions.append((dx/norm, dy/norm))
    return directions


def trace_characteristics(coefficients, extent, seeds=5, steps=240):
    """쌍곡선형 영역의 특성 곡선을 RK2로 추적합니다. 방향장이 뒤집히지 않도록 이전 방향과 맞춥니다."""
    step = 2*extent/steps*1.5
    grid = np.linspace(-.8*extent, .8*extent, seeds)
    sx, sy = [array.ravel() for array in np.meshgrid(grid, grid)]
    curves = []
    for family in (0, 1):
        for orientation in (1., -1.):
            x, y = sx.copy(), sy.copy()
            A, B, C = coefficients(x, y)
            alive = A*C - B*B < 0
            paths = [np.c_[x, y]]
            previous = None
            for _ in range(steps):
                A, B, C = coefficients(x, y)
                root = np.sqrt(np.maximum(B*B - A*C, 0.))
                dx, dy = characteristic_directions(A, B, C, root)[family]
                if previous is None:
                    dx, dy = orientation*dx, orientation*dy
                else:
                    flip = np.where(dx*previous[0] + dy*previous[1] < 0, -1., 1.)
                    dx, dy = flip*dx, flip*dy
                mx, my = x + .5*step*dx, y + .5*step*dy
                A2, B2, C2 = coefficients(mx, my)
                root2 = np.sqrt(np.maximum(B2*B2 - A2*C2, 0.))
                ex, ey = characteristic_directions(A2, B2, C2, root2)[family]
                flip = np.where(ex*dx + ey*dy < 0, -1., 1.)
                ex, ey = flip*ex, flip*ey
                x = np.where(alive, x + step*ex, x)
                y = np.where(alive, y + step*ey, y)
                previous = (ex, ey)
                A, B, C = coefficients(x, y)
                alive &= (A*C - B*B < 0) & (np.abs(x) <= extent) & (np.abs(y) <= extent)
                paths.append(np.where(alive[:, None], np.c_[x, y], np.nan))
                if not alive.any():
                    break
            stacked = np.stack(paths, axis=1)
            curves += [(family, path) for path in stacked if np.sum(np.isfinite(path[:, 0])) > 2]
    return curves


class PdeClassificationTopic:
    id = 'pde_classification'
    title = '12.4 · PDE의 분류, 특성, 표준형'
    description = 'A u_xx + 2B u_xy + C u_yy = F(x,y,u,u_x,u_y)의 형태를 판별식 AC−B²로 분류하고 특성과 표준형을 구합니다.'
    supports_animation = False
    inputs = (InputSpec('A', '계수 A(x,y)', 'text', '1'),
              InputSpec('B', '계수 B(x,y) (방정식에서는 2B u_xy)', 'text', '0'),
              InputSpec('C', '계수 C(x,y)', 'text', '-1'), PARAMETERS,
              InputSpec('extent', '분류 지도 범위 [−X,X]²', 'float', 2., .1, 100))
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x, y = symbols_for(('x', 'y'))
        A, B, C = (parameter_expression(p[key], p['parameters'], ('x', 'y')) for key in 'ABC')
        if A == 0 and B == 0 and C == 0:
            raise ValueError('A, B, C가 모두 0이면 2계 편미분방정식이 아닙니다.')
        discriminant = sp.simplify(A*C - B**2)
        grid = np.linspace(-p['extent'], p['extent'], 161)
        xx, yy = np.meshgrid(grid, grid, indexing='ij')

        def values(expression, X, Y):
            return np.broadcast_to(evaluate(expression, (X, Y), ('x', 'y')), np.shape(X)).astype(float)

        types = np.sign(np.round(values(discriminant, xx, yy), 12)).astype(int)
        coefficients = lambda X, Y: tuple(values(e, X, Y) for e in (A, B, C))
        rows = [['A', str(A)], ['B', str(B)], ['C', str(C)], ['AC − B²', str(discriminant)]]
        normal = []
        constant = not (A.free_symbols | B.free_symbols | C.free_symbols)
        if constant:
            kind = int(np.sign(round(float(discriminant), 12)))
            rows.append(['형태', TYPE_NAMES[kind]])
            normal = self.normal_form(A, B, C, kind, x, y)
            label = TYPE_NAMES[kind]
        else:
            present = sorted(set(types.ravel()))
            label = '혼합형 (영역에 따라 다름)' if len(present) > 1 else TYPE_NAMES[present[0]]
            rows.append(['형태 (표시 영역)', label])
            normal = self.variable_characteristics(A, B, C, x, y)
        curves = trace_characteristics(coefficients, p['extent']) if np.any(types < 0) else []
        fractions = {TYPE_NAMES[k]: float(np.mean(types == k)) for k in (-1, 0, 1)}
        notices = ['교재 식 (14)의 표기를 따릅니다: 혼합 도함수의 계수는 2B입니다. 특성방정식은 A y′² − 2B y′ + C = 0입니다(식 15, 부호에 주의).',
                   '쌍곡선형은 실수 특성이 두 족, 포물선형은 한 족, 타원형은 실수 특성이 없습니다. 그림의 곡선은 쌍곡선형 영역의 특성선입니다.']
        if not constant:
            notices.append('변수 계수는 점마다 형태가 다를 수 있습니다(혼합형, 예: Tricomi 방정식). 표준형 변환은 영역마다 따로 구해야 합니다.')
        metrics = {f'{name} 영역 비율': value for name, value in fractions.items()}
        return Result(self.id, p, metrics, {'분류': (['항목', '값'], rows), '특성과 표준형': (['항목', '값'], normal)},
                      dict(grid=grid, types=types, curves=curves, label=label,
                           symbolic='\n'.join(f'{a}: {b}' for a, b in rows + normal)), notices)

    @staticmethod
    def normal_form(A, B, C, kind, x, y):
        """상수 계수의 특성 Φ, Ψ와 교재 표의 새 변수·표준형입니다."""
        rows = []
        if A != 0:
            roots = sp.solve(sp.Eq(A*sp.Symbol('m')**2 - 2*B*sp.Symbol('m') + C, 0), sp.Symbol('m'))
            rows.append(['특성방정식', f'{A}·(y′)² − 2·({B})·y′ + {C} = 0'])
            rows.append(['기울기 y′', ', '.join(str(sp.nsimplify(root)) for root in roots)])
            invariants = [sp.simplify(y - root*x) for root in roots]
        else:
            # A=0이면 특성의 한 족은 x=상수입니다.
            invariants = [x] + ([sp.simplify(2*B*y - C*x)] if B != 0 else [])
            rows.append(['특성방정식', f'−2·({B})·y′ + {C} = 0 와 x = 상수'])
        if kind < 0:
            phi, psi = invariants[0], invariants[1]
            rows += [['특성 Φ = 상수', str(phi)], ['특성 Ψ = 상수', str(psi)],
                     ['새 변수', f'v = {phi},  w = {psi}'], ['표준형', 'u_vw = F₁(v, w, u, u_v, u_w)']]
        elif kind == 0:
            phi = invariants[0]
            # v=x, w=Φ로 바꾸면 B²=AC이므로 2계 부분은 A·u_vv만 남습니다(특성이 아닌 변수 v로 두 번 미분).
            rows += [['특성 Φ = Ψ = 상수', str(phi)], ['새 변수', f'v = x,  w = {phi}'],
                     ['표준형', 'u_vv = F₂(v, w, u, u_v, u_w)']]
        else:
            phi, psi = invariants[0], invariants[1]
            real = sp.simplify((phi + psi)/2)
            imaginary = sp.simplify((phi - psi)/(2*sp.I))
            rows += [['복소 특성 Φ, Ψ', f'{phi},  {psi}'], ['새 변수', f'v = {real},  w = {imaginary}'],
                     ['표준형', 'u_vv + u_ww = F₃(v, w, u, u_v, u_w)']]
        return rows

    @staticmethod
    def variable_characteristics(A, B, C, x, y):
        rows = [['특성방정식', f'({A})·(y′)² − 2·({B})·y′ + ({C}) = 0']]
        if A != 0:
            slopes = [sp.simplify((B + sign*sp.sqrt(B**2 - A*C))/A) for sign in (1, -1)]
            rows.append(['기울기 y′ = (B ± √(B²−AC))/A', ',  '.join(str(slope) for slope in slopes)])
        else:
            rows.append(['특성', f'x = 상수 와 y′ = ({C})/(2·({B}))'])
        rows.append(['표준형', '쌍곡선형 영역에서는 u_vw = F₁, 타원형 영역에서는 u_vv + u_ww = F₃로 영역마다 변환합니다.'])
        return rows

    def figures(self, result):
        from matplotlib.colors import ListedColormap
        d = result.data
        figure = Figure(figsize=(7, 6), layout='constrained')
        axis = figure.subplots()
        cmap = ListedColormap(['#f4a582', '#f7f7f7', '#92c5de'])
        mesh = axis.pcolormesh(d['grid'], d['grid'], d['types'].T, cmap=cmap, vmin=-1.5, vmax=1.5, shading='auto')
        colorbar = figure.colorbar(mesh, ax=axis, ticks=[-1, 0, 1])
        colorbar.ax.set_yticklabels(['hyperbolic', 'parabolic', 'elliptic'])
        colors = ('tab:red', 'tab:blue')
        for family, path in d['curves']:
            axis.plot(path[:, 0], path[:, 1], color=colors[family], lw=.8)
        axis.set(xlabel='x', ylabel='y', aspect='equal', title='Type map (sign of AC−B²) and characteristics',
                 xlim=(d['grid'][0], d['grid'][-1]), ylim=(d['grid'][0], d['grid'][-1]))
        return {'분류 지도와 특성선': figure}


# ── 12.4, 12.7: 무한 공간에서의 해 ────────────────────────────────
class DalembertTopic(EvolutionTopic):
    id = 'dalembert'
    title = '12.4 · D’Alembert 해와 특성선'
    description = '무한 현의 초기 변위·속도를 입력합니다. 화면 끝은 경계가 아니므로 반사되지 않습니다.'
    inputs = (InputSpec('initial', '초기 변위 f(x)', 'text', 'exp(-x^2)'),
              InputSpec('velocity', '초기 속도 g(x)', 'text', '0'), PARAMETERS,
              InputSpec('speed', '파동 속력 c', 'float', 1., .001, 20),
              InputSpec('extent', '표시 반폭 X', 'float', 5., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 501, 101, 2001),
              InputSpec('quadrature_points', '속도 적분 표본 수', 'int', 8001, 1001, 32001)) + TIME_INPUTS
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(-p['extent'], p['extent'], p['num_points'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        f, g = function(p['initial'], p['parameters']), function(p['velocity'], p['parameters'])
        frames, grid = dalembert(f, g, x, time, p['speed'], p['quadrature_points'])
        return evolution_result(self.id, p, x, time, frames, f(x),
            ['u=[f(x−ct)+f(x+ct)]/2 + (1/2c)∫[x−ct,x+ct]g(s)ds입니다.',
             f'초기 함수는 표시 구간 밖 [{grid[0]:g}, {grid[-1]:g}]에서도 정의되어야 합니다.',
             '속도 적분은 균일 격자의 누적 사다리꼴 적분과 보간입니다. 좁은 펄스·불연속에서는 적분 표본 수를 늘리십시오.'])

    def figures(self, result):
        figures = super().figures(result)
        d, p = result.data, result.params
        figure = Figure(figsize=(8, 5), layout='constrained')
        axis = figure.subplots()
        axis.plot(p['speed'] * d['time'], d['time'], label='x=ct')
        axis.plot(-p['speed'] * d['time'], d['time'], label='x=-ct')
        axis.set(xlabel='x', ylabel='t', title='Characteristics through the origin')
        axis.legend(); axis.grid(alpha=.3)
        figures['특성선'] = figure
        return figures


class InfiniteHeatTopic(EvolutionTopic):
    id = 'infinite_heat'
    title = '12.7 · 무한 막대와 열핵'
    description = '무한 막대의 열핵 합성곱을 계산합니다. Fourier 변환 해 F(ω,t)=F(ω,0)exp(−αω²t)와 동등합니다.'
    MODES = ('무한 막대', '반무한 막대 · u(0,t)=0')
    inputs = (InputSpec('mode', '영역', 'choice', MODES[0], choices=MODES),
              InputSpec('initial', '초기 온도 f(x)', 'text', 'exp(-x^2)'), PARAMETERS,
              InputSpec('alpha', '열확산계수 α', 'float', .2, .000001, 10),
              InputSpec('extent', '표시 반폭 X', 'float', 5., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 401, 101, 1001),
              InputSpec('quadrature_points', '합성곱 격자 점 수', 'int', 16385, 2049, 65537)) + TIME_INPUTS
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        semi = p['mode'] == self.MODES[1]
        time = np.linspace(0, p['end_time'], p['time_points'])
        f = function(p['initial'], p['parameters'])
        if semi:
            # 사인 변환 해 (20)은 f를 기함수로 확장한 무한 막대의 해와 같습니다. x=0의 도약도 자동 처리됩니다.
            source = lambda values: np.sign(values) * f(np.abs(values))
            half = p['num_points']
            full = np.linspace(-p['extent'], p['extent'], 2 * half - 1)
            x = full[half - 1:]
        else:
            source = f
            full = x = np.linspace(-p['extent'], p['extent'], p['num_points'])
        frames = infinite_heat(source, full, time, p['alpha'], p['quadrature_points'])
        refined = infinite_heat(source, full, time[-1:], p['alpha'], 2 * p['quadrature_points'] - 1)[0]
        if semi:
            frames, refined = frames[:, len(full) - len(x):], refined[len(full) - len(x):]
            frames[1:, 0] = 0.
        notices = ['열핵 u=(1/√(4παt))∫ exp(−(x−ξ)²/(4αt))f(ξ)dξ를 셀마다 정확히 적분한 가중치(erf 차이)로 합성곱합니다.',
                   '표시 경계에서 온도를 고정하지 않습니다. 함수는 실수축 전체에 정의되어야 합니다.',
                   '격자를 두 배로 세분한 결과와의 차이는 오차 상한이 아니라 수렴 확인용입니다. 도약이 있는 초기값도 격자 간격에 비례하여 수렴합니다.']
        if semi:
            notices[1] = ('반무한 막대 x≥0, u(0,t)=0: Fourier 사인 변환 해 u=(2/π)∫∫f(p)sin(wp)e^(−αw²t)sin(wx)dpdw (교재 식 20)는 '
                          '초기값을 기함수로 확장한 무한 막대의 해와 같습니다. 오른쪽 표시 끝은 경계가 아닙니다.')
        result = evolution_result(self.id, p, x, time, frames, f(x), notices,
            {'격자 세분화 최종 최대 차이': float(np.max(np.abs(frames[-1] - refined)))})
        mass = np.trapezoid(frames, x=x, axis=1)
        checked(mass)
        result.tables['표시 구간 내 적분'] = (['t', 'integral over displayed interval'], np.column_stack((time, mass)))
        return result


# ── 12.8~12.11의 공통 2차원 그림 ──────────────────────────────────
def field_figure(x, y, fields, titles, polar=False):
    figure = Figure(figsize=(9, 5), layout='constrained')
    axes = figure.subplots(1, len(fields), squeeze=False)[0]
    for axis, values, title in zip(axes, fields, titles):
        if polar:
            rr, tt = np.meshgrid(y, x, indexing='ij')
            # θ 주기 격자에 끝점을 추가하여 원판의 이음매를 닫습니다.
            xx, yy = rr * np.cos(tt), rr * np.sin(tt)
            mesh = axis.contourf(np.c_[xx, xx[:, :1]], np.c_[yy, yy[:, :1]],
                                 np.c_[values, values[:, :1]], levels=24, cmap='coolwarm')
        else:
            mesh = axis.pcolormesh(x, y, values.T, shading='auto', cmap='coolwarm')
        axis.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        figure.colorbar(mesh, ax=axis, label='u')
    return figure


def degenerate_groups(omega, tolerance=1e-9):
    """같은 진동수를 갖는 서로 다른 (m, n) 조합을 모읍니다. 대칭 쌍 (m,n)·(n,m)도 포함합니다."""
    values = omega.ravel()
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for previous, index in zip(order[:-1], order[1:]):
        if abs(values[index] - values[previous]) <= tolerance*max(1., values[previous]):
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)
    count = omega.shape[1]
    return [(float(values[group[0]]), [(i // count + 1, i % count + 1) for i in sorted(group)])
            for group in groups if len(group) > 1]


def nodal_figure(x, y, values, polar=False):
    """변위의 부호와 절선 u=0을 그립니다. 극좌표 입력은 (r, θ) 격자입니다."""
    figure = Figure(figsize=(6, 5), layout='constrained')
    axis = figure.subplots()
    if polar:
        rr, tt = np.meshgrid(y, np.r_[x, 2*np.pi], indexing='ij')
        closed = np.c_[values, values[:, :1]]
        xx, yy = rr*np.cos(tt), rr*np.sin(tt)
        limit = max(float(np.max(np.abs(closed))), 1e-12)
        axis.contourf(xx, yy, closed, levels=24, cmap='coolwarm', vmin=-limit, vmax=limit)
        interior = closed[:-1]
        if np.min(interior) < 0 < np.max(interior):
            axis.contour(xx[:-1], yy[:-1], interior, levels=[0], colors='k', linewidths=1.5)
    else:
        limit = max(float(np.max(np.abs(values))), 1e-12)
        axis.pcolormesh(x, y, values.T, shading='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
        interior = values[1:-1, 1:-1]
        if np.min(interior) < 0 < np.max(interior):
            axis.contour(x[1:-1], y[1:-1], interior.T, levels=[0], colors='k', linewidths=1.5)
    axis.set(title='Initial displacement: sign and nodal lines u=0', xlabel='x', ylabel='y', aspect='equal')
    return figure


def membrane_balance(a, v, omega, length, height, speed, moment):
    """교재 12.8 식 (1)~(3): 미소부분 ΔxΔy의 수직 합력 TΔy[u_x(x+Δx)−u_x(x)] + TΔx[u_y(y+Δy)−u_y(y)]를
    질량 ρΔxΔy로 나누면 c²[Δu_x/Δx + Δu_y/Δy]이고 Δ→0에서 u_tt입니다. (x₀, y₀)는 |u_tt|가 가장 큰 내부 점입니다.
    """
    count = a.shape[0]
    kx = np.arange(1, count + 1) * np.pi / length
    ky = np.arange(1, count + 1) * np.pi / height
    amplitude = a * np.cos(omega * moment) + np.divide(v, omega) * np.sin(omega * moment)

    def field(x, y, dx=0, dy=0):
        # dx, dy번 미분한 모드 합입니다(0 또는 1).
        bx = np.cos(np.outer(np.atleast_1d(x), kx)) * kx if dx else np.sin(np.outer(np.atleast_1d(x), kx))
        by = np.cos(np.outer(np.atleast_1d(y), ky)) * ky if dy else np.sin(np.outer(np.atleast_1d(y), ky))
        return np.einsum('im,mn,jn->ij', bx, amplitude, by)

    gx = np.linspace(.1 * length, .9 * length, 81)
    gy = np.linspace(.1 * height, .9 * height, 81)
    acceleration = -np.einsum('im,mn,jn->ij', np.sin(np.outer(gx, kx)), amplitude * omega ** 2, np.sin(np.outer(gy, ky)))
    i, j = np.unravel_index(np.argmax(np.abs(acceleration)), acceleration.shape)
    x0, y0, target = float(gx[i]), float(gy[j]), float(acceleration[i, j])
    rows = []
    for divisor in (8, 16, 32, 64, 128, 256):
        hx, hy = length / divisor, height / divisor
        quotient = speed ** 2 * ((field(x0 + hx / 2, y0, dx=1) - field(x0 - hx / 2, y0, dx=1))[0, 0] / hx
                                 + (field(x0, y0 + hy / 2, dy=1) - field(x0, y0 - hy / 2, dy=1))[0, 0] / hy)
        rows.append([hx, hy, float(quotient), target, abs(float(quotient) - target)])
    return dict(rows=rows, x0=x0, y0=y0, moment=moment, field=field)


class MembraneTopic:
    id = 'rectangle_membrane'
    title = '12.8–12.9 · 직사각형 막과 이중 급수'
    description = '가장자리가 고정된 균질 막의 u_tt=c²(u_xx+u_yy)를 이중 사인 급수로 계산합니다.'
    inputs = (InputSpec('initial', '초기 변위 f(x,y)', 'text', 'sin(pi*x/l)*sin(pi*y/height)'),
              InputSpec('velocity', '초기 속도 g(x,y)', 'text', '0'), PARAMETERS,
              InputSpec('length', '가로 길이 l', 'float', 1., .05, 20, help='수식에 l이라고 쓰면 이 값이 들어갑니다. 예: sin(pi*x/l)'),
              InputSpec('height', '세로 길이 height', 'float', 1., .05, 20, help='수식에 height라고 쓰면 이 값이 들어갑니다. 예: sin(pi*y/height)'),
              InputSpec('speed', '파동 속력 c', 'float', 1., .01, 20),
              InputSpec('N', '방향별 모드 수 N', 'int', 6, 1, 16),
              InputSpec('num_points', '방향별 공간 표본 수', 'int', 81, 41, 161)) + TIME_INPUTS
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(0, p['length'], p['num_points'])
        y = np.linspace(0, p['height'], p['num_points'])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        variables = ('x', 'y', 'l', 'height')
        initial = function(p['initial'], p['parameters'], variables)(xx, yy, p['length'], p['height'])
        velocity = function(p['velocity'], p['parameters'], variables)(xx, yy, p['length'], p['height'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames, energy, a, v, omega = rectangle_membrane(x, y, initial, velocity,
                    p['length'], p['height'], p['speed'], p['N'], time)
        modes = [[m + 1, n + 1, a[m, n], v[m, n], omega[m, n]] for m in range(p['N']) for n in range(p['N'])]
        result = self.make_result(p, x, y, time, frames, initial, energy,
                    (['m', 'n', 'A_mn', 'V_mn', 'omega'], modes))
        if max(np.max(np.abs(initial[[0,-1], :])), np.max(np.abs(initial[:, [0,-1]])),
               np.max(np.abs(velocity[[0,-1], :])), np.max(np.abs(velocity[:, [0,-1]]))) > 1e-8:
            result.notices.append('초기 변위·속도의 가장자리가 고정단 조건과 다릅니다. 급수 해의 가장자리는 0입니다.')
        result.data['omega'] = omega
        model = membrane_balance(a, v, omega, p['length'], p['height'], p['speed'], time[len(time) // 3])
        result.tables['모델링: 미소부분의 힘'] = (['Δx', 'Δy', 'c²(Δu_x/Δx + Δu_y/Δy)', 'u_tt (점 x₀, y₀)', '차이'], model['rows'])
        result.data['modeling'] = {key: model[key] for key in ('x0', 'y0', 'moment')}
        result.data['modal'] = (a, v)
        groups = degenerate_groups(omega)
        result.tables['중복 고유값'] = (['omega', 'c·π·√(m²/l²+n²/height²)의 같은 값을 갖는 (m, n)'],
                                       [[value, ', '.join(f'({m},{n})' for m, n in pairs)] for value, pairs in groups])
        if groups:
            result.notices.append(f'같은 진동수를 갖는 서로 다른 모드 조합이 {len(groups)}개 있습니다(교재 12.9 예제 1). '
                                  '같은 진동수라도 절선 모양이 전혀 다른 진동이 존재합니다.')
        self.warn_time_sampling(result, omega)
        return result

    @staticmethod
    def warn_time_sampling(result, omega):
        time = result.data['time']
        samples = 2*np.pi/(float(np.max(omega))*(time[1]-time[0]))
        if samples < 10:
            result.notices.append(f'최고 모드의 주기당 출력 표본이 {samples:.3g}개입니다. 시간 표본 수를 늘리거나 마지막 시간을 줄이십시오.')

    def make_result(self, p, x, y, time, frames, initial, energy, modes, polar=False):
        metrics = {'초기 재구성 RMSE': float(np.sqrt(np.mean((frames[0]-initial)**2))),
                   '에너지 최대 변화': float(np.max(np.abs(energy-energy[0])))}
        checked(list(metrics.values()))
        data = dict(x=x, y=y, time=time, frames=frames, initial=initial, energy=energy, polar=polar)
        samples = GridRows((y, x), (frames[-1],), (1, 0)) if polar else GridRows((x, y), (frames[-1],))
        tables = {'모드 계수': modes, '에너지': (['t', 'energy'], np.column_stack((time, energy))),
                  '최종 공간 표본': (['theta' if polar else 'x', 'r' if polar else 'y', 'u'], samples)}
        return Result(self.id, p, metrics, tables, data,
            ['고정 가장자리·일정한 속력·무감쇠·무외력의 유한 모드 해입니다. 임의 형태의 막은 지원하지 않습니다.',
             '에너지는 단위 면밀도 모드 에너지입니다. 출력 시간 간격이 크면 진동을 놓칠 수 있습니다.'])

    def figures(self, result):
        d = result.data
        figures = {'막의 변위': field_figure(d['x'], d['y'], [d['frames'][0], d['frames'][-1]],
                            ['t=0 (projection)', f"t={d['time'][-1]:g}"], d['polar'])}
        figure = Figure(figsize=(8, 4), layout='constrained')
        axis = figure.subplots()
        axis.plot(d['time'], d['energy'])
        axis.set(xlabel='t', ylabel='Energy', title='Modal energy'); axis.grid(alpha=.3)
        figures['에너지'] = figure
        figures['초기 변위의 절선'] = nodal_figure(d['x'], d['y'], d['frames'][0], d['polar'])
        figures['정규진동과 절선'] = self.mode_gallery(result)
        if 'modeling' in d:
            figures['모델링: 막의 힘'] = self.force_diagram(result)
        return figures

    @staticmethod
    def force_diagram(result):
        """교재 그림 301: 미소부분 ΔxΔy의 네 변에 장력 TΔy, TΔx가 접선 방향으로 작용합니다."""
        d, p = result.data, result.params
        model = membrane_balance(*d['modal'], d['omega'], p['length'], p['height'], p['speed'], d['modeling']['moment'])
        x0, y0 = model['x0'], model['y0']
        hx, hy = p['length'] / 8, p['height'] / 8
        x = np.linspace(0, p['length'], 121)
        y = np.linspace(0, p['height'], 121)
        surface = model['field'](x, y)
        figure = Figure(figsize=(10, 4.5), layout='constrained')
        left, right = figure.subplots(1, 2)
        limit = max(float(np.max(np.abs(surface))), 1e-12)
        left.pcolormesh(x, y, surface.T, shading='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
        corners = np.array([[x0 - hx/2, y0 - hy/2], [x0 + hx/2, y0 - hy/2], [x0 + hx/2, y0 + hy/2], [x0 - hx/2, y0 + hy/2], [x0 - hx/2, y0 - hy/2]])
        left.plot(corners[:, 0], corners[:, 1], 'k', lw=2)
        for (px, py, ux, uy, label) in ((x0 - hx/2, y0, -1, 0, 'TΔy'), (x0 + hx/2, y0, 1, 0, 'TΔy'),
                                        (x0, y0 - hy/2, 0, -1, 'TΔx'), (x0, y0 + hy/2, 0, 1, 'TΔx')):
            left.annotate('', xy=(px + ux*hx*.9, py + uy*hy*.9), xytext=(px, py), arrowprops=dict(arrowstyle='->', lw=1.8))
            left.text(px + ux*hx, py + uy*hy, label, ha='center', va='center', fontsize=8)
        left.set(title=f"u(x,y,t={model['moment']:.3g}) and element ΔxΔy (top view)", xlabel='x', ylabel='y', aspect='equal')
        section = model['field'](x, np.array([y0]))[:, 0]
        right.plot(x, section, color='0.5')
        segment = (x >= x0 - hx/2) & (x <= x0 + hx/2)
        right.plot(x[segment], section[segment], 'k', lw=2.5)
        ends = [section.min(), section.max()]
        for point, sign, color in ((x0 - hx/2, -1, 'tab:blue'), (x0 + hx/2, 1, 'tab:red')):
            height_value = model['field'](np.array([point]), np.array([y0]))[0, 0]
            slope = model['field'](np.array([point]), np.array([y0]), dx=1)[0, 0]
            direction = sign*np.array([1., slope])/np.hypot(1., slope)*.2*p['length']
            tip = height_value + direction[1]
            ends += [tip]
            # 화살표 끝이 곡선 범위 밖이어도 그려지도록 축 범위에 포함합니다.
            right.annotate('', xy=(point + direction[0], tip), xytext=(point, height_value), annotation_clip=False,
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
        margin = .08*max(max(ends) - min(ends), 1e-12)
        right.set_ylim(min(ends) - margin, max(ends) + margin)
        right.set(title=f'Section y = {y0:.3g}: tension TΔy at both ends (tangent)', xlabel='x', ylabel='u')
        right.grid(alpha=.3)
        return figure

    def mode_gallery(self, result):
        """교재 그림 303과 같이 낮은 정규진동 6개의 부호와 절선(u=0)을 보여 줍니다."""
        p = result.params
        x = np.linspace(0, p['length'], 121)
        y = np.linspace(0, p['height'], 121)
        figure = Figure(figsize=(9, 6), layout='constrained')
        axes = figure.subplots(2, 3)
        for axis, (m, n) in zip(axes.ravel(), [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]):
            mode = np.outer(np.sin(m*np.pi*x/p['length']), np.sin(n*np.pi*y/p['height']))
            axis.pcolormesh(x, y, mode.T, shading='auto', cmap='coolwarm', vmin=-1, vmax=1)
            if m > 1 or n > 1:
                # 가장자리의 u=0은 경계조건이므로 내부 절선만 그립니다.
                axis.contour(x[1:-1], y[1:-1], mode[1:-1, 1:-1].T, levels=[0], colors='k', linewidths=1.2, linestyles='--')
            omega = p['speed']*np.pi*np.hypot(m/p['length'], n/p['height'])
            axis.set(title=f'u{m}{n}, ω={omega:.4g}', aspect='equal', xticks=[], yticks=[])
        figure.suptitle('Normal modes and nodal lines (dashed)', fontsize=10)
        return figure


    def animation(self, result):
        d = result.data
        # 극좌표 재생에서도 마지막 각도와 첫 각도를 연결하여 이음매를 닫습니다.
        x = np.r_[d['x'], 2*np.pi] if d['polar'] else d['x']
        frames = np.concatenate((d['frames'], d['frames'][:, :, :1]), axis=2) if d['polar'] else d['frames'].transpose(0,2,1)
        return dict(kind='field', x=x, y=d['y'], polar=d['polar'], frames=frames,
                    labels=[f't={t:.5g}' for t in d['time']], xlabel='x', ylabel='y')


class DiskMembraneTopic(MembraneTopic):
    id = 'disk_membrane'
    title = '12.10 · 원형 막과 Fourier–Bessel 급수'
    description = '고정 원주를 갖는 원형 막입니다. r·theta·l을 입력에 사용하며 l은 반지름입니다.'
    inputs = (InputSpec('initial', '초기 변위 f(r,theta)', 'text', '1-(r/l)^2'),
              InputSpec('velocity', '초기 속도 g(r,theta)', 'text', '0'), PARAMETERS,
              InputSpec('radius', '반지름 l', 'float', 1., .05, 20, help='수식에 l이라고 쓰면 이 값이 들어갑니다. 예: sin(pi*r/l)'),
              InputSpec('speed', '파동 속력 c', 'float', 1., .01, 20),
              InputSpec('radial_count', '방사 모드 수', 'int', 6, 1, 16),
              InputSpec('angular_count', '최대 각도 차수 m', 'int', 2, 0, 6),
              InputSpec('num_points', '방사 표본 수', 'int', 101, 41, 201),
              InputSpec('angular_points', '각도 표본 수', 'int', 81, 33, 161)) + TIME_INPUTS
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        r = np.linspace(0, p['radius'], p['num_points'])
        theta = np.linspace(0, 2*np.pi, p['angular_points'], endpoint=False)
        rr, tt = np.meshgrid(r, theta, indexing='ij')
        variables = ('r', 'theta', 'l')
        f = function(p['initial'], p['parameters'], variables)
        g = function(p['velocity'], p['parameters'], variables)
        initial, velocity = f(rr, tt, p['radius']), g(rr, tt, p['radius'])
        if not np.allclose(initial[0], initial[0,0]) or not np.allclose(velocity[0], velocity[0,0]):
            raise ValueError('r=0에서 함수값은 theta에 관계없이 하나여야 합니다.')
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames, energy, a, v, omega, labels = disk_membrane(r, theta,
            lambda rho, angle: f(rho, angle, p['radius']), lambda rho, angle: g(rho, angle, p['radius']),
            p['radius'], p['speed'], p['radial_count'], p['angular_count'], time)
        rows = [[*label, a[i], v[i], omega[i]] for i, label in enumerate(labels)]
        result = self.make_result(p, theta, r, time, frames, initial, energy,
            (['m', 'n', 'angular', 'A', 'V', 'omega'], rows), polar=True)
        if not np.allclose(initial[-1], 0) or not np.allclose(velocity[-1], 0):
            result.notices.append('입력 원주 값이 고정 경계 0과 다릅니다. 모드 해의 원주는 0입니다.')
        result.data['omega'] = omega
        self.warn_time_sampling(result, omega)
        return result

    def mode_gallery(self, result):
        """교재 그림 309·310: 축대칭 모드 J₀(αₘr/R)의 동심원 절선과 비축대칭 모드의 지름 절선."""
        from scipy.special import jn_zeros, jv
        radius = result.params['radius']
        r = np.linspace(0, radius, 81)
        theta = np.linspace(0, 2*np.pi, 121)
        rr, tt = np.meshgrid(r, theta, indexing='ij')
        xx, yy = rr*np.cos(tt), rr*np.sin(tt)
        figure = Figure(figsize=(9, 6), layout='constrained')
        axes = figure.subplots(2, 3)
        for axis, (m, n) in zip(axes.ravel(), [(0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (3, 2)]):
            root = jn_zeros(m, n)[-1]
            mode = jv(m, root*rr/radius)*np.cos(m*tt)
            axis.contourf(xx, yy, mode, levels=20, cmap='coolwarm', vmin=-1, vmax=1)
            if m > 0 or n > 1:
                axis.contour(xx[:-1], yy[:-1], mode[:-1], levels=[0], colors='k', linewidths=1.2, linestyles='--')
            axis.set(title=f'm={m}, n={n}, ω={result.params["speed"]*root/radius:.4g}', aspect='equal', xticks=[], yticks=[])
        figure.suptitle('Normal modes J_m(α_mn r/R)cos(mθ) and nodal lines (dashed)', fontsize=10)
        return figure


# ── 12.6 후반: 직사각형의 정상 상태 열전도 (Laplace 방정식) ────────────
class RectangleLaplaceTopic:
    id = 'rectangle_laplace'
    title = '12.6 · 직사각형 정상 상태와 Laplace 방정식'
    description = '직사각형 [0,l]×[0,height]에서 u_xx+u_yy=0의 네 변 Dirichlet 문제를 sinh 급수의 중첩으로 계산합니다.'
    supports_animation = False
    SIDE_VARIABLES = {'top': ('x', 'l', 'height'), 'bottom': ('x', 'l', 'height'),
                      'left': ('y', 'l', 'height'), 'right': ('y', 'l', 'height')}
    inputs = (InputSpec('top', '윗변 u(x, height)', 'text', 'sin(pi*x/l)'),
              InputSpec('bottom', '아랫변 u(x, 0)', 'text', '0'),
              InputSpec('left', '왼쪽 변 u(0, y)', 'text', '0'),
              InputSpec('right', '오른쪽 변 u(l, y)', 'text', '0'), PARAMETERS,
              InputSpec('length', '가로 길이 l', 'float', 1., .01, 1000, help='수식에 l이라고 쓰면 이 값이 들어갑니다. 예: sin(pi*x/l)'),
              InputSpec('height', '세로 길이 height', 'float', 1., .01, 1000, help='수식에 height라고 쓰면 이 값이 들어갑니다. 예: sin(pi*y/height)'),
              InputSpec('N', '변마다 급수 차수 N', 'int', 30, 1, 200),
              InputSpec('num_points', '방향별 표시 표본 수', 'int', 101, 41, 301),
              InputSpec('quadrature_points', '경계 구적 노드 수', 'int', 1024, 128, 8192))
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(0, p['length'], p['num_points'])
        y = np.linspace(0, p['height'], p['num_points'])
        sides, functions = {}, {}
        for side, variables in self.SIDE_VARIABLES.items():
            expression = parameter_expression(p[side], p['parameters'], variables)
            functions[side] = (lambda e, v: lambda s: evaluate(e, (s, p['length'], p['height']), v))(expression, variables)
            if expression != 0:
                sides[side] = functions[side]
        field, coefficients = rectangle_laplace(sides, x, y, p['length'], p['height'], p['N'], p['quadrature_points'])
        targets = {'top': (field[:, -1], functions['top'](x)), 'bottom': (field[:, 0], functions['bottom'](x)),
                   'left': (field[0], functions['left'](y)), 'right': (field[-1], functions['right'](y))}
        # 모서리 두 점은 두 변의 값이 다를 수 있으므로 변 내부만 비교합니다.
        rmse = max(float(np.sqrt(np.mean((got[1:-1] - want[1:-1]) ** 2))) for got, want in targets.values())
        corners = [(functions['bottom'](np.array([0.]))[0], functions['left'](np.array([0.]))[0]),
                   (functions['bottom'](np.array([p['length']]))[0], functions['right'](np.array([0.]))[0]),
                   (functions['top'](np.array([0.]))[0], functions['left'](np.array([p['height']]))[0]),
                   (functions['top'](np.array([p['length']]))[0], functions['right'](np.array([p['height']]))[0])]
        notices = ['각 변의 문제 u=ΣAₙ sin(nπs/L)·sinh(…)/sinh(…)를 계산하여 더합니다(중첩 원리). 윗변만 주면 교재 식 (17)·(18)입니다.',
                   '정상 상태 열전도·정전기 퍼텐셜·고정된 막의 정적 변위가 같은 수학 모형입니다(교재 101쪽 "수학의 통합력").',
                   '경계 계수는 경계값의 도약점을 나눈 Gauss–Legendre 구적으로 계산합니다. 변 위의 재구성에는 Gibbs 진동이 남을 수 있습니다.']
        if any(abs(first - second) > 1e-9 for first, second in corners):
            notices.append('모서리에서 두 변의 경계값이 다릅니다. 모서리 근처에서 급수 수렴이 느리고 진동이 생깁니다.')
        center = field[len(x) // 2, len(y) // 2]
        rows = [[side, n + 1, value] for side, values in coefficients.items() for n, value in enumerate(values)]
        return Result(self.id, p, {'중심 온도': float(center), '경계 재구성 RMSE (최대 변)': rmse,
                                   '최고 온도': float(field.max()), '최저 온도': float(field.min())},
                      {'변별 사인 계수': (['side', 'n', 'coefficient'], rows),
                       '정상 상태 온도': (['x', 'y', 'u'], GridRows((x, y), (field,)))},
                      dict(x=x, y=y, field=field, coefficients=coefficients), notices)

    def figures(self, result):
        d = result.data
        figure = field_figure(d['x'], d['y'], [d['field']], ['Steady temperature u(x,y)'])
        axis = figure.axes[0]
        lines = axis.contour(d['x'], d['y'], d['field'].T, levels=12, colors='k', linewidths=.6)
        axis.clabel(lines, fontsize=7)
        axis.set_title('Steady temperature and isotherms')
        return {'정상 상태와 등온선': figure}


# ── 12.11: 원통·구 좌표에서의 내부 퍼텐셜 ──────────────────────────
class PotentialTopic:
    id = 'coordinate_potential'
    title = '12.11 · 원통·구 좌표의 Laplace 퍼텐셜'
    description = '원판 내부(z 독립 원통) 또는 축대칭 구 내부·외부의 Dirichlet 경계값 문제를 계산합니다.'
    supports_animation = False
    MODES = ('원판 · 원통 z 독립', '구 · 축대칭', '구 · 축대칭 외부')
    inputs = (InputSpec('mode', '영역', 'choice', MODES[0], choices=MODES),
              InputSpec('boundary', '경계값 h(theta) 또는 h(mu)', 'text', 'cos(2*theta)'), PARAMETERS,
              InputSpec('radius', '반지름 R', 'float', 1., .05, 20),
              InputSpec('N', '최대 차수 N', 'int', 8, 1, 30),
              InputSpec('num_points', '경계 구적 노드 수', 'int', 1001, 201, 8001))
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        exterior = p['mode'] == self.MODES[2]
        # 외부해는 R에서 3R까지 표시합니다. 무한대에서 0으로 감쇠합니다.
        r = np.linspace(p['radius'], 3*p['radius'], 81) if exterior else np.linspace(0, p['radius'], 81)
        theta = np.linspace(0, 2*np.pi, 241, endpoint=False)
        if p['mode'] == self.MODES[0]:
            boundary = function(p['boundary'], p['parameters'], ('theta',))
            field, constant, coefficients = disk_potential(boundary, r, theta, p['radius'], p['N'], p['num_points'])
            target = boundary(theta)
            rows = [[0, constant, 0], *coefficients]
            columns = ['n', 'a_n', 'b_n']
        else:
            boundary = function(p['boundary'], p['parameters'], ('mu',))
            field, coefficients = sphere_potential(boundary, r, theta, p['radius'], p['N']+1,
                                                   p['num_points'], exterior)
            target = boundary(np.cos(theta))
            rows, columns = list(enumerate(coefficients)), ['n', 'c_n']
        edge = 0 if exterior else -1
        rmse = float(np.sqrt(np.mean(checked((field[edge]-target)**2))))
        formula = ('구 외부: u=Σcₙ(R/r)ⁿ⁺¹Pₙ(cos φ), r≥R. 무한대에서 0입니다.' if exterior else
                   '원판: u=a₀+Σ(r/R)ⁿ(aₙcos nθ+bₙsin nθ). 구 내부: u=Σcₙ(r/R)ⁿPₙ(cos φ).')
        return Result(self.id, p, {'경계 재구성 RMSE': rmse},
            {'계수': (columns, rows), '퍼텐셜': (['r', 'theta', 'u'], GridRows((r, theta), (field,)))},
            # full_field는 이전 버전(노트북)과의 호환용 이름입니다.
            dict(x=theta, y=r, field=field, full_field=field, theta=theta, r=r, exterior=exterior),
            [formula,
             '경계 계수는 경계값의 도약점을 찾아 구간을 나눈 Gauss–Legendre 구적으로 계산합니다. 도약점 근처의 재구성에는 Gibbs 진동이 남습니다.',
             'Dirichlet 조건만 지원합니다. 구 그림은 대칭축을 포함하는 단면입니다. 일반 3차원 원통·비축대칭 구·Neumann·Robin 조건은 지원하지 않습니다.'])

    def figures(self, result):
        d = result.data
        title = 'Exterior potential (section, R ≤ r ≤ 3R)' if d.get('exterior') else 'Interior potential (section)'
        figure = field_figure(d['x'], d['y'], [d['field']], [title], True)
        if result.params['mode'].startswith('구'):
            figure.axes[0].set(xlabel='z (symmetry axis)', ylabel='Signed radial coordinate')
        return {'퍼텐셜 단면': figure}


# ── 12.12: Laplace 변환의 시간 지연과 반무한 열전도 ────────────────
class PdeLaplaceTopic(EvolutionTopic):
    id = 'pde_laplace'
    title = '12.12 · Laplace 변환으로 구하는 PDE 해'
    description = '초기값 0인 반무한 현의 경계 입력 전달과 반무한 막대의 계단 온도 응답을 확인합니다.'
    inputs = (InputSpec('mode', '방정식', 'choice', '반무한 현', choices=('반무한 현', '반무한 열전도')),
              InputSpec('boundary', '경계 변위 h(t)', 'text', 'Piecewise((sin(t),t<2*pi),(0,True))', visible_in=('반무한 현',)),
              PARAMETERS, InputSpec('speed', '속력 c', 'float', 1., .01, 20, visible_in=('반무한 현',)),
              InputSpec('alpha', '열확산계수 α', 'float', .2, .000001, 20, visible_in=('반무한 열전도',)),
              InputSpec('amplitude', '계단 경계 온도 A', 'float', 1., -100, 100, visible_in=('반무한 열전도',)),
              InputSpec('extent', '표시 길이 X', 'float', 8., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 501, 101, 2001)) + TIME_INPUTS
    examples = {}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(0, p['extent'], p['num_points'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames = np.zeros((len(time), len(x)))
        if p['mode'] == '반무한 현':
            h = function(p['boundary'], p['parameters'], ('t',))
            delayed = time[:, None] - x / p['speed']
            active = delayed >= 0
            # 인과 영역만 평가하여 sqrt(t)처럼 음의 시간에서 정의되지 않은 입력도 처리합니다.
            frames[active] = h(delayed[active])
            symbolic = '초기 변위·속도 0\ns²U=c²U_xx\nU(x,s)=H(s) exp(−xs/c)\nu(x,t)=h(t−x/c) Heaviside(t−x/c)\nH(s)=Laplace[h(t)] (형식적 표기)'
        else:
            frames[1:] = p['amplitude'] * erfc(x[None, :] / (2*np.sqrt(p['alpha']*time[1:, None])))
            frames[0, 0] = p['amplitude']
            symbolic = '초기 온도 0, u(0,t)=A\nsU=αU_xx\nU(x,s)=(A/s) exp(−x sqrt(s/α))\nu(x,t)=A erfc(x/(2 sqrt(αt))), t>0'
        result = evolution_result(self.id, p, x, time, frames, np.zeros_like(x),
            ['x≥0의 반무한 영역이며 화면 오른쪽 끝에서 반사·고정 경계조건을 부과하지 않습니다.',
             't=0의 모서리에서 초기값과 경계값이 다르면 경계값을 표시합니다. 파면 도착 시점에는 오른쪽 값을 사용합니다.',
             '표시 공식은 이 두 문제에서 유도한 해입니다. 임의 PDE의 기호 Laplace 자동 해법은 아닙니다.'])
        result.data['symbolic'] = symbolic
        result.tables['변환 과정'] = (['단계'], [[line] for line in symbolic.splitlines()])
        return result
