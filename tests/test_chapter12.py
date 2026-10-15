"""PDE 해석해·초기조건·경계조건·에너지와 화면·저장 계약 검증."""
import numpy as np
import pytest
from scipy.special import jn_zeros, jv, erfc
from engineering_math.topics import get_topic
from engineering_math.core.chapter12 import (dalembert, infinite_heat, rectangle_membrane,
    disk_membrane, disk_potential, sphere_potential)


@pytest.mark.parametrize('speed', [.3, 2.])
def test_dalembert_right_travelling_wave(speed):
    x, time = np.linspace(-2, 2, 101), np.linspace(0, 1, 21)
    frames, _ = dalembert(np.sin, lambda x: -speed*np.cos(x), x, time, speed, 32001)
    np.testing.assert_allclose(frames, np.sin(x[None,:]-speed*time[:,None]), atol=2e-8)
    np.testing.assert_allclose(frames[0], np.sin(x), atol=1e-14)


def test_constant_initial_velocity():
    x, time = np.linspace(-2, 2, 21), np.linspace(0, 1, 11)
    frames, _ = dalembert(lambda x: np.full_like(x, 3.), lambda x: np.full_like(x, 2.), x, time, .7, 1001)
    np.testing.assert_allclose(frames, np.broadcast_to(3+2*time[:,None], frames.shape), atol=1e-12)


@pytest.mark.parametrize('alpha', [.01, .2, 2.])
def test_infinite_heat_gaussian(alpha):
    x, time = np.linspace(-4, 4, 101), np.linspace(0, 2, 21)
    frames = infinite_heat(lambda x: np.exp(-x*x), x, time, alpha, 16385)
    width = 1+4*alpha*time[:,None]
    np.testing.assert_allclose(frames, np.exp(-x[None,:]**2/width)/np.sqrt(width), atol=1e-8)


def test_infinite_heat_constant_preservation():
    result = get_topic('infinite_heat').compute({'initial': '3'})
    np.testing.assert_allclose(result.data['frames'], 3, atol=1e-12)


@pytest.mark.parametrize('velocity_only', [False, True])
def test_rectangle_exact_single_mode(velocity_only):
    length, height, speed = 2., 3., .7
    x, y, time = np.linspace(0,length,81), np.linspace(0,height,81), np.linspace(0,2,31)
    basis = np.sin(2*np.pi*x[:,None]/length)*np.sin(3*np.pi*y[None,:]/height)
    initial, velocity = (np.zeros_like(basis), basis) if velocity_only else (basis, np.zeros_like(basis))
    frames, energy, a, v, _ = rectangle_membrane(x,y,initial,velocity,length,height,speed,4,time)
    omega = speed*np.pi*np.sqrt((2/length)**2+(3/height)**2)
    factor = np.sin(omega*time)/omega if velocity_only else np.cos(omega*time)
    np.testing.assert_allclose(frames, factor[:,None,None]*basis, atol=1e-12)
    np.testing.assert_allclose(frames[:,[0,-1],:], 0, atol=0)
    np.testing.assert_allclose(frames[:,:,[0,-1]], 0, atol=0)
    np.testing.assert_allclose(energy, energy[0], atol=1e-12)


@pytest.mark.parametrize('m', [0, 1, 2])
def test_disk_single_mode_and_energy(m):
    radius, speed = 2., .7
    r = np.linspace(0,radius,501)
    theta = np.linspace(0,2*np.pi,65,endpoint=False)
    time = np.linspace(0,1,11)
    root = jn_zeros(m,1)[0]
    basis = jv(m,root*r[:,None]/radius)*np.cos(m*theta[None,:])
    mode = lambda rho, angle: jv(m,root*rho/radius)*np.cos(m*angle)
    frames, energy, a, v, omega, labels = disk_membrane(r,theta,mode,lambda rho, angle: 0*rho,radius,speed,2,m,time)
    exact = np.cos(speed*root/radius*time)[:,None,None]*basis
    # Gauss–Legendre 방사 구적이므로 표시 격자 간격과 무관하게 정확합니다.
    np.testing.assert_allclose(frames,exact,atol=1e-12)
    np.testing.assert_allclose(energy,energy[0],rtol=1e-12)
    np.testing.assert_allclose(frames[:,-1],0,atol=0)


def test_disk_invalid_center():
    with pytest.raises(ValueError,match='r=0'):
        get_topic('disk_membrane').compute({'initial':'cos(theta)'})


def test_disk_potential_harmonic():
    r = np.linspace(0,2,31)
    theta = np.linspace(0,2*np.pi,301,endpoint=False)
    field, constant, coefficients = disk_potential(lambda theta: 3+2*np.cos(3*theta),r,theta,2.,8)
    expected = 3+2*(r[:,None]/2)**3*np.cos(3*theta)
    np.testing.assert_allclose(field,expected,atol=1e-12)
    assert constant == pytest.approx(3.)


def test_sphere_dipole_and_constant():
    r, theta = np.linspace(0,2,31), np.linspace(0,2*np.pi,61,endpoint=False)
    field, coefficients = sphere_potential(lambda mu: 2+mu,r,theta,2.,7,1001)
    exact = 2+r[:,None]/2*np.cos(theta)
    # 다항식 경계값은 Gauss–Legendre 구적으로 정확히 적분됩니다.
    np.testing.assert_allclose(field,exact,atol=1e-12)
    np.testing.assert_allclose(field[0],2,atol=1e-12)


def test_pde_residual_is_not_sample_proof():
    zero = get_topic('pde_basics').compute({})
    assert '잔차 = 0' in zero.data['symbolic']
    nonzero = get_topic('pde_basics').compute({'candidate':'x^2+y^2'})
    assert nonzero.metrics['표본 잔차 최대값'] == pytest.approx(4.)
    sampled = get_topic('pde_basics').compute({'mode':'열','candidate':'t^2','time':0.})
    assert sampled.metrics['표본 잔차 최대값'] == 0
    assert '기호적으로 0: False' in sampled.data['symbolic']


def test_laplace_boundary_causality_and_nonnegative_domain():
    result = get_topic('pde_laplace').compute({'boundary':'sqrt(t)','speed':2.,'end_time':1.})
    d = result.data
    delay = d['time'][:,None]-d['x'][None,:]/2
    np.testing.assert_allclose(d['frames'],np.sqrt(np.maximum(delay,0)),atol=1e-14)
    np.testing.assert_allclose(d['frames'][:,0],np.sqrt(d['time']))


def test_semiaxis_heat_step_and_corner():
    result = get_topic('pde_laplace').compute({'mode':'반무한 열전도','alpha':.5,'amplitude':2.})
    d = result.data
    np.testing.assert_allclose(d['frames'][:,0],2.)
    np.testing.assert_allclose(d['frames'][0,1:],0.)
    np.testing.assert_allclose(d['frames'][-1],2*erfc(d['x']/(2*np.sqrt(.5*d['time'][-1]))))


NEW_IDS = ['pde_basics','pde_classification','dalembert','rectangle_laplace','infinite_heat','rectangle_membrane','disk_membrane','coordinate_potential','pde_laplace']


@pytest.mark.parametrize('topic_id',NEW_IDS)
def test_figures_and_persistence(topic_id,tmp_path):
    from engineering_math.core.project import save_settings,load_settings,export_results
    from engineering_math.topics.lessons import lesson_for
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    topic = get_topic(topic_id)
    result = topic.compute({})
    save_settings(tmp_path/'settings.json',topic_id,result.params)
    assert load_settings(tmp_path/'settings.json')['params'] == result.params
    export_results(tmp_path/'result.zip',result)
    assert '12.' in lesson_for(topic_id)
    for figure in topic.figures(result).values():
        FigureCanvasAgg(figure).draw()
        plt.close(figure)


@pytest.mark.parametrize('topic_id',NEW_IDS)
def test_examples(topic_id):
    topic = get_topic(topic_id)
    for params in topic.examples.values():
        result = topic.compute(params)
        assert all(np.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize('topic_id,params',[
    ('dalembert',{'initial':'sqrt(x)'}),('infinite_heat',{'initial':'1/0'}),
    ('rectangle_membrane',{'initial':'1/0'}),('pde_basics',{'candidate':'1/0'}),
    ('coordinate_potential',{'boundary':'1/0'}),('pde_laplace',{'boundary':'1/0'})])
def test_bad_inputs(topic_id,params):
    with pytest.raises(ValueError):
        get_topic(topic_id).compute(params)


def test_infinite_heat_textbook_pulse_matches_erf():
    """교재 12.7 예제 1: U₀=100, |x|<1, c²=1의 닫힌 해 (식 13)."""
    from scipy.special import erf
    x, time = np.linspace(-3, 3, 241), np.array([0., 1/8, 1/2, 2.])
    frames = infinite_heat(lambda v: np.where(np.abs(v) < 1, 100., 0.), x, time, 1., 16385)
    root = 2*np.sqrt(time[1:, None])
    exact = 50*(erf((1-x)/root) + erf((1+x)/root))
    # 도약을 erfc로 분리하므로 기계 정밀도 수준입니다.
    np.testing.assert_allclose(frames[1:], exact, atol=1e-10)


def test_infinite_heat_one_sided_exponential():
    """도약과 매끄러운 꼬리가 함께 있는 e^(−x)H(x)의 닫힌 해와 비교합니다."""
    from scipy.special import erfc
    x, time, alpha = np.linspace(-3, 3, 121), np.array([.05, .3, 1.]), .7
    frames = infinite_heat(lambda v: np.where(v > 0, np.exp(-v), 0.), x, time, alpha, 16385)
    at = alpha*time[:, None]
    exact = .5*np.exp(at - x)*erfc((2*at - x)/(2*np.sqrt(at)))
    # 도약을 뺀 나머지에 꺾임이 남으므로 O(h²) 수준의 오차를 허용합니다.
    np.testing.assert_allclose(frames, exact, atol=2e-6)


def test_sphere_capacitor_textbook_example():
    """교재 12.11 예제 1: 상반구 110 V, 하반구 0 V → A₀=55, A₁=165/2, A₂=0, A₃=−385/8."""
    r, theta = np.array([0., .5, 1.]), np.linspace(0, 2*np.pi, 8, endpoint=False)
    _, coefficients = sphere_potential(lambda mu: np.where(mu > 0, 110., 0.), r, theta, 1., 4, 1001)
    np.testing.assert_allclose(coefficients, [55, 82.5, 0, -48.125], atol=1e-9)
    # 외부해는 무한대에서 점전하 55/r처럼 감쇠합니다 (교재 식 24).
    far = np.array([1e5])
    field, _ = sphere_potential(lambda mu: np.where(mu > 0, 110., 0.), far, theta, 1., 4, 1001, exterior=True)
    np.testing.assert_allclose(field, 55/1e5, rtol=1e-4)


def test_disk_potential_jump_boundary():
    """원판 경계의 도약(연습문제 12.10-5형): 계수가 해석값 aₙ=0, bₙ=2·220(1−cos nπ)/(nπ)·½와 일치합니다."""
    r, theta = np.linspace(0, 1, 5), np.linspace(0, 2*np.pi, 16, endpoint=False)
    boundary = lambda t: np.where(np.sin(t) > 0, 220., 0.)
    _, constant, coefficients = disk_potential(boundary, r, theta, 1., 5, 1024)
    assert constant == pytest.approx(110, abs=1e-9)
    for n, cosine, sine in coefficients:
        assert cosine == pytest.approx(0, abs=1e-9)
        assert sine == pytest.approx(220*(1-np.cos(n*np.pi))/(n*np.pi), abs=1e-9)


def test_disk_membrane_textbook_drum():
    """교재 12.10 예제 1: f=1−r², R=1 → Aₘ=8/(αₘ³J₁(αₘ)) = 1.10801, −0.13978, 0.04548."""
    result = get_topic('disk_membrane').compute({'initial': '1-r^2', 'radius': 1., 'speed': 2., 'angular_count': 0})
    roots = jn_zeros(0, 3)
    exact = 8/(roots**3*jv(1, roots))
    np.testing.assert_allclose([row[3] for row in result.tables['모드 계수'][1][:3]], exact, rtol=1e-10)


def test_retired_infinite_heat_order_setting(tmp_path):
    import json
    from engineering_math.core.project import load_settings
    path = tmp_path/'old.json'
    path.write_text(json.dumps({'version': 2, 'topic': 'infinite_heat', 'params': {'order': 80, 'alpha': .3}}), encoding='utf-8')
    params = load_settings(path)['params']
    assert 'order' not in params and params['alpha'] == .3


def test_rectangle_laplace_single_mode_exact():
    from engineering_math.core.chapter12 import rectangle_laplace
    x, y = np.linspace(0, 2, 41), np.linspace(0, 3, 61)
    field, _ = rectangle_laplace({'top': lambda s: np.sin(np.pi*s/2)}, x, y, 2., 3., 10)
    exact = np.sin(np.pi*x[:, None]/2)*np.sinh(np.pi*y[None, :]/2)/np.sinh(np.pi*3/2)
    np.testing.assert_allclose(field, exact, atol=1e-12)


def test_rectangle_laplace_symmetry_and_superposition():
    one = get_topic('rectangle_laplace').compute({'top': '1', 'N': 101, 'num_points': 101})
    assert one.metrics['중심 온도'] == pytest.approx(.25, abs=1e-12)
    allsides = get_topic('rectangle_laplace').compute({'top': '1', 'bottom': '1', 'left': '1', 'right': '1', 'N': 101})
    interior = allsides.data['field'][20:-20, 20:-20]
    np.testing.assert_allclose(interior, 1, atol=1e-3)


def test_rectangle_laplace_large_modes_do_not_overflow():
    result = get_topic('rectangle_laplace').compute({'top': '1', 'length': .05, 'height': 100., 'N': 200})
    assert np.all(np.isfinite(result.data['field']))


@pytest.mark.parametrize('params,kind,phi', [
    ({'C': '4'}, '타원형', None),
    ({'C': '-16'}, '쌍곡선형', {'-4*x + y', '4*x + y'}),
    ({'B': '1', 'C': '1'}, '포물선형', {'-x + y'}),
    ({'B': '5/2', 'C': '4'}, '쌍곡선형', {'-4*x + y', '-x + y'})])
def test_pde_classification_constant_coefficients(params, kind, phi):
    result = get_topic('pde_classification').compute(params)
    assert dict(result.tables['분류'][1])['형태'] == kind
    rows = dict(result.tables['특성과 표준형'][1])
    if kind == '쌍곡선형':
        assert {rows['특성 Φ = 상수'], rows['특성 Ψ = 상수']} == phi
    elif kind == '포물선형':
        assert {rows['특성 Φ = Ψ = 상수']} == phi and rows['표준형'].startswith('u_vv')
    else:
        assert rows['표준형'].startswith('u_vv + u_ww')


def test_tricomi_is_mixed_and_characteristics_stay_hyperbolic():
    result = get_topic('pde_classification').compute({'A': 'y', 'C': '1'})
    d = result.data
    assert d['label'].startswith('혼합형')
    assert result.metrics['쌍곡선형 영역 비율'] == pytest.approx(.5, abs=.02)
    assert d['curves']
    for _, path in d['curves']:
        finite = path[np.isfinite(path[:, 1])]
        assert np.all(finite[:, 1] <= 1e-9)


def test_pde_classification_rejects_zero_principal_part():
    with pytest.raises(ValueError, match='2계'):
        get_topic('pde_classification').compute({'A': '0', 'B': '0', 'C': '0'})


ODE_MODE = '상미분방정식처럼 풀기'
SOLVE_LIKE_ODE_EXERCISES = {
    '연습문제 17: u_xx + 16π²u = 0': {'mode': ODE_MODE, 'c': '16*pi^2'},
    '연습문제 19: u_y + y²u = 0': {'mode': ODE_MODE, 'variable': 'y', 'a': '0', 'b': '1', 'c': 'y^2'},
    '연습문제 21: u_yy + 6u_y + 13u = 4e^(3y)': {'mode': ODE_MODE, 'variable': 'y', 'b': '6', 'c': '13', 'g': '4*exp(3*y)'},
    '연습문제 22: u_xy = u_x': {'mode': ODE_MODE, 'variable': 'y', 'substitution': 'p = u_x로 치환', 'a': '0', 'b': '1', 'c': '-1'},
    '연습문제 23: x²u_xx + 2xu_x − 2u = 0': {'mode': ODE_MODE, 'a': 'x^2', 'b': '2*x', 'c': '-2'},
}


@pytest.mark.parametrize('name', list(SOLVE_LIKE_ODE_EXERCISES))
def test_solve_like_ode_examples_have_zero_residual(name):
    """12.1 연습문제(화면 예시에서는 지웠지만 풀이 검산은 유지합니다)."""
    result = get_topic('pde_basics').compute(SOLVE_LIKE_ODE_EXERCISES[name])
    assert dict(result.tables['풀이'][1])['대입 검산 잔차'] == '0'


def test_solve_like_ode_rejects_non_differential_equation():
    with pytest.raises(ValueError, match='미분방정식'):
        get_topic('pde_basics').compute({'mode': '상미분방정식처럼 풀기', 'a': '0', 'b': '0'})


def test_membrane_force_balance_converges_to_acceleration():
    """교재 12.8 식 (1)~(3): c²(Δu_x/Δx+Δu_y/Δy) → u_tt (중심차분이므로 O(Δ²))."""
    result = get_topic('rectangle_membrane').compute({'initial': '0.1*(4*x-x^2)*(2*y-y^2)', 'length': 4., 'height': 2.,
                                                      'speed': 2.23606797749979, 'N': 7})
    rows = result.tables['모델링: 미소부분의 힘'][1]
    errors = [row[4] for row in rows]
    assert errors[-1] < 1e-3*abs(rows[-1][3])
    assert all(later < earlier/3.5 for earlier, later in zip(errors, errors[1:]))
