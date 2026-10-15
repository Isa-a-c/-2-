"""화면과 독립적으로 해석해 및 입력 경계를 검증합니다."""
import json
import zipfile
import numpy as np
import pytest
from engineering_math.core.expr import parse_expression, evaluate
from engineering_math.core.fourier import calculate_fourier_coefficients, FourierSession
from engineering_math.core.project import save_settings, load_settings, export_results
from engineering_math.topics.fourier import FourierTopic, MODES
from engineering_math.topics.heat import HeatTopic


def test_odd_fourier_analytic_coefficients():
    x = np.linspace(-np.pi, np.pi, 20000)
    a0, an, bn = calculate_fourier_coefficients(x, x, np.pi, 100)
    n = np.arange(1, 101)
    assert abs(a0) < 1e-12
    np.testing.assert_allclose(an, 0, atol=1e-12)
    np.testing.assert_allclose(bn, 2*(-1.)**(n+1)/n, atol=6e-6)


@pytest.mark.parametrize('text', ['x', 'x**2', 'sin(x)', 'cos(x)', 'exp(x)', 'abs(x)', 'sign(x)', '3'])
def test_supported_functions_and_empty_comparison(text):
    result = FourierTopic().compute({'function': text, 'N': 100, 'comparison_N': ''})
    assert len(result.data['an']) == 100
    assert result.data['y'].shape == (5000,)
    assert result.tables['수렴 오차'][1] == []
    assert np.isfinite(result.metrics['MSE'])


@pytest.mark.parametrize('mode', MODES)
def test_advanced_calculations(mode):
    result = FourierTopic().compute({'mode': mode, 'function': 'sign(x)', 'N': 3, 'comparison_N': '50'})
    assert len(result.data['an']) == 50


def test_gibbs_overshoot():
    result = FourierTopic().compute({'function': 'sign(x)', 'N': 100})
    peak = result.data['sums'][100].max()
    assert 1.15 < peak < 1.20


def test_parser_multivariable_piecewise_and_special_function():
    x = np.array([-1., 0., 1.])
    expression = parse_expression('Piecewise((x+t, x<0), (t, True))', ('x', 't'))
    np.testing.assert_allclose(evaluate(expression, (x, 2), ('x', 't')), [1, 2, 2])
    np.testing.assert_allclose(evaluate(parse_expression('Heaviside(x,0.5)'), (x,)), [0, .5, 1])
    assert evaluate(parse_expression('besselj(0,x)'), (np.array([0.]),))[0] == 1


def test_complex_policy():
    expression = parse_expression('exp(I*z)', ('z',), allow_complex=True)
    np.testing.assert_allclose(evaluate(expression, ([0., np.pi],), ('z',), True), [1, -1], atol=1e-14)
    with pytest.raises(ValueError):
        parse_expression('I*x')
    expression = parse_expression('sqrt(z)', ('z',), allow_complex=True)
    np.testing.assert_allclose(evaluate(expression, ([-1.],), ('z',), True), [1j])


def test_symbolic_equality_and_chained_conditions():
    x = np.array([-1., 0., 1.])
    expression = parse_expression('Piecewise((3, x==0), (1, -2<x<0), (2, True))')
    np.testing.assert_allclose(evaluate(expression, (x,)), [1, 3, 2])


def test_fourier_even_analytic_coefficients():
    x = np.linspace(-np.pi, np.pi, 20000)
    a0, an, bn = calculate_fourier_coefficients(x, x*x, np.pi, 20)
    n = np.arange(1, 21)
    assert abs(a0-np.pi**2/3) < 2e-8
    np.testing.assert_allclose(an, 4*(-1.)**n/n**2, atol=4e-8)
    np.testing.assert_allclose(bn, 0., atol=1e-12)


@pytest.mark.parametrize('text', ["__import__('os')", 'x.__class__', 'sin(x,2)', 'unknown(x)', '[x]', 'x[0]', 'lambda x:x'])
def test_invalid_parser_input(text):
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        parse_expression(text)


@pytest.mark.parametrize('text', ['1/x', 'log(x)', 'exp(1000*x)'])
def test_nonfinite_input(text):
    with pytest.raises(ValueError):
        evaluate(parse_expression(text), (np.array([-1., 0., 1.]),))


def test_heat_single_mode_exact_solution():
    result = HeatTopic().compute({})
    data = result.data
    expected = np.exp(-.1*np.pi**2*data['time'][:, None])*np.sin(np.pi*data['x'])
    np.testing.assert_allclose(data['temperature'], expected, atol=1e-13)
    np.testing.assert_allclose(data['coefficients'][0], 1., atol=1e-14)
    np.testing.assert_allclose(data['coefficients'][1:], 0., atol=1e-14)


def test_heat_nonunit_length():
    result = HeatTopic().compute({'length': 3., 'alpha': .7, 'initial': '2*sin(2*pi*x/l)'})
    data = result.data
    expected = 2*np.exp(-.7*(2*np.pi/3)**2*data['time'][:, None])*np.sin(2*np.pi*data['x']/3)
    np.testing.assert_allclose(data['temperature'], expected, atol=1e-13)


def test_heat_steady_state_and_energy():
    result = HeatTopic().compute({'initial': '20+60*x/l', 'left': 20., 'right': 80.})
    np.testing.assert_allclose(result.data['temperature'], np.broadcast_to(result.data['steady'], result.data['temperature'].shape), atol=1e-12)
    result = HeatTopic().compute({'initial': '1-abs(2*x/l-1)'})
    energy = result.tables['잔차 에너지'][1][:, 1]
    assert np.all(np.diff(energy) <= 1e-14)
    assert np.all(result.data['temperature'][:, [0, -1]] == 0)
    assert any('끝점' in notice for notice in HeatTopic().compute({'initial': '1'}).notices)


@pytest.mark.parametrize('params', [{'alpha': 0}, {'length': -1}, {'N': True}, {'time_points': 1}, {'unknown': 3}])
def test_heat_invalid_parameters(params):
    with pytest.raises(ValueError):
        HeatTopic().compute(params)


def test_settings_and_export(tmp_path):
    result = HeatTopic().compute({})
    path = tmp_path/'설정.json'
    save_settings(path, result.topic, result.params)
    assert load_settings(path)['params'] == result.params
    output = tmp_path/'결과.zip'
    export_results(output, result)
    with zipfile.ZipFile(output) as archive:
        assert '온도.csv' in archive.namelist()
        assert json.loads(archive.read('settings.json'))['topic'] == 'heat'


def test_incremental_coefficients_preserved():
    session = FourierSession()
    first = session.prepare('x', np.pi, 5000, 10)
    second = session.prepare('x', np.pi, 5000, 100)
    np.testing.assert_array_equal(first[5], second[5][:10])
