"""확장 함수·매개변수·사용자 외력과 부분 성공 상태의 회귀 검증입니다."""
import numpy as np
import sympy as sp
import pytest
from scipy import special
from engineering_math.core.expr import parse_expression, parameter_expression, parse_parameters, evaluate
from engineering_math.topics.fourier import FourierTopic
from engineering_math.topics.heat import HeatTopic
from engineering_math.topics.wave import WaveTopic


@pytest.mark.parametrize('expression, expected', [
    ('asin(x)', np.arcsin), ('arccos(x)', np.arccos), ('atan(x)', np.arctan),
    ('asinh(x)', np.arcsinh), ('acosh(x+1)', lambda x: np.arccosh(x+1)),
    ('atanh(x)', np.arctanh), ('ln(x+1)', np.log1p),
    ('sinc(x)', lambda x: np.sinc(x/np.pi)), ('floor(x)', np.floor), ('ceil(x)', np.ceil),
    ('erfc(x)', special.erfc), ('erfi(x)', special.erfi),
    ('besseli(0,x)', lambda x: special.iv(0,x)), ('besselk(0,x+1)', lambda x: special.kv(0,x+1)),
    ('loggamma(x+1)', lambda x: special.loggamma(x+1)),
    ('beta(x+1,2)', lambda x: special.beta(x+1,2)),
])
def test_extended_functions(expression, expected):
    x = np.array([0., .2, .5])
    np.testing.assert_allclose(evaluate(parse_expression(expression), (x,)), expected(x), atol=1e-13)


def test_parameters_power_and_piecewise():
    expression = parameter_expression('A*Piecewise((x^2, x<0), (sin(w*x), True))', 'A=2; w=pi')
    np.testing.assert_allclose(evaluate(expression, (np.array([-1., 0., .5]),)), [2, 0, 2])


@pytest.mark.parametrize('text', ['x=2', 'sin=3', 'A=1; A=2', 'A=1; B=A', 'A=I', 'A=1/0', 'A=2,B=3', '__x=2'])
def test_invalid_parameter_declarations(text):
    with pytest.raises((ValueError, SyntaxError)):
        parse_parameters(text)


def test_parameter_change_invalidates_fourier_cache():
    topic = FourierTopic()
    first = topic.compute({'function': 'A*sin(x)', 'parameters': 'A=2'})
    second = topic.compute({'function': 'A*sin(x)', 'parameters': 'A=3'})
    assert first.data['bn'][0] == pytest.approx(2.)
    assert second.data['bn'][0] == pytest.approx(3.)
    assert second.data['cache_reused'] == 0


@pytest.mark.parametrize('topic', [HeatTopic(), WaveTopic()])
def test_pde_parameters(topic):
    result = topic.compute({'initial': 'A*sin(pi*x/l)', 'parameters': 'A=2'})
    np.testing.assert_allclose(result.data['initial'], 2*np.sin(np.pi*result.data['x']), atol=1e-14)


def test_parameter_settings_roundtrip(tmp_path):
    from engineering_math.core.project import save_settings, load_settings, export_results
    result = HeatTopic().compute({'initial': 'A*sin(pi*x/l)', 'parameters': 'A=2', 'num_points': 201, 'time_points': 3})
    path = tmp_path/'parameters.json'
    save_settings(path, 'heat', result.params)
    assert load_settings(path)['params']['parameters'] == 'A=2'
    export_results(tmp_path/'parameters.zip', result)


@pytest.mark.parametrize('topic', [HeatTopic(), WaveTopic()])
def test_length_variable_is_l_and_old_ell_still_works(topic):
    new = topic.compute({'initial': 'sin(pi*x/l)', 'length': 2.})
    old = topic.compute({'initial': 'sin(pi*x/ell)', 'length': 2.})
    np.testing.assert_allclose(new.data['initial'], np.sin(np.pi*new.data['x']/2), atol=1e-14)
    np.testing.assert_array_equal(new.data['initial'], old.data['initial'])
    assert 'ell' not in next(spec.label for spec in topic.inputs if spec.key == 'length')
