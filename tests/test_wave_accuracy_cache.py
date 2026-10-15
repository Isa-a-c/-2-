"""파동 해석해, 열 정확도 비교, 실제 계수 재사용을 검증합니다."""
import numpy as np
import pytest
from engineering_math.topics.wave import WaveTopic
from engineering_math.topics.heat import HeatTopic
from engineering_math.topics.fourier import FourierTopic
from engineering_math.core.project import save_settings, load_settings, export_results


def test_wave_exact_displacement_mode():
    result = WaveTopic().compute({})
    data = result.data
    exact = np.cos(np.pi*data['time'][:, None])*np.sin(np.pi*data['x'])
    np.testing.assert_allclose(data['displacement'], exact, atol=2e-14)
    np.testing.assert_allclose(data['energy'], np.pi**2/4, atol=1e-13)
    assert np.all(data['displacement'][:, [0, -1]] == 0)


def test_wave_initial_velocity_nonunit_length_and_speed():
    result = WaveTopic().compute({'initial': '0', 'velocity': '3*sin(2*pi*x/l)', 'length': 2., 'speed': 4.})
    data = result.data
    omega = 4*np.pi
    exact = 3/omega*np.sin(omega*data['time'][:, None])*np.sin(np.pi*data['x'])
    np.testing.assert_allclose(data['displacement'], exact, atol=2e-14)
    np.testing.assert_allclose(data['velocity_initial'], 3*np.sin(np.pi*data['x']), atol=2e-14)
    assert result.metrics['초기 속도 RMSE'] < 1e-13


def test_wave_multimode_energy_and_zero_solution():
    result = WaveTopic().compute({'initial': 'sin(pi*x/l)+0.5*sin(3*pi*x/l)', 'velocity': '2*sin(2*pi*x/l)'})
    energy = result.data['energy']
    np.testing.assert_allclose(energy, energy[0], rtol=2e-14)
    zero = WaveTopic().compute({'initial': '0'})
    assert np.all(zero.data['displacement'] == 0)
    assert zero.metrics['에너지 최대 변화'] == 0


def test_heat_wave_same_initial_shape():
    wave = WaveTopic().compute({'end_time': 1., 'time_points': 101})
    heat = HeatTopic().compute({})
    np.testing.assert_allclose(wave.data['heat'], heat.data['temperature'], atol=1e-14)
    np.testing.assert_allclose(wave.data['displacement'][0], heat.data['temperature'][0], atol=1e-14)
    assert wave.data['displacement'][-1, 250] < -.99
    assert 0 < heat.data['temperature'][-1, 250] < 1


def test_wave_boundary_and_temporal_sampling_notices():
    result = WaveTopic().compute({'initial': '1', 'velocity': '1', 'time_points': 2})
    assert any('끝점' in notice for notice in result.notices)
    assert any('한 주기당' in notice for notice in result.notices)


@pytest.mark.parametrize('params', [{'speed': 0}, {'length': -1}, {'N': True}, {'velocity': '1/0'}])
def test_wave_invalid_inputs(params):
    with pytest.raises((ValueError, TypeError)):
        WaveTopic().compute(params)


def test_heat_accuracy_isolates_grid_and_order_changes():
    result = HeatTopic().compute({'initial': '1-abs(2*x/l-1)', 'N': 5, 'num_points': 201})
    baseline, refined, higher = result.data['accuracy']['rows']
    assert baseline[1:3] == [201, 5]
    assert refined[1:3] == [401, 5]
    assert higher[1:3] == [201, 10]
    assert higher[3] < baseline[3]
    assert baseline[4:] == [0., 0.]
    assert refined[4] > 0
    assert result.data['accuracy']['x'].size == 2049


def test_heat_accuracy_single_mode_and_upper_limit():
    result = HeatTopic().compute({'N': 100, 'num_points': 201})
    rows = result.data['accuracy']['rows']
    assert rows[2][2] == 199
    assert max(row[3] for row in rows) < 1e-13


def test_fourier_topic_cache_only_integrates_missing_orders(monkeypatch):
    import engineering_math.core.fourier as core
    original = core.integrate_harmonics
    calls = []
    def traced(x, y, length, first, last):
        calls.append((first, last))
        return original(x, y, length, first, last)
    monkeypatch.setattr(core, 'integrate_harmonics', traced)
    topic = FourierTopic()
    first = topic.compute({'N': 10, 'comparison_N': ''})
    restored = FourierTopic()
    restored.restore_cache(topic.snapshot_cache())
    extended = restored.compute({'N': 15, 'comparison_N': ''})
    shortened = restored.compute({'N': 3, 'comparison_N': ''})
    assert calls == [(1, 10), (11, 15)]
    assert extended.data['cache_reused'] == 10
    assert shortened.data['cache_integrated'] == 0
    np.testing.assert_array_equal(first.data['an'], extended.data['an'][:10])
    restored.compute({'N': 3, 'comparison_N': '', 'num_points': 6000})
    assert calls[-1] == (1, 3)


def test_wave_settings_and_export(tmp_path):
    import zipfile
    result = WaveTopic().compute({})
    path = tmp_path/'wave.json'
    save_settings(path, 'wave', result.params)
    assert load_settings(path)['params'] == result.params
    output = tmp_path/'wave.zip'
    export_results(output, result)
    with zipfile.ZipFile(output) as archive:
        assert '변위 및 열 비교.csv' in archive.namelist()
