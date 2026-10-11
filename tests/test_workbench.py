"""분류·입력 전환·설정 호환성 및 메모리 공유를 검증합니다."""
import csv
import io
import pickle
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from engineering_math.topics.catalog import EXPERIMENTS, GROUPS, experiment_index
from engineering_math.topics import get_topic
from engineering_math.core.models import validate_params
from engineering_math.core.tables import GridRows
from engineering_math.core.project import save_settings, export_results


@pytest.mark.parametrize('entry', EXPERIMENTS, ids=lambda item: item.topic + '-' + item.problem)
def test_supported_configuration(entry):
    topic = get_topic(entry.topic)
    assert topic.id == entry.topic
    params = validate_params(topic.inputs, entry.defaults)
    assert EXPERIMENTS[experiment_index(entry.topic, params)] == entry


def test_single_topic_import_is_lazy():
    code = ('import sys; from engineering_math.topics import get_topic; get_topic("fourier"); '
            'assert "engineering_math.topics.chapter12" not in sys.modules; '
            'assert "engineering_math.topics.ode" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_grid_rows_coordinates_pickle_and_csv(tmp_path):
    x, y = np.array([1., 3.]), np.array([2., 4., 6.])
    values = x[:, None] + y
    rows = GridRows((x, y), (values,), (1, 0))
    assert rows.fields[0] is values
    np.testing.assert_equal(list(rows), np.column_stack((np.tile(y, 2), np.repeat(x, 3), values.ravel())))
    np.testing.assert_equal(pickle.loads(pickle.dumps(rows))[-1], [6., 3., 9.])
    with pytest.raises(IndexError):
        rows[len(rows)]
    result = get_topic('heat').compute({'num_points': 201, 'time_points': 3})
    assert result.tables['온도'][1].fields[0] is result.data['temperature']
    path = tmp_path / 'rows.zip'
    export_results(path, result)
    with zipfile.ZipFile(path) as archive:
        table = list(csv.reader(io.StringIO(archive.read('온도.csv').decode('utf-8-sig'))))
    assert len(table) == 604
    np.testing.assert_allclose(list(map(float, table[-1])), result.tables['온도'][1][-1])


def test_common_ui_routes_settings_and_lazy_canvas(tmp_path):
    from PySide6.QtWidgets import QApplication
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from engineering_math.ui.window import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    try:
        assert window.selector.group.count() == len(GROUPS)
        for index, entry in enumerate(EXPERIMENTS):
            window.selector.setCurrentIndex(index)
            assert window.topic.id == entry.topic
            assert all(window.form.values()[key] == value for key, value in entry.defaults.items())
        window.selector.select('heat')
        window.form.fields['num_points'].setValue(701)
        assert window.form.rows['num_points'].isHidden()
        window.form.details.click()
        assert not window.form.rows['num_points'].isHidden()
        window.form.details.click()
        assert window.form.values()['num_points'] == 701
        window.selector.select('wave')
        window.selector.select('heat')
        assert window.form.values()['num_points'] == 701

        # 사용자 선택 경로도 같은 명세로 연결됩니다.
        window.selector.problem.setCurrentText('파동')
        window.selector.configuration.setCurrentIndex(window.selector.configuration.findText('원형 막 · 원주 고정'))
        assert window.topic.id == 'disk_membrane'
        window.selector.select('heat')
        result = window.topic.compute(window.form.values())
        window.present(result)
        app.processEvents()
        assert len(window.tabs.findChildren(FigureCanvasQTAgg)) == 1
        window.tabs.setCurrentIndex(1)
        app.processEvents()
        assert len(window.tabs.findChildren(FigureCanvasQTAgg)) == 2
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
        assert len(window.tabs.findChildren(FigureCanvasQTAgg)) == len(window.figures)
        window.calculate()
        assert window.result is result and not window.job.busy

        # 기존 JSON v2의 반무한 열전도 모드는 파동으로 덮어쓰지 않습니다.
        path = tmp_path / 'heat.json'
        save_settings(path, 'pde_laplace', {'mode': '반무한 열전도', 'amplitude': 3.})
        window.calculate = lambda: None
        window.load_path(path)
        assert window.selector.problem.currentText() == '열전도'
        assert window.form.values()['mode'] == '반무한 열전도'
        assert window.form.values()['amplitude'] == 3.
        window.form.set_values({})
        assert window.form.values()['mode'] == '반무한 열전도'
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize('topic_id', list(dict.fromkeys(item.topic for item in EXPERIMENTS)))
def test_animation_contract(topic_id):
    """재생 버튼이 켜지는 주제는 animation()이 프레임과 라벨을 돌려주어야 합니다."""
    topic = get_topic(topic_id)
    if not getattr(topic, 'supports_animation', True):
        return
    spec = topic.animation(topic.compute({}))
    assert len(spec['frames']) == len(spec['labels']) > 0
