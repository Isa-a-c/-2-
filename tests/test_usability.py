"""목차 트리, 예제 즉시 계산, 자동 재계산, 표·지표 표기, 글자 확대를 검증합니다."""
import time
import numpy as np
import pytest
from engineering_math.topics.catalog import EXPERIMENTS
from engineering_math.topics import get_topic


@pytest.fixture
def window(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from engineering_math.ui.window import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow(QSettings(str(tmp_path/'settings.ini'), QSettings.Format.IniFormat))
    window.auto.setChecked(True)
    window.show()
    yield window
    window.set_zoom(1., redraw=False)
    window.close()
    app.processEvents()


def wait(window, seconds=60):
    from PySide6.QtWidgets import QApplication
    deadline = time.monotonic()+seconds
    while (window.job.busy or window.auto_timer.isActive()) and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert not window.job.busy


def test_tree_groups_entries_by_chapter(window):
    tree = window.selector.tree
    chapters = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert chapters[0].startswith('11장') and chapters[1].startswith('12장')
    for index, entry in enumerate(EXPERIMENTS):
        parent = window.selector.entries[index].parent().text(0)
        assert parent.startswith(entry.lesson[:2])
    assert len(chapters) == 2


def test_examples_match_locked_configuration(window):
    from engineering_math.ui.experiments import entry_examples
    for index, entry in enumerate(EXPERIMENTS):
        names = window.selector.example_names(index)
        assert names == entry_examples(entry, get_topic(entry.topic))
    fixed = window.selector.example_names(next(i for i, e in enumerate(EXPERIMENTS) if e.lesson.startswith('12.5')))
    insulated = window.selector.example_names(next(i for i, e in enumerate(EXPERIMENTS) if e.lesson == '12.6 · 단열 막대'))
    assert fixed and insulated and not set(fixed) & set(insulated)


def test_search_finds_textbook_example_number(window):
    window.selector.search.setText('11.2-3')
    visible = [item.text(0) for item in window.selector.entries if not item.isHidden()]
    assert visible == ['11.2 · 반구간 전개'] or all(v.startswith('11.') for v in visible)
    item = next(item for item in window.selector.entries if not item.isHidden())
    shown = [item.child(i).text(0) for i in range(item.childCount()) if not item.child(i).isHidden()]
    assert shown and all('11.2-3' in name for name in shown)
    window.selector.search.clear()
    assert all(item.isHidden() == entry.hidden for item, entry in zip(window.selector.entries, EXPERIMENTS))


def test_choosing_example_calculates_immediately(window):
    window.selector.select('half_range')
    name = window.selector.example_names(window.selector.currentIndex())[0]
    window.selector.choose_example(name)
    assert window.job.busy or window.result is not None
    wait(window)
    assert window.result is not None
    assert window.result.params == {**window.result.params, **window.topic.examples[name]}


def test_value_change_recalculates_and_keeps_tab(window):
    window.selector.select('fourier')
    window.calculate()
    wait(window)
    window.tabs.setCurrentIndex(2)
    tab = window.tabs.tabText(2)
    window.form.fields['N'].setValue(window.result.params['N']+3)
    assert window.auto_timer.isActive()
    wait(window)
    assert window.result.params['N'] == window.form.values()['N']
    assert window.tabs.tabText(window.tabs.currentIndex()) == tab
    assert '일치' in window.input_state.text()


def test_slow_configuration_skips_auto(window):
    window.selector.select('fourier')
    window.durations[window.state_key] = 5.
    window.form.fields['N'].setValue(window.form.values()['N']+1)
    assert not window.auto_timer.isActive()
    window.auto.setChecked(False)
    window.durations.clear()
    window.form.fields['N'].setValue(window.form.values()['N']+1)
    assert not window.auto_timer.isActive()


def test_auto_error_is_not_modal(window):
    window.selector.select('fourier')
    window.form.fields['function'].setText('sin(')
    window.auto_calculate()
    wait(window)
    assert '자동' in window.statusBar().currentMessage()


def test_number_formatting():
    from engineering_math.ui.window import format_number, pretty_number, ResultTable
    assert format_number(3.0) == '3'
    assert format_number(np.int64(7)) == '7'
    assert format_number(0.123456789) == '0.123457'
    assert pretty_number(1.25e-5) == '1.25×10⁻⁵'
    assert pretty_number(2.5) == '2.5'
    model = ResultTable(['a'], [[1/3]])
    assert model.raw(0, 0) == repr(1/3)


def test_table_copy_uses_full_precision(window):
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QTabWidget
    from engineering_math.ui.window import ResultView
    window.present(window.topic.compute(window.form.values()))
    table = window.tabs.findChild(QTabWidget, 'tables').widget(0)
    assert isinstance(table, ResultView)
    table.selectAll()
    table.copy_selection()
    lines = QGuiApplication.clipboard().text().splitlines()
    assert lines[0].split('\t') == list(table.model().columns)
    assert len(lines) == table.model().rowCount()+1


def test_zoom_changes_fonts_and_is_remembered(window):
    from PySide6.QtWidgets import QApplication
    import matplotlib
    base = window.base_point_size
    window.step_zoom(1)
    assert QApplication.font().pointSizeF() == pytest.approx(base*1.15)
    assert matplotlib.rcParams['font.size'] == pytest.approx(11.5)
    assert float(window.settings.value('zoom')) == pytest.approx(1.15)
    window.set_zoom(1.)
    assert QApplication.font().pointSizeF() == pytest.approx(base)


def test_custom_entry_for_free_input(window):
    from engineering_math.ui.experiments import CUSTOM_GROUP, ROLE
    selector = window.selector
    for index, entry in enumerate(selector.entries):
        selector.ensure_examples(index)
        names = [entry.child(i).text(0) for i in range(entry.childCount())]
        # 교재 예제만 남고 마지막에 직접 입력하는 '사용자 지정' 항목 하나가 있습니다.
        assert names[-1] == CUSTOM_GROUP and entry.child(entry.childCount()-1).childCount() == 0
        assert all(name.startswith('교재 ') for name in names[:-1])
    selector.select('heat')
    selector.choose_custom()
    assert window.custom_mode and window.form.values()['initial'] == 'sin(pi*x/l)'
    assert selector.tree.currentItem().text(0) == CUSTOM_GROUP
    window.form.fields['initial'].setText('x*(1-x)')
    window.form.fields['alpha'].setValue(.3)
    wait(window)
    assert window.result is not None and window.result.params['initial'] == 'x*(1-x)'
    # 교재 예제를 골랐다가 사용자 지정으로 돌아오면 직접 입력한 값이 남아 있습니다.
    selector.choose_example(selector.example_names(selector.currentIndex())[0])
    wait(window)
    assert not window.custom_mode and window.form.values()['initial'] != 'x*(1-x)'
    selector.choose_custom()
    assert window.form.values()['initial'] == 'x*(1-x)' and window.form.values()['alpha'] == .3
    # 다른 절의 사용자 지정은 따로 기억합니다.
    selector.select('wave')
    selector.choose_custom()
    assert window.form.values()['initial'] == 'sin(pi*x/l)'


def test_player_buttons_are_icon_only(app=None):
    from PySide6.QtWidgets import QApplication
    from matplotlib.figure import Figure
    from engineering_math.ui.player import FigureAnimator, PlayerBar, ZoomCanvas
    QApplication.instance() or QApplication([])
    figure = Figure()
    figure.add_subplot().plot([0, 1, 2], [0, 1, 0])
    player = PlayerBar(FigureAnimator(figure), ZoomCanvas(figure))
    assert all(button.text() == '' and not button.icon().isNull() and button.toolTip()
               for button in player.buttons.values())


def test_wheel_changes_values_only_in_focused_field(window):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    window.selector.select('heat')
    widget = window.form.fields['alpha']
    before = widget.value()

    def wheel(target):
        center = QPointF(target.width()/2, target.height()/2)
        event = QWheelEvent(center, target.mapToGlobal(center), QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(target, event)

    window.setFocus()
    wheel(widget)
    assert widget.value() == before and not window.auto_timer.isActive()
    from PySide6.QtWidgets import QSlider, QSpinBox, QComboBox
    n = window.form.values()['N']
    wheel(window.form.rows['N'].findChild(QSlider))
    wheel(window.form.rows['N'].findChild(QSpinBox))
    assert window.form.values()['N'] == n
    mode = window.form.fields['mode']
    text = mode.currentText()
    wheel(mode)
    assert mode.currentText() == text
    widget.setFocus()
    QApplication.processEvents()
    wheel(widget)
    assert widget.value() != before


def test_hidden_section_stays_out_of_tree_but_still_loads(window, tmp_path):
    from engineering_math.core.project import save_settings
    index = next(i for i, entry in enumerate(EXPERIMENTS) if entry.topic == 'transform_table')
    assert EXPERIMENTS[index].hidden and window.selector.entries[index].isHidden()
    for text in ('11.10', '변환', ''):
        window.selector.search.setText(text)
        assert window.selector.entries[index].isHidden()
    window.selector.search.clear()
    assert sum(not item.isHidden() for item in window.selector.entries) == len(EXPERIMENTS)-1
    # 예전에 저장한 11.10 설정 파일은 그대로 열립니다.
    path = tmp_path/'table.json'
    save_settings(path, 'transform_table', {})
    window.calculate = lambda: None
    window.load_path(path)
    assert window.topic.id == 'transform_table'


def test_search_finds_alias_example_numbers(window):
    for number, label in [('12.7-3', '교재 예제 12.7-1·3'), ('11.5-3', '교재 예제 11.5-1·3'), ('11.9-5', '교재 예제 11.9-4·5')]:
        window.selector.search.setText(number)
        shown = [item.child(i).text(0) for item in window.selector.entries if not item.isHidden()
                 for i in range(item.childCount()) if not item.child(i).isHidden()]
        assert any(name.startswith(label) for name in shown), number
    window.selector.search.clear()
