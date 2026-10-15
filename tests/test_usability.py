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
    assert all(not item.isHidden() for item in window.selector.entries)


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


def test_custom_examples_are_grouped(window):
    from engineering_math.ui.experiments import CUSTOM_GROUP, ROLE
    selector = window.selector
    for index, entry in enumerate(selector.entries):
        selector.ensure_examples(index)
        groups = [entry.child(i) for i in range(entry.childCount()) if entry.child(i).text(0) == CUSTOM_GROUP]
        direct = [entry.child(i).text(0) for i in range(entry.childCount()) if entry.child(i).text(0) != CUSTOM_GROUP]
        assert all(name.startswith('교재 ') for name in direct)
        assert len(groups) <= 1
        for group in groups:
            names = [group.child(j).text(0) for j in range(group.childCount())]
            assert names and not any(name.startswith('교재 ') for name in names)
    # 사용자 지정 예시를 고르면 묶음이 펼쳐지고 바로 계산합니다.
    selector.select('heat')
    group = next(selector.entries[selector.currentIndex()].child(i) for i in range(selector.entries[selector.currentIndex()].childCount())
                 if selector.entries[selector.currentIndex()].child(i).text(0) == CUSTOM_GROUP)
    name = group.child(0).text(0)
    selector.choose_example(name)
    assert group.isExpanded() and selector.tree.currentItem().text(0) == name
    wait(window)
    assert window.result is not None
    selector.search.setText(name)
    assert not group.isHidden() and not group.child(0).isHidden()
    selector.search.clear()


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
