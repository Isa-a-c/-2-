"""그래프 아래 재생 막대, 변화 과정 탭, Ctrl+휠 확대를 검증합니다."""
import numpy as np
import pytest
from engineering_math.topics import get_topic
from engineering_math.topics.catalog import EXPERIMENTS
from engineering_math.ui.player import FigureAnimator, SWEEP_FRAMES, evolution_figure

CONFIGURATIONS = list(dict.fromkeys((entry.topic, tuple(sorted(entry.defaults.items()))) for entry in EXPERIMENTS))


@pytest.fixture(scope='module')
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def snapshot(figure):
    state = []
    for axes in figure.axes:
        for line in axes.lines:
            if line.get_label() != '_재생 점':
                state.append((line.get_visible(), np.array(line.get_xdata(), float).copy(),
                              np.array(line.get_ydata(), float).copy()))
        for collection in axes.collections:
            if hasattr(collection, 'get_segments'):
                state.append(len(collection.get_segments()))
        state.extend(patch.get_visible() for patch in axes.patches)
    return state


def same(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        if isinstance(a, tuple):
            assert a[0] == b[0]
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])
        else:
            assert a == b


@pytest.mark.parametrize('topic_id, defaults', CONFIGURATIONS, ids=lambda value: str(value)[:30])
def test_every_figure_returns_to_original_at_last_frame(topic_id, defaults):
    import matplotlib.pyplot as plt
    topic = get_topic(topic_id)
    for name, figure in topic.figures(topic.compute(dict(defaults))).items():
        original = snapshot(figure)
        animator = FigureAnimator(figure)
        for index in (0, animator.frames//2, animator.frames-1):
            label = animator.show(index)
            assert isinstance(label, str)
        same(original, snapshot(figure))
        dots = [line for axes in figure.axes for line in axes.lines if line.get_label() == '_재생 점']
        assert not any(dot.get_visible() for dot in dots), name
        plt.close(figure)


def test_classification_of_known_figures():
    import matplotlib.pyplot as plt
    heat = get_topic('heat')
    figures = heat.figures(heat.compute({}))
    sequence = FigureAnimator(figures['시간별 온도'])
    assert sequence.frames == 5
    sequence.show(0)
    solid = [line for line in figures['시간별 온도'].axes[0].lines if line.get_linestyle() == '-']
    assert [line.get_visible() for line in solid] == [True, False, False, False, False]
    from engineering_math.ui.player import _TimeLine
    assert isinstance(FigureAnimator(figures['시공간 히트맵']).parts[0], _TimeLine)
    fourier = get_topic('fourier')
    sweep = FigureAnimator(fourier.figures(fourier.compute({}))['현재 부분합'])
    assert sweep.frames == SWEEP_FRAMES
    sweep.show(0)
    line = sweep.figure.axes[0].lines[0]
    assert len(line.get_xdata()) < 0.05*len(sweep.parts[0].lines[0][1])
    plt.close('all')


def test_evolution_figure_frames():
    data = get_topic('heat').animation(get_topic('heat').compute({}))
    figure, custom = evolution_figure(data)
    animator = FigureAnimator(figure, custom)
    assert animator.frames == len(data['frames']) and animator.animated
    animator.show(animator.frames-1)
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_ydata(), data['frames'][-1])
    field = get_topic('rectangle_membrane').animation(get_topic('rectangle_membrane').compute({}))
    figure, custom = evolution_figure(field)
    assert custom['show'](3).startswith('4/')


def test_zoom_keeps_cursor_point_fixed(app):
    from matplotlib.figure import Figure
    from engineering_math.ui.player import ZoomCanvas
    figure = Figure()
    axes = figure.add_subplot()
    axes.plot([0, 10], [0, 4])
    axes.set(xlim=(0, 10), ylim=(0, 4))
    canvas = ZoomCanvas(figure)
    canvas.draw()
    point = axes.transData.transform((2.5, 1.))
    ZoomCanvas.zoom(axes, point, .5)
    assert axes.get_xlim() == pytest.approx((1.25, 6.25))
    assert axes.get_ylim() == pytest.approx((.5, 2.5))
    axes.set_xscale('log')
    axes.set_xlim(1, 100)
    canvas.draw()
    ZoomCanvas.zoom(axes, axes.transData.transform((10, 1.)), .5)
    assert axes.get_xlim() == pytest.approx((10**.5, 10**1.5))
    ZoomCanvas.pan(axes, .5)
    assert axes.get_xlim() == pytest.approx((10, 100))


def test_ctrl_wheel_and_double_click_home(app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    from engineering_math.ui.player import ZoomCanvas
    figure = Figure()
    axes = figure.add_subplot()
    axes.plot([0, 10], [0, 10])
    canvas = ZoomCanvas(figure)
    toolbar = NavigationToolbar2QT(canvas, None)
    canvas.resize(600, 400)
    canvas.draw()
    before = axes.get_xlim()
    center = QPointF(canvas.width()/2, canvas.height()/2)

    def wheel(modifiers):
        event = QWheelEvent(center, center, QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, modifiers,
                            Qt.ScrollPhase.NoScrollPhase, False)
        canvas.wheelEvent(event)
        return event

    wheel(Qt.KeyboardModifier.NoModifier)
    assert axes.get_xlim() == before
    wheel(Qt.KeyboardModifier.ControlModifier)
    zoomed = axes.get_xlim()
    assert zoomed[1]-zoomed[0] == pytest.approx((before[1]-before[0])*.8)
    wheel(Qt.KeyboardModifier.ShiftModifier)
    assert axes.get_xlim()[0] < zoomed[0]
    toolbar.home()
    assert axes.get_xlim() == pytest.approx(before)


def test_player_loop_modes(app):
    from matplotlib.figure import Figure
    from engineering_math.ui.player import PlayerBar, ZoomCanvas
    figure = Figure()
    axes = figure.add_subplot()
    for k in range(4):
        axes.plot(np.linspace(0, 1, 20), k*np.linspace(0, 1, 20), label=f'k = {k}')
    canvas = ZoomCanvas(figure)
    player = PlayerBar(FigureAnimator(figure), canvas)
    assert player.slider.maximum() == 3 and player.slider.value() == 3
    assert player.label.text().startswith('4/4') and 'k = 3' in player.label.text()
    player.toggle()
    assert player.playing and player.slider.value() == 0
    for _ in range(3):
        player.tick()
    player.tick()
    assert not player.playing and player.slider.value() == 3
    player.loop.setCurrentText('반복')
    player.toggle()
    player.go(3)
    player.tick()
    assert player.slider.value() == 0
    player.loop.setCurrentText('왕복')
    player.go(3)
    player.direction = 1
    player.tick()
    assert player.slider.value() == 2 and player.direction == -1
    player.go(0)
    player.tick()
    assert player.slider.value() == 1 and player.direction == 1
    player.pause()
    player.speed.setCurrentText('4×')
    assert player.timer.interval() == int(min(max(4000/4, 30), 700)/4)


def test_static_figure_disables_bar(app):
    from matplotlib.figure import Figure
    from engineering_math.ui.player import PlayerBar, ZoomCanvas
    figure = Figure()
    figure.add_subplot().pcolormesh(np.zeros((3, 3)))
    player = PlayerBar(FigureAnimator(figure), ZoomCanvas(figure))
    assert not player.buttons['play'].isEnabled() and player.label.text() == '정지 그래프'


def test_window_tabs_have_players(app, tmp_path):
    from PySide6.QtCore import QSettings
    from engineering_math.ui.window import MainWindow
    from engineering_math.ui.player import PlayerBar
    window = MainWindow(QSettings(str(tmp_path/'s.ini'), QSettings.Format.IniFormat))
    window.show()
    try:
        window.selector.select('heat')
        window.present(window.topic.compute({}))
        names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert '변화 과정' in names
        # 그래프 탭들 뒤에 변화 과정 탭이 오고, 모두 그래프 바로 아래에 재생 막대가 있습니다.
        for index in range(names.index('변화 과정')+1):
            window.tabs.setCurrentIndex(index)
            app.processEvents()
            page = window.tabs.currentWidget()
            assert isinstance(page.player, PlayerBar)
            assert page.layout().indexOf(page.player) == page.layout().count()-1
        playing = window.tabs.widget(names.index('시간별 온도')).player
        window.tabs.setCurrentIndex(names.index('시간별 온도'))
        playing.toggle()
        window.tabs.setCurrentIndex(names.index('변화 과정'))
        assert not playing.playing
        window.selector.select('half_range')
        window.present(window.topic.compute({}))
        assert '변화 과정' not in [window.tabs.tabText(i) for i in range(window.tabs.count())]
    finally:
        window.close()
        app.processEvents()


def test_heatmap_time_line_moves_through_time():
    import matplotlib.pyplot as plt
    heat = get_topic('heat')
    result = heat.compute({})
    figure = heat.figures(result)['시공간 히트맵']
    animator = FigureAnimator(figure)
    times = result.data['time']
    assert animator.animated and animator.frames == min(len(times), 120)
    np.testing.assert_allclose(animator.parts[0].times, times, atol=1e-12)
    line = animator.parts[0].line
    assert animator.show(0).endswith(f't = {times[0]:.4g}') and line.get_visible()
    assert line.get_ydata()[0] == pytest.approx(times[0])
    animator.show(animator.frames//2)
    assert times[0] < line.get_ydata()[0] < times[-1]
    animator.show(animator.frames-1)
    assert not line.get_visible()
    # x–y 평면의 막 변위는 시간축이 아니므로 정지 그래프입니다.
    membrane = get_topic('rectangle_membrane')
    assert FigureAnimator(membrane.figures(membrane.compute({}))['막의 변위']).frames == 1
    plt.close('all')


def test_play_pause_stop_buttons_and_seek(app):
    from matplotlib.figure import Figure
    from engineering_math.ui.player import PlayerBar, ZoomCanvas
    figure = Figure()
    figure.add_subplot().plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    player = PlayerBar(FigureAnimator(figure), ZoomCanvas(figure))
    buttons = player.buttons
    assert player.seek == 6 and player.slider.pageStep() == 6
    assert buttons['play'].isEnabled() and not buttons['pause'].isEnabled() and not buttons['stop'].isEnabled()
    buttons['play'].click()
    assert player.playing and player.slider.value() == 0
    assert not buttons['play'].isEnabled() and buttons['pause'].isEnabled() and buttons['stop'].isEnabled()
    player.tick()
    buttons['pause'].click()
    assert not player.playing and player.slider.value() == 1 and buttons['stop'].isEnabled()
    buttons['next'].click()
    assert player.slider.value() == 7
    buttons['previous'].click()
    assert player.slider.value() == 1
    buttons['stop'].click()
    assert not player.playing and player.slider.value() == player.slider.maximum()
    assert not buttons['stop'].isEnabled() and buttons['play'].isEnabled()


def test_time_graph_moves_single_curve_without_trails():
    import matplotlib.pyplot as plt
    wave = get_topic('wave')
    result = wave.compute({})
    figure = wave.figures(result)['시간별 변위']
    series = wave.animation(result)
    animator = FigureAnimator(figure, series=series)
    assert animator.frames == len(result.data['time'])
    live = animator.parts[0].line
    snapshots = animator.parts[0].snapshots
    assert animator.show(10).endswith(series['labels'][10])
    assert live.get_visible() and not any(line.get_visible() for line in snapshots)
    np.testing.assert_array_equal(live.get_ydata(), series['frames'][10])
    low, high = figure.axes[0].get_ylim()
    assert low <= series['frames'].min() and high >= series['frames'].max()
    animator.show(animator.frames-1)
    assert not live.get_visible() and all(line.get_visible() for line in snapshots)
    plt.close('all')


def test_time_snapshots_without_series_show_only_current():
    import matplotlib.pyplot as plt
    heat = get_topic('heat')
    figure = heat.figures(heat.compute({}))['시간별 온도']
    animator = FigureAnimator(figure)
    animator.show(2)
    solid = [line for line in figure.axes[0].lines if line.get_linestyle() == '-']
    assert [line.get_visible() for line in solid] == [False, False, True, False, False]
    animator.show(animator.frames-1)
    assert all(line.get_visible() for line in solid)
    # 차수별 곡선은 쌓아서 비교합니다.
    fourier = get_topic('fourier')
    orders = FigureAnimator(fourier.figures(fourier.compute({}))['여러 차수 비교'])
    orders.show(2)
    lines = orders.parts[0].lines
    assert [line.get_visible() for line in lines] == [True, True, True, False, False]
    plt.close('all')
