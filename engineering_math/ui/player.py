"""그래프 탭 아래의 재생 막대와 Ctrl+휠 확대 캔버스입니다.

주제 코드를 고치지 않고 그래프에 이미 그려진 선을 조사해 재생 방식을 정합니다.

- 실선이 셋 이상인 축: 그린 순서대로 한 개씩 나타납니다(시간별·차수별 곡선, 기저 함수).
- 실선 한두 개, 표식 점, 줄기(stem), 막대: 왼쪽부터 그려 나갑니다(끝점에 움직이는 점 표시).
- 점선·파선은 기준선이므로 그대로 두고, 색 지도·등고선만 있는 축은 정지 그래프입니다.

주제의 animation() 데이터(시간에 따른 해, 막의 변위)는 evolution_figure로 '변화 과정' 그래프를 만들어
같은 재생 막대로 움직입니다.
"""
import math
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QStyle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SWEEP_FRAMES = 60
SPEEDS = (('0.25×', .25), ('0.5×', .5), ('1×', 1.), ('2×', 2.), ('4×', 4.))
LOOPS = ('한 번', '반복', '왕복')
ZOOM_STEP = .8
HINT = 'Ctrl+휠 확대 · Shift+휠 좌우 이동 · 더블클릭 원래대로'


def _points(line):
    x = line.get_xdata()
    return len(x) if np.ndim(x) else 1


def _solid(line):
    return line.get_linestyle() in ('-', 'solid') and _points(line) > 2


def _markers(line):
    return line.get_linestyle() in ('None', 'none', '', ' ') and line.get_marker() not in (None, 'None', 'none', '') \
        and _points(line) > 2


def _data_axes(figure):
    return [axes for axes in figure.axes if axes.get_label() != '<colorbar>']


class _Sequence:
    """선을 그린 순서대로 하나씩 보입니다."""

    def __init__(self, lines):
        self.lines = lines

    def show(self, progress):
        k = int(round(progress*(len(self.lines)-1)))
        for i, line in enumerate(self.lines):
            line.set_visible(i <= k)
        return self.lines[k].get_label()

    @property
    def frames(self):
        return len(self.lines)


class _Sweep:
    """선·점·줄기·막대를 왼쪽부터 그려 나갑니다."""

    def __init__(self, axes, lines, markers, stems, bars):
        self.lines = [(line, np.asarray(line.get_xdata()), np.asarray(line.get_ydata())) for line in lines]
        self.markers = [(line, np.asarray(line.get_xdata()), np.asarray(line.get_ydata())) for line in markers]
        self.stems = [(collection, list(collection.get_segments())) for collection in stems]
        self.bars = sorted(bars, key=lambda patch: patch.get_x())
        self.dots = []
        for line, x, y in self.lines:
            dot, = axes.plot([x[-1]], [y[-1]], 'o', color=line.get_color(), markersize=6, zorder=line.get_zorder()+1)
            dot.set_label('_재생 점')
            dot.set_visible(False)
            self.dots.append(dot)
        reference = self.lines[0][1] if self.lines else self.markers[0][1] if self.markers else None
        self.reference = reference

    frames = SWEEP_FRAMES

    def show(self, progress):
        done = progress >= 1
        for (line, x, y), dot in zip(self.lines, self.dots):
            n = max(1, math.ceil(progress*len(x)))
            line.set_data(x[:n], y[:n])
            dot.set_data([x[n-1]], [y[n-1]])
            dot.set_visible(not done)
        for line, x, y in self.markers:
            n = max(1, math.ceil(progress*len(x)))
            line.set_data(x[:n], y[:n])
        for collection, segments in self.stems:
            collection.set_segments(segments[:max(1, math.ceil(progress*len(segments)))])
        for i, patch in enumerate(self.bars):
            patch.set_visible(i < max(1, math.ceil(progress*len(self.bars))))
        if self.reference is not None and len(self.reference) and not done:
            return f'x = {self.reference[max(1, math.ceil(progress*len(self.reference)))-1]:.4g}까지'
        return None


class FigureAnimator:
    """그래프 하나의 프레임을 계산합니다. frames가 1이면 정지 그래프입니다."""

    def __init__(self, figure, custom=None):
        self.figure = figure
        self.custom = custom
        self.parts = []
        if custom is not None:
            self.frames = custom['frames']
            return
        for axes in _data_axes(figure):
            solid = [line for line in axes.lines if _solid(line)]
            markers = [line for line in axes.lines if _markers(line)]
            stems = [c for c in axes.collections if isinstance(c, LineCollection) and len(c.get_segments()) > 2]
            bars = [p for p in axes.patches if isinstance(p, Rectangle)]
            if len(solid) >= 3:
                self.parts.append(_Sequence(solid))
            elif solid or markers or stems or len(bars) > 2:
                self.parts.append(_Sweep(axes, solid, markers, stems, bars if len(bars) > 2 else []))
        self.frames = max((part.frames for part in self.parts), default=1)

    @property
    def animated(self):
        return self.frames > 1

    def show(self, index):
        """index번째 프레임을 그리고 표시할 설명을 돌려줍니다. 마지막 프레임은 원래 그래프입니다."""
        if self.custom is not None:
            return self.custom['show'](index)
        progress = 1. if self.frames <= 1 else index/(self.frames-1)
        progress = max(progress, 1/SWEEP_FRAMES)
        labels = [part.show(progress) for part in self.parts]
        label = next((text for text in labels if text and not text.startswith('_')), '')
        return f'{index+1}/{self.frames}  {label}'.rstrip()


def evolution_figure(data):
    """주제의 animation() 데이터로 '변화 과정' 그래프와 프레임 함수를 만듭니다."""
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 4.8), layout='constrained')
    field = data.get('kind') == 'field'
    axes = figure.add_subplot(111, projection='polar' if field and data.get('polar') else None)
    frames = data['frames']
    if field:
        # 모든 프레임에 같은 색상 범위를 사용하여 진폭 변화를 보존합니다.
        limit = max(float(np.max(np.abs(frames))), 1e-9)
        axes.grid(False)
        mesh = axes.pcolormesh(data['x'], data['y'], frames[0], shading='gouraud', cmap='coolwarm', vmin=-limit, vmax=limit)
        figure.colorbar(mesh, ax=axes, label='Displacement')
        if not data.get('polar'):
            axes.set(xlabel=data['xlabel'], ylabel=data['ylabel'], aspect='equal')
        update = mesh.set_array
    else:
        axes.plot(data['x'], data['reference'], 'k--', label='Initial / original')
        line, = axes.plot(data['x'], frames[0], label='Approximation')
        values = np.r_[np.ravel(frames), data['reference']]
        low, high = float(values.min()), float(values.max())
        margin = max((high-low)*.08, 1e-6)
        axes.set(xlabel=data['xlabel'], ylabel=data['ylabel'], ylim=(low-margin, high+margin))
        axes.grid(alpha=.3)
        axes.legend()
        update = line.set_ydata

    def show(index):
        update(frames[index])
        return f'{index+1}/{len(frames)}  {data["labels"][index]}'

    return figure, {'frames': len(frames), 'show': show, 'start': 0}


class ZoomCanvas(FigureCanvasQTAgg):
    """Ctrl+휠로 마우스 위치를 중심으로 확대·축소하고, Shift+휠로 좌우 이동합니다."""

    def __init__(self, figure):
        super().__init__(figure)
        self.mpl_connect('button_press_event', self._double_click)

    def _double_click(self, event):
        if event.dblclick and self.toolbar is not None:
            self.toolbar.home()

    def _axes_at(self, event):
        ratio = self.device_pixel_ratio
        point = event.position()
        x, y = point.x()*ratio, (self.height()-point.y())*ratio
        axes = self.inaxes((x, y))
        if axes is None or axes.get_label() == '<colorbar>' or axes.name != 'rectilinear':
            return None, None
        return axes, (x, y)

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        steps = event.angleDelta().y()/120 or event.angleDelta().x()/120
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        shift = modifiers & Qt.KeyboardModifier.ShiftModifier
        if not (ctrl or shift) or not steps:
            return super().wheelEvent(event)
        axes, point = self._axes_at(event)
        if axes is None:
            return event.ignore()
        if self.toolbar is not None:
            self.toolbar.push_current()
        if ctrl:
            self.zoom(axes, point, ZOOM_STEP**steps)
        else:
            self.pan(axes, -steps*.1)
        self.draw_idle()
        event.accept()

    @staticmethod
    def zoom(axes, point, factor):
        """화면 좌표 point를 고정점으로 factor배(1보다 작으면 확대) 범위를 바꿉니다."""
        inverse = axes.transScale.inverted()
        center = (axes.transScale + axes.transLimits).inverted().transform(
            axes.transAxes.inverted().transform(point))
        cx, cy = axes.transScale.transform(center)
        (x0, y0), (x1, y1) = axes.transScale.transform([(axes.get_xlim()[0], axes.get_ylim()[0]),
                                                         (axes.get_xlim()[1], axes.get_ylim()[1])])
        new = inverse.transform([(cx+(x0-cx)*factor, cy+(y0-cy)*factor), (cx+(x1-cx)*factor, cy+(y1-cy)*factor)])
        axes.set_xlim(new[0][0], new[1][0])
        axes.set_ylim(new[0][1], new[1][1])

    @staticmethod
    def pan(axes, fraction):
        low, high = axes.transScale.transform([(axes.get_xlim()[0], 0), (axes.get_xlim()[1], 0)])[:, 0]
        shift = (high-low)*fraction
        new = axes.transScale.inverted().transform([(low+shift, 0), (high+shift, 0)])[:, 0]
        axes.set_xlim(*new)


class PlayerBar(QWidget):
    """처음·이전·재생·다음·끝 단추, 프레임 슬라이더, 속도와 반복 방식을 고르는 재생 막대입니다."""
    frameChanged = Signal(int)

    def __init__(self, animator, canvas, parent=None):
        super().__init__(parent)
        self.animator, self.canvas = animator, canvas
        self.direction = 1
        style = self.style()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self.buttons = {}
        for key, icon, tip in [('first', QStyle.StandardPixmap.SP_MediaSkipBackward, '처음 (Home)'),
                               ('previous', QStyle.StandardPixmap.SP_MediaSeekBackward, '이전 프레임 (←)'),
                               ('play', QStyle.StandardPixmap.SP_MediaPlay, '재생 / 일시정지 (Space)'),
                               ('next', QStyle.StandardPixmap.SP_MediaSeekForward, '다음 프레임 (→)'),
                               ('last', QStyle.StandardPixmap.SP_MediaSkipForward, '끝 (End)')]:
            button = QPushButton()
            button.setIcon(style.standardIcon(icon))
            button.setToolTip(tip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            layout.addWidget(button)
            self.buttons[key] = button
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(animator.frames-1, 0))
        layout.addWidget(self.slider, 1)
        self.label = QLabel()
        self.label.setMinimumWidth(self.fontMetrics().horizontalAdvance('0'*22))
        layout.addWidget(self.label)
        self.speed = QComboBox()
        for text, value in SPEEDS:
            self.speed.addItem(text, value)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip('재생 속도')
        self.loop = QComboBox()
        self.loop.addItems(LOOPS)
        self.loop.setToolTip('한 번: 끝에서 멈춤 · 반복: 처음부터 다시 · 왕복: 앞뒤로 되감기')
        layout.addWidget(self.speed)
        layout.addWidget(self.loop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.speed.currentIndexChanged.connect(self.update_interval)
        self.buttons['first'].clicked.connect(lambda: self.go(0))
        self.buttons['previous'].clicked.connect(lambda: self.go(self.slider.value()-1))
        self.buttons['play'].clicked.connect(self.toggle)
        self.buttons['next'].clicked.connect(lambda: self.go(self.slider.value()+1))
        self.buttons['last'].clicked.connect(lambda: self.go(self.slider.maximum()))
        self.slider.valueChanged.connect(self.show_frame)
        self.update_interval()
        if animator.animated:
            start = (animator.custom or {}).get('start', animator.frames-1)
            self.slider.setValue(start)
            self.show_frame(start)
        else:
            for widget in (*self.buttons.values(), self.slider, self.speed, self.loop):
                widget.setEnabled(False)
            self.label.setText('정지 그래프')
            self.setToolTip('색 지도·등고선만 있는 그래프는 재생할 프레임이 없습니다. ' + HINT)

    @property
    def playing(self):
        return self.timer.isActive()

    def update_interval(self):
        # 프레임 수와 관계없이 1× 속도에서 약 4초 동안 재생합니다.
        base = min(max(4000/max(self.animator.frames, 1), 30), 700)
        self.timer.setInterval(int(base/self.speed.currentData()))

    def go(self, index):
        self.slider.setValue(min(max(index, 0), self.slider.maximum()))

    def show_frame(self, index):
        self.label.setText(self.animator.show(index))
        self.canvas.draw_idle()
        self.frameChanged.emit(index)

    def toggle(self):
        if self.playing:
            self.pause()
            return
        if not self.animator.animated:
            return
        if self.loop.currentText() != '왕복' and self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self.direction = 1 if self.loop.currentText() != '왕복' or self.slider.value() < self.slider.maximum() else -1
        self.timer.start()
        self.buttons['play'].setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def pause(self):
        self.timer.stop()
        self.buttons['play'].setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def tick(self):
        value, last, mode = self.slider.value(), self.slider.maximum(), self.loop.currentText()
        target = value+self.direction
        if mode == '왕복':
            if target > last or target < 0:
                self.direction *= -1
                target = value+self.direction
        elif target > last:
            if mode == '한 번':
                self.pause()
                return
            target = 0
        self.slider.setValue(min(max(target, 0), last))

    def keyPressEvent(self, event):
        actions = {Qt.Key.Key_Space: self.toggle, Qt.Key.Key_Left: lambda: self.go(self.slider.value()-1),
                   Qt.Key.Key_Right: lambda: self.go(self.slider.value()+1), Qt.Key.Key_Home: lambda: self.go(0),
                   Qt.Key.Key_End: lambda: self.go(self.slider.maximum())}
        action = actions.get(event.key())
        if action is None:
            return super().keyPressEvent(event)
        action()
