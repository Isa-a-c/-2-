"""주제 교체에 독립적인 공업수학 학습 화면입니다."""
import math
import time
from html import escape
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QSignalBlocker, QSettings, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QGuiApplication
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QScrollArea, QTabWidget, QProgressBar, QMessageBox, QCheckBox,
    QFileDialog, QTableView, QTextBrowser, QApplication)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from engineering_math import VERSION
from engineering_math.topics import get_topic
from engineering_math.topics.lessons import lesson_for
from engineering_math.core.project import save_settings, load_settings, export_results
from engineering_math.core.models import validate_params
from .forms import InputForm
from .jobs import CalculationJob
from .player import FigureAnimator, PlayerBar, ZoomCanvas, evolution_figure, HINT
from .experiments import ExperimentSelector

# 마지막 계산이 이 시간(초) 안에 끝난 구성만 입력 변경 시 자동으로 다시 계산합니다.
AUTO_LIMIT = 1.0
AUTO_DELAY_MS = 450
ZOOM_STEPS = (.85, 1., 1.15, 1.3, 1.5, 1.75, 2.)
SUPERSCRIPT = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


KOREAN_FONTS = ('Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR')


def use_korean_fallback_font():
    """그래프 글꼴은 DejaVu Sans를 쓰고, 한글은 설치된 한글 글꼴로 대신 그립니다(글자 깨짐 방지)."""
    import matplotlib
    from matplotlib import font_manager
    installed = {font.name for font in font_manager.fontManager.ttflist}
    fallback = [name for name in KOREAN_FONTS if name in installed]
    if fallback:
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', fallback[0]]
    return fallback


def format_number(value, digits=6):
    """표·지표용 짧은 숫자 표기입니다. 정수는 소수점 없이, 아주 크거나 작은 값은 e 표기입니다."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.{digits}g}'
    return str(value)


def pretty_number(value):
    """지표 줄에서 1.2e-05를 1.2×10⁻⁵처럼 보여 줍니다."""
    text = format_number(value)
    if 'e' in text and isinstance(value, (float, np.floating)) and math.isfinite(value):
        mantissa, exponent = text.split('e')
        return f'{mantissa}×10{str(int(exponent)).translate(SUPERSCRIPT)}'
    return text


class ResultTable(QAbstractTableModel):
    """표의 셀마다 위젯을 생성하지 않고 필요한 값만 표시합니다.

    화면에는 6자리로 줄여 보이고, 풍선 도움말과 복사(Ctrl+C)에는 전체 정밀도를 씁니다.
    """
    def __init__(self, columns, rows, parent=None):
        super().__init__(parent)
        self.columns, self.rows = columns, rows

    def rowCount(self, parent=None):
        return len(self.rows)

    def columnCount(self, parent=None):
        return len(self.columns)

    def raw(self, row, column):
        value = self.rows[row][column]
        return repr(float(value)) if isinstance(value, (float, np.floating)) else format_number(value)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return format_number(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.raw(index.row(), index.column())
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float, np.number)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section] if orientation == Qt.Orientation.Horizontal else str(section+1)


class ResultView(QTableView):
    """선택한 셀을 탭으로 구분한 전체 정밀도 텍스트로 복사합니다(엑셀에 붙여넣기 가능)."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        indexes = self.selectedIndexes()
        if not indexes:
            return
        model = self.model()
        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        chosen = {(index.row(), index.column()) for index in indexes}
        lines = ['\t'.join(model.columns[c] for c in columns)] if len(rows) > 1 else []
        lines += ['\t'.join(model.raw(r, c) if (r, c) in chosen else '' for c in columns) for r in rows]
        QGuiApplication.clipboard().setText('\n'.join(lines))


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle(f'공업수학 학습 스튜디오 {VERSION}')
        self.resize(1400, 900)
        self.states = {}
        # 절마다 사용자 지정 입력을 따로 기억합니다. 교재 예제를 골라도 지워지지 않습니다.
        self.custom_states = {}
        self.custom_mode = False
        self.state_key = None
        self.topic = None
        self.form = None
        self.result = None
        self.figures = []
        self.job = CalculationJob(self)
        self.job.completed.connect(self.present)
        self.job.failed.connect(self.show_error)
        # 글자 배율과 자동 계산 선택을 기억합니다. 테스트는 임시 파일을 넘깁니다.
        use_korean_fallback_font()
        self.settings = settings if settings is not None else QSettings('EngineeringMathStudio', 'studio')
        self.durations = {}
        self.started = None
        self.auto_job = False
        self.auto_waiting = False
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(AUTO_DELAY_MS)
        self.auto_timer.timeout.connect(self.auto_calculate)
        # 확대 배율을 곱하기 전의 기본 글자 크기를 한 번만 기록합니다(창을 다시 만들어도 누적되지 않음).
        app = QApplication.instance()
        if app.property('base_point_size') is None:
            app.setProperty('base_point_size', QApplication.font().pointSizeF())
        self.base_point_size = float(app.property('base_point_size'))
        self.zoom = float(self.settings.value('zoom', 1.))

        # 왼쪽 위는 교재 목차 트리, 아래는 입력 폼입니다. 경계는 끌어서 조절합니다.
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QSplitter(Qt.Orientation.Vertical)
        self.selector = ExperimentSelector()
        left.addWidget(self.selector)
        inputs = QWidget()
        layout = QVBoxLayout(inputs)
        layout.setContentsMargins(0, 0, 0, 0)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        controls = QHBoxLayout()
        calculate = QPushButton('계산하기')
        calculate.setToolTip('Ctrl+Enter')
        calculate.setDefault(True)
        cancel = QPushButton('계산 취소')
        cancel.setToolTip('Esc')
        reset = QPushButton('초기화')
        for widget in (calculate, cancel, reset):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.auto = QCheckBox('값을 바꾸면 자동으로 다시 계산')
        self.auto.setToolTip(f'입력을 바꾸고 잠시 멈추면 다시 계산합니다. 마지막 계산이 {AUTO_LIMIT:g}초보다 '
                             '오래 걸린 구성은 계산하기를 직접 누르십시오.')
        self.auto.setChecked(self.settings.value('auto', 'true') in (True, 'true'))
        self.auto.toggled.connect(lambda checked: self.settings.setValue('auto', checked))
        layout.addWidget(self.auto)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        left.addWidget(inputs)
        left.setStretchFactor(0, 2)
        left.setStretchFactor(1, 3)
        splitter.addWidget(left)

        right = QWidget()
        output = QVBoxLayout(right)
        self.metrics = QLabel('입력값을 확인한 후 계산하기를 누르십시오.')
        self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output.addWidget(self.metrics)
        self.input_state = QLabel()
        self.input_state.setWordWrap(True)
        output.addWidget(self.input_state)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.render_tab)
        output.addWidget(self.tabs)
        self.notices = QLabel()
        self.notices.setWordWrap(True)
        self.notices.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output.addWidget(self.notices)
        splitter.addWidget(right)
        splitter.setSizes([430, 970])
        self.selector.currentIndexChanged.connect(self.change_topic)
        self.selector.exampleChosen.connect(self.apply_example)
        self.selector.customChosen.connect(self.open_custom)
        calculate.clicked.connect(self.calculate)
        cancel.clicked.connect(self.cancel)
        reset.clicked.connect(lambda: self.form.set_values({}))
        for keys in ('Ctrl+Return', 'Ctrl+Enter'):
            QShortcut(QKeySequence(keys), self, self.calculate)
        QShortcut(QKeySequence('Esc'), self, self.cancel)

        menu = self.menuBar().addMenu('파일')
        menu.addAction('설정 저장…', self.save)
        menu.addAction('설정 불러오기…', self.load)
        menu.addAction('결과 CSV 묶음 저장…', self.export)
        menu.addSeparator()
        menu.addAction('종료', self.close)
        view = self.menuBar().addMenu('보기')
        for text, keys, slot in [('글자 크게', ('Ctrl++', 'Ctrl+='), lambda: self.step_zoom(1)),
                                 ('글자 작게', ('Ctrl+-',), lambda: self.step_zoom(-1)),
                                 ('글자 기본 크기', ('Ctrl+0',), lambda: self.set_zoom(1.))]:
            action = view.addAction(text, slot)
            action.setShortcuts([QKeySequence(key) for key in keys])
        self.menuBar().addAction('학습 안내', self.help)
        self.set_zoom(self.zoom, redraw=False)
        self.change_topic()

    def clear_output(self):
        import matplotlib.pyplot as plt
        blocker = QSignalBlocker(self.tabs)
        self.result = None
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        for figure in self.figures:
            plt.close(figure)
        self.figures.clear()

    def change_topic(self, *_):
        self.cancel()
        self.custom_mode = False
        if self.topic is not None:
            self.states[self.state_key] = self.form.values()
        self.state_key = self.selector.currentIndex()
        entry = self.selector.experiment
        self.topic = get_topic(entry.topic)
        self.description.setText(self.topic.description)
        self.build_form(self.states.get(self.state_key, {}))
        self.selector.ensure_examples(self.state_key)
        self.clear_output()
        self.add_lesson()
        self.update_input_state()
        self.auto_timer.stop()
        self.auto_waiting = False
        self.metrics.setText('왼쪽 목록에서 예제를 고르거나 입력값을 확인한 후 계산하기를 누르십시오.')
        self.notices.clear()

    def build_form(self, values):
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.form = InputForm(self.topic.inputs, self.selector.experiment.defaults)
        self.form.set_values(values)
        self.form.changed.connect(self.update_input_state)
        self.form.changed.connect(self.schedule_auto)
        self.form.changed.connect(self.remember_custom)
        self.scroll.setWidget(self.form)

    def apply_example(self, name):
        """예제를 입력에 채우고 바로 계산합니다."""
        self.custom_mode = False
        self.form.set_values(self.topic.examples[name])
        self.calculate()

    def open_custom(self):
        """사용자 지정: 이 절에서 마지막으로 직접 입력한 값(처음에는 기본값)을 불러와 편집합니다."""
        self.custom_mode = True
        self.form.set_values(self.custom_states.get(self.state_key, {}))
        self.auto_timer.stop()
        first = next((widget for spec, widget in zip(self.topic.inputs, self.form.fields.values())
                      if spec.kind == 'text' and not self.form.rows[spec.key].isHidden()), None)
        if first is not None:
            first.setFocus()
            first.selectAll()
        self.statusBar().showMessage('사용자 지정: 함수와 매개변수를 직접 입력한 뒤 계산하기(Ctrl+Enter)를 누르십시오. '
                                     '자동 계산이 켜져 있으면 입력을 멈출 때 계산합니다.')

    def remember_custom(self):
        if self.custom_mode:
            self.custom_states[self.state_key] = self.form.values()

    def schedule_auto(self):
        if self.auto.isChecked() and self.durations.get(self.state_key, 0.) <= AUTO_LIMIT:
            self.auto_timer.start()

    def auto_calculate(self):
        if self.job.busy:
            # 진행 중인 계산을 끊으면 작업 프로세스를 다시 띄워야 하므로 끝난 뒤에 계산합니다.
            self.auto_waiting = True
            return
        try:
            params = validate_params(self.topic.inputs, self.form.values())
        except ValueError as exc:
            self.statusBar().showMessage(f'자동 계산 보류: {exc}')
            return
        if self.result is None or self.result.params != params:
            self.calculate(auto=True)

    def resume_auto(self):
        if self.auto_waiting:
            self.auto_waiting = False
            self.schedule_auto()

    def add_lesson(self):
        browser = QTextBrowser()
        browser.setMarkdown(lesson_for(self.topic.id))
        self.tabs.addTab(browser, '학습 설명')

    def update_input_state(self):
        try:
            current = validate_params(self.topic.inputs, self.form.values())
        except ValueError:
            current = None
        if self.result is None:
            self.input_state.setText('아직 계산 결과가 없습니다.')
            self.input_state.setStyleSheet('')
        elif current != self.result.params:
            self.input_state.setText('입력값이 변경되었습니다. 다시 계산하십시오. 표시·저장되는 결과는 이전 계산의 결과입니다.')
            self.input_state.setStyleSheet('color: #9a4d00; font-weight: bold;')
        else:
            self.input_state.setText('현재 입력값과 계산 결과가 일치합니다.')
            self.input_state.setStyleSheet('')

    def calculate(self, auto=False):
        self.auto_timer.stop()
        self.auto_waiting = False
        self.auto_job = auto
        try:
            params = validate_params(self.topic.inputs, self.form.values())
        except ValueError as exc:
            self.show_error(str(exc))
            return
        if self.result is not None and self.result.topic == self.topic.id and self.result.params == params:
            self.cancel()
            self.statusBar().showMessage('입력이 동일하여 현재 결과를 재사용하였습니다.')
            return
        self.progress.show()
        self.statusBar().showMessage('계산 중입니다. 취소하거나 다른 주제로 이동할 수 있습니다.')
        self.started = time.monotonic()
        try:
            self.job.start(self.topic.id, params)
        except Exception as exc:
            self.show_error(str(exc))

    def cancel(self):
        self.auto_timer.stop()
        self.auto_waiting = False
        self.job.cancel()
        self.progress.hide()
        self.statusBar().showMessage('계산이 중지되었습니다.')

    def present(self, result):
        self.progress.hide()
        if result.topic != self.topic.id:
            return
        auto, self.auto_job = self.auto_job, False
        # 같은 구성을 다시 계산하면 보던 탭(그래프·표)에 그대로 머뭅니다.
        previous = self.result is not None and self.result.topic == result.topic
        current_tab = self.tabs.tabText(self.tabs.currentIndex()) if previous else None
        current_table = self.table_tab_name() if previous else None
        try:
            self.clear_output()
            blocker = QSignalBlocker(self.tabs)
            figures = self.topic.figures(result)
            for name, figure in figures.items():
                page = QWidget()
                page.pending_figure = figure
                self.tabs.addTab(page, name)
                self.figures.append(figure)
            if getattr(self.topic, 'supports_animation', True):
                # 시간·차수에 따른 해를 같은 재생 막대로 보는 탭입니다. 프레임은 탭을 열 때 계산합니다.
                page = QWidget()
                page.pending_evolution = lambda topic=self.topic, result=result: topic.animation(result)
                self.tabs.addTab(page, '변화 과정')
            table_tabs = QTabWidget()
            table_tabs.setObjectName('tables')
            for name, (columns, rows) in result.tables.items():
                table = ResultView()
                table.setModel(ResultTable(columns, rows, table))
                table.setAlternatingRowColors(True)
                table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
                table.setToolTip('셀을 선택하고 Ctrl+C를 누르면 전체 자릿수로 복사됩니다.')
                table_tabs.addTab(table, name)
                if name == current_table:
                    table_tabs.setCurrentIndex(table_tabs.count()-1)
            table_tabs.currentChanged.connect(lambda i: self.fit_table(table_tabs.widget(i)))
            self.tabs.addTab(table_tabs, '수치 표')
            if 'symbolic' in result.data:
                symbolic_view = QTextBrowser()
                symbolic_view.setPlainText(result.data['symbolic'])
                self.tabs.addTab(symbolic_view, '기호 계산')
            settings_view = QTextBrowser()
            lines = ['계산 당시 입력값', '']
            for spec in self.topic.inputs:
                lines.append(f'{spec.label}: {result.params[spec.key]}')
            settings_view.setPlainText('\n'.join(lines))
            self.tabs.addTab(settings_view, '계산 설정')
            self.add_lesson()
            names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
            self.tabs.setCurrentIndex(names.index(current_tab) if current_tab in names else 0)
            del blocker
            self.show_metrics(result)
            self.result = result
            self.update_input_state()
            self.render_tab(self.tabs.currentIndex())
            elapsed = None
            if self.started is not None:
                # 그래프를 그리는 시간까지 포함해야 자동 계산이 화면을 막는지 판단할 수 있습니다.
                # 작업 프로세스를 처음 띄운 계산은 가져오기 시간이 섞이므로 기록하지 않습니다.
                elapsed = time.monotonic() - self.started
                if not self.job.cold:
                    self.durations[self.state_key] = elapsed
                self.started = None
            message = '계산이 완료되었습니다.' if elapsed is None else f'계산이 완료되었습니다 ({elapsed:.2f}초).'
            elapsed = self.durations.get(self.state_key, 0.)
            if self.auto.isChecked() and elapsed > AUTO_LIMIT:
                message += ' 계산이 오래 걸려 이 구성은 자동 계산을 쉽니다. 값을 바꾼 뒤 계산하기를 누르십시오.'
            self.statusBar().showMessage(message)
            self.resume_auto()
        except Exception as exc:
            self.auto_job = auto
            self.show_error(str(exc))

    def show_metrics(self, result):
        """주요 지표를 굵은 이름과 값의 묶음으로 보여 주고, 안내는 글머리표로 나눕니다."""
        cells = [f'<span style="white-space:nowrap"><b>{escape(name)}</b>&nbsp; {escape(pretty_number(value))}</span>'
                 for name, value in result.metrics.items()]
        self.metrics.setText('&nbsp;&nbsp;&nbsp;&nbsp; '.join(cells))
        self.notices.setText('<br>'.join(f'• {escape(notice)}' for notice in result.notices))

    def table_tab_name(self):
        tables = self.tabs.findChild(QTabWidget, 'tables')
        return tables.tabText(tables.currentIndex()) if tables is not None else None

    @staticmethod
    def fit_table(table):
        if isinstance(table, QTableView) and not getattr(table, 'fitted', False):
            table.fitted = True
            table.resizeColumnsToContents()

    def render_tab(self, index):
        """선택한 탭에서만 Qt 캔버스를 만들고 처음 한 번 그립니다."""
        page = self.tabs.widget(index)
        if isinstance(page, QTabWidget):
            self.fit_table(page.currentWidget())
        for player in self.tabs.findChildren(PlayerBar):
            if page is None or not page.isAncestorOf(player):
                player.pause()
        if page is None:
            return
        custom = None
        if hasattr(page, 'pending_evolution'):
            try:
                figure, custom = evolution_figure(page.pending_evolution())
            except Exception as exc:
                del page.pending_evolution
                self.show_error(str(exc))
                return
            del page.pending_evolution
            self.figures.append(figure)
        elif hasattr(page, 'pending_figure'):
            figure = page.pending_figure
            del page.pending_figure
        else:
            return
        # 그래프 바로 아래에 재생 막대를 둡니다. 확대·이동은 Ctrl/Shift+휠과 도구 모음을 함께 씁니다.
        layout = QVBoxLayout(page)
        canvas = ZoomCanvas(figure)
        toolbar = NavigationToolbar2QT(canvas, page)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(HINT))
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        series = None
        if custom is None and self.result is not None and self.tabs.tabText(index).startswith('시간별')                 and getattr(self.topic, 'supports_animation', True):
            # 시간별 그래프는 몇 개의 스냅숏 대신 전체 시간 데이터로 곡선을 움직입니다.
            try:
                series = self.topic.animation(self.result)
            except Exception:
                series = None
        player = PlayerBar(FigureAnimator(figure, custom, series), canvas, page)
        layout.addWidget(player)
        page.player = player
        canvas.draw()

    def show_error(self, message):
        self.progress.hide()
        self.started = None
        if self.auto_job:
            # 입력하는 도중의 오류는 창을 띄우지 않고 상태 표시줄과 안내 줄에만 보여 줍니다.
            self.auto_job = False
            self.statusBar().showMessage(f'자동 계산 실패: {message}')
            self.input_state.setText(f'자동 계산 실패: {message}')
            self.input_state.setStyleSheet('color: #b00020; font-weight: bold;')
            self.resume_auto()
            return
        self.statusBar().showMessage('계산 또는 파일 작업에 실패했습니다.')
        QMessageBox.warning(self, '입력 및 계산 확인', message)
        self.resume_auto()

    def step_zoom(self, step):
        nearest = min(range(len(ZOOM_STEPS)), key=lambda i: abs(ZOOM_STEPS[i]-self.zoom))
        self.set_zoom(ZOOM_STEPS[max(0, min(len(ZOOM_STEPS)-1, nearest+step))])

    def set_zoom(self, zoom, redraw=True):
        """화면 글자와 그래프 글자 크기를 함께 바꾸고 다음 실행에도 기억합니다."""
        import matplotlib
        self.zoom = min(max(float(zoom), ZOOM_STEPS[0]), ZOOM_STEPS[-1])
        self.settings.setValue('zoom', self.zoom)
        font = QApplication.font()
        font.setPointSizeF(self.base_point_size*self.zoom)
        QApplication.setFont(font)
        matplotlib.rcParams['font.size'] = 10*self.zoom
        self.statusBar().showMessage(f'글자 크기 {self.zoom:.0%}')
        if redraw and self.form is not None:
            # 이미 만든 입력 칸은 이전 글자 크기의 폭을 유지하므로 같은 값으로 다시 만듭니다.
            details = self.form.details.isChecked()
            self.build_form(self.form.values())
            self.form.details.setChecked(details)
            self.auto_timer.stop()
        if redraw and self.result is not None:
            self.present(self.result)

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, '설정 저장', '', 'JSON (*.json)')
        if path:
            try:
                save_settings(path, self.topic.id, self.form.values())
            except Exception as exc:
                self.show_error(str(exc))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, '설정 불러오기', '', 'JSON (*.json)')
        if path:
            try:
                self.load_path(path)
            except Exception as exc:
                self.show_error(str(exc))

    def load_path(self, path):
        """대화상자와 분리하여 실제 설정 로딩 경로를 검사할 수 있습니다."""
        data = load_settings(path)
        self.selector.select(data['topic'], data['params'])
        self.form.set_values(data['params'])
        self.calculate()

    def export(self):
        if self.result is None:
            self.show_error('먼저 계산을 완료하십시오.')
            return
        path, _ = QFileDialog.getSaveFileName(self, '결과 저장', '', 'ZIP (*.zip)')
        if path:
            try:
                export_results(path, self.result)
                self.statusBar().showMessage(f'저장되었습니다: {Path(path).name}')
            except Exception as exc:
                self.show_error(str(exc))

    def help(self):
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == '학습 설명':
                self.tabs.setCurrentIndex(index)
                break

    def closeEvent(self, event):
        self.cancel()
        self.job.shutdown()
        self.clear_output()
        event.accept()
