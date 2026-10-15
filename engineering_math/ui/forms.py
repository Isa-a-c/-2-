"""주제의 입력 명세로 공통 폼을 생성합니다."""
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QFormLayout, QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QPushButton, QSizePolicy)
from engineering_math.topics.catalog import NUMERICAL_KEYS


class InputForm(QWidget):
    changed = Signal()

    def __init__(self, specs, locked_values=None):
        super().__init__()
        self.specs = specs
        self.locked_values = dict(locked_values or {})
        self.fields = {}
        self.rows = {}
        self.form = QFormLayout(self)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.details = QPushButton('수치 설정 펼치기')
        self.details.setCheckable(True)
        self.details.setToolTip('적분·출력 표본과 계산 정밀도 설정입니다. 접어도 값은 유지됩니다.')
        self.details.toggled.connect(self.update_visibility)
        self.form.addRow(self.details)
        self.details.setVisible(any(spec.key in NUMERICAL_KEYS for spec in specs))
        for spec in specs:
            if spec.kind == 'int':
                widget = QSpinBox()
                widget.setRange(int(spec.minimum), int(spec.maximum))
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(spec.minimum), int(spec.maximum))
                slider.valueChanged.connect(widget.setValue)
                widget.valueChanged.connect(slider.setValue)
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                # 숫자 칸은 글자 크기에 맞춘 폭을 유지하고 슬라이더가 남는 폭을 씁니다.
                widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
                metrics = widget.fontMetrics()
                widget.setMinimumWidth(metrics.horizontalAdvance('8'*(len(str(int(spec.maximum)))+1)) + 4*metrics.height())
                layout.addWidget(slider, 1)
                layout.addWidget(widget)
            elif spec.kind == 'float':
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(spec.minimum, spec.maximum)
                row = widget
            elif spec.kind == 'choice':
                widget = QComboBox()
                widget.addItems(spec.choices)
                row = widget
            else:
                widget = QLineEdit()
                widget.setMaxLength(1000)
                row = widget
            widget.setToolTip(spec.help)
            self.fields[spec.key] = widget
            self.rows[spec.key] = row
            self.form.addRow(spec.label, row)
            if spec.kind in ('int', 'float'):
                widget.valueChanged.connect(lambda *_: self.changed.emit())
            elif spec.kind == 'choice':
                widget.currentTextChanged.connect(lambda *_: self.changed.emit())
            else:
                widget.textChanged.connect(lambda *_: self.changed.emit())
        self.set_values({})
        if 'mode' in self.fields:
            self.fields['mode'].currentTextChanged.connect(self.update_visibility)
        self.update_visibility()

    def values(self):
        values = {}
        for spec in self.specs:
            widget = self.fields[spec.key]
            values[spec.key] = widget.value() if spec.kind in ('int', 'float') else widget.currentText() if spec.kind == 'choice' else widget.text()
        return values

    def set_values(self, values):
        blocker = QSignalBlocker(self)
        values = {**values, **self.locked_values}
        for spec in self.specs:
            widget, value = self.fields[spec.key], values.get(spec.key, spec.default)
            if spec.kind in ('int', 'float'):
                widget.setValue(value)
            elif spec.kind == 'choice':
                widget.setCurrentText(value)
            else:
                widget.setText(value)
        self.update_visibility()
        del blocker
        self.changed.emit()

    def update_visibility(self, *_):
        mode = self.fields['mode'].currentText() if 'mode' in self.fields else ''
        self.details.setText('수치 설정 접기' if self.details.isChecked() else '수치 설정 펼치기')
        for spec in self.specs:
            visible = (not spec.visible_in or mode in spec.visible_in)
            visible &= spec.key not in self.locked_values
            visible &= spec.key not in NUMERICAL_KEYS or self.details.isChecked()
            self.form.setRowVisible(self.rows[spec.key], visible)
