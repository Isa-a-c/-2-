"""교재 장 → 절 → 예제 순서의 트리 선택기입니다.

절 항목을 누르면 그 실험 구성으로 바뀌고, 교재 예제 항목을 누르면 구성을 바꾼 뒤
exampleChosen 신호로 예제 이름을 알립니다. 각 절의 마지막 '사용자 지정' 항목은 사용자가 함수·매개변수를
직접 입력하는 자리이며 customChosen 신호를 보냅니다. 예제 목록은 주제 모듈을 가져와야 하므로
절을 처음 펼치거나 선택할 때 채웁니다.
"""
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QLabel
from engineering_math.topics.catalog import EXPERIMENTS, experiment_index
from engineering_math.topics.textbook import EXAMPLES as TEXTBOOK

CHAPTERS = (('11.', '11장 Fourier 해석'), ('12.', '12장 편미분방정식'))
ROLE = Qt.ItemDataRole.UserRole
CUSTOM_GROUP = '사용자 지정'


def chapter_of(entry):
    return next(title for prefix, title in CHAPTERS if entry.lesson.startswith(prefix))


def entry_examples(entry, topic):
    """잠긴 입력(defaults)이 이 구성과 맞는 예제만 돌려줍니다."""
    defaults = {spec.key: spec.default for spec in topic.inputs}
    return [name for name, params in topic.examples.items()
            if all(params.get(key, defaults.get(key)) == value for key, value in entry.defaults.items())]


class ExperimentSelector(QWidget):
    currentIndexChanged = Signal(int)
    exampleChosen = Signal(str)
    customChosen = Signal()

    def __init__(self):
        super().__init__()
        self._index = -1
        self.search = QLineEdit()
        self.search.setPlaceholderText('검색: 절 번호, 주제, 예제 번호 (예: 11.2, 열, 12.6-1)')
        self.search.setClearButtonEnabled(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.method = QLabel()
        self.method.setWordWrap(True)
        self.method.setTextFormat(Qt.TextFormat.RichText)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.method)

        self.entries = []
        chapters = {}
        for title in dict.fromkeys(chapter_of(entry) for entry in EXPERIMENTS):
            chapters[title] = QTreeWidgetItem(self.tree, [title])
            chapters[title].setFlags(Qt.ItemFlag.ItemIsEnabled)
        for i, entry in enumerate(EXPERIMENTS):
            item = QTreeWidgetItem(chapters[chapter_of(entry)], [entry.lesson])
            item.setData(0, ROLE, ('entry', i))
            item.setToolTip(0, f'{entry.problem} · {entry.configuration}\n해법: {entry.method}')
            # 펼침 화살표가 보이도록 자리표시 자식을 둡니다.
            QTreeWidgetItem(item, ['…']).setFlags(Qt.ItemFlag.NoItemFlags)
            self.entries.append(item)
        self.loaded = set()
        self.tree.expandToDepth(0)
        self.tree.itemExpanded.connect(self.on_expanded)
        self.tree.itemClicked.connect(self.on_clicked)
        self.tree.itemActivated.connect(self.on_clicked)
        self.search.textChanged.connect(self.apply_filter)
        self.setCurrentIndex(0)

    # ── 기존 API ───────────────────────────────────────────────
    @property
    def experiment(self):
        return EXPERIMENTS[self._index]

    def currentData(self):
        return self.experiment.topic

    def currentIndex(self):
        return self._index

    def findData(self, topic):
        return experiment_index(topic)

    def select(self, topic, params=None):
        self.setCurrentIndex(experiment_index(topic, params))

    def setCurrentIndex(self, index):
        if index == self._index:
            return
        if not 0 <= index < len(EXPERIMENTS):
            raise ValueError('지원하지 않는 실험 구성입니다.')
        entry = EXPERIMENTS[index]
        self._index = index
        with QSignalBlocker(self.tree):
            self.tree.setCurrentItem(self.entries[index])
        self.method.setText(f'<b>{entry.problem}</b> · {entry.configuration}<br>해법: {entry.method}')
        self.currentIndexChanged.emit(index)

    # ── 예제 항목 ──────────────────────────────────────────────
    def ensure_examples(self, index):
        """절 아래에 교재 예제를 두고, 마지막에 직접 입력하는 '사용자 지정' 항목을 둡니다."""
        if index in self.loaded:
            return
        from engineering_math.topics import get_topic
        self.loaded.add(index)
        item = self.entries[index]
        item.takeChildren()
        for name in entry_examples(EXPERIMENTS[index], get_topic(EXPERIMENTS[index].topic)):
            child = QTreeWidgetItem(item, [name])
            child.setData(0, ROLE, ('example', index, name))
            child.setToolTip(0, name)
        custom = QTreeWidgetItem(item, [CUSTOM_GROUP])
        custom.setData(0, ROLE, ('custom', index))
        custom.setToolTip(0, '함수와 매개변수를 직접 입력해 계산합니다. 입력한 내용은 이 절에서 다시 열 때 그대로 남아 있습니다.')
        font = custom.font(0)
        font.setItalic(True)
        custom.setFont(0, font)

    def example_items(self, index):
        """교재 예제 항목을 화면 순서대로 돌려줍니다('사용자 지정' 제외)."""
        self.ensure_examples(index)
        item = self.entries[index]
        return [item.child(i) for i in range(item.childCount()) if item.child(i).data(0, ROLE)[0] == 'example']

    def custom_item(self, index):
        self.ensure_examples(index)
        item = self.entries[index]
        return next(item.child(i) for i in range(item.childCount()) if item.child(i).data(0, ROLE)[0] == 'custom')

    def choose_custom(self):
        """현재 구성의 '사용자 지정' 항목을 선택하고 customChosen을 보냅니다."""
        with QSignalBlocker(self.tree):
            self.tree.setCurrentItem(self.custom_item(self._index))
        self.customChosen.emit()

    def example_names(self, index):
        return [child.text(0) for child in self.example_items(index)]

    def choose_example(self, name):
        """현재 구성의 예제 항목을 선택하고 exampleChosen을 보냅니다."""
        for child in self.example_items(self._index):
            if child.text(0) == name:
                with QSignalBlocker(self.tree):
                    self.tree.setCurrentItem(child)
                self.exampleChosen.emit(name)
                return
        raise ValueError(f'이 구성에 없는 예제입니다: {name}')

    def on_expanded(self, item):
        data = item.data(0, ROLE)
        if data and data[0] == 'entry':
            self.ensure_examples(data[1])

    def on_clicked(self, item, _column=0):
        data = item.data(0, ROLE)
        if not data:
            return
        self.setCurrentIndex(data[1])
        if data[0] == 'entry':
            self.ensure_examples(data[1])
            item.setExpanded(True)
            return
        with QSignalBlocker(self.tree):
            self.tree.setCurrentItem(item)
        if data[0] == 'custom':
            self.customChosen.emit()
        else:
            self.exampleChosen.emit(data[2])

    def apply_filter(self, text):
        """절 이름·문제·구성 또는 예제 이름에 검색어가 들어 있는 항목만 보입니다."""
        text = text.strip().lower()
        for index, item in enumerate(self.entries):
            entry = EXPERIMENTS[index]
            if text and index not in self.loaded and any(
                    example.topic == entry.topic and text in example.label.lower() for example in TEXTBOOK):
                self.ensure_examples(index)
            own = not text or any(text in value.lower() for value in
                                  (entry.lesson, entry.problem, entry.configuration, entry.method))
            matches = 0
            for i in range(item.childCount()):
                child = item.child(i)
                hit = own or text in child.text(0).lower()
                child.setHidden(not hit)
                matches += hit
            item.setHidden(not (own or matches))
            if text and matches and not own:
                item.setExpanded(True)
        for i in range(self.tree.topLevelItemCount()):
            chapter = self.tree.topLevelItem(i)
            chapter.setHidden(all(chapter.child(j).isHidden() for j in range(chapter.childCount())))
            if text:
                chapter.setExpanded(True)
