"""수식 평가와 계산을 종료 가능한 상주 작업 프로세스에서 수행합니다.

작업 프로세스는 완료 후에도 유지하여 Python·NumPy·SymPy 가져오기 비용을
계산마다 반복하지 않습니다. 계산 중 취소·새 요청은 프로세스를 강제 종료하고
다음 요청에서 새로 시작합니다. 따라서 종료되지 않는 입력도 화면을 막지 않습니다.
"""
import multiprocessing as mp
import queue
from PySide6.QtCore import QObject, QTimer, Signal


def compute_worker(output, topic_id, params, cache):
    """한 번의 계산을 수행하고 (성공 여부, 결과 또는 오류, 캐시)를 보냅니다."""
    try:
        from engineering_math.topics import get_topic
        topic = get_topic(topic_id)
        if hasattr(topic, 'restore_cache'):
            topic.restore_cache(cache)
        result = topic.compute(params)
        snapshot = topic.snapshot_cache() if hasattr(topic, 'snapshot_cache') else None
        output.put((True, result, snapshot))
    except Exception as exc:
        output.put((False, f'{type(exc).__name__}: {exc}', None))


class _Tagged:
    """compute_worker의 출력에 작업 번호를 붙여 결과 큐로 전달합니다."""

    def __init__(self, results, job_id):
        self.results, self.job_id = results, job_id

    def put(self, message):
        self.results.put((self.job_id, *message))


def worker_loop(requests, results):
    """요청을 순서대로 처리합니다. 쉬는 동안 나머지 주제 모듈을 미리 가져옵니다."""
    from importlib import import_module
    from engineering_math.topics.catalog import EXPERIMENTS
    pending = list(dict.fromkeys(f'engineering_math.topics.{item.module}' for item in EXPERIMENTS))
    while True:
        try:
            message = requests.get(timeout=.02 if pending else None)
        except queue.Empty:
            try:
                import_module(pending.pop(0))
            except Exception:
                pass  # 미리 가져오기 실패는 실제 요청에서 오류로 보고됩니다.
            continue
        if message is None:
            return
        job_id, topic_id, params, cache = message
        compute_worker(_Tagged(results, job_id), topic_id, params, cache)


class CalculationJob(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.requests = None
        self.results = None
        self.job_id = 0
        self.pending = None
        # 마지막 성공 결과의 캐시만 보관합니다. 실패·취소는 캐시를 교체하지 않습니다.
        self.caches = {}
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.poll)

    @property
    def busy(self):
        return self.pending is not None

    def ensure_worker(self):
        if self.process is not None and self.process.is_alive():
            return
        self.stop_worker()
        context = mp.get_context('spawn')
        self.requests, self.results = context.Queue(), context.Queue()
        self.process = context.Process(target=worker_loop, args=(self.requests, self.results), daemon=True)
        self.process.start()

    def start(self, topic_id, params):
        self.cancel()
        self.ensure_worker()
        self.job_id += 1
        self.pending = (self.job_id, topic_id)
        self.requests.put((self.job_id, topic_id, params, self.caches.get(topic_id)))
        self.timer.start()

    def poll(self):
        if not self.busy:
            self.timer.stop()
            return
        while True:
            try:
                job_id, success, payload, snapshot = self.results.get_nowait()
            except queue.Empty:
                if not self.process.is_alive():
                    self.pending = None
                    self.stop_worker()
                    self.failed.emit('계산 프로세스가 결과 없이 종료되었습니다.')
                return
            if job_id == self.pending[0]:
                break
        self.pending = None
        self.timer.stop()
        if success:
            if snapshot is not None:
                self.caches[payload.topic] = snapshot
            self.completed.emit(payload)
        else:
            self.failed.emit(payload)

    def cancel(self):
        """진행 중인 계산만 중단합니다. 쉬고 있는 작업 프로세스는 유지합니다."""
        self.timer.stop()
        if self.busy:
            self.pending = None
            self.stop_worker()

    def stop_worker(self, graceful=False):
        if self.process is not None:
            if graceful and self.process.is_alive():
                try:
                    self.requests.put(None)
                    self.process.join(timeout=.5)
                except (OSError, ValueError):
                    pass
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            self.process.close()
            self.process = None
        for channel in (self.requests, self.results):
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()
        self.requests = self.results = None

    def shutdown(self):
        self.cancel()
        self.stop_worker(graceful=True)
