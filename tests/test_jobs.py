"""상주 작업 프로세스의 재사용·취소·오류 복구를 검증합니다."""
import time
import pytest


@pytest.fixture
def job():
    from PySide6.QtWidgets import QApplication
    from engineering_math.ui.jobs import CalculationJob
    app = QApplication.instance() or QApplication([])
    job = CalculationJob()
    job.outcomes = []
    job.completed.connect(lambda result: job.outcomes.append(('ok', result)))
    job.failed.connect(lambda message: job.outcomes.append(('error', message)))
    job.app = app
    yield job
    job.shutdown()


def wait(job, limit=60):
    deadline = time.monotonic() + limit
    while job.busy and time.monotonic() < deadline:
        job.app.processEvents()
        time.sleep(.005)
    assert not job.busy, '계산 완료 대기 시간이 초과되었습니다.'
    return job.outcomes.pop()


def test_worker_is_reused_and_faster_after_first_job(job):
    job.start('heat', {'num_points': 201, 'time_points': 3})
    kind, result = wait(job)
    assert kind == 'ok' and result.topic == 'heat'
    pid = job.process.pid
    start = time.monotonic()
    job.start('heat', {'num_points': 203, 'time_points': 3})
    kind, result = wait(job)
    assert kind == 'ok' and result.params['num_points'] == 203
    assert job.process.pid == pid
    # 프로세스 생성·모듈 가져오기를 반복하지 않으므로 수백 ms 이내에 끝납니다.
    assert time.monotonic() - start < .5


def test_error_keeps_worker_and_cache(job):
    job.start('fourier', {'N': 10})
    assert wait(job)[0] == 'ok'
    pid = job.process.pid
    job.start('fourier', {'function': '1/0'})
    kind, message = wait(job)
    assert kind == 'error' and message
    job.start('fourier', {'N': 20})
    kind, result = wait(job)
    assert kind == 'ok' and job.process.pid == pid
    assert result.data['cache_reused'] == 10 and result.data['cache_integrated'] == 10


def test_cancel_terminates_running_job_and_next_job_restarts(job):
    job.start('heat', {'num_points': 201, 'time_points': 3})
    wait(job)
    job.start('fourier', {'N': 100, 'num_points': 20000})
    job.cancel()
    assert not job.busy and job.process is None
    job.app.processEvents()
    assert not job.outcomes
    job.start('heat', {'num_points': 201, 'time_points': 3})
    assert wait(job)[0] == 'ok'


def test_new_request_discards_previous_result(job):
    job.start('heat', {'num_points': 201, 'time_points': 3})
    job.start('heat', {'num_points': 205, 'time_points': 3})
    kind, result = wait(job)
    assert kind == 'ok' and result.params['num_points'] == 205
    job.app.processEvents()
    assert not job.outcomes


def test_shutdown_stops_idle_worker(job):
    job.start('heat', {'num_points': 201, 'time_points': 3})
    wait(job)
    assert job.process.is_alive()
    job.shutdown()
    assert job.process is None and job.requests is None and job.results is None
