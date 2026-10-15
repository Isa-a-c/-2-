"""소스와 실행 파일에서 같은 Qt 통합 검사를 수행합니다."""
import json
import time
import tempfile
from pathlib import Path
from engineering_math.topics.fourier import MODES
from engineering_math.core.project import save_settings, load_settings, export_results
from engineering_math.topics.catalog import EXPERIMENTS


def wait_for_job(application, window):
    deadline = time.monotonic()+90
    pulses = 0
    while window.job.busy and time.monotonic() < deadline:
        application.processEvents()
        pulses += 1
        time.sleep(.01)
    assert not window.job.busy, '계산 완료 대기 시간이 초과되었습니다.'
    return pulses


def open_tab(application, window, name):
    names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    window.tabs.setCurrentIndex(names.index(name))
    application.processEvents()
    return window.tabs.currentWidget()


def play_to_end(application, player):
    """재생 단추로 몇 프레임 진행한 뒤 마지막 프레임(원래 그래프)으로 이동합니다."""
    assert player.animator.animated
    player.go(0)
    player.toggle()
    deadline = time.monotonic()+5
    while player.slider.value() < 2 and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.01)
    player.pause()
    assert player.slider.value() >= 2
    player.go(player.slider.maximum())
    application.processEvents()


def run(application, window, report_path):
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    checks = []
    errors = []
    window.show_error = errors.append
    window.job.failed.disconnect()
    window.job.failed.connect(errors.append)
    try:
        window.show()
        application.processEvents()
        # 실제 Qt 이벤트 루프를 펌프하며 별도 계산 프로세스의 완료를 기다립니다.
        window.calculate()
        pulses = wait_for_job(application, window)
        assert window.result is not None and not errors, errors
        assert pulses > 2
        checks.append('Qt 비동기 푸리에 계산 및 화면 응답')
        for key, value in [('N', 37), ('num_points', 7311)]:
            window.form.fields[key].setValue(value)
            from PySide6.QtWidgets import QSlider
            slider = window.form.rows[key].findChild(QSlider)
            assert slider.value() == value
            slider.setValue(value+1)
            assert window.form.fields[key].value() == value+1
        checks.append('N 및 표본 수 숫자 입력·슬라이더 양방향 동기화')
        assert '변경' in window.input_state.text()
        window.form.set_values({'N': 20})
        window.calculate()
        # 계산 중 수정한 입력을 완료된 결과와 혼동하지 않아야 합니다.
        window.form.fields['N'].setValue(21)
        wait_for_job(application, window)
        assert window.result.params['N'] == 20
        assert window.result.data['cache_reused'] == 10
        assert window.result.data['cache_integrated'] == 10
        assert '변경' in window.input_state.text()
        window.form.fields['N'].setValue(20)
        assert '일치' in window.input_state.text()
        checks.append('프로세스 사이 계수 재사용 및 계산 중 입력 변경 안내')
        window.form.fields['function'].setText('1/0')
        window.calculate()
        wait_for_job(application, window)
        assert errors
        errors.clear()
        previous = window.result
        window.form.set_values({'N': 20})
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.result is previous and not window.job.busy
        checks.append('잘못된 입력 이후 복구 및 성공 캐시 보존')
        for mode in MODES:
            result = window.topic.compute({'mode': mode})
            window.present(result)
            assert window.result is result and not errors, errors
            application.processEvents()
        checks.append('푸리에 기본 및 추가 분석 6종 그래프 표시')
        play_to_end(application, open_tab(application, window, '변화 과정').player)
        play_to_end(application, open_tab(application, window, window.tabs.tabText(0)).player)
        checks.append('푸리에 부분합 변화 과정 탭과 그래프 아래 재생 막대')
        window.calculate()
        window.cancel()
        assert window.job.process is None
        checks.append('계산 취소')
        window.selector.setCurrentIndex(window.selector.findData('heat'))
        result = window.topic.compute({})
        window.present(result)
        assert window.tabs.count() == 8 and not errors, errors
        checks.append('열방정식 곡선·히트맵·수치 표 표시')
        application.processEvents()
        screenshot = Path(report_path).with_suffix('.png')
        window.grab().save(str(screenshot))
        play_to_end(application, open_tab(application, window, '시간별 온도').player)
        play_to_end(application, open_tab(application, window, '변화 과정').player)
        checks.append('열방정식 시간별 곡선 순차 재생과 변화 과정 마지막 프레임')
        window.calculate()
        window.selector.setCurrentIndex(window.selector.findData('wave'))
        assert not window.job.busy and window.result is None
        window.calculate()
        wait_for_job(application, window)
        result = window.result
        assert result.topic == 'wave' and window.tabs.count() == 11 and not errors, errors
        checks.append('계산 중 주제 변경 및 파동 비동기 계산·열 비교 그래프')
        window.resize(1000, 700)
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('wave-small.png')))
        window.help()
        assert window.tabs.tabText(window.tabs.currentIndex()) == '학습 설명'
        assert '초기 속도' in window.tabs.currentWidget().toPlainText()
        play_to_end(application, open_tab(application, window, '변화 과정').player)
        checks.append('파동 변화 과정 재생·학습 설명 및 작은 창 표시')
        with tempfile.TemporaryDirectory(prefix='공업수학 검증 ') as folder:
            path = Path(folder)/'설정.json'
            save_settings(path, result.topic, result.params)
            assert load_settings(path)['topic'] == 'wave'
            export_results(Path(folder)/'결과.zip', result)
            window.selector.setCurrentIndex(window.selector.findData('fourier'))
            window.load_path(path)
            wait_for_job(application, window)
            assert window.result.topic == 'wave' and not errors
        checks.append('한글 경로 설정 및 결과 저장')
        from engineering_math.core.expr import parse_expression, evaluate
        assert evaluate(parse_expression('besselj(0,x)'), ([0.],))[0] == 1.
        checks.append('SciPy 특수함수 평가')
        window.selector.setCurrentIndex(window.selector.findData('fourier'))
        window.form.set_values({'function': 'A*sinc(x)', 'parameters': 'A=2'})
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.result.params['parameters'] == 'A=2', errors
        window.help()
        assert '매개변수' in window.tabs.currentWidget().toPlainText()
        checks.append('확장 함수 및 공통 매개변수 입력·도움말')
        # 교재 순서의 새 주제 모두를 실제 작업 프로세스와 Qt 화면으로 검증합니다.
        chapter11_ids = list(dict.fromkeys(item.topic for item in EXPERIMENTS
                                           if item.lesson.startswith('11.') and item.topic != 'fourier'))
        for topic_id in chapter11_ids:
            window.selector.setCurrentIndex(window.selector.findData(topic_id))
            window.calculate()
            wait_for_job(application, window)
            assert window.result is not None and window.result.topic == topic_id and not errors, errors
            assert window.figures and '변화 과정' not in [window.tabs.tabText(i) for i in range(window.tabs.count())]
            assert open_tab(application, window, window.tabs.tabText(0)).player is not None
            window.help()
            assert '11.' in window.tabs.currentWidget().toPlainText()
        window.tabs.setCurrentIndex(0)
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('chapter11-window.png')))
        checks.append('11.2~11.10 전체 주제(DFT 포함)의 비동기 계산·그래프·학습 설명')
        chapter12_ids = ['pde_basics', 'dalembert', 'rectangle_laplace', 'infinite_heat', 'rectangle_membrane',
                         'disk_membrane', 'coordinate_potential', 'pde_laplace']
        for topic_id in chapter12_ids:
            window.selector.setCurrentIndex(window.selector.findData(topic_id))
            window.calculate()
            wait_for_job(application, window)
            assert window.result is not None and window.result.topic == topic_id and not errors, errors
            assert window.figures
            window.help()
            assert '12.' in window.tabs.currentWidget().toPlainText()
            if topic_id in ('rectangle_membrane', 'disk_membrane'):
                page = open_tab(application, window, '변화 과정')
                play_to_end(application, page.player)
                page.player.canvas.draw()
                window.grab().save(str(Path(report_path).with_name(topic_id + '-animation.png')))
        window.tabs.setCurrentIndex(0)
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('chapter12-window.png')))
        checks.append('12장 새 PDE 주제·기호 잔차·막의 2차원 애니메이션')
        window.form.fields['end_time'].setValue(3.)
        window.calculate()
        window.close()
        assert window.job.process is None
        checks.append('계산 중 창 종료 시 작업 프로세스 정리')
        import sys
        report = dict(passed=True, frozen=bool(getattr(sys, 'frozen', False)), checks=checks)
        code = 0
    except Exception as exc:
        import traceback
        report = dict(passed=False, checks=checks, error=str(exc), traceback=traceback.format_exc())
        code = 1
    finally:
        window.close()
        application.processEvents()
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return code
