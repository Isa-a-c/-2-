"""GUI와 독립적인 설정 파일 및 CSV 묶음 저장입니다."""
import csv
import io
import json
import zipfile
import math
from pathlib import Path
from .models import validate_params
from engineering_math.topics import get_topic


# 계산 방식이 바뀌어 의미가 사라진 이전 설정 항목입니다. 불러올 때 제거하고 기본값을 씁니다.
RETIRED_KEYS = {'infinite_heat': ('order',)}


def settings(topic, params):
    return {'version': 2, 'topic': topic, 'params': validate_params(get_topic(topic).inputs, params)}


def load_settings(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('설정 파일은 JSON 객체여야 합니다.')
    if data.get('version') == 1:
        params = {key: data[key] for key in ('function', 'L', 'N', 'num_points', 'comparison_N') if key in data}
        params['comparison_N'] = ','.join(map(str, params.get('comparison_N', [])))
        params.update(data.get('advanced', {}))
        for key in ('center', 'width', 'frequency', 'rate'):
            if key in params:
                params[key] = float(params[key])
        if 'M' in params:
            params['M'] = int(params['M'])
        data = {'version': 2, 'topic': 'fourier', 'params': params}
    if data.get('version') != 2:
        raise ValueError('지원하지 않는 설정 버전입니다.')
    params = data.get('params')
    if isinstance(params, dict):
        params = {key: value for key, value in params.items()
                  if key not in RETIRED_KEYS.get(data.get('topic'), ())}
        # 주제가 이전 값의 이름 변경 규칙(migrate)을 제공하면 적용합니다.
        topic = get_topic(data['topic']) if isinstance(data.get('topic'), str) else None
        if topic is not None and hasattr(topic, 'migrate'):
            params = topic.migrate(params)
    return settings(data['topic'], params)


def save_settings(path, topic, params):
    Path(path).write_text(json.dumps(settings(topic, params), ensure_ascii=False, indent=2), encoding='utf-8')


def export_results(path, result):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('settings.json', json.dumps(settings(result.topic, result.params), ensure_ascii=False, indent=2))
        archive.writestr('metrics.json', json.dumps(result.metrics, ensure_ascii=False, indent=2))
        for name, (columns, rows) in result.tables.items():
            # 전체 CSV 문자열을 메모리에 복제하지 않고 ZIP 항목에 순서대로 씁니다.
            with archive.open(name+'.csv', 'w') as raw:
                with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(['' if isinstance(value, float) and not math.isfinite(value) else value for value in row] for row in rows)
