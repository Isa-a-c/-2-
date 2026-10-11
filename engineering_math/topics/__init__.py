"""필요한 계산 모듈만 가져오는 주제 등록소입니다."""
from importlib import import_module
from .catalog import EXPERIMENTS, experiment_index
from .textbook import examples_for


def get_topics():
    """전체 검증·기존 API용입니다. 화면에서는 get_topic을 사용합니다."""
    return {key: get_topic(key) for key in dict.fromkeys(item.topic for item in EXPERIMENTS)}


def get_topic(topic_id):
    entry = EXPERIMENTS[experiment_index(topic_id)]
    module = import_module(f'{__name__}.{entry.module}')
    # 상태를 가진 계산 객체는 호출자별로 분리합니다.
    topic = getattr(module, entry.class_name)()
    textbook = examples_for(topic_id)
    if textbook:
        # 교재 본문 예제를 먼저 보여 주고 기존 예시는 그 뒤에 둡니다.
        topic.examples = {**textbook, **topic.examples}
    return topic
