import json
import re
from datetime import datetime

import pytest

import feedback
from feedback import (
    FEEDBACK_MAX_LENGTH,
    TAIWAN_TIMEZONE,
    build_record,
    feedbacks_filename,
    save_feedback,
)


@pytest.fixture
def feedbacks_path(tmp_path, monkeypatch):
    path = tmp_path / 'feedbacks'
    monkeypatch.setattr(feedback, 'FEEDBACKS_PATH', str(path))
    monkeypatch.setattr(feedback, '_filenames', {})
    return path


def only_file(path):
    files = list(path.iterdir())
    assert len(files) == 1
    return files[0]


def read_lines(path):
    lines = []
    for line in only_file(path).read_text(encoding='utf-8').splitlines():
        lines.append(json.loads(line))
    return lines


def test_feedbacks_filename_format():
    now = datetime(2026, 11, 17, 9, 5, 3, 7000, tzinfo=TAIWAN_TIMEZONE)
    assert feedbacks_filename('mt', now) == 'mt-20261117090503007.jsonl'


def test_feedbacks_filename_defaults_to_now():
    assert re.fullmatch(r'asr-kaldi-\d{17}\.jsonl', feedbacks_filename('asr-kaldi'))


def test_save_feedback_writes_one_json_line(feedbacks_path):
    save_feedback('mt', '阿美_南勢', '我們以前的生活', 'sosowalen', '需修改', '  建議  ')
    assert read_lines(feedbacks_path) == [{
        'service': 'mt',
        'language': '阿美_南勢',
        'input': '我們以前的生活',
        'output': 'sosowalen',
        'rate': '需修改',
        'feedback': '建議',
    }]


def test_save_feedback_appends_and_keeps_chinese(feedbacks_path):
    save_feedback('mt', '阿美_南勢', 'a', 'b', '可用', '')
    save_feedback('mt', '泰雅_萬大', 'c', 'd', '我不確定', None)
    lines = read_lines(feedbacks_path)
    assert len(lines) == 2
    assert lines[1]['language'] == '泰雅_萬大'
    assert lines[1]['feedback'] == ''
    assert '泰雅' in only_file(feedbacks_path).read_text(encoding='utf-8')
    assert re.fullmatch(r'mt-\d{17}\.jsonl', only_file(feedbacks_path).name)


@pytest.mark.parametrize('service', ['asr-kaldi', 'asr', 'tts', 'mt'])
def test_allowed_services(service):
    assert build_record(service, '鄒', 'a', 'b', '可用', '')['service'] == service


def test_rejects_unknown_service():
    with pytest.raises(ValueError):
        build_record('nllb', '鄒', 'a', 'b', '可用', '')


@pytest.mark.parametrize('rate', [None, '', '不可用'])
def test_rejects_unknown_rate(rate):
    with pytest.raises(ValueError):
        build_record('mt', '鄒', 'a', 'b', rate, '')


def test_feedback_length_limit():
    assert build_record('mt', '鄒', 'a', 'b', '需修改', '字' * FEEDBACK_MAX_LENGTH)
    with pytest.raises(ValueError):
        build_record('mt', '鄒', 'a', 'b', '需修改', '字' * (FEEDBACK_MAX_LENGTH + 1))
