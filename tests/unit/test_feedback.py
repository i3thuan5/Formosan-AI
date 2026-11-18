import json
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

NOW = datetime(2026, 11, 17, 9, 5, 3, 7000, tzinfo=TAIWAN_TIMEZONE)


class FakeDatetime(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.current


@pytest.fixture
def feedbacks_path(tmp_path, monkeypatch):
    path = tmp_path / 'feedbacks'
    monkeypatch.setattr(feedback, 'FEEDBACKS_PATH', str(path))
    monkeypatch.setattr(feedback, 'datetime', FakeDatetime)
    FakeDatetime.current = NOW
    return path


def read_lines(path):
    lines = []
    for line in path.read_text(encoding='utf-8').splitlines():
        lines.append(json.loads(line))
    return lines


def test_feedbacks_filename_is_monthly():
    assert feedbacks_filename('mt', NOW) == 'mt-202611.jsonl'
    assert feedbacks_filename('asr-kaldi', NOW) == 'asr-kaldi-202611.jsonl'


def test_save_feedback_writes_one_json_line(feedbacks_path):
    save_feedback(
        'mt', '阿美_南勢', '我們以前的生活', 'sosowalen', '需修改', '  建議  ',
        option='華翻族',
    )
    assert read_lines(feedbacks_path / 'mt-202611.jsonl') == [{
        'service': 'mt',
        'option': '華翻族',
        'language': '阿美_南勢',
        'input': '我們以前的生活',
        'output': 'sosowalen',
        'rate': '需修改',
        'feedback': '建議',
        'time': '2026-11-17 09:05:03',
    }]


def test_save_feedback_appends_to_same_month_and_keeps_chinese(feedbacks_path):
    save_feedback('mt', '阿美_南勢', 'a', 'b', '可用', '', option='族翻華')
    save_feedback('mt', '泰雅_萬大', 'c', 'd', '我不確定', None, option='華翻族')
    tong = feedbacks_path / 'mt-202611.jsonl'
    lines = read_lines(tong)
    assert len(lines) == 2
    assert lines[1]['language'] == '泰雅_萬大'
    assert lines[1]['feedback'] == ''
    assert '泰雅' in tong.read_text(encoding='utf-8')


def test_save_feedback_starts_new_file_next_month(feedbacks_path):
    FakeDatetime.current = datetime(2026, 11, 30, 23, 59, 59, tzinfo=TAIWAN_TIMEZONE)
    save_feedback('mt', '鄒', 'a', 'b', '可用', '', option='族翻華')
    FakeDatetime.current = datetime(2026, 12, 1, 0, 0, 0, tzinfo=TAIWAN_TIMEZONE)
    save_feedback('mt', '鄒', 'c', 'd', '可用', '', option='族翻華')
    assert sorted(p.name for p in feedbacks_path.iterdir()) == [
        'mt-202611.jsonl', 'mt-202612.jsonl']
    assert read_lines(feedbacks_path / 'mt-202612.jsonl')[0]['time'] == '2026-12-01 00:00:00'


def test_option_defaults_to_empty(feedbacks_path):
    record = save_feedback('asr-kaldi', '鄒', 'a', 'b', '可用', '')
    assert record['option'] == ''


@pytest.mark.parametrize('service', ['asr-kaldi', 'asr', 'tts', 'mt'])
def test_allowed_services(service):
    assert build_record(service, '鄒', 'a', 'b', '可用', '', '', NOW)['service'] == service


def test_rejects_unknown_service():
    with pytest.raises(ValueError):
        build_record('nllb', '鄒', 'a', 'b', '可用', '', '', NOW)


@pytest.mark.parametrize('option', ['族翻華', '華翻族', ''])
def test_allowed_options(option):
    assert build_record('mt', '鄒', 'a', 'b', '可用', '', option, NOW)['option'] == option


@pytest.mark.parametrize('option', [None, '族語翻華語', 'to_zh'])
def test_rejects_unknown_option(option):
    with pytest.raises(ValueError):
        build_record('mt', '鄒', 'a', 'b', '可用', '', option, NOW)


@pytest.mark.parametrize('rate', [None, '', '不可用'])
def test_rejects_unknown_rate(rate):
    with pytest.raises(ValueError):
        build_record('mt', '鄒', 'a', 'b', rate, '', '', NOW)


def test_feedback_length_limit():
    assert build_record('mt', '鄒', 'a', 'b', '需修改', '字' * FEEDBACK_MAX_LENGTH, '', NOW)
    with pytest.raises(ValueError):
        build_record('mt', '鄒', 'a', 'b', '需修改', '字' * (FEEDBACK_MAX_LENGTH + 1), '', NOW)
