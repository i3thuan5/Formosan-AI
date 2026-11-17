"""使用者對 AI 結果的回饋問卷，一筆一行存成 jsonl。"""
import json
import threading
from datetime import datetime, timedelta, timezone
from os import getenv
from pathlib import Path

FEEDBACKS_PATH = getenv('FEEDBACKS_PATH', default='./feedbacks')
TAIWAN_TIMEZONE = timezone(timedelta(hours=8), 'Asia/Taipei')

SERVICES = ('asr-kaldi', 'asr', 'tts', 'mt')
RATES = ('可用', '需修改', '我不確定')
FEEDBACK_MAX_LENGTH = 500

# 同一個 process 內的多個 request 同時寫，愛鎖起來才袂兩行黏做伙
_write_lock = threading.Lock()
# service → 這個 process 寫ê檔名。process 第一擺寫 service ê回饋時才決定，
# 了後攏 append 佇仝一个檔；重開服務就開新檔
_filenames = {}


def feedbacks_filename(service, now=None):
    # service-YYYYMMDDHHmmssSSS.jsonl
    if now is None:
        now = datetime.now(TAIWAN_TIMEZONE)
    timestamp = now.strftime('%Y%m%d%H%M%S') + f'{now.microsecond // 1000:03d}'
    return f'{service}-{timestamp}.jsonl'


def build_record(service, language, input_text, output_text, rate, feedback):
    if service not in SERVICES:
        raise ValueError(f'service 只能是 {SERVICES}：{service!r}')
    if rate not in RATES:
        raise ValueError(f'rate 只能是 {RATES}：{rate!r}')
    feedback = (feedback or '').strip()
    if len(feedback) > FEEDBACK_MAX_LENGTH:
        raise ValueError(f'feedback 最多 {FEEDBACK_MAX_LENGTH} 字')
    return {
        'service': service,
        'language': language,
        'input': input_text,
        'output': output_text,
        'rate': rate,
        'feedback': feedback,
    }


def save_feedback(service, language, input_text, output_text, rate, feedback):
    record = build_record(
        service, language, input_text, output_text, rate, feedback)
    line = json.dumps(record, ensure_ascii=False) + '\n'
    folder = Path(FEEDBACKS_PATH)
    with _write_lock:
        folder.mkdir(parents=True, exist_ok=True)
        if service not in _filenames:
            _filenames[service] = feedbacks_filename(service)
        with open(folder / _filenames[service], 'a', encoding='utf-8') as tong:
            tong.write(line)
    return record
