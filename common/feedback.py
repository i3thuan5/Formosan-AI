"""使用者對 AI 結果的回饋問卷，一筆一行存成 jsonl。"""
import json
import threading
from datetime import datetime, timedelta, timezone
from os import getenv
from pathlib import Path

FEEDBACKS_PATH = getenv('FEEDBACKS_PATH', default='./feedbacks')
TAIWAN_TIMEZONE = timezone(timedelta(hours=8), 'Asia/Taipei')

FEEDBACK_MAX_LENGTH = 500
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# 同一個 process 內的多個 request 同時寫，愛鎖起來才袂兩行黏做伙
_write_lock = threading.Lock()


def feedbacks_filename(service, now):
    return f'{service}-{now.strftime("%Y%m")}.jsonl'


def build_record(
    service, language, input_text, output_text, rate, feedback, option, now,
):
    feedback = (feedback or '').strip()
    if len(feedback) > FEEDBACK_MAX_LENGTH:
        raise ValueError(f'feedback 最多 {FEEDBACK_MAX_LENGTH} 字')
    return {
        'service': service,
        'option': option,
        'language': language,
        'input': input_text,
        'output': output_text,
        'rate': rate,
        'feedback': feedback,
        'time': now.strftime(TIME_FORMAT),
    }


def save_feedback(
    service, language, input_text, output_text, rate, feedback, option='',
):
    # 檔名佮 time 欄位用仝一个時間，跨月彼一刻才袂一筆分兩月
    now = datetime.now(TAIWAN_TIMEZONE)
    record = build_record(
        service, language, input_text, output_text, rate, feedback, option, now)
    line = json.dumps(record, ensure_ascii=False) + '\n'
    folder = Path(FEEDBACKS_PATH)
    with _write_lock:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / feedbacks_filename(service, now), 'a', encoding='utf-8') as tong:
            tong.write(line)
    return record
