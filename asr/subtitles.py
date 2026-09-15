"""Render the application's bilingual SRT template."""


def timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def render_srt(segments, translations=None):
    if translations is not None and len(translations) != len(segments):
        raise ValueError("Each segment must have exactly one translation")
    cues = []
    for index, segment in enumerate(segments):
        text = segment["text"].strip()
        if not text or segment["end"] <= segment["start"]:
            continue
        translation = ""
        if translations is not None:
            translation = translations[index].strip()
        cues.append(
            f"{len(cues) + 1}\n"
            f"{timestamp(segment['start'])} --> {timestamp(segment['end'])}\n"
            f"族語：{text}\n華語：{translation}"
        )
    return "\n\n".join(cues)
