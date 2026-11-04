import json
import os
import re
from pathlib import Path
from contextlib import contextmanager

import gradio as gr
import httpx

from colors import sa_orange_color, sa_zinc_color

STATIC_DIR_NAME = 'common_static'
COMMON_STATIC_ROOT = Path(__file__).parent / STATIC_DIR_NAME
SAPOLITA_WEBSITE_HOST = os.environ.get('SAPOLITA_WEBSITE_HOST')
GA_MEASUREMENT_ID = os.environ.get('GA_MEASUREMENT_ID', '')
# 成果網站ê選單佮 footer，由 common/liah_khing.py 掠
KHING_DIR = COMMON_STATIC_ROOT / 'khing'
GA_PV_TOKEN = '{{ ga_pv }}'
GA_PV_TIMEOUT_SECONDS = 3


def tsoo_ga_head(ga_id):
    if not ga_id:
        return None
    return (
        '<script async '
        'src="https://www.googletagmanager.com/gtag/js?id={gid}"></script>'
        '<script>'
        'window.dataLayer = window.dataLayer || [];'
        'function gtag(){{dataLayer.push(arguments);}}'
        "gtag('js', new Date());"
        "gtag('config', '{gid}');"
        '</script>'
    ).format(gid=ga_id)


def tok_khing(filename):
    # 「/」開頭ê網址改做連去成果網站
    html = (KHING_DIR / filename).read_text(encoding='utf-8')
    if SAPOLITA_WEBSITE_HOST:
        html = re.sub(
            r'\b(href|src)="/(?!/)',
            rf'\1="https://{SAPOLITA_WEBSITE_HOST}/',
            html,
        )
    return html


def tsoo_khing_head(header, footer):
    # 瀏覽人次ê數字位置，等 demo.load 提著數字才填
    footer = footer.replace(GA_PV_TOKEN, '<span class="sa-ga-pv"></span>')
    khing = json.dumps({'header': header, 'footer': footer}, ensure_ascii=False)
    # 「</」愛閃避，若無 HTML 內底ê </nav> 會予瀏覽器當做 </script>
    khing = khing.replace('</', '<\\/')
    khing_css = (COMMON_STATIC_ROOT / 'css' / 'khing.css').read_text(encoding='utf-8')
    khing_js = (COMMON_STATIC_ROOT / 'js' / 'khing.js').read_text(encoding='utf-8')
    # css 毋用 css_paths：Gradio 會共 @media 內底ê規則加 .gradio-container 前綴，
    # 選單佮 footer 佇 <gradio-app> 外口，就套袂著
    return (
        f'<style>{khing_css}</style>'
        f'<script>window.SA_KHING = {khing};</script>'
        f'<script>{khing_js}</script>'
    )


def tsha_ga_pv():
    # 對成果網站提瀏覽人次，提無就回傳 None
    try:
        response = httpx.get(
            f'https://{SAPOLITA_WEBSITE_HOST}/ga-pv/',
            timeout=GA_PV_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        pv = response.json()['pv']
    except Exception:
        return None
    if pv is None:
        return None
    return int(pv)


@contextmanager
def render_demo(demo_md_filename="", js=None, css_paths=[]):
    gr.set_static_paths(
        paths=[
            COMMON_STATIC_ROOT / "image",
            COMMON_STATIC_ROOT / "pdf", ]
    )

    common_css_paths = [COMMON_STATIC_ROOT / 'css' / 'common.css', ]
    footer = tok_khing('footer.html')
    head = tsoo_khing_head(tok_khing('header.html'), footer)
    if GA_MEASUREMENT_ID:
        head = tsoo_ga_head(GA_MEASUREMENT_ID) + head

    demo = gr.Blocks(
        title=get_title(demo_md_filename),
        delete_cache=(3600, 3600),
        css_paths=(common_css_paths + css_paths),
        head=head,
        theme=gr.themes.Default(
            primary_hue=sa_orange_color,
            neutral_hue=sa_zinc_color,
            font=(
                "tauhu-oo",
                gr.themes.GoogleFont("Source Sans Pro"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            )
        ),
        js=js,
    )

    with demo:
        gr.HTML(
            "<noscript class='sa-alert alert-success'>"
            "很抱歉，本站某些功能須在JavaScript啟用的狀態下才能正常操作。"
            "</noscript>"
        )

        with open(demo_md_filename, 'r', encoding='utf-8') as tong:
            gr.Markdown(tong.readline(), elem_id="main")
            gr.Markdown(tong.read())

        yield demo

        if GA_PV_TOKEN in footer and SAPOLITA_WEBSITE_HOST:
            ga_pv = gr.Number(visible=False)
            demo.load(
                tsha_ga_pv, outputs=ga_pv,
                queue=False, show_progress='hidden',
            ).then(None, inputs=ga_pv, js='(pv) => window.saKhingGaPv(pv)')

    demo.launch(
        allowed_paths=[
            'ilrdf-logo.png',
            '族語AI翻譯計畫網站-系統操作教學手冊.pdf',
            'SRT字幕語音辨識系統操作手冊.pdf',
        ],
        favicon_path=COMMON_STATIC_ROOT / 'favicon' / 'favicon.svg',
    )


def get_title(demo_md_filename):
    with open(demo_md_filename, 'r', encoding='utf-8') as tong:
        return tong.readline().strip("# ")
