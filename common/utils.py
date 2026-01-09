import os
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import gradio as gr

from colors import sa_orange_color, sa_zinc_color

STATIC_DIR_NAME = 'common_static'
COMMON_STATIC_ROOT = Path(__file__).parent / STATIC_DIR_NAME
SAPOLITA_WEBSITE_HOST = os.environ.get('SAPOLITA_WEBSITE_HOST')


@contextmanager
def render_demo(title, js=None, css_paths=[]):
    gr.set_static_paths(
        paths=[
            COMMON_STATIC_ROOT / "image",
            COMMON_STATIC_ROOT / "pdf", ]
    )

    common_css_paths = [COMMON_STATIC_ROOT / 'css' / 'common.css', ]

    demo = gr.Blocks(
        title=title,
        css_paths=(common_css_paths + css_paths),
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
        gr.HTML(
            """
            <nav aria-label="無障礙選單" class="visually-hidden-focusable" role="navigation">
                <div>
                    <a href="#main" class="sa-link p-1 m-1">跳去主內容</a>
                    <a href="https://ai-labs.ilrdf.org.tw/sitemap/" class="sa-link p-1 m-1">網站導覽</a>
                </div>
            </nav>"""
        )
        gr.HTML("""
            <a href="https://{site}/" class="sa-link">
                < 返回成果網站首頁
            </a>
            """.format(site=SAPOLITA_WEBSITE_HOST))

        yield demo

        with gr.Row(equal_height=True):
            gr.HTML(
                "<div>"
                "<hr>"
                "<p class='text-center'>Copyright &copy; {} "
                "財團法人原住民族語言研究發展基金會 版權所有</p>"
                "</div>".format(datetime.now().year))

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                    <img class='img-fluid' src='/gradio_api/file={}/image/ilrdf-logo.png'
                        alt='財團法人原住民族語言研究發展基金會logo'>
                    """.format(STATIC_DIR_NAME))
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                    <p>電話：(02)2341-8508</p>
                    <p>傳真：(02)2341-8256</p>
                    <p>信箱：ilrdf@ilrdf.org.tw</p>
                    <p>地址：100029台北市中正區羅斯福路一段63號</p>
                    """)
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                        <p><a href="https://{site}/copyright/" class="sa-link">著作權聲明</a></p>
                        <p><a href="https://{site}/termofuse/" class="sa-link">網站使用條款</a></p>
                    """.format(site=SAPOLITA_WEBSITE_HOST))

    demo.launch(
        allowed_paths=[
            'ilrdf-logo.png',
            '族語AI翻譯計畫網站-系統操作教學手冊.pdf',
            'SRT字幕語音辨識系統操作手冊.pdf',
        ],
        favicon_path=COMMON_STATIC_ROOT / 'favicon' / 'favicon.svg',
    )
