from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import gradio as gr

from colors import sa_orange_color, sa_zinc_color


@contextmanager
def render_demo(title):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "static" / "image"])

    demo = gr.Blocks(
        title=title,
        css_paths=[Path(__file__).parent / 'static' / 'css' / 'app.css', ],
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
        )
    )

    with demo:
        gr.HTML("""
            <a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">
                < 返回成果網站首頁
            </a>
            """)

        yield demo

        with gr.Row(equal_height=True):
            gr.HTML(
                "<div>"
                "<hr>"
                "<p class='text-center'>Copy &copy; {} "
                "財團法人原住民族語言研究發展基金會 版權所有</p>"
                "</div>".format(datetime.now().year))

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                    <img class='img-fluid' src='/gradio_api/file=static/image/ilrdf-logo.png'
                        alt='財團法人原住民族語言研究發展基金會logo'>
                    """)
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                    <p>電話：(02)2341-8508</p>
                    <p>傳真：(02)2341-8256</p>
                    <p>信箱：ilrdf@ilrdf.org.tw</p>
                    <p>地址：100029台北市中正區羅斯福路一段63號</p>
                    """)
            with gr.Column(scale=1, min_width=300):
                gr.HTML("""
                        <p><a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">著作權聲明</a></p>
                        <p><a href="https://ai-no-ilrdf.ithuankhoki.tw/" class="sa-link">網站使用條款</a></p>
                    """)

    demo.launch(
        allowed_paths=['ilrdf-logo.png'],
        favicon_path=Path(__file__).parent / 'static' /
        'favicon' / 'logo.svg'
    )
