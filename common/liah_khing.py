"""
掠成果網站（ai-labs）頂懸ê選單佮下跤ê footer，存做 Gradio 服務共用ê HTML。

佇本機執行，執行了檢查 git diff，閣 commit：

    python3 common/liah_khing.py
    python3 common/liah_khing.py --url http://localhost:8000/comming_soon/

預設掠 /comming_soon/：伊有完整ê選單佮 footer，而且選單無任何一項是「目前頁面」。

存出來ê檔案：
- common/static/khing/header.html：無障礙選單、主選單（main 外口ê <nav>）
- common/static/khing/footer.html：<footer>

會改ê所在（佇 common/utils.py 執行ê時陣才換做真正ê值）：
- 版權年份 → {{ ni }}
- 瀏覽人次ê數字 → {{ ga_pv }}
- 「/」開頭ê網址保留原樣，由 utils.py 加 https://SAPOLITA_WEBSITE_HOST
"""
import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

TSHING_SIAT_URL = 'https://ai-labs.ilrdf.org.tw/comming_soon/'
KHING_DIR = Path(__file__).parent / 'static' / 'khing'
TIMEOUT_SECONDS = 30

NI_RE = re.compile(r'(Copyright &copy; )\d{4}')
GA_PV_RE = re.compile(r'(<li\b[^>]*\bid="ga-pv"[^>]*>[^<]*?：)[\d,]+')


class KhingParser(HTMLParser):
    """揣 <main> 外口ê <nav> 佮 <footer>，記in佇原始碼ê起迄位置。"""

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        # HTMLParser.getpos() 是用「\n」算行數，所以這馬嘛干焦用「\n」切
        self.line_starts = [0]
        for tsua in source.split('\n'):
            self.line_starts.append(self.line_starts[-1] + len(tsua) + 1)
        self.main_tshim = 0
        self.tag = None
        self.tshim = 0
        self.khiloh = None
        self.navs = []
        self.footers = []

    def ui_tsi(self):
        tsua, lia = self.getpos()
        return self.line_starts[tsua - 1] + lia

    def handle_starttag(self, tag, attrs):
        if tag == 'main':
            self.main_tshim += 1
        if self.tag is not None:
            if tag == self.tag:
                self.tshim += 1
            return
        if self.main_tshim == 0 and tag in ('nav', 'footer'):
            self.tag = tag
            self.tshim = 1
            self.khiloh = self.ui_tsi()

    def handle_endtag(self, tag):
        if tag == 'main':
            self.main_tshim -= 1
        if tag != self.tag:
            return
        self.tshim -= 1
        if self.tshim > 0:
            return
        kiat = self.source.index('>', self.ui_tsi()) + 1
        html = self.source[self.khiloh:kiat]
        if tag == 'nav':
            self.navs.append(html)
        else:
            self.footers.append(html)
        self.tag = None


def theh_bang_iah(url):
    if urllib.parse.urlsplit(url).scheme not in ('http', 'https'):
        raise ValueError(f'干焦接受 http/https 網址：{url}')
    request = urllib.request.Request(
        url, headers={'User-Agent': 'formosan-ai-liah-khing'})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode('utf-8')


def tshing_khang_tsua(html):
    # Django 模板產生ê空行佮行尾空白提掉，diff 較清氣
    tsuas = []
    for tsua in html.splitlines():
        tsua = tsua.rstrip()
        if tsua.strip():
            tsuas.append(tsua)
    return '\n'.join(tsuas) + '\n'


def sia_khing(filename, url, html):
    tong = KHING_DIR / filename
    tong.parent.mkdir(parents=True, exist_ok=True)
    thau = f'<!-- 由 common/liah_khing.py 對 {url} 掠的，毋通手改 -->\n'
    tong.write_text(thau + tshing_khang_tsua(html), encoding='utf-8')
    print(f'已寫入 {tong}')


def main():
    parser = argparse.ArgumentParser(description='掠成果網站ê選單佮 footer')
    parser.add_argument('--url', default=TSHING_SIAT_URL)
    args = parser.parse_args()

    try:
        source = theh_bang_iah(args.url)
    except (ValueError, urllib.error.URLError, TimeoutError) as e:
        sys.exit(f'掠 {args.url} 失敗：{e}')
    khing = KhingParser(source)
    khing.feed(khing.source)

    if not khing.navs or len(khing.footers) != 1:
        sys.exit(
            f'掠無選單抑是 footer：<nav> {len(khing.navs)} 个、'
            f'<footer> {len(khing.footers)} 个，請確認網址 {args.url}')

    footer, ni_soo = NI_RE.subn(r'\1{{ ni }}', khing.footers[0])
    if ni_soo != 1:
        sys.exit('footer 揣無版權年份「Copyright &copy; 年份」')
    footer, pv_soo = GA_PV_RE.subn(r'\1{{ ga_pv }}', footer)
    if pv_soo == 0:
        print('注意：footer 無瀏覽人次，Gradio 頁面就毋顯示瀏覽人次')

    sia_khing('header.html', args.url, '\n'.join(khing.navs))
    sia_khing('footer.html', args.url, footer)


if __name__ == '__main__':
    main()
