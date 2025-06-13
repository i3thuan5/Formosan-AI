import csv
from io import BytesIO

import requests
from omegaconf import OmegaConf

EXTRA_FORMOSAN_G2P = {
    "z": "z",
    "o": "o",
    "h": "h",
    "g": "g",
    "y": "j",
    "w": "w",
    "c": "ʦ",
    "u": "u",
    "f": "f",
    "v": "v",
    "j": "ɟ",
    "b": "b",
    "q": "q",
    "e": "e",
    "l": "l",
    "d": "d",
}


def gh_download(repo, path):
    headers = {
        "Accept": "application/vnd.github.raw+json",
    }

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download {path} from {repo}, response: {response}")
    response.encoding = "utf-8-sig"

    return response.text


def load_g2p(g2p_string):
    g2p = dict()

    csv_reader = csv.DictReader(g2p_string.split("\n"))

    for row in csv_reader:
        language = row["Language"]
        dialect = row["Dialect"]

        if dialect == "-":
            lang_tag = f"{language}"
        else:
            lang_tag = f"{language}_{dialect}"

        for key in row:
            if key in ["Language", "Dialect"]:
                continue

            if row[key] == "-":
                continue

            g2p[lang_tag] = g2p.get(lang_tag, {})
            g2p[lang_tag][key] = row[key].split(",")

        for g, p in EXTRA_FORMOSAN_G2P.items():
            if g not in g2p[lang_tag]:
                g2p[lang_tag][g] = p

        for lang_tag in g2p:
            # 按照 key 的字元長度排序
            g2p[lang_tag] = dict(
                sorted(g2p[lang_tag].items(), key=lambda x: len(x[0]), reverse=True)
            )

    return g2p


OmegaConf.register_new_resolver("gh_download", gh_download)
OmegaConf.register_new_resolver("load_g2p", load_g2p)
