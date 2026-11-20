"""Inference for ILRDF/chatterbox-formosan-all-finetune (Chatterbox T3 finetuned on 42 Formosan dialects).

    tts = FormosanTTS(repo_dir, device="cuda")
    wav, sr = tts.generate("Nga'ay ho!", "ami-x-skl", "ref.wav")

Text goes through the G2P table g2p.csv (orthography -> IPA) and Chatterbox's punctuation normalization (minus its
capitalization and its ':' -> ',' rewrite), exactly as the training data was prepared.
Needs the chatterbox-tts package (github.com/resemble-ai/chatterbox, commit 5de7a54).
"""
import csv
import json
import re
from pathlib import Path

import librosa
import torch
from safetensors.torch import load_file

from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

# graphemes every dialect falls back to when its own row of g2p.csv has no rule for them
EXTRA_G2P = {"z": "z", "o": "o", "h": "h", "g": "g", "y": "j", "w": "w", "c": "ʦ", "u": "u",
             "f": "f", "v": "v", "j": "ɟ", "b": "b", "q": "q", "e": "e", "l": "l", "d": "d"}
META_COLUMNS = {"language", "subgroup", "subgroup_eng", "lang_code"}
S3GEN_VOCAB = 6561  # speech token ids S3Gen has an embedding for


def load_g2p(path):
    """lang_code -> {grapheme: IPA}, longest graphemes first; multi-valued cells use the first IPA."""
    table = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        m = {g: p.split(",")[0].strip() for g, p in row.items()
             if g not in META_COLUMNS and p and p.strip() != "-"}
        for g, p in EXTRA_G2P.items():
            m.setdefault(g, p)
        table[row["lang_code"]] = dict(sorted(m.items(), key=lambda x: len(x[0]), reverse=True))
    return table


def lowercase(text, lang_code):
    # Saisiyat: S (/s/) and s (/θ/) are different letters; Kavalan: R (/ʁ/) is always upper case.
    if lang_code.startswith("xsy"):
        return "".join(c if c == "S" else c.lower() for c in text)
    if lang_code.startswith("ckv"):
        return "".join("R" if c in "Rr" else c.lower() for c in text)
    return text.lower()


def normalize_punctuation(text):
    """Chatterbox's punc_norm() without its capitalization and ':' -> ',' (':' marks vowel length)."""
    text = " ".join(text.split())
    for old, new in [("...", ", "), ("…", ", "), (" - ", ", "), (";", ", "), ("—", "-"), ("–", "-"),
                     (" ,", ","), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if text and not text.endswith((".", "!", "?", "-", ",", "、", "，", "。", "？", "！")):
        text += "."
    return text


class FormosanTTS:
    def __init__(self, repo_dir, device="cuda"):
        repo_dir = Path(repo_dir)
        self.device = device
        self.g2p = load_g2p(repo_dir / "g2p.csv")
        self.dialect_tags = json.load(open(repo_dir / "dialect_tags.json", encoding="utf-8"))
        self.tokenizer = MTLTokenizer(str(repo_dir / "grapheme_mtl_merged_expanded_v1.json"))
        # 2496 = Chatterbox's 2454 multilingual text tokens + 42 dialect tags
        self.hp = T3Config(text_tokens_dict_size=len(self.tokenizer.tokenizer.get_vocab()))
        self.t3 = T3(self.hp)
        self.t3.load_state_dict(load_file(str(repo_dir / "t3_formosan.safetensors")))
        self.t3.to(device).eval()
        self.s3gen = S3Gen()
        self.s3gen.load_state_dict(torch.load(repo_dir / "s3gen.pt", map_location="cpu", weights_only=True))
        self.s3gen.to(device).eval()
        self.ve = VoiceEncoder()
        self.ve.load_state_dict(torch.load(repo_dir / "ve.pt", map_location="cpu", weights_only=True))
        self.ve.to(device).eval()

    def text_to_ipa(self, text, lang_code):
        """Orthographic text -> the model's IPA input. Raises ValueError for characters without a G2P rule."""
        g2p = self.g2p[lang_code]
        text = lowercase(text, lang_code)
        # ellipses (… / ...) are only written, not heard (no pause or intonation change in recordings): drop them
        text = re.sub(r"…+|\.{2,}", " ", text)
        text = re.sub(r"\s+([!?.;,])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\"\-“”]", "", text)
        text = re.sub(r"[ʼ’']", "ʼ", text).replace("^", "⌃")
        words = []
        for word in text.split():
            punct = ""
            if word[-1] in "!?.;,":
                word, punct = word[:-1], word[-1]
            phones, i = [], 0
            while i < len(word):
                key = next((k for k in g2p if word.startswith(k, i)), None)
                if key is None:
                    raise ValueError(f"{lang_code}: no G2P rule for {word[i]!r} in {word!r}")
                phones.append(g2p[key])
                i += len(key)
            words.append("".join(phones) + punct)
        return normalize_punctuation(" ".join(words))

    @torch.inference_mode()
    def generate(self, text, lang_code, ref_audio, temperature=0.8, cfg_weight=0.3, max_new_tokens=1000):
        """text: orthographic text; lang_code: e.g. "ami-x-skl"; ref_audio: path to a recording of the target
        voice. Returns (waveform as a 1-D numpy array, 24000)."""
        hp, device = self.hp, self.device
        tag = self.dialect_tags[lang_code]["tag"].strip("[]")
        ids = self.tokenizer.encode(self.text_to_ipa(text, lang_code), language_id=tag, lowercase=False)
        text_tokens = torch.tensor([[hp.start_text_token] + ids + [hp.stop_text_token]], device=device)
        text_tokens = torch.cat([text_tokens, text_tokens])  # conditional + unconditional rows for CFG

        ref_24k, _ = librosa.load(ref_audio, sr=S3GEN_SR, mono=True)
        ref_16k = librosa.resample(ref_24k, orig_sr=S3GEN_SR, target_sr=S3_SR)
        ref_dict = self.s3gen.embed_ref(ref_24k[: 10 * S3GEN_SR], S3GEN_SR, device=device)
        prompt, _ = self.s3gen.tokenizer.forward([ref_16k[: 6 * S3_SR]], max_len=hp.speech_cond_prompt_len)
        speaker = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)).mean(0, keepdim=True)
        cond = T3Cond(speaker_emb=speaker, cond_prompt_speech_tokens=torch.atleast_2d(prompt),
                      emotion_adv=0.5 * torch.ones(1, 1, 1)).to(device=device)

        speech_tokens = self.t3.inference(t3_cond=cond, text_tokens=text_tokens, max_new_tokens=max_new_tokens,
                                          temperature=temperature, cfg_weight=cfg_weight)[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < S3GEN_VOCAB].to(device)
        wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=ref_dict)
        return wav.squeeze(0).cpu().numpy(), S3GEN_SR
