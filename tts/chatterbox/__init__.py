# Model code of Chatterbox (https://github.com/resemble-ai/chatterbox, commit 5de7a54, MIT License, see LICENSE):
# src/chatterbox/models/{t3,s3gen,s3tokenizer,voice_encoder,tokenizers,utils.py}.
# The package's own __init__ (which imports the TTS/VC wrappers and the watermarker) is left out.
#
# 相對於上游只做了讓 flake8 通過的修改，沒有改變行為：空白與縮排、移除未使用的 import 與變數、
# 補 __all__、變數 l 改名、拆開過長的字串，以及補上 utils/mask.py 漏掉的 import logging
# （原本只有在 chunk mask 全為 False 時才會因 NameError 而出錯）。
