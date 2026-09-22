"""NLLB loading and batched Formosan-to-Chinese translation."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from languages import TRANSLATION_LANGUAGE_CODES

TRANSLATION_MODEL_ID = "ILRDF/nllb-600m-formosan-all-finetune-v2"
TRANSLATION_TARGET_LANGUAGE = "zho_Hant"


class Translator:
    """Load NLLB immediately on the device selected by the application."""

    def __init__(self, device, batch_size=8):
        if batch_size < 1:
            raise ValueError("TRANSLATION_BATCH_SIZE must be positive")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATION_MODEL_ID,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def translate_segments(self, segments, whisper_language):
        """Translate ASR segments to Traditional Chinese with the NLLB fine-tune."""
        try:
            source_language = TRANSLATION_LANGUAGE_CODES[whisper_language]
        except KeyError as error:
            raise ValueError(
                f"No NLLB language token mapping for Whisper language: {whisper_language}"
            ) from error

        texts = [segment["text"].strip() for segment in segments]
        if not texts:
            return []
        if self.batch_size < 1:
            raise ValueError("self.batch_size must be positive")

        translator, tokenizer, device = self.model, self.tokenizer, self.device
        tokenizer.src_lang = source_language
        tokenizer.tgt_lang = TRANSLATION_TARGET_LANGUAGE
        translated_texts = []
        for start in range(0, len(texts), self.batch_size):
            text_batch = texts[start:start + self.batch_size]
            inputs = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.inference_mode():
                translated = translator.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                        TRANSLATION_TARGET_LANGUAGE
                    ),
                    max_new_tokens=256,
                    num_beams=5,
                    no_repeat_ngram_size=4,
                    renormalize_logits=True,
                )
            translated_texts.extend(
                tokenizer.batch_decode(translated, skip_special_tokens=True)
            )
        return translated_texts
