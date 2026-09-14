"""WhisperX 3.8.7rc1 extension for Formosan subtitles.

ASR overrides adapted from WhisperX.

BSD 2-Clause License

Copyright (c) 2024, Max Bain

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
from dataclasses import replace
from typing import List, Optional, Union

import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions
from transformers.pipelines.pt_utils import PipelineIterator
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.schema import SingleSegment, TranscriptionResult, ProgressCallback
from whisperx.asr import (
    WhisperModel as BaseWhisperModel,
    FasterWhisperPipeline as BasePipeline,
    load_model as load_base_model,
    find_numeral_symbol_tokens,
)
from whisperx.vads import Pyannote, Silero, Vad
from whisperx.vads.pyannote import Binarize


class FormosanTokenizer(Tokenizer):
    """Tokenizer that accepts language tokens added by the Formosan model."""

    def __init__(
        self,
        tokenizer,
        multilingual: bool,
        task: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.tokenizer = tokenizer

        if not multilingual:
            self.task = None
            self.language = None
            self.language_code = "en"
            return

        if task not in ("transcribe", "translate"):
            raise ValueError(
                f"'{task}' is not a valid task (accepted tasks: transcribe, translate)"
            )

        language_token = f"<|{language}|>"
        language_id = tokenizer.token_to_id(language_token)
        if language_id is None:
            raise ValueError(
                f"Language token does not exist in the model: {language_token}"
            )

        self.task = tokenizer.token_to_id(f"<|{task}|>")
        self.language = language_id
        self.language_code = language


class WhisperModel(BaseWhisperModel):
    """
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    """

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ):
        batch_size = features.shape[0]
        previous_tokens = []
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            previous_tokens = tokenizer.encode(initial_prompt)
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
            hotwords=options.hotwords,
        )

        encoder_output = self.encode(features)

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
        )

        tokens_batch = [x.sequences_ids[0] for x in result]

        text = tokenizer.tokenizer.decode_batch(
            [
                [token for token in tokens if token < tokenizer.eot]
                for tokens in tokens_batch
            ]
        )

        return encoder_output, text, tokens_batch


class FasterWhisperPipeline(BasePipeline):
    """WhisperX ASR with Whisper word timing and VAD subtitle boundaries."""

    def preprocess(self, input_dict):
        audio = input_dict["inputs"]

        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {
            "inputs": features,
            "start": input_dict["start"],
            "end": input_dict["end"],
            "segment_size": input_dict["segment_size"],
        }

    def _forward(self, model_inputs):
        encoder_output, _text, tokens = self.model.generate_segment_batched(
            model_inputs["inputs"], self.tokenizer, self.options
        )
        outputs = [
            [
                {
                    "tokens": tokens[i],
                    "start": model_inputs["start"][i],
                    "end": model_inputs["end"][i],
                    "seek": int(model_inputs["start"][i] * 100),
                }
            ]
            for i in range(len(tokens))
        ]

        self.last_speech_timestamp = self.model.add_word_timestamps(
            outputs,
            self.tokenizer,
            encoder_output,
            num_frames=model_inputs["segment_size"],
            prepend_punctuations=self.options.prepend_punctuations,
            append_punctuations=self.options.append_punctuations,
            last_speech_timestamp=self.last_speech_timestamp,
        )

        outputs = [outputs[i][0]["words"] for i in range(len(outputs))]
        return {
            "words": outputs,
        }

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        def stack(items):
            return {
                "inputs": torch.stack([x["inputs"] for x in items]),
                "start": [x["start"] for x in items],
                "end": [x["end"] for x in items],
                "segment_size": [x["segment_size"] for x in items],
            }

        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack
        )
        model_iterator = PipelineIterator(
            dataloader, self.forward, forward_params, loader_batch_size=batch_size
        )
        return PipelineIterator(model_iterator, self.postprocess, postprocess_params)

    def transcribe(self, audio, *args, **kwargs):
        """Keep request-specific tokenizer and timestamp state isolated on failure."""
        tokenizer, options = self.tokenizer, self.options
        self.last_speech_timestamp = 0.0
        try:
            return self._transcribe(audio, *args, **kwargs)
        finally:
            self.tokenizer, self.options = tokenizer, options
            self.last_speech_timestamp = 0.0

    def _transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if not 0 < chunk_size <= 30:
            raise ValueError("chunk_size must be greater than 0 and at most 30 seconds")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.last_speech_timestamp = 0.0
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg["start"] * SAMPLE_RATE)
                f2 = int(seg["end"] * SAMPLE_RATE)

                yield {
                    "inputs": audio[f1:f2],
                    "start": seg["start"],
                    "end": seg["end"],
                    "segment_size": int(
                        round((f2 - f1) / SAMPLE_RATE * self.model.frames_per_second)
                    ),
                }

        # Pre-process audio and merge chunks as defined by the respective VAD child class
        # In case vad_model is manually assigned (see 'load_model') follow the functionality of pyannote toolkit
        if isinstance(self.vad_model, Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        pre_merge_vad_segments = self.vad_model(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        )
        vad_segments = merge_chunks(
            pre_merge_vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if not vad_segments:
            if progress_callback is not None:
                progress_callback(100.0)
            return {
                "segments": [],
                "language": language or self.preset_language or "unknown",
            }
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = FormosanTokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            current_task = (
                self.tokenizer.task
                if isinstance(self.tokenizer.task, str)
                else self.tokenizer.tokenizer.id_to_token(self.tokenizer.task)[2:-2]
            )
            task = task or current_task
            if task != current_task or language != self.tokenizer.language_code:
                self.tokenizer = FormosanTokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        binarize = Binarize(
            max_duration=chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        segments = (
            pre_merge_vad_segments
            if isinstance(self.vad_model, Silero)
            else binarize(pre_merge_vad_segments).get_timeline()
        )
        segments: List[SingleSegment] = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": "",
            }
            for seg in segments
        ]

        batch_size = batch_size or self._batch_size or 1
        total_segments = len(vad_segments)
        for idx, out in enumerate(
            self.__call__(
                data(audio, vad_segments),
                batch_size=batch_size,
                num_workers=num_workers,
            )
        ):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = (
                    base_progress / 2 if combined_progress else base_progress
                )
                print(f"Progress: {percent_complete:.2f}%...")

            if progress_callback is not None:
                progress_callback(100 * (idx + 1) / total_segments)
            words = out["words"] if batch_size > 1 else out["words"][0]
            last_speech_timestamp_index = 0
            next_last_speech_timestamp_index = 0
            for word in words:
                candidate_indices = []

                for i, segment in enumerate(segments[last_speech_timestamp_index:]):
                    if segment["end"] < word["start"]:
                        next_last_speech_timestamp_index = (
                            last_speech_timestamp_index + i + 1
                        )
                    overlap_start = max(segment["start"], word["start"])
                    overlap_end = min(segment["end"], word["end"])
                    if overlap_start <= overlap_end:
                        candidate_indices.append(last_speech_timestamp_index + i)
                last_speech_timestamp_index = next_last_speech_timestamp_index

                if not candidate_indices:
                    print(
                        f"Warning: Word '{word['word']}' at [{round(word['start'], 3)} --> {round(word['end'], 3)}] is not in any segment."
                    )
                else:
                    largest_overlap = -1
                    best_segment_index = None
                    for i in candidate_indices:
                        segment = segments[i]
                        overlap_start = max(segment["start"], word["start"])
                        overlap_end = min(segment["end"], word["end"])
                        overlap_duration = overlap_end - overlap_start
                        if overlap_duration > largest_overlap:
                            largest_overlap = overlap_duration
                            best_segment_index = i
                    segments[best_segment_index]["text"] += word["word"]
        return {
            "segments": [s for s in segments if s["text"].strip()],
            "language": language,
        }


def load_model(whisper_arch: str, device: str, **kwargs) -> FasterWhisperPipeline:
    """Use WhisperX 3.8.7rc1 loading/options/VAD with our word-timing model."""
    if kwargs.get("model") is None:
        compute_type = kwargs.get("compute_type", "default")
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "float32"
        kwargs["model"] = WhisperModel(
            whisper_arch,
            device=device,
            device_index=kwargs.get("device_index", 0),
            compute_type=compute_type,
            download_root=kwargs.get("download_root"),
            local_files_only=kwargs.get("local_files_only", False),
            cpu_threads=kwargs.get("threads", 4),
            use_auth_token=kwargs.get("use_auth_token"),
        )
    base = load_base_model(whisper_arch, device, **kwargs)
    return FasterWhisperPipeline(
        model=base.model,
        vad=base.vad_model,
        vad_params=base._vad_params,
        options=base.options,
        tokenizer=base.tokenizer,
        language=base.preset_language,
        suppress_numerals=base.suppress_numerals,
    )
