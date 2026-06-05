from __future__ import annotations

import os

# Reduce fragmentation so a 4-bit 7B model loads cleanly on a 16 GB GPU. Must be
# set before torch initializes CUDA, hence at import time.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from .charset import DEFAULT_CHARSET
from .postprocess import KazakhWordCorrector


@dataclass
class OCRResponse:
    raw_text: str
    corrected_text: str
    changed_tokens: list[tuple[str, str]]


# Exact special letters that appear in the Kazakh Latin data. We prime the model
# with this inventory so it does not drift into Turkish/other diacritics.
SPECIAL_LETTERS = "áäçéğıiñóöşúüý"

_SYSTEM_PROMPT = (
    "You are an expert transcriber of handwritten Kazakh written in the Latin "
    "alphabet. You read connected cursive handwriting accurately."
)

_PROMPT = (
    "Transcribe the HANDWRITTEN Kazakh-Latin text in this image, exactly as written.\n\n"
    "CRITICAL: The text is written in the LATIN script. Output LATIN letters only. "
    "NEVER convert to Cyrillic. NEVER translate. Copy the letters you see.\n\n"
    "Rules:\n"
    "1. Transcribe ONLY the handwriting. Ignore any printed/typed text, headers, "
    "and page numbers that may also appear in the image.\n"
    "2. Preserve the original line breaks of the handwriting.\n"
    "3. Use ONLY these characters (lowercase and uppercase forms): "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z and the special "
    f"Kazakh-Latin letters {SPECIAL_LETTERS} (and their uppercase versions). "
    "Do not output any Cyrillic character.\n"
    "4. Pay close attention to diacritics. Distinguish: i vs ı (dotless), "
    "o/ó/ö, u/ú/ü, g/ğ, n/ñ, s/ş, a/á/ä, c/ç, e/é, y/ý.\n"
    "5. Keep punctuation (. , ! ? ; : - ' ) and spacing as written.\n"
    "6. Do NOT translate, explain, or add commentary. Output the transcription "
    "and nothing else."
)

_LINE_PROMPT = (
    "This image is ONE line of handwritten Kazakh written in the LATIN alphabet. "
    "Transcribe exactly the letters you see, on a single line.\n"
    "Output LATIN letters only — NEVER Cyrillic, NEVER a translation.\n"
    "Allowed characters: a-z and the special Kazakh-Latin letters "
    f"{SPECIAL_LETTERS} (plus uppercase) and punctuation. Distinguish i/ı, o/ó/ö, "
    "u/ú/ü, g/ğ, n/ñ, s/ş, a/á/ä, c/ç, e/é, y/ý.\n"
    "Output only the transcription of this one line, nothing else."
)


def _resolve_dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class VLMKazOCR:
    """Local vision-language OCR for handwritten Kazakh Latin.

    Defaults to Qwen2.5-VL-7B, loaded int8 (torchao) so it fits a 16 GB GPU.
    Override with KAZOCR_VLM_MODEL and KAZOCR_VLM_QUANT (auto|torchao|bnb|none).
    """

    def __init__(self, lexicon_path: str | None = None, model_name: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor

        # Default: 7B quantized to int8 with torchao (~9 GB, fits a 16 GB GPU and
        # is far more accurate than 3B on cursive). torchao uses torch's own CUDA
        # kernels, so unlike bitsandbytes it does not break torch on Windows.
        self.model_name = model_name or os.environ.get(
            "KAZOCR_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # KAZOCR_VLM_QUANT: auto | torchao | bnb | none
        quant = os.environ.get("KAZOCR_VLM_QUANT", "auto").lower()
        if quant == "auto":
            big = any(s in self.model_name for s in ("7B", "32B", "72B"))
            quant = "torchao" if (big and self.device == "cuda") else "none"

        load_kwargs: dict = {"dtype": _resolve_dtype()}
        if quant == "torchao" and self.device == "cuda":
            from transformers import TorchAoConfig

            # int8 weight-only: torch-native kernels (no extra native deps like
            # the int4 "mslk" packing lib), near-lossless, and a 7B fits ~9 GB.
            try:
                from torchao.quantization import Int8WeightOnlyConfig

                ao_cfg = TorchAoConfig(Int8WeightOnlyConfig())
            except Exception:
                ao_cfg = TorchAoConfig("int8_weight_only")
            load_kwargs["quantization_config"] = ao_cfg
            load_kwargs["device_map"] = "cuda"
            load_kwargs["low_cpu_mem_usage"] = True
        elif quant == "bnb" and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "cuda"
        else:
            load_kwargs["device_map"] = self.device
        self.quant = quant

        self.model = self._load_model(load_kwargs)
        self.model.eval()

        # Cap the visual tokens so a full A4 page stays fast without losing the
        # handwriting detail (28*28 px per token).
        min_pixels = 256 * 28 * 28
        max_pixels = 1600 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        # Left padding is required for correct batched generation.
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        # How many line crops to transcribe per model call.
        self.batch_size = int(os.environ.get("KAZOCR_LINE_BATCH", "8"))
        self.corrector = KazakhWordCorrector(lexicon_path=lexicon_path)

    def _load_model(self, load_kwargs: dict):
        import transformers

        # Pick the dedicated class for this architecture; the Auto class is a
        # last resort for transformers builds that lack the named class.
        for cls_name in ("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText"):
            if hasattr(transformers, cls_name):
                cls = getattr(transformers, cls_name)
                break
        else:  # pragma: no cover
            cls = transformers.AutoModel

        try:
            return cls.from_pretrained(self.model_name, **load_kwargs)
        except TypeError:
            # Older transformers expect torch_dtype instead of dtype.
            if "dtype" in load_kwargs:
                alt = dict(load_kwargs)
                alt["torch_dtype"] = alt.pop("dtype")
                return cls.from_pretrained(self.model_name, **alt)
            raise

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Light autocontrast helps faint ballpoint ink without distorting shapes.
        return ImageOps.autocontrast(image, cutoff=1)

    @staticmethod
    def _segment_lines(image: Image.Image) -> list[tuple[int, int]]:
        """Split a page into handwritten text-line bands.

        Dense cursive lines are joined by ascenders/descenders, so a simple ink
        threshold merges them. Instead we estimate the line pitch from the ink
        profile's autocorrelation, locate each line's peak, and cut at the ink
        valley between adjacent peaks. Returns (y0, y1) bands top-to-bottom, or
        an empty list when the page is not a clean multi-line layout.
        """
        gray = np.asarray(ImageOps.autocontrast(image.convert("L")), dtype=np.uint8)
        h, w = gray.shape
        thresh = max(120, int(gray.mean() - 25))
        ink = (gray < thresh).sum(axis=1).astype(np.float32) / w  # ink per row

        inked = np.where(ink > 0.01)[0]
        if len(inked) < 5:
            return []

        # Line pitch from autocorrelation of the lightly-smoothed profile.
        ks = max(2, h // 400)
        ink_s = np.convolve(ink, np.ones(2 * ks + 1) / (2 * ks + 1), mode="same")
        centered = ink_s - ink_s.mean()
        autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        lo = 40
        hi = max(lo + 10, h // 3)
        pitch = lo + int(np.argmax(autocorr[lo:hi]))

        # Smooth to ~1/6 of a line so each line is a single broad peak.
        k = max(2, pitch // 6)
        smooth = np.convolve(ink, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
        min_dist = max(8, int(pitch * 0.55))
        peak_floor = 0.2 * smooth.max()

        peaks: list[int] = []
        for y in range(1, h - 1):
            if smooth[y] >= smooth[y - 1] and smooth[y] > smooth[y + 1] and smooth[y] > peak_floor:
                if peaks and y - peaks[-1] < min_dist:
                    if smooth[y] > smooth[peaks[-1]]:
                        peaks[-1] = y
                else:
                    peaks.append(y)
        if len(peaks) < 2:
            return []

        cuts = [int(inked[0])]
        for a, b in zip(peaks, peaks[1:]):
            cuts.append(a + int(np.argmin(smooth[a:b])))
        cuts.append(int(inked[-1]) + 1)

        bands: list[tuple[int, int]] = []
        for i in range(len(cuts) - 1):
            y0, y1 = cuts[i], cuts[i + 1]
            if y1 - y0 >= 12:
                bands.append((max(0, y0 - 3), min(h, y1 + 3)))
        return bands

    def _generate_batch(
        self, images: list[Image.Image], prompt: str, max_new_tokens: int
    ) -> list[str]:
        """Transcribe several crops in one (or a few) batched model calls."""
        import torch
        from qwen_vl_utils import process_vision_info

        results: list[str] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            texts: list[str] = []
            chunk_images: list = []
            for image in chunk:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                img_inputs, _ = process_vision_info(messages)
                chunk_images.extend(img_inputs)

            inputs = self.processor(
                text=texts,
                images=chunk_images,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
            # Left padding makes the prompt length identical across the batch.
            prompt_len = inputs.input_ids.shape[1]
            trimmed = generated[:, prompt_len:]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.extend(d.strip() for d in decoded)
        return results

    def _raw_transcribe(self, image: Image.Image) -> str:
        # Multi-line pages degenerate in a single shot, so transcribe line by
        # line; each crop also gets the full pixel budget, which sharpens detail.
        # Lines are batched through the model together to keep it fast.
        lines = self._segment_lines(image)
        if len(lines) >= 2:
            crops = [image.crop((0, y0, image.width, y1)) for y0, y1 in lines]
            texts = self._generate_batch(crops, _LINE_PROMPT, max_new_tokens=128)
            out = [t.replace("\n", " ").strip() for t in texts]
            out = [t for t in out if t]
            if out:
                return "\n".join(out)
        # Fallback: single short crop or segmentation found nothing usable.
        return self._generate_batch([image], _PROMPT, max_new_tokens=512)[0]

    def recognize(self, image: Image.Image) -> OCRResponse:
        prepared = self._prepare_image(image)
        raw_text = self._raw_transcribe(prepared)
        corrected = self.corrector.correct_text(raw_text)
        return OCRResponse(
            raw_text=raw_text,
            corrected_text=corrected.corrected_text,
            changed_tokens=corrected.changed_tokens,
        )


@lru_cache(maxsize=1)
def get_engine() -> VLMKazOCR:
    return VLMKazOCR()


# Unused import guard kept intentionally: DEFAULT_CHARSET documents the inventory
# the prompt is derived from and is handy for callers building eval reports.
_ = DEFAULT_CHARSET
