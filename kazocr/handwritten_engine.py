from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .postprocess import KazakhWordCorrector


@dataclass
class OCRResponse:
    raw_text: str
    corrected_text: str
    changed_tokens: list[tuple[str, str]]


class HandwrittenKazOCR:
    def __init__(self, lexicon_path: str | None = None) -> None:
        cache_root = Path(__file__).resolve().parents[1] / ".paddlex_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("paddleocr is not installed. Run `pip install -r requirements.txt`.") from exc

        self.ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.corrector = KazakhWordCorrector(lexicon_path=lexicon_path)

    def _resize_for_page_ocr(self, image: Image.Image) -> Image.Image:
        max_width = 1280
        if image.width <= max_width:
            return image
        scale = max_width / max(1, image.width)
        target_size = (max_width, max(1, int(image.height * scale)))
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def _prepare_variants(self, image: Image.Image) -> list[Image.Image]:
        base = ImageOps.autocontrast(self._resize_for_page_ocr(image).convert("L"))
        variants: list[Image.Image] = [base]

        # Keep the default path fast. Extra variants help mostly on small crops.
        if base.width < 900 and base.height < 260:
            variants.append(base.filter(ImageFilter.SHARPEN))
            variants.append(ImageEnhance.Contrast(base).enhance(1.45))

        arr = np.asarray(base, dtype=np.uint8)
        if base.width < 900 and base.height < 260:
            binary = np.where(arr < 185, 0, 255).astype(np.uint8)
            variants.append(Image.fromarray(binary, mode="L"))

        prepared: list[Image.Image] = []
        seen = set()
        for variant in variants:
            rgb = Image.merge("RGB", (variant, variant, variant))
            key = hash(rgb.tobytes()[:: max(1, len(rgb.tobytes()) // 2048)])
            if key in seen:
                continue
            seen.add(key)
            prepared.append(rgb)
        return prepared

    def _extract_lines(self, result: object) -> list[tuple[float, float, str, float]]:
        lines: list[tuple[float, float, str, float]] = []
        if not result:
            return lines

        pages = result if isinstance(result, list) else [result]
        for page in pages:
            entries = page
            if isinstance(page, dict):
                entries = page.get("rec_texts") or page.get("texts") or []
                boxes = page.get("rec_polys") or page.get("dt_polys") or []
                scores = page.get("rec_scores") or page.get("scores") or []
                if entries and boxes:
                    for idx, text in enumerate(entries):
                        box = boxes[idx]
                        score = float(scores[idx]) if idx < len(scores) else 0.0
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        lines.append((min(ys), min(xs), str(text), score))
                    continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not entry or len(entry) < 2:
                    continue
                box = entry[0]
                rec = entry[1]
                text = rec[0] if isinstance(rec, (list, tuple)) and rec else str(rec)
                score = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.0
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                lines.append((min(ys), min(xs), str(text), score))
        return lines

    def _recognize_variant(self, image: Image.Image) -> tuple[str, float]:
        result = self.ocr.predict(np.asarray(image))
        lines = self._extract_lines(result)
        if not lines:
            return "", -10.0

        lines.sort(key=lambda item: (round(item[0] / 12), item[1]))
        raw_text = "\n".join(text.strip() for _, _, text, _ in lines if text.strip())
        avg_conf = sum(score for *_, score in lines) / max(1, len(lines))
        corrected = self.corrector.correct_text(raw_text)
        combined_score = corrected.score + avg_conf * 1.5
        return raw_text, combined_score

    def recognize(self, image: Image.Image) -> OCRResponse:
        best_raw = ""
        best_score = -10.0
        variants = self._prepare_variants(image)

        # Full-page handwriting on CPU should stay responsive: use a single fast pass.
        if image.width >= 1000 or image.height >= 420:
            variants = variants[:1]

        for variant in variants:
            raw_text, score = self._recognize_variant(variant)
            if score > best_score:
                best_raw = raw_text
                best_score = score

        corrected = self.corrector.correct_text(best_raw)
        return OCRResponse(
            raw_text=best_raw,
            corrected_text=corrected.corrected_text,
            changed_tokens=corrected.changed_tokens,
        )
