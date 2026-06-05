from __future__ import annotations

"""Batch-run the local VLM OCR over a folder of images and write a report.

Usage:
    python evaluate.py --images "D:/Projects/PycharmProjects/Kazakh Latin Handwriting Dataset"
    python evaluate.py --images <dir> --crop-top 0.26   # drop printed header band

If a sibling <name>.gt.txt file exists next to an image, character error rate
(CER) against it is reported as well.
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from kazocr.vlm_engine import VLMKazOCR

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _cer(ref: str, hyp: str) -> float:
    ref = " ".join(ref.split())
    hyp = " ".join(hyp.split())
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein over characters.
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def _crop_top(image: Image.Image, frac: float) -> Image.Image:
    if frac <= 0:
        return image
    top = int(image.height * frac)
    return image.crop((0, top, image.width, image.height))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Batch OCR evaluation for KazOCR")
    parser.add_argument("--images", required=True, help="Folder of images")
    parser.add_argument("--crop-top", type=float, default=0.0,
                        help="Fraction of the top to crop (removes printed header)")
    parser.add_argument("--out", default="ocr_results.txt", help="Report output path")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images")
    args = parser.parse_args()

    image_dir = Path(args.images)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Loading model... (first run downloads weights)")
    t0 = time.time()
    engine = VLMKazOCR()
    print(f"Model ready in {time.time() - t0:.1f}s. Processing {len(images)} images.\n")

    report_lines: list[str] = []
    cers: list[float] = []
    for idx, path in enumerate(images, 1):
        image = _crop_top(Image.open(path), args.crop_top)
        t = time.time()
        result = engine.recognize(image)
        dt = time.time() - t

        gt_path = path.with_suffix(".gt.txt")
        block = [f"=== {path.name}  ({dt:.1f}s) ==="]
        block.append("[corrected]")
        block.append(result.corrected_text)
        if result.raw_text != result.corrected_text:
            block.append("[raw]")
            block.append(result.raw_text)
        if gt_path.exists():
            gt = gt_path.read_text(encoding="utf-8").strip()
            cer = _cer(gt, result.corrected_text)
            cers.append(cer)
            block.append(f"[CER vs ground-truth] {cer:.3f}")
        block.append("")
        chunk = "\n".join(block)
        report_lines.append(chunk)
        print(f"[{idx}/{len(images)}] {path.name}  {dt:.1f}s"
              + (f"  CER={cers[-1]:.3f}" if gt_path.exists() else ""))

    if cers:
        avg = sum(cers) / len(cers)
        summary = f"\nMean CER over {len(cers)} labelled images: {avg:.3f}"
        report_lines.append(summary)
        print(summary)

    Path(args.out).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
