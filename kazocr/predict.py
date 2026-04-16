from __future__ import annotations

import argparse
import sys

import torch

from .charset import KazakhLatinCharset
from .config import OCRConfig
from .model import CRNN
from .preprocess import image_to_tensor, load_image, resize_preserving_aspect


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict text from image with KazOCR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    charset = KazakhLatinCharset(alphabet=checkpoint["charset"])
    config = OCRConfig(**checkpoint["config"])
    model = CRNN(charset.vocab_size, config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = load_image(args.image)
    image = resize_preserving_aspect(image, config.image_height, config.min_width, config.max_width)
    tensor = image_to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        text = charset.decode_ctc(logits.log_softmax(dim=-1).argmax(dim=-1).squeeze(1).tolist())
    print(text)


if __name__ == "__main__":
    main()
