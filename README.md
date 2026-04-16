# KazOCR

Kazakh Latin OCR toolkit with two paths:

- handwritten recognition app based on a pretrained handwriting model
- trainable CRNN baseline for printed text and future fine-tuning

## Best path if you have no dataset

Run the desktop app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

On first start the app downloads a pretrained handwriting OCR model, then you can:

- open an image
- see raw OCR text
- see corrected Kazakh Latin text
- inspect which words were auto-corrected

## CLI baseline

There is also a trainable CRNN + CTC baseline in case you later collect your own data:

```powershell
python -m kazocr.train --epochs 10 --steps-per-epoch 300 --save-dir runs\kazocr
python -m kazocr.predict --checkpoint runs\kazocr\best.pt --image path\to\image.png
```

## How correction works

The handwritten model reads Latin handwriting roughly like English OCR would. Then the postprocessor:

- fixes common OCR confusions such as `0 -> o` and `1 -> i`
- compares words against a built-in Kazakh Latin lexicon
- replaces near-matches with more plausible Kazakh forms

## Reality check

- Without your own dataset, no system can promise true 100 percent on all handwriting.
- This app is designed to be strong for neat handwritten Latin text and noticeably better than raw OCR because of the Kazakh word correction layer.
- If you later bring 50 to 300 real examples from your handwriting style, quality can jump a lot with fine-tuning.
