# KazOCR — Handwritten Kazakh-Latin OCR

Local, GPU-accelerated recognition of **handwritten Kazakh written in the Latin
alphabet** (connected cursive). Built around a local vision-language model
(Qwen2.5-VL) plus a Kazakh lexicon corrector. Runs fully offline after a one-time
model download.

Connected cursive with diacritics (`á ä ç é ğ ı ñ ó ö ş ú ü ý`) is one of the
hardest OCR problems — classic engines (Tesseract / PaddleOCR / EasyOCR) fail on
it. A local VLM reads it far more reliably, and we prime it with the exact
Kazakh-Latin letter inventory so the accents land correctly.

---

## 1. What you get

- **Desktop app** (`app.py`) — open an image, get the transcription.
- **Batch tool** (`evaluate.py`) — run a whole folder, write a report, optionally
  measure error rate.
- Typical quality on neat cursive: **~85 % word accuracy**, diacritics mostly
  correct. Speed: **~4–8 s per full page** (after a ~15–35 s one-time model load).

---

## 2. How it works (pipeline)

The engine lives in `kazocr/vlm_engine.py` (`VLMKazOCR`). Three stages:

1. **Line segmentation** (`_segment_lines`)
   A full page fed to the model in one shot makes it lose track and loop after a
   couple of lines. So the page is first split into individual text lines: we take
   the horizontal ink-density profile, estimate the line spacing from its
   autocorrelation, find each line's peak, and cut at the ink valley between lines.

2. **Recognition** (Qwen2.5-VL, `_generate_batch`)
   Each line crop is sent to the local vision-language model with a prompt that
   pins the output to the Kazakh-Latin alphabet ("Latin only, never Cyrillic,
   keep diacritics"). All line crops of a page are processed in **batched** model
   calls for speed.

3. **Correction** (`kazocr/postprocess.py`)
   A conservative pass: fixes obvious garbage (e.g. `0lardyñ → olardyñ`,
   `qyzmet1 → qyzmeti`) and snaps near-exact matches to a built-in Kazakh lexicon,
   **without stripping the diacritics the model already produced**. Add your own
   words to `user_lexicon.txt`.

> Note on the alphabet: this project uses the `ç ş ğ ä ö ü ñ ı ú ý` romanization
> (matches `kazocr/charset.py`), not the 2021 acute-accent reform (Á Ǵ Ń Ó Ú).

---

## 3. Requirements

- **NVIDIA GPU with CUDA**, ideally **≥16 GB VRAM** (developed on an RTX 5080).
  - 12 GB GPU: use the 3B model (see §6).
  - CPU-only works but is very slow — not recommended.
- **Python 3.10**.
- ~16 GB free disk for the model weights, plus ~6 GB for the Python packages.

---

## 4. Install

```powershell
# 1) Torch built for your CUDA version. cu128 is for RTX 40/50-series (Blackwell/Ada).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Everything else (transformers, accelerate, torchao, qwen-vl-utils, pillow, numpy)
pip install -r requirements.txt
```

Verify the install picked up CUDA:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect something like: 2.11.0+cu128 True
```

> **PyCharm users:** make sure the project interpreter is the Python where you ran
> the install above (Settings → Project → Python Interpreter). A common failure is
> PyCharm running a different/old venv that has a **CPU-only** torch and is missing
> `accelerate` — see Troubleshooting.

---

## 5. Running

### Desktop app

```powershell
python app.py
```

1. Wait until the **Recognize** button becomes enabled (~15–35 s while the model
   loads into the GPU; the status bar shows progress).
2. **Open Image** → pick a photo/scan of handwriting.
3. **Recognize** → you get three panes: raw model output, corrected Kazakh-Latin,
   and the list of word fixes.

The app is meant for images that contain **handwriting only**. (The sample dataset
images also have a printed reference at the top — for those, use the batch tool
with `--crop-top`, below.)

### Batch tool

```powershell
# Transcribe every image in a folder, write ocr_results.txt
python evaluate.py --images "D:\Projects\PycharmProjects\Kazakh Latin Handwriting Dataset"

# Drop a printed header band (top 26%) so only the handwriting is read:
python evaluate.py --images <folder> --crop-top 0.26

# Limit how many images to process:
python evaluate.py --images <folder> --limit 5
```

If a ground-truth file `0001.gt.txt` sits next to `0001.jpg`, the report also
prints the character error rate (CER) for that image.

---

## 6. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `KAZOCR_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Which model to use. |
| `KAZOCR_VLM_QUANT` | `auto` | `auto` \| `torchao` (int8, torch-native) \| `bnb` \| `none`. |
| `KAZOCR_LINE_BATCH` | `8` | Line crops per model call. Lower it if you hit VRAM limits. |

Defaults target a 16 GB GPU: the 7B model loads in **int8 weight-only via torchao**
(~9 GB VRAM, torch-native CUDA kernels, near-lossless quality).

Lighter / smaller GPU — run the 3B model at full precision:

```powershell
$env:KAZOCR_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"   # auto-selects no quant
python app.py
```

(3B is faster and fits ~12 GB, but follows instructions less reliably than 7B.)

---

## 7. Running on another PC

You need three things: **the code, the Python packages, and the model weights.**

1. Copy the `ocr2` project folder.
2. Install the stack (§4) on the new machine — pick the torch build that matches
   that machine's CUDA / GPU.
3. **Model weights (~16 GB)** download automatically on the **first run** from
   HuggingFace into:
   ```
   C:\Users\<user>\.cache\huggingface\hub
   ```
   Internet is needed only once; afterwards it runs offline.

**Offline / no re-download:** copy the cached model folder from this PC to the new
one (same path):

```
C:\Users\<user>\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-7B-Instruct
```

---

## 8. Troubleshooting

- **GUI stuck on "model still loading" / `ValueError: requires accelerate`**
  PyCharm is running the wrong interpreter (a CPU torch / missing `accelerate`).
  Point the project at the Python where you installed the stack, and confirm
  `torch.cuda.is_available()` is `True`.

- **`CUDA out of memory` on load**
  Use the 3B model (§6), or set `KAZOCR_VLM_QUANT=torchao` explicitly, or lower
  `KAZOCR_LINE_BATCH`.

- **torch stops importing (`WinError 127`) after installing `bitsandbytes`**
  `bitsandbytes` clashes with torch's CUDA DLLs on Windows. Uninstall it and
  force-reinstall torch:
  ```powershell
  pip uninstall -y bitsandbytes
  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
  We use **torchao int8** instead, which has no such issue.

- **`transformers` 5.x errors during quantized load**
  Pin a 4.x release: `pip install "transformers>=4.56,<5.0"`.

- **`UnicodeEncodeError` printing Kazakh letters in a terminal**
  The Windows console is cp1251. Set `PYTHONIOENCODING=utf-8` (the scripts already
  reconfigure stdout where it matters).

---

## 9. Project layout

| Path | Role |
|---|---|
| `app.py` | Tkinter desktop app. |
| `evaluate.py` | Batch OCR + CER report. |
| `kazocr/vlm_engine.py` | **Main engine**: line segmentation + Qwen2.5-VL + corrector. |
| `kazocr/postprocess.py` | Conservative Kazakh lexicon corrector. |
| `kazocr/charset.py` | The Kazakh-Latin character inventory. |
| `kazocr/resources/kazakh_lexicon.txt` | Built-in word list. |
| `user_lexicon.txt` | Your extra words (auto-loaded if present). |
| `requirements.txt` | Python dependencies. |
| `kazocr/handwritten_engine.py` | **Legacy** PaddleOCR engine (not used). |
| `kazocr/model.py`, `train.py`, `predict.py`, `dataset.py` | **Legacy** CRNN baseline (not used). |

---

## 10. Accuracy & limits

- Best on neat, reasonably separated cursive. Very messy or overlapping lines hurt
  the line segmentation.
- Common error types: `g/q` confusion, occasional wrong suffix on rare words.
- Improve quality by adding domain words to `user_lexicon.txt`.
- No system is 100 % on free handwriting; compare the **raw** and **corrected**
  panes when in doubt.
