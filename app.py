from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from kazocr.handwritten_engine import HandwrittenKazOCR


class KazOCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KazOCR Handwritten")
        self.root.geometry("1100x760")
        self.root.configure(bg="#f4efe7")

        self.engine: HandwrittenKazOCR | None = None
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="Model is not loaded yet.")
        self.raw_var = tk.StringVar()
        self.corrected_var = tk.StringVar()
        self.changes_var = tk.StringVar()

        self._build_styles()
        self._build_layout()
        self._load_engine_async()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#f4efe7")
        style.configure("Card.TFrame", background="#fffaf2", relief="flat")
        style.configure("Title.TLabel", background="#f4efe7", foreground="#2f241f", font=("Segoe UI Semibold", 22))
        style.configure("Body.TLabel", background="#fffaf2", foreground="#3f342c", font=("Segoe UI", 11))
        style.configure("Action.TButton", font=("Segoe UI Semibold", 11), padding=10)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="Main.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Main.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="KazOCR Handwritten", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Rukopis latin text -> raw OCR -> kazakh-oriented correction",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        actions = ttk.Frame(outer, style="Main.TFrame")
        actions.pack(fill="x", pady=(18, 12))
        ttk.Button(actions, text="Open Image", command=self.open_image, style="Action.TButton").pack(side="left")
        ttk.Button(actions, text="Recognize", command=self.recognize_current, style="Action.TButton").pack(side="left", padx=10)
        ttk.Button(actions, text="Clear", command=self.clear_state, style="Action.TButton").pack(side="left")

        status = ttk.Label(outer, textvariable=self.status_var, style="Body.TLabel")
        status.pack(fill="x", pady=(0, 12))

        body = ttk.Frame(outer, style="Main.TFrame")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Card.TFrame", padding=14)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(body, style="Card.TFrame", padding=14)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Preview", style="Body.TLabel").pack(anchor="w")
        self.preview = tk.Label(left, bg="#f7f1e7", bd=0)
        self.preview.pack(fill="both", expand=True, pady=(10, 0))

        ttk.Label(right, text="Raw OCR", style="Body.TLabel").pack(anchor="w")
        self.raw_text = tk.Text(right, height=8, wrap="word", font=("Consolas", 12), bg="#f9f5ee", fg="#1f1a16")
        self.raw_text.pack(fill="x", pady=(8, 14))

        ttk.Label(right, text="Corrected Kazakh Latin", style="Body.TLabel").pack(anchor="w")
        self.corrected_text = tk.Text(right, height=8, wrap="word", font=("Consolas", 12), bg="#eef7ef", fg="#16210f")
        self.corrected_text.pack(fill="x", pady=(8, 14))

        ttk.Label(right, text="Word Fixes", style="Body.TLabel").pack(anchor="w")
        self.changes_text = tk.Text(right, height=10, wrap="word", font=("Consolas", 11), bg="#fff4e6", fg="#432f19")
        self.changes_text.pack(fill="both", expand=True, pady=(8, 0))

    def _load_engine_async(self) -> None:
        self.status_var.set("Loading handwritten OCR model. First run can take a while because weights are downloaded.")

        def worker() -> None:
            try:
                engine = HandwrittenKazOCR()
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"Model load failed: {exc}"))
                return
            self.root.after(0, lambda: self._set_engine(engine))

        threading.Thread(target=worker, daemon=True).start()

    def _set_engine(self, engine: HandwrittenKazOCR) -> None:
        self.engine = engine
        self.status_var.set("Model is ready. Open an image with handwritten Kazakh Latin text.")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        image = Image.open(path)
        self.current_image = image
        self._show_preview(image)
        self.status_var.set(f"Loaded: {Path(path).name}")

    def _show_preview(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((520, 520))
        self.current_photo = ImageTk.PhotoImage(preview)
        self.preview.configure(image=self.current_photo)

    def recognize_current(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("KazOCR", "Open an image first.")
            return
        if self.engine is None:
            messagebox.showinfo("KazOCR", "Model is still loading.")
            return
        self.status_var.set("Recognizing text...")

        def worker() -> None:
            try:
                result = self.engine.recognize(self.current_image.copy())
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"Recognition failed: {exc}"))
                return
            self.root.after(0, lambda: self._apply_result(result.raw_text, result.corrected_text, result.changed_tokens))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_result(self, raw_text: str, corrected_text: str, changed_tokens: list[tuple[str, str]]) -> None:
        self._fill_text(self.raw_text, raw_text)
        self._fill_text(self.corrected_text, corrected_text)
        if changed_tokens:
            lines = [f"{src} -> {dst}" for src, dst in changed_tokens]
            self._fill_text(self.changes_text, "\n".join(lines))
        else:
            self._fill_text(self.changes_text, "No lexicon-based corrections were applied.")
        self.status_var.set("Done. Compare raw OCR and corrected text.")

    def _fill_text(self, widget: tk.Text, value: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def clear_state(self) -> None:
        self.current_image = None
        self.current_photo = None
        self.preview.configure(image="")
        self._fill_text(self.raw_text, "")
        self._fill_text(self.corrected_text, "")
        self._fill_text(self.changes_text, "")
        self.status_var.set("State cleared.")


def main() -> None:
    root = tk.Tk()
    app = KazOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
