from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .charset import KazakhLatinCharset
from .config import OCRConfig
from .dataset import ManifestOCRDataset, SyntheticKazakhDataset, collate_ocr_batch
from .model import CRNN


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(preds: list[str], refs: list[str]) -> float:
    errors = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    total = sum(max(1, len(r)) for r in refs)
    return errors / total


def build_datasets(args: argparse.Namespace, charset: KazakhLatinCharset, config: OCRConfig):
    if args.train_manifest:
        train_ds = ManifestOCRDataset(args.train_manifest, charset, config)
    else:
        train_ds = SyntheticKazakhDataset(
            size=args.steps_per_epoch * args.batch_size,
            charset=charset,
            config=config,
            corpus_path=args.corpus,
            fonts_dir=args.fonts_dir,
        )
    if args.val_manifest:
        val_ds = ManifestOCRDataset(args.val_manifest, charset, config)
    else:
        val_ds = SyntheticKazakhDataset(
            size=max(128, args.batch_size * 32),
            charset=charset,
            config=config,
            corpus_path=args.corpus,
            fonts_dir=args.fonts_dir,
        )
    return train_ds, val_ds


def greedy_decode(log_probs: torch.Tensor, charset: KazakhLatinCharset) -> list[str]:
    pred_ids = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    return [charset.decode_ctc(row) for row in pred_ids]


def evaluate(model: CRNN, loader: DataLoader, criterion: nn.Module, charset: KazakhLatinCharset, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[str] = []
    all_refs: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            input_lengths = torch.full(
                (images.shape[0],),
                fill_value=logits.shape[0],
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            all_preds.extend(greedy_decode(log_probs, charset))
            all_refs.extend(batch["texts"])
    return total_loss / max(1, len(loader)), cer(all_preds, all_refs)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train KazOCR")
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--fonts-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="runs/kazocr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    charset = KazakhLatinCharset()
    config = OCRConfig()
    train_ds, val_ds = build_datasets(args, charset, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ocr_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ocr_batch,
    )

    model = CRNN(charset.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CTCLoss(blank=charset.blank_id, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_cer = float("inf")
    metadata = {
        "charset": charset.alphabet,
        "config": asdict(config),
        "args": vars(args),
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        train_loss = 0.0
        for batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                log_probs = logits.log_softmax(dim=-1)
                input_lengths = torch.full(
                    (images.shape[0],),
                    fill_value=logits.shape[0],
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))
        val_loss, val_cer = evaluate(model, val_loader, criterion, charset, device)
        scheduler.step(val_cer)

        checkpoint = {
            "model_state": model.state_dict(),
            "charset": charset.alphabet,
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_cer": val_cer,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(checkpoint, save_dir / "best.pt")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_cer={val_cer:.4f}")


if __name__ == "__main__":
    main()
