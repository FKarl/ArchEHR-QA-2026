"""
Train final HYDRA model for ArchEHR-QA Subtask 2 with best config:
- Dual-head (binary + 3-class)
- Synthetic:real = 1:1 (downsample synthetic)
- Real upsample x3
- Freeze first 8 encoder layers

Usage:
    python train_final.py --output_dir output/final
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
LABEL2ID = {"essential": 0, "supplementary": 1, "not-relevant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def make_binary_labels(labels_3class: torch.Tensor) -> torch.Tensor:
    return (labels_3class == LABEL2ID["essential"]).long()


def load_real_data(split_dir: str) -> list[dict]:
    split_dir = Path(split_dir)
    tree = ET.parse(split_dir / "archehr-qa.xml")
    root = tree.getroot()
    case_map = {}
    for case in root.findall("case"):
        cid = case.attrib["id"]
        pq_el = case.find("patient_question")
        patient_q = (
            "".join(pq_el.itertext()).strip() if pq_el is not None else ""
        )
        clinician_q = (case.findtext("clinician_question") or "").strip()
        sentences = {}
        for sent in case.find("note_excerpt_sentences").findall("sentence"):
            sid = sent.attrib["id"]
            text = "".join(sent.itertext()).strip()
            sentences[sid] = text
        case_map[cid] = {
            "patient_question": patient_q,
            "clinician_question": clinician_q,
            "sentences": sentences,
        }
    with open(split_dir / "archehr-qa_key.json") as f:
        keys = json.load(f)
    examples = []
    for entry in keys:
        cid = entry["case_id"]
        if cid not in case_map:
            continue
        info = case_map[cid]
        if "answers" not in entry:
            continue
        for ans in entry["answers"]:
            sid = ans["sentence_id"]
            if sid not in info["sentences"]:
                continue
            examples.append(
                {
                    "case_id": cid,
                    "sentence_id": sid,
                    "patient_question": info["patient_question"],
                    "clinician_question": info["clinician_question"],
                    "sentence": info["sentences"][sid],
                    "label": LABEL2ID[ans["relevance"]],
                }
            )
    return examples


def load_synthetic_data(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    examples = []
    for case in data:
        pq = case["patient_question"]
        cq = case["clinician_question"]
        sent_map = {s["id"]: s["text"] for s in case["sentences"]}
        for lbl in case["relevance_labels"]:
            sid = lbl["sentence_id"]
            examples.append(
                {
                    "patient_question": pq,
                    "clinician_question": cq,
                    "sentence": sent_map[sid],
                    "label": LABEL2ID[lbl["relevance"]],
                }
            )
    return examples


class EvidenceDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        sep = self.tokenizer.sep_token
        text = (
            f"{ex['patient_question']} {sep} "
            f"{ex['clinician_question']} {sep} "
            f"{ex['sentence']}"
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }


class HydraSentenceClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head_dropout = nn.Dropout(dropout)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_output = self.head_dropout(outputs.last_hidden_state[:, 0, :])
        return self.binary_head(cls_output), self.fine_head(cls_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default="../data/dev")
    parser.add_argument(
        "--synthetic_path",
        type=str,
        default="synthetic_data/good_synthetic.json",
    )
    parser.add_argument("--output_dir", type=str, default="output/final")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--binary_alpha", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    real = load_real_data(args.real_dir)
    synthetic = load_synthetic_data(args.synthetic_path)
    print(f"Loaded {len(real)} real, {len(synthetic)} synthetic")

    # Downsample synthetic to 1:1
    max_synthetic = len(real)
    rng = np.random.RandomState(args.seed)
    syn_sample = list(
        rng.choice(
            synthetic, size=min(len(synthetic), max_synthetic), replace=False
        )
    )
    # Upsample real x3
    train_examples = real * 3 + syn_sample
    print(f"Train set: {len(train_examples)} (real x3 + synthetic 1:1)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = EvidenceDataset(
        train_examples, tokenizer, max_length=args.max_length
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = HydraSentenceClassifier(
        MODEL_NAME, num_labels=3, dropout=args.dropout
    )
    # Freeze first 8 layers
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False
    for i in range(8):
        for param in model.encoder.encoder.layer[i].parameters():
            param.requires_grad = False
    model.to(device)

    # Loss functions
    label_counts = Counter(e["label"] for e in train_examples)
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [
            total_samples / (len(label_counts) * label_counts.get(i, 1))
            for i in range(3)
        ],
        dtype=torch.float,
    ).to(device)
    loss_fn_fine = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    n_essential = label_counts.get(LABEL2ID["essential"], 1)
    n_rest = total_samples - n_essential
    bin_weights = torch.tensor(
        [
            total_samples / (2 * max(n_rest, 1)),
            total_samples / (2 * max(n_essential, 1)),
        ],
        dtype=torch.float,
    ).to(device)
    loss_fn_binary = nn.CrossEntropyLoss(
        weight=bin_weights, label_smoothing=args.label_smoothing
    )

    encoder_params = [
        p
        for n, p in model.named_parameters()
        if "_head" not in n and p.requires_grad
    ]
    head_params = [
        p
        for n, p in model.named_parameters()
        if "_head" in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )
    accum = args.grad_accum_steps
    effective_steps = (len(train_loader) // accum) * args.epochs
    warmup_steps = int(0.15 * effective_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=effective_steps,
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            binary_logits, fine_logits = model(input_ids, attention_mask)
            labels_bin = make_binary_labels(labels)
            loss = (
                loss_fn_fine(fine_logits, labels)
                + args.binary_alpha * loss_fn_binary(binary_logits, labels_bin)
            ) / accum
            loss.backward()
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running_loss += loss.item() * accum
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"Model saved to {out_dir / 'final_model.pt'}")
    # Save tokenizer for inference
    tokenizer.save_pretrained(out_dir)
    print(f"Tokenizer saved to {out_dir}")

    # --- Predict on dev set and save JSON ---
    print("Predicting on dev set and saving JSON ...")
    model.eval()
    dev_examples = real  # Only real dev data
    dev_ds = EvidenceDataset(
        dev_examples, tokenizer, max_length=args.max_length
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)
    all_fine_preds = []
    all_fine_probs = []
    all_bin_preds = []
    all_bin_probs = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, fine_logits = model(input_ids, attention_mask)
            all_fine_preds.extend(fine_logits.argmax(dim=-1).cpu().numpy())
            all_fine_probs.extend(
                torch.softmax(fine_logits, dim=-1).cpu().numpy()
            )
            all_bin_preds.extend(binary_logits.argmax(dim=-1).cpu().numpy())
            all_bin_probs.extend(
                torch.softmax(binary_logits, dim=-1).cpu().numpy()
            )
    dev_results = []
    for i, ex in enumerate(dev_examples):
        pred_fine_label = ID2LABEL[int(all_fine_preds[i])]
        prob_fine = [round(float(p), 4) for p in all_fine_probs[i]]
        pred_binary_label = (
            "essential" if int(all_bin_preds[i]) == 1 else "not-essential"
        )
        prob_binary = [round(float(p), 4) for p in all_bin_probs[i]]
        pred_bin_from_fine = (
            "essential"
            if int(all_fine_preds[i]) == LABEL2ID["essential"]
            else "not-essential"
        )
        dev_results.append(
            {
                "case_id": ex["case_id"],
                "sentence_id": ex["sentence_id"],
                "sentence": ex["sentence"],
                "pred_fine": pred_fine_label,
                "prob_fine": prob_fine,
                "pred_binary": pred_binary_label,
                "prob_binary": prob_binary,
                "pred_binary_from_fine": pred_bin_from_fine,
            }
        )
    dev_json_path = out_dir / "dev_predictions_both_heads.json"
    with open(dev_json_path, "w") as f:
        json.dump(dev_results, f, indent=2)
    print(f"Dev predictions saved to {dev_json_path}")

    # --- Print metrics ---
    from sklearn.metrics import f1_score, classification_report

    # True labels: essential=1, rest=0
    y_true_bin = [
        (ex["label"] == LABEL2ID["essential"]) for ex in dev_examples
    ]
    y_pred_bin = [
        1 if r["pred_binary"] == "essential" else 0 for r in dev_results
    ]
    y_pred_bin_from_fine = [
        1 if r["pred_binary_from_fine"] == "essential" else 0
        for r in dev_results
    ]
    y_true_fine = [ex["label"] for ex in dev_examples]
    y_pred_fine = [
        (
            list(ID2LABEL.keys())[
                list(ID2LABEL.values()).index(r["pred_fine"])
            ]
            if r["pred_fine"] in LABEL2ID
            else -1
        )
        for r in dev_results
    ]
    # F1 scores
    f1_bin = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_bin_from_fine = f1_score(
        y_true_bin, y_pred_bin_from_fine, pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(
        y_true_fine,
        [LABEL2ID.get(r["pred_fine"], -1) for r in dev_results],
        average="macro",
        zero_division=0,
    )
    print(f"\nDev set metrics:")
    print(f"  Binary F1 (binary head):      {f1_bin:.4f}")
    print(f"  Binary F1 (from 3-class):     {f1_bin_from_fine:.4f}")
    print(f"  Macro F1 (3-class head):      {macro_f1:.4f}")
    print("\n3-class classification report:")
    print(
        classification_report(
            y_true_fine,
            [LABEL2ID.get(r["pred_fine"], -1) for r in dev_results],
            target_names=list(LABEL2ID.keys()),
            digits=4,
        )
    )
    print("\nBinary classification report (binary head):")
    print(
        classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=["not-essential", "essential"],
            digits=4,
        )
    )
    print("\nBinary classification report (from 3-class head):")
    print(
        classification_report(
            y_true_bin,
            y_pred_bin_from_fine,
            target_names=["not-essential", "essential"],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
