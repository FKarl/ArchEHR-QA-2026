#!/usr/bin/env python3
"""
Pipeline for ArchEHR-QA Subtask 2 with official scorer integration.
"""

import argparse
import gc
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.scoring_subtask_2 import (  # noqa: E402
    compute_evidence_scores,
    get_leaderboard,
    load_key,
)


BIO_CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"
DEBERTA_BASE = "microsoft/deberta-base"

LABEL2ID = {"essential": 0, "supplementary": 1, "not-relevant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class EvidenceRow:
    case_id: str
    sentence_id: str
    patient_question: str
    clinician_question: str
    sentence_text: str
    label: int  # essential=0, supplementary=1, not-relevant=2


class EvidenceDataset(Dataset):
    def __init__(self, rows: List[EvidenceRow], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sep = self.tokenizer.sep_token or "[SEP]"
        text = (
            f"Patient question: {row.patient_question} "
            f"{sep} Clinician question: {row.clinician_question} "
            f"{sep} Note sentence: {row.sentence_text}"
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row.label, dtype=torch.long),
        }


class HydraSentenceClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        mlp_hidden_size: int,
        dropout: float,
        local_files_only: bool,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden = int(self.encoder.config.hidden_size)
        self.pre_dropout = nn.Dropout(dropout)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 2),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 3),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.pre_dropout(pooled)
        return self.binary_head(pooled), self.fine_head(pooled)


def make_binary_labels(labels_3class: torch.Tensor) -> torch.Tensor:
    return (labels_3class == LABEL2ID["essential"]).long()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_key(v: str) -> Tuple[int, object]:
    s = str(v)
    return (0, int(s)) if s.isdigit() else (1, s)


def sanitize_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_real_rows(split_dir: Path) -> List[EvidenceRow]:
    tree = ET.parse(split_dir / "archehr-qa.xml")
    root = tree.getroot()

    case_map: Dict[str, dict] = {}
    for case in root.findall("case"):
        cid = str(case.attrib["id"])
        pq_el = case.find("patient_question")
        patient_q = (
            "".join(pq_el.itertext()).strip() if pq_el is not None else ""
        )
        clinician_q = (case.findtext("clinician_question") or "").strip()

        sentences: Dict[str, str] = {}
        sent_parent = case.find("note_excerpt_sentences")
        if sent_parent is not None:
            for sent in sent_parent.findall("sentence"):
                sid = str(sent.attrib["id"])
                text = "".join(sent.itertext()).strip()
                sentences[sid] = text

        case_map[cid] = {
            "patient_question": patient_q,
            "clinician_question": clinician_q,
            "sentences": sentences,
        }

    with open(split_dir / "archehr-qa_key.json", "r") as f:
        key_data = json.load(f)

    rows: List[EvidenceRow] = []
    for entry in key_data:
        cid = str(entry["case_id"])
        if cid not in case_map:
            continue
        answers = entry.get("answers", [])
        info = case_map[cid]
        for ans in answers:
            sid = str(ans["sentence_id"])
            rel = str(ans["relevance"])
            if sid not in info["sentences"]:
                continue
            if rel not in LABEL2ID:
                continue
            rows.append(
                EvidenceRow(
                    case_id=cid,
                    sentence_id=sid,
                    patient_question=info["patient_question"],
                    clinician_question=info["clinician_question"],
                    sentence_text=info["sentences"][sid],
                    label=LABEL2ID[rel],
                )
            )
    return rows


def load_synthetic_rows(path: Path) -> List[EvidenceRow]:
    with open(path, "r") as f:
        data = json.load(f)

    rows: List[EvidenceRow] = []
    for case_idx, case in enumerate(data, start=1):
        patient_q = str(case["patient_question"])
        clinician_q = str(case["clinician_question"])
        sent_map = {str(s["id"]): str(s["text"]) for s in case["sentences"]}
        for lbl in case["relevance_labels"]:
            sid = str(lbl["sentence_id"])
            rel = str(lbl["relevance"])
            if sid not in sent_map or rel not in LABEL2ID:
                continue
            rows.append(
                EvidenceRow(
                    case_id=f"syn_{case_idx}",
                    sentence_id=sid,
                    patient_question=patient_q,
                    clinician_question=clinician_q,
                    sentence_text=sent_map[sid],
                    label=LABEL2ID[rel],
                )
            )
    return rows


def compute_multiclass_weights(
    labels: np.ndarray,
    num_classes: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    counts = np.bincount(labels, minlength=num_classes)
    max_count = int(max(1, counts.max()))
    weights = []
    for c in range(num_classes):
        n = int(counts[c])
        if n <= 0:
            w = 1.0
        else:
            ratio = max_count / float(n)
            w = ratio if mode == "balanced" else math.sqrt(ratio)
        weights.append(float(max(1.0, w)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def maybe_freeze_layers(
    model: HydraSentenceClassifier, freeze_layers: int
) -> None:
    if freeze_layers <= 0:
        return

    encoder = model.encoder
    if hasattr(encoder, "embeddings"):
        for param in encoder.embeddings.parameters():
            param.requires_grad = False

    layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "layer"):
        layers = encoder.layer

    if layers is None:
        return

    for i in range(min(freeze_layers, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False


@torch.no_grad()
def predict_head_outputs(
    model: nn.Module,
    rows: List[EvidenceRow],
    tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = EvidenceDataset(rows, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs_bin: List[np.ndarray] = []
    preds_fine: List[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        binary_logits, fine_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        probs_bin.append(
            torch.softmax(binary_logits, dim=-1)[:, 1].detach().cpu().numpy()
        )
        preds_fine.append(
            torch.argmax(fine_logits, dim=-1).detach().cpu().numpy()
        )

    if not probs_bin:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(probs_bin, axis=0), np.concatenate(
        preds_fine, axis=0
    )


def rows_to_submission(
    rows: List[EvidenceRow], probs: np.ndarray, threshold: float
) -> List[dict]:
    per_case_all: Dict[str, set] = {}
    per_case_pred: Dict[str, set] = {}

    for row, prob in zip(rows, probs):
        per_case_all.setdefault(row.case_id, set()).add(row.sentence_id)
        if prob >= threshold:
            per_case_pred.setdefault(row.case_id, set()).add(row.sentence_id)

    out = []
    for cid in sorted(per_case_all.keys(), key=numeric_key):
        pred_ids = sorted(per_case_pred.get(cid, set()), key=numeric_key)
        out.append(
            {"case_id": str(cid), "prediction": [str(sid) for sid in pred_ids]}
        )
    return out


def rows_to_submission_from_fine(
    rows: List[EvidenceRow], fine_preds: np.ndarray
) -> List[dict]:
    per_case_all: Dict[str, set] = {}
    per_case_pred: Dict[str, set] = {}
    essential_id = LABEL2ID["essential"]

    for row, pred in zip(rows, fine_preds):
        per_case_all.setdefault(row.case_id, set()).add(row.sentence_id)
        if int(pred) == essential_id:
            per_case_pred.setdefault(row.case_id, set()).add(row.sentence_id)

    out = []
    for cid in sorted(per_case_all.keys(), key=numeric_key):
        pred_ids = sorted(per_case_pred.get(cid, set()), key=numeric_key)
        out.append(
            {"case_id": str(cid), "prediction": [str(sid) for sid in pred_ids]}
        )
    return out


def score_with_key_map(
    submission: List[dict], key_map: Dict[str, dict]
) -> dict:
    scores = compute_evidence_scores(submission=submission, key_map=key_map)
    return get_leaderboard(scores)


def tune_threshold_official(
    rows: List[EvidenceRow], probs: np.ndarray, key_map: Dict[str, dict]
) -> Tuple[float, dict]:
    best = {
        "threshold": 0.5,
        "strict_micro_f1": -1.0,
        "strict_micro_precision": 0.0,
        "strict_micro_recall": 0.0,
        "lenient_micro_f1": 0.0,
    }
    for thr in np.arange(0.05, 0.96, 0.01):
        submission = rows_to_submission(
            rows=rows, probs=probs, threshold=float(thr)
        )
        leaderboard = score_with_key_map(
            submission=submission, key_map=key_map
        )
        metric = float(leaderboard["strict_micro_f1"])
        if metric > best["strict_micro_f1"]:
            best = {
                "threshold": float(thr),
                "strict_micro_f1": metric,
                "strict_micro_precision": float(
                    leaderboard["strict_micro_precision"]
                ),
                "strict_micro_recall": float(
                    leaderboard["strict_micro_recall"]
                ),
                "lenient_micro_f1": float(leaderboard["lenient_micro_f1"]),
            }
    return best["threshold"], best


def make_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def build_train_rows(
    real_train_rows: List[EvidenceRow],
    synthetic_rows: List[EvidenceRow],
    synthetic_ratio: float,
    real_upsample: int,
    seed: int,
) -> List[EvidenceRow]:
    rng = random.Random(seed)

    train_rows = list(real_train_rows) * max(1, int(real_upsample))

    if synthetic_rows and synthetic_ratio > 0:
        max_synth = int(len(real_train_rows) * synthetic_ratio)
        if max_synth > 0:
            if max_synth < len(synthetic_rows):
                syn = rng.sample(synthetic_rows, max_synth)
            else:
                syn = list(synthetic_rows)
            train_rows.extend(syn)

    rng.shuffle(train_rows)
    return train_rows


def train_for_cv_fold(
    model_name: str,
    train_rows: List[EvidenceRow],
    val_rows: List[EvidenceRow],
    val_key_map: Dict[str, dict],
    run_cfg: dict,
    args,
    base_seed: int,
    fold_idx: int,
    device: torch.device,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    set_seed(base_seed + fold_idx)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )

    train_ds = EvidenceDataset(
        train_rows, tokenizer=tokenizer, max_length=run_cfg["max_length"]
    )
    val_ds = EvidenceDataset(
        val_rows, tokenizer=tokenizer, max_length=run_cfg["max_length"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=run_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=run_cfg["batch_size"], shuffle=False
    )

    model = HydraSentenceClassifier(
        model_name=model_name,
        mlp_hidden_size=run_cfg["mlp_hidden_size"],
        dropout=run_cfg["dropout"],
        local_files_only=args.local_files_only,
    ).to(device)

    maybe_freeze_layers(model, args.freeze_layers)

    optimizer = make_optimizer(
        model=model,
        learning_rate=run_cfg["learning_rate"],
        weight_decay=run_cfg["weight_decay"],
    )

    train_labels_3 = np.array([r.label for r in train_rows], dtype=int)
    train_labels_bin = (train_labels_3 == LABEL2ID["essential"]).astype(int)

    fine_weights = compute_multiclass_weights(
        labels=train_labels_3,
        num_classes=3,
        mode=run_cfg["class_weight_mode"],
        device=device,
    )
    bin_weights = compute_multiclass_weights(
        labels=train_labels_bin,
        num_classes=2,
        mode=run_cfg["class_weight_mode"],
        device=device,
    )

    loss_fn_fine = nn.CrossEntropyLoss(
        weight=fine_weights, label_smoothing=args.label_smoothing
    )
    loss_fn_binary = nn.CrossEntropyLoss(
        weight=bin_weights, label_smoothing=args.label_smoothing
    )

    accum = max(1, int(args.grad_accum_steps))
    update_steps = max(
        1, math.ceil(len(train_loader) / accum) * run_cfg["epochs"]
    )
    warmup_steps = int(0.15 * update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=update_steps,
    )

    best = {
        "epoch": -1,
        "threshold": 0.5,
        "strict_micro_f1": -1.0,
        "strict_micro_precision": 0.0,
        "strict_micro_recall": 0.0,
        "lenient_micro_f1": 0.0,
    }
    best_state = None

    for epoch in range(1, run_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_3 = batch["labels"].to(device)

            binary_logits, fine_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            labels_bin = make_binary_labels(labels_3)

            if args.binary_only:
                loss = loss_fn_binary(binary_logits, labels_bin) / accum
            else:
                loss = (
                    loss_fn_fine(fine_logits, labels_3)
                    + args.binary_alpha
                    * loss_fn_binary(binary_logits, labels_bin)
                ) / accum
            loss.backward()

            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), run_cfg["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += float(loss.detach().cpu().item()) * accum
            n_batches += 1

        val_probs, _ = predict_head_outputs(
            model=model,
            rows=val_rows,
            tokenizer=tokenizer,
            batch_size=run_cfg["batch_size"],
            max_length=run_cfg["max_length"],
            device=device,
        )
        threshold, epoch_metrics = tune_threshold_official(
            rows=val_rows,
            probs=val_probs,
            key_map=val_key_map,
        )

        avg_loss = total_loss / max(1, n_batches)
        print(
            f"    epoch={epoch} train_loss={avg_loss:.4f} "
            f"strict_micro_f1={epoch_metrics['strict_micro_f1']:.2f} "
            f"strict_micro_p={epoch_metrics['strict_micro_precision']:.2f} "
            f"strict_micro_r={epoch_metrics['strict_micro_recall']:.2f} "
            f"lenient_micro_f1={epoch_metrics['lenient_micro_f1']:.2f} thr={threshold:.2f}"
        )

        if epoch_metrics["strict_micro_f1"] > best["strict_micro_f1"]:
            best = {
                "epoch": epoch,
                "threshold": threshold,
                "strict_micro_f1": epoch_metrics["strict_micro_f1"],
                "strict_micro_precision": epoch_metrics[
                    "strict_micro_precision"
                ],
                "strict_micro_recall": epoch_metrics["strict_micro_recall"],
                "lenient_micro_f1": epoch_metrics["lenient_micro_f1"],
            }
            best_state = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No best model state found during CV fold training."
        )

    model.load_state_dict(best_state)
    final_probs, final_fine_preds = predict_head_outputs(
        model=model,
        rows=val_rows,
        tokenizer=tokenizer,
        batch_size=run_cfg["batch_size"],
        max_length=run_cfg["max_length"],
        device=device,
    )

    del model, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    clear_device_cache(device)
    return best, final_probs, final_fine_preds


def default_sweep_configs() -> List[dict]:
    # Mirrors Subtask 4 sweep shape; "negative_ratio" is adapted to "synthetic_ratio" for Subtask 2.
    return [
        {
            "epochs": 1,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "max_length": 256,
            "synthetic_ratio": 2.0,
            "dropout": 0.10,
            "weight_decay": 0.01,
            "class_weight_mode": "sqrt_balanced",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
        {
            "epochs": 1,
            "learning_rate": 3e-5,
            "batch_size": 16,
            "max_length": 256,
            "synthetic_ratio": 3.0,
            "dropout": 0.10,
            "weight_decay": 0.01,
            "class_weight_mode": "sqrt_balanced",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
        {
            "epochs": 2,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "max_length": 256,
            "synthetic_ratio": 2.0,
            "dropout": 0.10,
            "weight_decay": 0.01,
            "class_weight_mode": "sqrt_balanced",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
        {
            "epochs": 1,
            "learning_rate": 2e-5,
            "batch_size": 8,
            "max_length": 320,
            "synthetic_ratio": 2.0,
            "dropout": 0.20,
            "weight_decay": 0.01,
            "class_weight_mode": "balanced",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
        {
            "epochs": 1,
            "learning_rate": 5e-5,
            "batch_size": 16,
            "max_length": 192,
            "synthetic_ratio": 4.0,
            "dropout": 0.20,
            "weight_decay": 0.0,
            "class_weight_mode": "none",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
        {
            "epochs": 2,
            "learning_rate": 1e-5,
            "batch_size": 8,
            "max_length": 320,
            "synthetic_ratio": 3.0,
            "dropout": 0.20,
            "weight_decay": 0.01,
            "class_weight_mode": "balanced",
            "max_grad_norm": 1.0,
            "mlp_hidden_size": 1024,
        },
    ]


def config_tag(cfg: dict) -> str:
    return (
        f"e{cfg['epochs']}_lr{cfg['learning_rate']}_b{cfg['batch_size']}_"
        f"l{cfg['max_length']}_sr{cfg['synthetic_ratio']}_do{cfg['dropout']}_"
        f"cw{cfg['class_weight_mode']}"
    )


def run_single_cv(
    model_name: str,
    run_cfg: dict,
    args,
    run_dir: Path,
    real_rows_by_case: Dict[str, List[EvidenceRow]],
    case_ids: List[str],
    dev_key_path: Path,
    synthetic_rows: List[EvidenceRow],
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    case_ids_sorted = sorted(case_ids, key=numeric_key)
    groups = np.array(case_ids_sorted)
    indices = np.arange(len(case_ids_sorted))
    dummy_y = np.arange(len(case_ids_sorted))
    gkf = GroupKFold(n_splits=args.k)

    oof_rows: List[EvidenceRow] = []
    oof_probs: List[float] = []
    oof_fine_preds: List[int] = []
    fold_thresholds: List[float] = []
    fold_summaries: List[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(indices, dummy_y, groups), start=1
    ):
        train_case_ids = [case_ids_sorted[i] for i in tr_idx]
        val_case_ids = [case_ids_sorted[i] for i in va_idx]

        real_train_rows = []
        for cid in train_case_ids:
            real_train_rows.extend(real_rows_by_case[cid])

        val_rows = []
        for cid in val_case_ids:
            val_rows.extend(real_rows_by_case[cid])

        train_rows = build_train_rows(
            real_train_rows=real_train_rows,
            synthetic_rows=[] if args.no_synthetic else synthetic_rows,
            synthetic_ratio=run_cfg["synthetic_ratio"],
            real_upsample=args.real_upsample,
            seed=args.seed + fold_idx,
        )

        val_key_map = load_key(
            str(dev_key_path), case_ids_to_score=set(val_case_ids)
        )

        print(
            f"  fold={fold_idx}/{args.k} device={device} "
            f"train_rows={len(train_rows)} (real={len(real_train_rows)} upsample={args.real_upsample} syn_ratio={run_cfg['synthetic_ratio']}) "
            f"val_rows={len(val_rows)} val_cases={len(val_case_ids)}"
        )

        best_fold, val_probs, val_fine_preds = train_for_cv_fold(
            model_name=model_name,
            train_rows=train_rows,
            val_rows=val_rows,
            val_key_map=val_key_map,
            run_cfg=run_cfg,
            args=args,
            base_seed=args.seed + (1000 * args._run_counter),
            fold_idx=fold_idx,
            device=device,
        )

        fold_submission_binary = rows_to_submission(
            val_rows, val_probs, best_fold["threshold"]
        )
        fold_eval_binary = score_with_key_map(
            fold_submission_binary, val_key_map
        )

        fold_submission_multiclass = rows_to_submission_from_fine(
            val_rows, val_fine_preds
        )
        fold_eval_multiclass = score_with_key_map(
            fold_submission_multiclass, val_key_map
        )

        fold_summary = {
            "fold": fold_idx,
            "threshold": best_fold["threshold"],
            "tuning_strict_micro_f1": best_fold["strict_micro_f1"],
            "tuning_strict_micro_precision": best_fold[
                "strict_micro_precision"
            ],
            "tuning_strict_micro_recall": best_fold["strict_micro_recall"],
            "official_leaderboard": fold_eval_binary,
            "official_leaderboard_binary_head": fold_eval_binary,
            "official_leaderboard_multiclass_as_binary": fold_eval_multiclass,
        }
        fold_summaries.append(fold_summary)
        fold_thresholds.append(float(best_fold["threshold"]))
        oof_rows.extend(val_rows)
        oof_probs.extend(val_probs.tolist())
        oof_fine_preds.extend(val_fine_preds.tolist())

        print(
            f"    fold_binary_strict_micro_f1={fold_eval_binary['strict_micro_f1']:.2f} "
            f"fold_multiclass_strict_micro_f1={fold_eval_multiclass['strict_micro_f1']:.2f} "
            f"thr={best_fold['threshold']:.2f}"
        )

    avg_thr = float(np.mean(fold_thresholds)) if fold_thresholds else 0.5
    oof_submission_binary = rows_to_submission(
        oof_rows, np.array(oof_probs), avg_thr
    )
    oof_submission_multiclass = rows_to_submission_from_fine(
        oof_rows, np.array(oof_fine_preds, dtype=int)
    )
    full_key_map = load_key(str(dev_key_path), case_ids_to_score=None)
    oof_eval_binary = score_with_key_map(oof_submission_binary, full_key_map)
    oof_eval_multiclass = score_with_key_map(
        oof_submission_multiclass, full_key_map
    )

    summary = {
        "model_name": model_name,
        "k": args.k,
        "run_config": run_cfg,
        "avg_threshold": avg_thr,
        "no_synthetic": bool(args.no_synthetic),
        "synthetic_path": None if args.no_synthetic else args.synthetic_path,
        "real_upsample": int(args.real_upsample),
        "folds": fold_summaries,
        "oof_leaderboard": oof_eval_binary,
        "oof_leaderboard_binary_head": oof_eval_binary,
        "oof_leaderboard_multiclass_as_binary": oof_eval_multiclass,
    }
    save_json(summary, run_dir / "cv_summary.json")
    save_json(oof_submission_binary, run_dir / "dev_oof_submission.json")
    save_json(oof_eval_binary, run_dir / "dev_oof_scores.json")
    save_json(
        oof_submission_multiclass,
        run_dir / "dev_oof_submission_multiclass_as_binary.json",
    )
    save_json(
        oof_eval_multiclass,
        run_dir / "dev_oof_scores_multiclass_as_binary.json",
    )
    return summary


def parse_models(raw: str) -> List[str]:
    resolved = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        key = item.lower()
        if key in {"bioclinicalbert", "bio_clinicalbert", "bio-clinicalbert"}:
            resolved.append(BIO_CLINICALBERT)
        elif key in {"deberta", "deberta-base", "deberta_base"}:
            resolved.append(DEBERTA_BASE)
        else:
            resolved.append(item)
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subtask 2 paper pipeline: k-fold sweep with official scorer"
    )
    parser.add_argument(
        "--models", type=str, default="bioclinicalbert,deberta"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_runs_per_model", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data/dev")
    parser.add_argument(
        "--synthetic_path",
        type=str,
        default="subtask2_evidence/synthetic_data/good_synthetic.json",
    )
    parser.add_argument(
        "--no_synthetic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--real_upsample", type=int, default=3)
    parser.add_argument(
        "--binary_only", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--binary_alpha", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--freeze_layers", type=int, default=8)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default="subtask2_evidence/output/paper_bert_kfold",
    )
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args._run_counter = 0

    models = parse_models(args.models)
    if not models:
        raise ValueError("No models provided.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    real_rows = load_real_rows(Path(args.data_dir))
    if not real_rows:
        raise RuntimeError(f"No real dev rows loaded from {args.data_dir}")

    case_ids = sorted({r.case_id for r in real_rows}, key=numeric_key)
    real_rows_by_case: Dict[str, List[EvidenceRow]] = {}
    for row in real_rows:
        real_rows_by_case.setdefault(row.case_id, []).append(row)

    synthetic_rows: List[EvidenceRow] = []
    if not args.no_synthetic:
        synthetic_rows = load_synthetic_rows(Path(args.synthetic_path))

    print(
        f"Loaded real rows={len(real_rows)} cases={len(case_ids)} "
        f"synthetic_rows={len(synthetic_rows)} no_synthetic={args.no_synthetic}"
    )

    all_cfgs = default_sweep_configs()
    max_runs = max(1, min(int(args.max_runs_per_model), 6))
    selected_cfgs = all_cfgs[:max_runs]

    print(
        f"Running sweep: models={models} runs_per_model={len(selected_cfgs)} k={args.k}"
    )

    all_runs: List[dict] = []
    dev_key_path = Path(args.data_dir) / "archehr-qa_key.json"

    for model_name in models:
        model_slug = sanitize_name(model_name)
        for i, cfg in enumerate(selected_cfgs, start=1):
            args._run_counter += 1
            run_slug = f"run_{i:02d}_{config_tag(cfg)}"
            run_dir = output_root / "sweep" / model_slug / run_slug
            summary_path = run_dir / "cv_summary.json"
            print(
                f"\n[run] model={model_name} config={i}/{len(selected_cfgs)} -> {run_slug}"
            )

            if args.skip_existing and summary_path.exists():
                run_summary = json.loads(summary_path.read_text())
                oof_binary = run_summary.get(
                    "oof_leaderboard_binary_head",
                    run_summary["oof_leaderboard"],
                )
                oof_multi = run_summary.get(
                    "oof_leaderboard_multiclass_as_binary", oof_binary
                )
                print(
                    "  reused existing summary, "
                    f"binary_strict_micro_f1={oof_binary['strict_micro_f1']:.2f} "
                    f"multiclass_strict_micro_f1={oof_multi['strict_micro_f1']:.2f}"
                )
            else:
                run_summary = run_single_cv(
                    model_name=model_name,
                    run_cfg=cfg,
                    args=args,
                    run_dir=run_dir,
                    real_rows_by_case=real_rows_by_case,
                    case_ids=case_ids,
                    dev_key_path=dev_key_path,
                    synthetic_rows=synthetic_rows,
                )
                oof_binary = run_summary.get(
                    "oof_leaderboard_binary_head",
                    run_summary["oof_leaderboard"],
                )
                oof_multi = run_summary.get(
                    "oof_leaderboard_multiclass_as_binary", oof_binary
                )
                print(
                    "  oof_binary_strict_micro_f1="
                    f"{oof_binary['strict_micro_f1']:.2f} "
                    f"oof_multiclass_strict_micro_f1={oof_multi['strict_micro_f1']:.2f}"
                )

            oof_binary = run_summary.get(
                "oof_leaderboard_binary_head", run_summary["oof_leaderboard"]
            )
            oof_multi = run_summary.get(
                "oof_leaderboard_multiclass_as_binary", oof_binary
            )
            all_runs.append(
                {
                    "model_name": model_name,
                    "run_dir": str(run_dir),
                    "run_config": run_summary["run_config"],
                    "avg_threshold": run_summary["avg_threshold"],
                    "oof_leaderboard": oof_binary,
                    "oof_leaderboard_binary_head": oof_binary,
                    "oof_leaderboard_multiclass_as_binary": oof_multi,
                    "no_synthetic": run_summary.get(
                        "no_synthetic", bool(args.no_synthetic)
                    ),
                    "real_upsample": run_summary.get(
                        "real_upsample", int(args.real_upsample)
                    ),
                }
            )

    best = max(
        all_runs,
        key=lambda x: x["oof_leaderboard_binary_head"]["strict_micro_f1"],
    )
    sweep_summary = {
        "models": models,
        "k": args.k,
        "max_runs_per_model": len(selected_cfgs),
        "runs": all_runs,
        "best": best,
    }
    sweep_summary_path = output_root / "sweep_summary.json"
    save_json(sweep_summary, sweep_summary_path)

    print(
        f"\nBest sweep run: model={best['model_name']} "
        f"binary_strict_micro_f1={best['oof_leaderboard_binary_head']['strict_micro_f1']:.2f} "
        f"run_dir={best['run_dir']}"
    )
    print(f"Saved sweep summary: {sweep_summary_path}")


if __name__ == "__main__":
    main()
