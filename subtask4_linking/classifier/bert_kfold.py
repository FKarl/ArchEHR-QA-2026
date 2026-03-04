#!/usr/bin/env python3
"""
Paper pipeline for Subtask 4 with a transformer + MLP pair classifier.

"""

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import (
    build_gold_links_from_key,
    load_cases_from_xml,
    load_subtask1_predictions,
    save_json,
    score_links_official,
    split_answer_sentences,
)


BIO_CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"
DEBERTA_BASE = "microsoft/deberta-base"


@dataclass
class PairRow:
    case_id: str
    answer_id: str
    evidence_id: str
    patient_question: str
    clinician_question: str
    answer_text: str
    evidence_text: str
    label: int  # 0/1 for training or eval; -1 for inference


class LinkDataset(Dataset):
    def __init__(self, rows: List[PairRow], tokenizer, max_length: int):
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
            f"{sep} Answer sentence: {row.answer_text} "
            f"{sep} Evidence sentence: {row.evidence_text}"
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if row.label >= 0:
            out["labels"] = torch.tensor(row.label, dtype=torch.long)
        return out


class TransformerMLPLinker(nn.Module):
    """
    Transformer encoder + single hidden layer MLP head (hidden_size=1024 by default).
    """

    def __init__(
        self,
        model_name: str,
        mlp_hidden_size: int,
        dropout: float,
        local_files_only: bool,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.pre_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 2),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.pre_dropout(pooled))
        return logits


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


def build_gold_train_rows(
    cases: Dict[str, object],
    answer_texts: Dict[str, Dict[int, str]],
    gold_links: Dict[str, Dict[int, List[str]]],
    negative_ratio: float,
    seed: int,
) -> List[PairRow]:
    rng = random.Random(seed)
    rows: List[PairRow] = []

    for cid, case in cases.items():
        if cid not in answer_texts:
            continue
        all_eids = sorted(case.note_sentences.keys(), key=numeric_key)
        if not all_eids:
            continue

        for aid_int, answer_text in sorted(
            answer_texts[cid].items(), key=lambda x: x[0]
        ):
            pos_set = set(gold_links[cid].get(aid_int, []))
            positives = [eid for eid in all_eids if eid in pos_set]
            negatives = [eid for eid in all_eids if eid not in pos_set]

            if negative_ratio > 0 and positives:
                want = int(max(1, round(len(positives) * negative_ratio)))
                if want < len(negatives):
                    negatives = rng.sample(negatives, want)

            for eid in positives:
                rows.append(
                    PairRow(
                        case_id=cid,
                        answer_id=str(aid_int),
                        evidence_id=eid,
                        patient_question=case.patient_question,
                        clinician_question=case.clinician_question,
                        answer_text=answer_text,
                        evidence_text=case.note_sentences[eid],
                        label=1,
                    )
                )
            for eid in negatives:
                rows.append(
                    PairRow(
                        case_id=cid,
                        answer_id=str(aid_int),
                        evidence_id=eid,
                        patient_question=case.patient_question,
                        clinician_question=case.clinician_question,
                        answer_text=answer_text,
                        evidence_text=case.note_sentences[eid],
                        label=0,
                    )
                )
    return rows


def build_gold_eval_rows(
    cases: Dict[str, object],
    answer_texts: Dict[str, Dict[int, str]],
    gold_links: Dict[str, Dict[int, List[str]]],
) -> List[PairRow]:
    rows: List[PairRow] = []
    for cid, case in cases.items():
        if cid not in answer_texts:
            continue
        all_eids = sorted(case.note_sentences.keys(), key=numeric_key)
        if not all_eids:
            continue
        for aid_int, answer_text in sorted(
            answer_texts[cid].items(), key=lambda x: x[0]
        ):
            pos_set = set(gold_links[cid].get(aid_int, []))
            for eid in all_eids:
                rows.append(
                    PairRow(
                        case_id=cid,
                        answer_id=str(aid_int),
                        evidence_id=eid,
                        patient_question=case.patient_question,
                        clinician_question=case.clinician_question,
                        answer_text=answer_text,
                        evidence_text=case.note_sentences[eid],
                        label=1 if eid in pos_set else 0,
                    )
                )
    return rows


def build_inference_rows(
    cases: Dict[str, object],
    answers_by_case: Dict[str, Dict[str, str]],
) -> List[PairRow]:
    rows: List[PairRow] = []
    for cid, case in cases.items():
        answer_map = answers_by_case.get(cid, {})
        if not answer_map:
            continue

        candidate_ids = sorted(case.note_sentences.keys(), key=numeric_key)

        for aid, answer_text in sorted(
            answer_map.items(), key=lambda x: numeric_key(x[0])
        ):
            for eid in candidate_ids:
                rows.append(
                    PairRow(
                        case_id=cid,
                        answer_id=str(aid),
                        evidence_id=eid,
                        patient_question=case.patient_question,
                        clinician_question=case.clinician_question,
                        answer_text=answer_text,
                        evidence_text=case.note_sentences[eid],
                        label=-1,
                    )
                )
    return rows


def load_answers_for_split(
    split: str,
    key_path: Optional[Path],
    subtask1_predictions: Optional[str],
    use_gold_answers_if_key: bool,
) -> Dict[str, Dict[str, str]]:
    if subtask1_predictions:
        preds = load_subtask1_predictions(subtask1_predictions)
        out: Dict[str, Dict[str, str]] = {}
        for cid, answer in preds.items():
            sents = split_answer_sentences(answer)
            out[cid] = {str(i): s for i, s in enumerate(sents, start=1)}
        return out

    if key_path and (split == "dev" or use_gold_answers_if_key):
        answer_texts, _ = build_gold_links_from_key(str(key_path))
        return {
            cid: {
                str(aid): txt
                for aid, txt in sorted(answer_map.items(), key=lambda x: x[0])
            }
            for cid, answer_map in answer_texts.items()
        }

    raise ValueError(
        f"Need --subtask1_predictions for split={split} when gold key answers are not enabled."
    )


def tune_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> Tuple[float, dict]:
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
    return best["threshold"], best


def rows_to_submission(
    rows: List[PairRow], probs: np.ndarray, threshold: float
) -> List[dict]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    case_answer_ids: Dict[str, set] = {}

    for row, prob in zip(rows, probs):
        case_answer_ids.setdefault(row.case_id, set()).add(row.answer_id)
        if prob >= threshold:
            grouped.setdefault(row.case_id, {}).setdefault(
                row.answer_id, []
            ).append(row.evidence_id)

    out: List[dict] = []
    for cid in sorted(case_answer_ids.keys(), key=numeric_key):
        prediction = []
        for aid in sorted(case_answer_ids[cid], key=numeric_key):
            eids = sorted(
                set(grouped.get(cid, {}).get(aid, [])), key=numeric_key
            )
            if eids:
                prediction.append(
                    {
                        "answer_id": str(aid),
                        "evidence_id": [str(eid) for eid in eids],
                    }
                )
        out.append({"case_id": str(cid), "prediction": prediction})
    return out


def make_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def compute_class_weights(
    train_rows: List[PairRow], class_weight_mode: str, device: torch.device
) -> torch.Tensor:
    labels = np.array([r.label for r in train_rows], dtype=int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if class_weight_mode == "none" or pos == 0:
        pos_weight = 1.0
    elif class_weight_mode == "balanced":
        pos_weight = float(max(1.0, neg / max(1, pos)))
    else:
        pos_weight = float(math.sqrt(max(1.0, neg / max(1, pos))))
    return torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    rows: List[PairRow],
    tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    ds = LinkDataset(rows, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs: List[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    if not probs:
        return np.array([], dtype=float)
    return np.concatenate(probs, axis=0)


def train_for_cv_fold(
    model_name: str,
    train_rows: List[PairRow],
    val_rows: List[PairRow],
    run_cfg: dict,
    base_seed: int,
    fold_idx: int,
    device: torch.device,
    local_files_only: bool,
) -> Tuple[dict, np.ndarray]:
    set_seed(base_seed + fold_idx)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )

    train_ds = LinkDataset(
        train_rows, tokenizer=tokenizer, max_length=run_cfg["max_length"]
    )
    val_ds = LinkDataset(
        val_rows, tokenizer=tokenizer, max_length=run_cfg["max_length"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=run_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=run_cfg["batch_size"], shuffle=False
    )

    model = TransformerMLPLinker(
        model_name=model_name,
        mlp_hidden_size=run_cfg["mlp_hidden_size"],
        dropout=run_cfg["dropout"],
        local_files_only=local_files_only,
    ).to(device)
    optimizer = make_optimizer(
        model=model,
        learning_rate=run_cfg["learning_rate"],
        weight_decay=run_cfg["weight_decay"],
    )
    class_weights = compute_class_weights(
        train_rows, run_cfg["class_weight_mode"], device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best = {
        "epoch": -1,
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    best_state = None

    for epoch in range(1, run_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), run_cfg["max_grad_norm"]
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        model.eval()
        val_probs: List[np.ndarray] = []
        val_labels: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.append(probs)
                val_labels.append(labels)

        probs = (
            np.concatenate(val_probs)
            if val_probs
            else np.array([], dtype=float)
        )
        labels = (
            np.concatenate(val_labels)
            if val_labels
            else np.array([], dtype=int)
        )
        threshold, epoch_metrics = tune_threshold(labels, probs)

        avg_loss = total_loss / max(1, n_batches)
        print(
            f"    epoch={epoch} train_loss={avg_loss:.4f} val_f1={epoch_metrics['f1']:.4f} "
            f"val_p={epoch_metrics['precision']:.4f} val_r={epoch_metrics['recall']:.4f} thr={threshold:.2f}"
        )

        if epoch_metrics["f1"] > best["f1"]:
            best = {
                "epoch": epoch,
                "threshold": threshold,
                "f1": epoch_metrics["f1"],
                "precision": epoch_metrics["precision"],
                "recall": epoch_metrics["recall"],
            }
            best_state = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No best model state found during CV fold training."
        )

    model.load_state_dict(best_state)
    final_probs = predict_probs(
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
    return best, final_probs


def train_full_dev_model(
    model_name: str,
    train_rows: List[PairRow],
    run_cfg: dict,
    seed: int,
    device: torch.device,
    local_files_only: bool,
) -> Tuple[nn.Module, object]:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    train_ds = LinkDataset(
        train_rows, tokenizer=tokenizer, max_length=run_cfg["max_length"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=run_cfg["batch_size"], shuffle=True
    )

    model = TransformerMLPLinker(
        model_name=model_name,
        mlp_hidden_size=run_cfg["mlp_hidden_size"],
        dropout=run_cfg["dropout"],
        local_files_only=local_files_only,
    ).to(device)
    optimizer = make_optimizer(
        model=model,
        learning_rate=run_cfg["learning_rate"],
        weight_decay=run_cfg["weight_decay"],
    )
    class_weights = compute_class_weights(
        train_rows, run_cfg["class_weight_mode"], device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, run_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), run_cfg["max_grad_norm"]
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
        print(
            f"  full-dev epoch={epoch} train_loss={total_loss / max(1, n_batches):.4f}"
        )

    return model, tokenizer


def default_sweep_configs() -> List[dict]:
    return [
        {
            "epochs": 1,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "max_length": 256,
            "negative_ratio": 2.0,
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
            "negative_ratio": 3.0,
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
            "negative_ratio": 2.0,
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
            "negative_ratio": 2.0,
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
            "negative_ratio": 4.0,
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
            "negative_ratio": 3.0,
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
        f"l{cfg['max_length']}_nr{cfg['negative_ratio']}_do{cfg['dropout']}_"
        f"cw{cfg['class_weight_mode']}"
    )


def run_single_cv(
    model_name: str,
    run_cfg: dict,
    args,
    run_dir: Path,
    dev_cases: Dict[str, object],
    dev_answers: Dict[str, Dict[int, str]],
    dev_links: Dict[str, Dict[int, List[str]]],
    dev_key_path: Path,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    case_ids = sorted(dev_cases.keys(), key=numeric_key)
    groups = np.array(case_ids)
    indices = np.arange(len(case_ids))
    dummy_y = np.arange(len(case_ids))
    gkf = GroupKFold(n_splits=args.k)

    oof_rows: List[PairRow] = []
    oof_probs: List[float] = []
    fold_thresholds: List[float] = []
    fold_summaries: List[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(indices, dummy_y, groups), start=1
    ):
        train_case_ids = [case_ids[i] for i in tr_idx]
        val_case_ids = [case_ids[i] for i in va_idx]

        train_cases = {cid: dev_cases[cid] for cid in train_case_ids}
        val_cases = {cid: dev_cases[cid] for cid in val_case_ids}
        train_answers = {cid: dev_answers[cid] for cid in train_case_ids}
        val_answers = {cid: dev_answers[cid] for cid in val_case_ids}
        train_links = {cid: dev_links[cid] for cid in train_case_ids}
        val_links = {cid: dev_links[cid] for cid in val_case_ids}

        train_rows = build_gold_train_rows(
            cases=train_cases,
            answer_texts=train_answers,
            gold_links=train_links,
            negative_ratio=run_cfg["negative_ratio"],
            seed=args.seed + fold_idx,
        )
        val_rows = build_gold_eval_rows(
            cases=val_cases,
            answer_texts=val_answers,
            gold_links=val_links,
        )

        print(
            f"  fold={fold_idx}/{args.k} device={device} "
            f"train_rows={len(train_rows)} val_rows={len(val_rows)}"
        )
        best_fold, val_probs = train_for_cv_fold(
            model_name=model_name,
            train_rows=train_rows,
            val_rows=val_rows,
            run_cfg=run_cfg,
            base_seed=args.seed + (1000 * args._run_counter),
            fold_idx=fold_idx,
            device=device,
            local_files_only=args.local_files_only,
        )

        fold_submission = rows_to_submission(
            val_rows, val_probs, best_fold["threshold"]
        )
        fold_eval = score_links_official(
            fold_submission,
            key_path=str(dev_key_path),
            case_ids_to_score=val_case_ids,
        )
        fold_summary = {
            "fold": fold_idx,
            "threshold": best_fold["threshold"],
            "tuning_f1": best_fold["f1"],
            "tuning_precision": best_fold["precision"],
            "tuning_recall": best_fold["recall"],
            "official_leaderboard": fold_eval["leaderboard"],
        }
        fold_summaries.append(fold_summary)
        fold_thresholds.append(float(best_fold["threshold"]))
        oof_rows.extend(val_rows)
        oof_probs.extend(val_probs.tolist())
        print(
            f"    fold_official_micro_f1={fold_eval['leaderboard']['micro_f1']:.2f} "
            f"thr={best_fold['threshold']:.2f}"
        )

    avg_thr = float(np.mean(fold_thresholds)) if fold_thresholds else 0.5
    oof_submission = rows_to_submission(oof_rows, np.array(oof_probs), avg_thr)
    oof_eval = score_links_official(oof_submission, key_path=str(dev_key_path))

    summary = {
        "model_name": model_name,
        "k": args.k,
        "run_config": run_cfg,
        "avg_threshold": avg_thr,
        "folds": fold_summaries,
        "oof_leaderboard": oof_eval["leaderboard"],
        "oof_scores": oof_eval["scores"],
    }
    save_json(summary, str(run_dir / "cv_summary.json"))
    save_json(oof_submission, str(run_dir / "dev_oof_submission.json"))
    save_json(oof_eval["leaderboard"], str(run_dir / "dev_oof_scores.json"))
    return summary


def train_best_and_predict_test(
    args,
    best_entry: dict,
    output_root: Path,
    dev_cases: Dict[str, object],
    dev_answers: Dict[str, Dict[int, str]],
    dev_links: Dict[str, Dict[int, List[str]]],
) -> dict:
    model_name = best_entry["model_name"]
    run_cfg = best_entry["run_config"]
    threshold = float(best_entry["avg_threshold"])
    device = pick_device(args.device)

    print(f"\n[final] training best model on full dev: {model_name}")
    train_rows = build_gold_train_rows(
        cases=dev_cases,
        answer_texts=dev_answers,
        gold_links=dev_links,
        negative_ratio=run_cfg["negative_ratio"],
        seed=args.seed + 777,
    )
    model, tokenizer = train_full_dev_model(
        model_name=model_name,
        train_rows=train_rows,
        run_cfg=run_cfg,
        seed=args.seed + 777,
        device=device,
        local_files_only=args.local_files_only,
    )

    final_dir = output_root / "final_best_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")
    tokenizer.save_pretrained(final_dir / "tokenizer")
    save_json(
        {
            "model_name": model_name,
            "run_config": run_cfg,
            "threshold": threshold,
            "source_best_entry": best_entry,
        },
        str(final_dir / "final_config.json"),
    )

    test_split_dir = Path(args.data_dir) / args.test_split
    test_xml = test_split_dir / "archehr-qa.xml"
    test_key = test_split_dir / "archehr-qa_key.json"
    test_cases = load_cases_from_xml(str(test_xml))
    test_answers = load_answers_for_split(
        split=args.test_split,
        key_path=test_key if test_key.exists() else None,
        subtask1_predictions=args.subtask1_test_predictions,
        use_gold_answers_if_key=args.use_gold_answers_if_key,
    )

    test_rows = build_inference_rows(
        cases=test_cases,
        answers_by_case=test_answers,
    )
    test_probs = predict_probs(
        model=model,
        rows=test_rows,
        tokenizer=tokenizer,
        batch_size=run_cfg["batch_size"],
        max_length=run_cfg["max_length"],
        device=device,
    )
    test_submission = rows_to_submission(
        test_rows, test_probs, threshold=threshold
    )
    test_pred_path = output_root / "best_model_test_predictions.json"
    save_json(test_submission, str(test_pred_path))
    print(f"[final] saved test predictions: {test_pred_path}")

    out = {
        "model_name": model_name,
        "threshold": threshold,
        "test_prediction_path": str(test_pred_path),
        "final_model_dir": str(final_dir),
    }

    if test_key.exists() and args.score_test_with_key:
        try:
            test_eval = score_links_official(
                test_submission, key_path=str(test_key)
            )
            score_path = (
                output_root / "best_model_test_predictions.scores.json"
            )
            save_json(test_eval["leaderboard"], str(score_path))
            print(
                f"[final] test micro_F1={test_eval['leaderboard']['micro_f1']:.2f} "
                f"micro_P={test_eval['leaderboard']['micro_precision']:.2f} "
                f"micro_R={test_eval['leaderboard']['micro_recall']:.2f}"
            )
            out["test_scores_path"] = str(score_path)
            out["test_leaderboard"] = test_eval["leaderboard"]
        except Exception as exc:
            print(
                f"[warn] Skipping test scoring due to key/scorer mismatch: {exc}"
            )
            out["test_scoring_error"] = str(exc)
    elif test_key.exists() and not args.score_test_with_key:
        print("[final] test scoring skipped (prediction-only mode).")

    return out


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
        description="Paper pipeline: BERT-like k-fold sweep + final test prediction"
    )
    parser.add_argument(
        "--mode", choices=["all", "sweep", "finalize"], default="all"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="bioclinicalbert,deberta",
        help="Comma-separated model names or aliases (bioclinicalbert,deberta).",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_runs_per_model", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--test_split", choices=["test", "test-2026"], default="test"
    )
    parser.add_argument("--subtask1_test_predictions", type=str, default=None)
    parser.add_argument(
        "--use_gold_answers_if_key",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        default="subtask4_linking/output/paper_bert_kfold",
    )
    parser.add_argument("--sweep_summary_path", type=str, default=None)
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--score_test_with_key",
        action=argparse.BooleanOptionalAction,
        default=False,
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

    dev_dir = Path(args.data_dir) / "dev"
    dev_xml = dev_dir / "archehr-qa.xml"
    dev_key = dev_dir / "archehr-qa_key.json"
    dev_cases = load_cases_from_xml(str(dev_xml))
    dev_answers, dev_links = build_gold_links_from_key(str(dev_key))

    sweep_summary = None
    if args.mode in {"all", "sweep"}:
        all_cfgs = default_sweep_configs()
        max_runs = max(1, min(int(args.max_runs_per_model), 6))
        selected_cfgs = all_cfgs[:max_runs]
        print(
            f"Running sweep: models={models} runs_per_model={len(selected_cfgs)} k={args.k}"
        )

        all_runs: List[dict] = []
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
                    print(
                        f"  reused existing summary, micro_F1={run_summary['oof_leaderboard']['micro_f1']:.2f}"
                    )
                else:
                    run_summary = run_single_cv(
                        model_name=model_name,
                        run_cfg=cfg,
                        args=args,
                        run_dir=run_dir,
                        dev_cases=dev_cases,
                        dev_answers=dev_answers,
                        dev_links=dev_links,
                        dev_key_path=dev_key,
                    )
                    print(
                        f"  oof_micro_F1={run_summary['oof_leaderboard']['micro_f1']:.2f}"
                    )

                all_runs.append(
                    {
                        "model_name": model_name,
                        "run_dir": str(run_dir),
                        "run_config": run_summary["run_config"],
                        "avg_threshold": run_summary["avg_threshold"],
                        "oof_leaderboard": run_summary["oof_leaderboard"],
                    }
                )

        best = max(all_runs, key=lambda x: x["oof_leaderboard"]["micro_f1"])
        sweep_summary = {
            "models": models,
            "k": args.k,
            "max_runs_per_model": len(selected_cfgs),
            "runs": all_runs,
            "best": best,
        }
        sweep_summary_path = output_root / "sweep_summary.json"
        save_json(sweep_summary, str(sweep_summary_path))
        print(
            f"\nBest sweep run: model={best['model_name']} "
            f"micro_F1={best['oof_leaderboard']['micro_f1']:.2f} "
            f"run_dir={best['run_dir']}"
        )
        print(f"Saved sweep summary: {sweep_summary_path}")

    if args.mode in {"all", "finalize"}:
        if sweep_summary is None:
            path = Path(
                args.sweep_summary_path
                or (Path(args.output_root) / "sweep_summary.json")
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Sweep summary not found: {path}. Run --mode sweep or --mode all first."
                )
            sweep_summary = json.loads(path.read_text())

        best_entry = sweep_summary["best"]
        final_summary = train_best_and_predict_test(
            args=args,
            best_entry=best_entry,
            output_root=output_root,
            dev_cases=dev_cases,
            dev_answers=dev_answers,
            dev_links=dev_links,
        )
        final_meta = {
            "best_from_sweep": best_entry,
            "final_outputs": final_summary,
        }
        save_json(final_meta, str(output_root / "final_summary.json"))
        print(f"Saved final summary: {output_root / 'final_summary.json'}")


if __name__ == "__main__":
    main()
