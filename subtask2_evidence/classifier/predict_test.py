"""
Run a trained HYDRA model on the ArchEHR-QA test set and save predictions.

Usage:
    python predict_test.py \
        --model_dir output/best_model_binary \
        --test_dir ../data/test \
        --out_dir output/test
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
LABEL2ID = {"essential": 0, "supplementary": 1, "not-relevant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class HydraSentenceClassifier(nn.Module):
    def __init__(
        self, model_name: str, num_labels: int = 3, dropout: float = 0.3
    ):
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
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.head_dropout(cls_output)
        binary_logits = self.binary_head(cls_output)
        fine_logits = self.fine_head(cls_output)
        return binary_logits, fine_logits


def load_test_data(test_dir: str) -> list[dict]:
    """Load test sentences from XML. No labels needed."""
    test_dir = Path(test_dir)
    tree = ET.parse(test_dir / "archehr-qa.xml")
    root = tree.getroot()

    examples = []
    for case in root.findall("case"):
        cid = case.attrib["id"]

        pq_el = case.find("patient_question")
        patient_q = (
            "".join(pq_el.itertext()).strip() if pq_el is not None else ""
        )
        clinician_q = (case.findtext("clinician_question") or "").strip()

        for sent in case.find("note_excerpt_sentences").findall("sentence"):
            sid = sent.attrib["id"]
            text = "".join(sent.itertext()).strip()
            examples.append(
                {
                    "case_id": cid,
                    "sentence_id": sid,
                    "patient_question": patient_q,
                    "clinician_question": clinician_q,
                    "sentence": text,
                }
            )
    return examples


# ---------------------------------------------------------------------------
# Dataset (same tokenisation as train.py, but no label)
# ---------------------------------------------------------------------------
class TestEvidenceDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
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
        }


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_fine_preds = []
    all_fine_probs = []
    all_bin_preds = []
    all_bin_probs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        binary_logits, fine_logits = model(input_ids, attention_mask)

        all_fine_preds.extend(fine_logits.argmax(dim=-1).cpu().numpy())
        all_fine_probs.extend(torch.softmax(fine_logits, dim=-1).cpu().numpy())
        all_bin_preds.extend(binary_logits.argmax(dim=-1).cpu().numpy())
        all_bin_probs.extend(
            torch.softmax(binary_logits, dim=-1).cpu().numpy()
        )

    return all_fine_preds, all_fine_probs, all_bin_preds, all_bin_probs


def main():
    parser = argparse.ArgumentParser(
        description="Run trained HYDRA model on the test set"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="output/final",
        help="Directory with final_model.pt + tokenizer files",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../data/test",
        help="Directory with archehr-qa.xml",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/test",
        help="Where to save test predictions JSON",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load tokenizer from the saved checkpoint
    model_dir = Path(args.model_dir)
    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Build model and load weights
    print(f"Loading model from {model_dir / 'final_model.pt'} ...")
    model = HydraSentenceClassifier(MODEL_NAME)
    model.load_state_dict(
        torch.load(model_dir / "final_model.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    # Load test data
    print(f"Loading test data from {args.test_dir} ...")
    examples = load_test_data(args.test_dir)
    print(f"  {len(examples)} sentences across test cases")

    ds = TestEvidenceDataset(examples, tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size)

    # Run predictions
    print("Running predictions ...")
    fine_preds, fine_probs, bin_preds, bin_probs = predict(
        model, loader, device
    )

    # Build output
    results = []
    for i, ex in enumerate(examples):
        pred_fine_label = ID2LABEL[int(fine_preds[i])]
        prob_fine = [round(float(p), 4) for p in fine_probs[i]]
        pred_binary_label = (
            "essential" if int(bin_preds[i]) == 1 else "not-essential"
        )
        prob_binary = [round(float(p), 4) for p in bin_probs[i]]
        # Binary-from-fine: only essential is relevant, rest not-essential
        pred_bin_from_fine = (
            "essential"
            if int(fine_preds[i]) == LABEL2ID["essential"]
            else "not-essential"
        )
        results.append(
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

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions_both_heads.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    from collections import Counter

    fine_counts = Counter(r["pred_fine"] for r in results)
    bin_counts = Counter(r["pred_binary"] for r in results)
    print(f"\nSaved {len(results)} predictions to {out_path}")
    print(f"Fine-grained distribution: {dict(fine_counts)}")
    print(f"Binary distribution:       {dict(bin_counts)}")


if __name__ == "__main__":
    main()
