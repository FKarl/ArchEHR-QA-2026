# Subtask 2: Evidence Identification

This folder contains a BERT-based approach for **Subtask 2: Evidence Identification** of the ArchEHR-QA 2026 shared task.

We frame this as a **sentence-pair classification** task:
- **Input:** `[CLS] question [SEP] sentence [SEP]`
- **Output:** Binary label (relevant / not-relevant)

We use a pre-trained biomedical BERT model fine-tuned on the task.
