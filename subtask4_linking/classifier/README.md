# Subtask 4: Evidence Alignment

This folder contains a BERT-based approach for **Subtask 4: Evidence Alignment** of the ArchEHR-QA 2026 shared task.

We frame this as a **sentence-pair classification** task:
- **Input:** `[CLS] Patient question [SEP] Clinician question [SEP] Answer sentence [SEP] Evidence sentence`
- **Output:** Binary label (relevant / not-relevant)

We use a pre-trained biomedical BERT model fine-tuned on the task.