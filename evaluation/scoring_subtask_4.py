"""
Scoring script for Subtask 4: Evidence Alignment

This subtask evaluates system-predicted alignments between answer sentences
and clinical note excerpt sentences against reference alignments.

Prediction format:
- prediction: array of objects with answer_id and evidence_id

Example submission.json:
[
  {
    "case_id": "1",
    "prediction": [
      {
        "answer_id": "1",
        "evidence_id": ["1"]
      },
      {
        "answer_id": "2",
        "evidence_id": ["5", "6"]
      },
      ...
    ]
  },
  ...
]

Usage:
    # Manual run with command-line arguments
    python scoring_subtask_4.py \
        --submission_path submission.json \
        --key_path archehr-qa_key.json \
        --out_file_path scores.json
"""

import json
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import sys


def parse_case_ids(case_id_args):
    """
    Parse case ID arguments supporting both individual IDs and ranges.
    
    Args:
        case_id_args: List of strings like ["1", "2-4", "5", "7-9"]
        
    Returns:
        Set of case ID strings, e.g., {"1", "2", "3", "4", "5", "7", "8", "9"}
    """
    if not case_id_args:
        return None
    
    case_ids = set()
    for arg in case_id_args:
        if "-" in arg and not arg.startswith("-"):
            # Handle range like "1-3"
            try:
                start, end = arg.split("-")
                for i in range(int(start), int(end) + 1):
                    case_ids.add(str(i))
            except ValueError as e:
                raise ValueError(
                    f"Invalid case ID range '{arg}'. Expected format like '1-3'. Error: {e}"
                )
        else:
            case_ids.add(arg)
    
    return case_ids if case_ids else None


def load_submission(path, case_ids_to_score=None):
    """
    Load and validate a submission file.

    Args:
        path: Path to the submission JSON file
        case_ids_to_score: Optional set of case IDs to score (filters others)

    Returns:
        List of dictionaries with case_id and prediction
    """
    submission = []
    with open(path, "r") as file:
        submission_json = json.load(file)

    print(f"Number of cases in submission: {len(submission_json)}")

    for case in submission_json:
        case_id = case["case_id"]
        if case_ids_to_score and case_id not in case_ids_to_score:
            continue
        prediction = case["prediction"]

        if not isinstance(prediction, list):
            error_str = f"[case {case_id}]: Prediction must be a list of alignment objects."
            raise ValueError(error_str)

        # Validate alignment format
        for alignment in prediction:
            if not isinstance(alignment, dict):
                error_str = f"[case {case_id}]: Each alignment must be an object."
                raise ValueError(error_str)

            if "answer_id" not in alignment:
                error_str = f"[case {case_id}]: Alignment missing 'answer_id' field."
                raise ValueError(error_str)

            if "evidence_id" not in alignment:
                error_str = f"[case {case_id}]: Alignment missing 'evidence_id' field."
                raise ValueError(error_str)

            if not isinstance(alignment["answer_id"], str):
                error_str = f"[case {case_id}]: 'answer_id' must be a string."
                raise ValueError(error_str)

            if not isinstance(alignment["evidence_id"], list):
                error_str = f"[case {case_id}]: 'evidence_id' must be a list."
                raise ValueError(error_str)

            for evidence_id in alignment["evidence_id"]:
                if not isinstance(evidence_id, str):
                    error_str = f"[case {case_id}]: All evidence IDs must be strings."
                    raise ValueError(error_str)

        submission.append(
            {
                "case_id": case_id,
                "prediction": prediction,
            }
        )

    if case_ids_to_score:
        print(f"Number of cases in submission after filtering: {len(submission)}")
    return submission


def load_key(path, case_ids_to_score=None):
    """
    Load the key file containing reference alignments.

    Args:
        path: Path to the key JSON file
        case_ids_to_score: Optional set of case IDs to score (filters others)

    Returns:
        Dictionary mapping case_id to dict containing:
        - alignments: set of (answer_id, evidence_id) tuples
        - valid_answer_ids: set of valid answer sentence IDs
        - valid_evidence_ids: set of valid evidence sentence IDs
    """
    with open(path, "r") as file:
        key_json = json.load(file)

    key_map = {}
    for case in key_json:
        case_id = case["case_id"]
        if case_ids_to_score and case_id not in case_ids_to_score:
            continue

        alignments = set()
        valid_answer_ids = set()

        # Parse answer_sentences to extract alignments and valid answer IDs
        clinician_answer_sentences = case["clinician_answer_sentences"]
        for answer_sentence in clinician_answer_sentences:
            answer_sent_id = answer_sentence["id"]
            valid_answer_ids.add(answer_sent_id)
            citation_ids = answer_sentence["citations"]

            for evidence_id in citation_ids:
                alignments.add((answer_sent_id, evidence_id))

        # Extract valid evidence IDs from the answers field
        # (which contains all sentences in the clinical note excerpt)
        valid_evidence_ids = set()
        for answer in case.get("answers", []):
            valid_evidence_ids.add(answer["sentence_id"])

        key_map[case_id] = {
            "alignments": alignments,
            "valid_answer_ids": valid_answer_ids,
            "valid_evidence_ids": valid_evidence_ids,
        }

    return key_map


def compute_alignment_scores(submission, key_map):
    """
    Compute alignment scores using micro-averaged and macro-averaged precision, recall, and F1.

    Args:
        submission: List of submission cases with predictions
        key_map: Dictionary mapping case_id to reference alignments

    Returns:
        Dictionary with micro and macro precision, recall, and f1 scores
    """
    total_true_positives = 0
    total_predicted = 0
    total_gold = 0

    # For macro averaging
    case_precisions = []
    case_recalls = []
    case_f1s = []

    for case in submission:
        case_id = case["case_id"]
        gold_alignments = key_map[case_id]["alignments"]

        # Convert predictions to set of (answer_id, evidence_id) tuples
        predicted_alignments = set()
        for alignment in case["prediction"]:
            answer_id = alignment["answer_id"]
            for evidence_id in alignment["evidence_id"]:
                predicted_alignments.add((answer_id, evidence_id))

        # Compute true positives for this case
        true_positives = len(predicted_alignments & gold_alignments)
        total_true_positives += true_positives
        total_predicted += len(predicted_alignments)
        total_gold += len(gold_alignments)

        # Compute per-case precision, recall, and F1 for macro averaging
        if len(predicted_alignments) > 0:
            case_precision = true_positives / len(predicted_alignments)
        else:
            case_precision = 0.0

        if len(gold_alignments) > 0:
            case_recall = true_positives / len(gold_alignments)
        else:
            case_recall = 0.0

        if case_precision + case_recall > 0:
            case_f1 = 2 * case_precision * case_recall / (case_precision + case_recall)
        else:
            case_f1 = 0.0

        case_precisions.append(case_precision)
        case_recalls.append(case_recall)
        case_f1s.append(case_f1)

    # Compute micro-averaged precision, recall, and F1
    if total_predicted > 0:
        micro_precision = total_true_positives / total_predicted
    else:
        micro_precision = 0.0

    if total_gold > 0:
        micro_recall = total_true_positives / total_gold
    else:
        micro_recall = 0.0

    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    # Compute macro-averaged precision, recall, and F1
    macro_precision = np.mean(case_precisions) if case_precisions else 0.0
    macro_recall = np.mean(case_recalls) if case_recalls else 0.0
    macro_f1 = np.mean(case_f1s) if case_f1s else 0.0

    scores = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    return scores


def get_leaderboard(scores):
    """
    Convert raw scores to leaderboard format.

    Args:
        scores: Dictionary of metric scores

    Returns:
        Dictionary with leaderboard scores
    """
    leaderboard = {
        "micro_precision": scores["micro_precision"] * 100,
        "micro_recall": scores["micro_recall"] * 100,
        "micro_f1": scores["micro_f1"] * 100,
        "macro_precision": scores["macro_precision"] * 100,
        "macro_recall": scores["macro_recall"] * 100,
        "macro_f1": scores["macro_f1"] * 100,
    }

    # Overall score is the micro F1
    leaderboard["overall_score"] = leaderboard["micro_f1"]

    return leaderboard


def score_submission(
    submission_path,
    key_path,
    out_file_path,
    case_ids_to_score=None,
):
    """
    Score a submission file against the reference key.

    Args:
        submission_path: Path to submission JSON file
        key_path: Path to key JSON file with reference alignments
        out_file_path: Path to output scores JSON file
        case_ids_to_score: Optional set of case IDs to score (filters others)
    """
    out_file_path = Path(out_file_path)

    if case_ids_to_score:
        print(f"Scoring only the following case IDs: {case_ids_to_score}")

    print()
    print("=" * 40)
    print("Loading submission")
    print("=" * 40)
    submission = load_submission(submission_path, case_ids_to_score=case_ids_to_score)

    print()
    print("=" * 40)
    print("Loading reference key")
    print("=" * 40)
    key_map = load_key(key_path, case_ids_to_score=case_ids_to_score)
    print(f"Number of cases in key: {len(key_map)}")

    # Validate submission against key
    print()
    print("=" * 40)
    print("Validating submission")
    print("=" * 40)
    key_case_ids = set(key_map.keys())
    submission_case_ids = {case["case_id"] for case in submission}

    if key_case_ids != submission_case_ids:
        missing_case_ids = key_case_ids - submission_case_ids
        extra_case_ids = submission_case_ids - key_case_ids
        error_msg = []
        if missing_case_ids:
            error_msg.append(
                f"Case IDs in key but not in submission: {sorted(missing_case_ids)}"
            )
        if extra_case_ids:
            error_msg.append(
                f"Case IDs in submission but not in key: {sorted(extra_case_ids)}"
            )

        error_str = "\n".join(error_msg)
        raise ValueError(error_str)

    # Validate that all predicted answer IDs and evidence IDs are valid
    invalid_id_errors = []
    for case in submission:
        case_id = case["case_id"]
        valid_answer_ids = key_map[case_id]["valid_answer_ids"]
        valid_evidence_ids = key_map[case_id]["valid_evidence_ids"]

        for alignment in case["prediction"]:
            answer_id = alignment["answer_id"]
            if answer_id not in valid_answer_ids:
                invalid_id_errors.append(
                    f"[case {case_id}]: Invalid answer_id: '{answer_id}'. "
                    f"Valid answer IDs are: {sorted(valid_answer_ids)}"
                )

            for evidence_id in alignment["evidence_id"]:
                if evidence_id not in valid_evidence_ids:
                    invalid_id_errors.append(
                        f"[case {case_id}]: Invalid evidence_id: '{evidence_id}'. "
                        f"Valid evidence IDs are: {sorted(valid_evidence_ids)}"
                    )

    if invalid_id_errors:
        error_str = "\n".join(invalid_id_errors)
        raise ValueError(error_str)

    print("Submission validation passed!")

    print()
    print("=" * 40)
    print("Computing alignment scores")
    print("=" * 40)
    scores = compute_alignment_scores(submission, key_map)

    print()
    print("=" * 40)
    print("Computing leaderboard scores")
    print("=" * 40)
    leaderboard = get_leaderboard(scores)

    for metric, score in leaderboard.items():
        print(f"  {metric}: {score:.2f}")

    print()
    print("=" * 40)
    print("Saving scores")
    print("=" * 40)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_path, "w") as out_file:
        json.dump(leaderboard, out_file, indent=2)
    print(f"Scores saved to: {out_file_path}")


def main_argparse(case_ids_to_score=None):
    """
    Main function for manual run with command-line arguments.

    Sample usage:
        python scoring_subtask_4.py \
            --submission_path submission.json \
            --key_path archehr-qa_key.json \
            --out_file_path scores.json
    """
    parser = ArgumentParser(description="Score a Subtask 4 submission.")
    parser.add_argument(
        "--submission_path",
        type=str,
        help="Path to the submission file",
        required=True,
    )
    parser.add_argument(
        "--key_path",
        type=str,
        help="Path to the key file with reference alignments",
        required=True,
    )
    parser.add_argument(
        "--out_file_path",
        type=str,
        help="Path to the output results file",
        required=True,
    )
    args = parser.parse_args()

    score_submission(
        submission_path=args.submission_path,
        key_path=args.key_path,
        out_file_path=args.out_file_path,
        case_ids_to_score=case_ids_to_score,
    )


def main_codabench(case_ids_to_score=None):
    """
    Main function for Codabench competition run.
    Uses fixed paths expected by the Codabench environment.
    """
    reference_dir = Path("/app/input/ref")
    result_dir = Path("/app/input/res")
    score_dir = Path("/app/output/")

    submission_path = result_dir / "submission.json"
    key_path = reference_dir / "archehr-qa_key.json"
    out_file_path = score_dir / "scores.json"

    score_submission(
        submission_path=submission_path,
        key_path=key_path,
        out_file_path=out_file_path,
        case_ids_to_score=case_ids_to_score,
    )


def main():
    """
    Entry point that determines whether to run in argparse or codabench mode.
    """
    parser = ArgumentParser(description="Score a Subtask 4 submission.")
    parser.add_argument(
        "--codabench",
        action="store_true",
        help="Run in Codabench mode with fixed paths",
    )
    parser.add_argument(
        "--case_ids_to_score",
        type=str,
        nargs="+",
        help="Optional list of case IDs to score (supports ranges like '1-3 5 7-9')",
        required=False,
        default=None,
    )

    # Check if --codabench is in the arguments
    args, remaining = parser.parse_known_args()

    if args.codabench:
        case_ids_to_score = parse_case_ids(args.case_ids_to_score)
        main_codabench(case_ids_to_score=case_ids_to_score)
    else:
        case_ids_to_score = parse_case_ids(args.case_ids_to_score)
        # Reset sys.argv to include only remaining arguments for the argparse main
        sys.argv = [sys.argv[0]] + remaining
        main_argparse(case_ids_to_score=case_ids_to_score)


if __name__ == "__main__":
    main()
