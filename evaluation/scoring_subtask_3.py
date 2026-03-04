"""
Scoring script for Subtask 3: Answer Generation

This subtask evaluates system-generated answers against reference
clinician-authored answers.

Prediction format:
- prediction: string containing the generated answer (≤ 75 words)

Example submission.json:
[
  {
    "case_id": "1",
    "prediction": "ERCP was used to relieve a bile duct obstruction caused by stones and sludge by placing a common bile duct stent. Because liver tests and bilirubin continued to worsen, the patient required a repeat ERCP, which found the stent obstructed by sludge and stones. Once the INR normalized, a sphincterotomy was performed and stones were removed, improving drainage."
  },
  ...
]

Usage:
    # Manual run with command-line arguments
    python scoring_subtask_3.py \
        --submission_path submission.json \
        --key_path archehr-qa_key.json \
        --data_path archehr-qa.xml \
        --quickumls_path quickumls/ \
        --out_file_path scores.json
"""

import json
from argparse import ArgumentParser
from lxml import etree
from enum import Enum
import numpy as np
from pathlib import Path
import torch
import nltk
import sys

nltk.download("punkt", quiet=True)

from scorers.medcon_scorer import MedconScorer
from scorers.align_scorer import AlignScorer
from scorers.bert_scorer import BertScorer
from scorers.rouge_scorer import RougeScorer
from scorers.bleu_scorer import BleuScorer
from scorers.sari_scorer import SariScorer


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


# Maximum word limit for answer predictions
MAX_PREDICTION_WORDS = 75


class MetricType(Enum):
    BLEU = "bleu"
    ROUGE = "rouge"
    SARI = "sari"
    BERTSCORE = "bertscore"
    MEDCON = "medcon"
    ALIGNSCORE = "alignscore"


def load_submission(
    path,
    max_prediction_words=MAX_PREDICTION_WORDS,
    case_ids_to_score=None,
):
    """
    Load and validate a submission file.

    Args:
        path: Path to the submission JSON file
        max_prediction_words: Maximum allowed words in prediction
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
        prediction_text = case["prediction"].strip()

        if not prediction_text:
            error_str = f"[case {case_id}]: Empty prediction."
            raise ValueError(error_str)

        # Check word count and truncate if necessary
        prediction_words = prediction_text.split()
        if len(prediction_words) > max_prediction_words:
            print(
                f"[case {case_id}]: Prediction has {len(prediction_words)} words, "
                f"truncating to {max_prediction_words} words."
            )
            prediction_text = " ".join(prediction_words[:max_prediction_words])

        submission.append(
            {
                "case_id": case_id,
                "prediction": prediction_text,
            }
        )

    if case_ids_to_score:
        print(f"Number of cases in submission after filtering: {len(submission)}")
    return submission


def load_key(path, case_ids_to_score=None):
    """
    Load the key file containing reference clinician answers.

    Args:
        path: Path to the key JSON file
        case_ids_to_score: Optional set of case IDs to score (filters others)

    Returns:
        Dictionary mapping case_id to reference answer (without citations)
    """
    with open(path, "r") as file:
        key_json = json.load(file)

    key_map = {}
    for case in key_json:
        case_id = case["case_id"]
        if case_ids_to_score and case_id not in case_ids_to_score:
            continue

        key_map[case_id] = case["clinician_answer_without_citations"].strip()

    return key_map


def load_sources(data_path, case_ids_to_score=None):
    """
    Load the source data (patient questions) from XML file for SARI computation.

    Args:
        data_path: Path to the XML file
        case_ids_to_score: Optional set of case IDs to score (filters others)

    Returns:
        Dictionary mapping case_id to patient question
    """
    tree_data = etree.parse(data_path)
    root_data = tree_data.getroot()
    cases_data = root_data.findall(".//case")
    source_map = {}

    for case_elem_data in cases_data:
        case_id = case_elem_data.attrib["id"]
        if case_ids_to_score and case_id not in case_ids_to_score:
            continue
        
        patient_narrative = case_elem_data.find("patient_narrative").text.strip()
        source_map[case_id] = patient_narrative

    return source_map


def compute_text_similarity_scores(
    references,
    predictions,
    sources,
    metrics=[m.value for m in MetricType],
    device="cpu",
    quickumls_path="quickumls/",
):
    """
    Compute text similarity scores between references and predictions.

    Args:
        references: List of reference clinician answers
        predictions: List of predicted answers
        sources: List of source texts (patient questions) for SARI
        metrics: List of metric types to compute
        device: Device for neural models (cpu/cuda)
        quickumls_path: Path to QuickUMLS data for MEDCON scorer

    Returns:
        Dictionary with metric scores
    """
    scores = {}
    metric_types = [MetricType(m) for m in metrics]

    for metric_type in metric_types:
        print(f"{' ' + metric_type.value.upper() + ' ':-^30}")

        if metric_type == MetricType.BLEU:
            scorer = BleuScorer()
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions
            )
        elif metric_type == MetricType.ROUGE:
            scorer = RougeScorer()
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions
            )
        elif metric_type == MetricType.SARI:
            scorer = SariScorer()
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions, sources
            )
        elif metric_type == MetricType.BERTSCORE:
            scorer = BertScorer(device=device)
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions
            )
        elif metric_type == MetricType.MEDCON:
            scorer = MedconScorer(quickumls_fp=quickumls_path)
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions
            )
        elif metric_type == MetricType.ALIGNSCORE:
            scorer = AlignScorer(device=device)
            scores[metric_type.value] = scorer.compute_overall_score(
                references, predictions
            )

        print(f"  Score: {scores[metric_type.value]}")

    return scores


def get_leaderboard(scores):
    """
    Convert raw scores to leaderboard format.

    Args:
        scores: Dictionary of metric scores

    Returns:
        Dictionary with leaderboard scores
    """
    leaderboard = {}

    # Process each metric
    for metric_type, metric_scores in scores.items():
        if metric_type == MetricType.ROUGE.value:
            # ROUGE returns a dictionary of different ROUGE types
            for rouge_type, score in metric_scores.items():
                leaderboard[rouge_type] = score * 100
        elif metric_type == MetricType.SARI.value:
            leaderboard[metric_type] = metric_scores
        else:
            leaderboard[metric_type] = metric_scores * 100

    # Calculate overall score (average of all metrics)
    # Use rougeLsum as the representative ROUGE score
    overall_score = np.mean(
        [
            leaderboard["bleu"],
            leaderboard["rougeLsum"],
            leaderboard["sari"],
            leaderboard["bertscore"],
            leaderboard["alignscore"],
            leaderboard["medcon"],
        ]
    )
    leaderboard["overall_score"] = overall_score

    return leaderboard


def score_submission(
    submission_path,
    key_path,
    out_file_path,
    data_path,
    quickumls_path="quickumls/",
    case_ids_to_score=None,
    device="cpu",
):
    """
    Score a submission file against the reference key.

    Args:
        submission_path: Path to submission JSON file
        key_path: Path to key JSON file with reference answers
        out_file_path: Path to output scores JSON file
        data_path: Path to XML file for source data (for SARI)
        quickumls_path: Path to QuickUMLS data
        case_ids_to_score: Optional set of case IDs to score (filters others)
        device: Device for neural models (cpu/cuda)
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

    # Load sources for SARI
    print()
    print("=" * 40)
    print("Loading source data for SARI")
    print("=" * 40)
    source_map = load_sources(data_path, case_ids_to_score=case_ids_to_score)
    print(f"Number of cases with source data: {len(source_map)}")

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

    print("Submission validation passed!")

    # Prepare references, predictions, and sources lists
    references = []
    predictions = []
    sources = []
    for case in submission:
        case_id = case["case_id"]
        references.append(key_map[case_id])
        predictions.append(case["prediction"])
        sources.append(source_map[case_id])

    print()
    print("=" * 40)
    print("Computing text similarity scores")
    print("=" * 40)
    scores = compute_text_similarity_scores(
        references,
        predictions,
        sources=sources,
        metrics=[m.value for m in MetricType],
        device=device,
        quickumls_path=quickumls_path,
    )

    print()
    print("=" * 40)
    print("Computing leaderboard scores")
    print("=" * 40)
    leaderboard = get_leaderboard(scores)

    for metric, score in sorted(leaderboard.items()):
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
        python scoring_subtask_3.py \
            --submission_path submission.json \
            --key_path archehr-qa_key.json \
            --data_path archehr-qa.xml \
            --quickumls_path quickumls/ \
            --out_file_path scores.json
    """
    parser = ArgumentParser(description="Score a Subtask 3 submission.")
    parser.add_argument(
        "--submission_path",
        type=str,
        help="Path to the submission file",
        required=True,
    )
    parser.add_argument(
        "--key_path",
        type=str,
        help="Path to the key file with reference answers",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data XML file for source data (for SARI metric)",
        required=True,
    )
    parser.add_argument(
        "--quickumls_path",
        type=str,
        help="Path to the QuickUMLS data for MEDCON scorer",
        required=False,
        default="quickumls/",
    )
    parser.add_argument(
        "--out_file_path",
        type=str,
        help="Path to the output results file",
        required=True,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    score_submission(
        submission_path=args.submission_path,
        key_path=args.key_path,
        out_file_path=args.out_file_path,
        data_path=args.data_path,
        quickumls_path=args.quickumls_path,
        case_ids_to_score=case_ids_to_score,
        device=device,
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
    data_path = reference_dir / "archehr-qa.xml"
    quickumls_path = reference_dir / "quickumls"
    out_file_path = score_dir / "scores.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    score_submission(
        submission_path=submission_path,
        key_path=key_path,
        out_file_path=out_file_path,
        data_path=data_path,
        quickumls_path=quickumls_path,
        case_ids_to_score=case_ids_to_score,
        device=device,
    )


def main():
    """
    Entry point that determines whether to run in argparse or codabench mode.
    """
    parser = ArgumentParser(description="Score a Subtask 3 submission.")
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
