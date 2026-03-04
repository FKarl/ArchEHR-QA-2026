import json
import importlib.util
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CaseData:
    case_id: str
    patient_question: str
    clinician_question: str
    note_sentences: Dict[str, str]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_answer_sentences(answer_text: str) -> List[str]:
    text = _clean_text(answer_text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    parts = [_clean_text(p) for p in parts if _clean_text(p)]
    return parts if parts else [text]


def extract_citation_ids(text: str) -> List[str]:
    ids: List[str] = []
    for block in re.findall(r"\[([^\]]+)\]", text):
        ids.extend(re.findall(r"\d+", block))
    return sorted(set(ids), key=lambda x: int(x))


def strip_citations(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def load_cases_from_xml(xml_path: str) -> Dict[str, CaseData]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases: Dict[str, CaseData] = {}

    for case in root.findall("case"):
        case_id = case.attrib["id"]
        pq_el = case.find("patient_question")
        patient_q = _clean_text("".join(pq_el.itertext()) if pq_el is not None else "")
        clinician_q = _clean_text(case.findtext("clinician_question"))

        note_sentences: Dict[str, str] = {}
        note_el = case.find("note_excerpt_sentences")
        if note_el is not None:
            for sent in note_el.findall("sentence"):
                sid = sent.attrib["id"]
                note_sentences[sid] = _clean_text("".join(sent.itertext()))

        cases[case_id] = CaseData(
            case_id=case_id,
            patient_question=patient_q,
            clinician_question=clinician_q,
            note_sentences=note_sentences,
        )
    return cases


def load_subtask1_predictions(path: str) -> Dict[str, str]:
    data = json.loads(Path(path).read_text())
    return {str(item["case_id"]): _clean_text(item["prediction"]) for item in data}


def load_subtask2_candidates(path: str) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text())
    result: Dict[str, List[str]] = {}

    if not data:
        return result

    if "evidence_sentence_ids" in data[0]:
        for item in data:
            cid = str(item["case_id"])
            ids = [str(x) for x in item.get("evidence_sentence_ids", [])]
            result[cid] = sorted(set(ids), key=lambda x: int(x))
        return result

    grouped: Dict[str, List[str]] = {}
    for item in data:
        cid = str(item.get("case_id"))
        sid = str(item.get("sentence_id"))
        if not cid or not sid:
            continue

        fine_label = str(item.get("pred_fine", "")).lower()
        bin_label = str(item.get("pred_binary", "")).lower()
        bin_from_fine = str(item.get("pred_binary_from_fine", "")).lower()

        is_relevant = (
            fine_label in {"essential", "supplementary"}
            or bin_label == "essential"
            or bin_from_fine == "essential"
        )
        if is_relevant:
            grouped.setdefault(cid, []).append(sid)

    for cid, ids in grouped.items():
        result[cid] = sorted(set(ids), key=lambda x: int(x))
    return result


def build_gold_links_from_key(
    key_path: str,
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, List[str]]]]:
    key_data = json.loads(Path(key_path).read_text())
    answer_texts: Dict[str, Dict[int, str]] = {}
    links: Dict[str, Dict[int, List[str]]] = {}

    for item in key_data:
        cid = str(item["case_id"])
        sentence_map: Dict[int, str] = {}
        link_map: Dict[int, List[str]] = {}

        # Prefer structured sentence/citation annotations in the newer key format.
        clinician_answer_sentences = item.get("clinician_answer_sentences")
        if clinician_answer_sentences:
            for idx, sent_obj in enumerate(clinician_answer_sentences, start=1):
                raw_id = str(sent_obj.get("id", idx))
                aid = int(raw_id) if raw_id.isdigit() else idx
                sentence_map[aid] = _clean_text(strip_citations(str(sent_obj.get("text", ""))))
                citations = [str(x) for x in sent_obj.get("citations", [])]
                citations = sorted(set(citations), key=_numeric_string_sort_key)
                link_map[aid] = citations
        else:
            for idx, sent in enumerate(split_answer_sentences(item["clinician_answer"]), start=1):
                sentence_map[idx] = strip_citations(sent)
                link_map[idx] = extract_citation_ids(sent)

        answer_texts[cid] = sentence_map
        links[cid] = link_map
    return answer_texts, links


def build_pair_set(submission_like: List[dict]) -> set:
    pairs = set()
    for case in submission_like:
        cid = str(case["case_id"])
        for pair in case.get("answer_to_evidence", []):
            aid = str(pair["answer_id"])
            eid = str(pair["evidence_id"])
            pairs.add((cid, aid, eid))
    return pairs


def score_links(gold_submission: List[dict], pred_submission: List[dict]) -> dict:
    gold_pairs = build_pair_set(gold_submission)
    pred_pairs = build_pair_set(pred_submission)

    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gold_pairs": len(gold_pairs),
        "num_pred_pairs": len(pred_pairs),
    }


def gold_links_to_submission(gold_links: Dict[str, Dict[int, List[str]]]) -> List[dict]:
    submission = []
    for cid, answer_map in sorted(gold_links.items(), key=lambda x: int(x[0])):
        pairs = []
        for aid, eids in sorted(answer_map.items(), key=lambda x: x[0]):
            for eid in eids:
                pairs.append({"answer_id": str(aid), "evidence_id": str(eid)})
        submission.append({"case_id": str(cid), "answer_to_evidence": pairs})
    return submission


def save_json(obj, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2))


def _numeric_string_sort_key(value: str) -> Tuple[int, object]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


_SCORER_MODULE = None


def _load_official_scorer_module():
    global _SCORER_MODULE
    if _SCORER_MODULE is not None:
        return _SCORER_MODULE

    scorer_path = Path(__file__).resolve().parents[1] / "evaluation" / "scoring_subtask_4.py"
    if not scorer_path.exists():
        raise FileNotFoundError(
            f"Official scorer not found at {scorer_path}. "
            "Add evaluation/scoring_subtask_4.py first."
        )

    spec = importlib.util.spec_from_file_location("scoring_subtask_4", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load scorer module from {scorer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SCORER_MODULE = module
    return module


def to_official_submission(submission_like: List[dict]) -> List[dict]:
    """
    Normalize submission objects to the official scorer format:
    [{"case_id": "...", "prediction": [{"answer_id": "...", "evidence_id": ["..."]}, ...]}, ...]
    """
    out: List[dict] = []

    for case in submission_like:
        cid = str(case["case_id"])

        if "prediction" in case:
            grouped: Dict[str, set] = {}
            for alignment in case.get("prediction", []):
                aid = str(alignment.get("answer_id", ""))
                if not aid:
                    continue
                for eid in alignment.get("evidence_id", []):
                    grouped.setdefault(aid, set()).add(str(eid))
        else:
            grouped = {}
            for pair in case.get("answer_to_evidence", []):
                aid = str(pair.get("answer_id", ""))
                eid = str(pair.get("evidence_id", ""))
                if aid and eid:
                    grouped.setdefault(aid, set()).add(eid)

        prediction = []
        for aid in sorted(grouped.keys(), key=_numeric_string_sort_key):
            eids = sorted(grouped[aid], key=_numeric_string_sort_key)
            prediction.append({"answer_id": aid, "evidence_id": eids})

        out.append({"case_id": cid, "prediction": prediction})

    out.sort(key=lambda x: _numeric_string_sort_key(str(x["case_id"])))
    return out


def score_links_official(
    submission_like: List[dict],
    key_path: str,
    case_ids_to_score: Optional[List[str]] = None,
) -> dict:
    scorer = _load_official_scorer_module()

    case_filter = {str(cid) for cid in case_ids_to_score} if case_ids_to_score else None
    submission = to_official_submission(submission_like)
    if case_filter:
        submission = [case for case in submission if case["case_id"] in case_filter]

    key_map = scorer.load_key(key_path, case_ids_to_score=case_filter)
    key_case_ids = set(key_map.keys())
    submission_case_ids = {case["case_id"] for case in submission}

    # Ensure all key cases are present when scoring (missing -> empty prediction).
    for cid in sorted(key_case_ids - submission_case_ids, key=_numeric_string_sort_key):
        submission.append({"case_id": cid, "prediction": []})
    submission = [case for case in submission if case["case_id"] in key_case_ids]
    submission.sort(key=lambda x: _numeric_string_sort_key(str(x["case_id"])))

    scores = scorer.compute_alignment_scores(submission, key_map)
    leaderboard = scorer.get_leaderboard(scores)
    return {"scores": scores, "leaderboard": leaderboard}
