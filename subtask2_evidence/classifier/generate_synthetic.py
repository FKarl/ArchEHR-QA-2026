"""
Synthetic Data Generation for Subtask 2: Evidence Identification

This script uses a local LLM (via Ollama) to generate synthetic training data
for improving the BERT-based evidence identification model.

The pipeline:
1. Generate synthetic variations using the LLM
2. Apply heuristic quality filters
3. Apply LLM-based quality filter
"""

import json
import sys
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Add LLM_inference to path for the client
sys.path.insert(0, str(Path(__file__).parent.parent / "LLM_inference"))

from llm_client import LLMClient
import re


def repair_json(text: str) -> str:
    """Attempt to repair common JSON issues from LLM output."""
    # Remove any trailing incomplete content after the last complete structure
    text = text.strip()

    # Try to find the outermost JSON object
    if not text.startswith("{"):
        # Try to find JSON start
        match = re.search(r"\{", text)
        if match:
            text = text[match.start() :]

    # Count braces and brackets to find complete JSON
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape = False
    last_valid_pos = 0

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i + 1
                break
        elif char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1

    if last_valid_pos > 0:
        return text[:last_valid_pos]

    # If we couldn't find complete JSON, try to close it
    # Remove any unterminated string at the end
    if in_string:
        # Find the last quote and truncate there, then close
        last_quote = text.rfind('"')
        if last_quote > 0:
            # Check if this quote is escaped
            num_backslashes = 0
            pos = last_quote - 1
            while pos >= 0 and text[pos] == "\\":
                num_backslashes += 1
                pos -= 1
            if num_backslashes % 2 == 0:  # Not escaped
                text = text[: last_quote + 1]

    # Close any open structures
    text = text.rstrip(",\n\t ")

    # Count remaining open braces/brackets
    brace_count = text.count("{") - text.count("}")
    bracket_count = text.count("[") - text.count("]")

    # Close them
    text += "]" * bracket_count + "}" * brace_count

    return text


# LLM settings
OLLAMA_MODEL = "llama3.1:70b"  # Use a fast model for generation
TEMPERATURE = 0.8  # Higher for more diversity
MAX_TOKENS = 4096  # Large enough for 15-30 sentence cases

# Generation settings
NUM_SYNTHETIC_PER_CASE = 5  # How many synthetic examples per real case
MIN_SENTENCES = 10
MAX_SENTENCES = 20

# Quality thresholds
MIN_SENTENCE_LENGTH = 10
MAX_SENTENCE_LENGTH = 500
MIN_ESSENTIAL_RATIO = 0.10  # At least 10% essential
MAX_ESSENTIAL_RATIO = 0.40  # At most 40% essential (real ≈ 28%)
MIN_NOT_RELEVANT_RATIO = 0.45  # At least 45% not-relevant (real ≈ 66%)
MAX_SUPPLEMENTARY_RATIO = 0.15  # At most 15% supplementary (real ≈ 6%)


@dataclass
class SyntheticCase:
    """A synthetic training case."""

    case_id: str
    patient_question: str
    clinician_question: str
    sentences: List[Dict[str, str]]  # [{"id": "1", "text": "..."}]
    relevance_labels: List[
        Dict[str, str]
    ]  # [{"sentence_id": "1", "relevance": "essential"}]
    source_case_id: str  # Original case this was based on
    generation_method: str


# PROMPTS

NUM_FEW_SHOT_EXAMPLES = 3  # Number of real examples to show in the prompt

GENERATION_PROMPT = """You are a medical data augmentation assistant. Your task is to create a synthetic clinical note excerpt with relevance labels for training a medical evidence identification model.

TASK: Study the real examples below, then generate a NEW synthetic case with DIFFERENT medical content (different condition, treatment, or scenario). Pay close attention to the label distributions in the examples — most sentences are "not-relevant".

{examples_block}

INSTRUCTIONS:
1. Create a NEW patient question about a DIFFERENT medical topic from any of the examples above
2. Create a corresponding clinician question (concise, max 15 words)
3. Write a LONG clinical note excerpt with 15-30 sentences. Real clinical notes are long and contain many sections (History of Present Illness, Hospital Course, Medications, Labs, Vitals, Plan, etc.). Include realistic clinical detail.
4. Label each sentence as "essential", "supplementary", or "not-relevant":
   - essential: Directly answers the patient's question (~20-35% of sentences)
   - supplementary: Provides helpful context but not critical (~5-10% of sentences, this label is RARE)
   - not-relevant: Does NOT help answer the question (~55-70% of sentences — this should be the MAJORITY)

IMPORTANT DISTRIBUTION GUIDANCE:
- Most sentences in a clinical note are NOT relevant to the patient's specific question. The note covers many topics (vitals, labs, medications, procedures, consults, discharge planning) and only a fraction directly address the question.
- "supplementary" is the RAREST label — only a few sentences at most.
- "not-relevant" should be the MOST COMMON label — typically 60%+ of sentences.
- Look at the examples above: notice how the majority of sentences are labeled "not-relevant".

OUTPUT FORMAT (use exactly this JSON format):
{{
    "patient_question": "...",
    "clinician_question": "...",
    "sentences": [
        {{"id": "1", "text": "..."}},
        {{"id": "2", "text": "..."}}
    ],
    "relevance_labels": [
        {{"sentence_id": "1", "relevance": "essential"}},
        {{"sentence_id": "2", "relevance": "not-relevant"}}
    ]
}}

Generate a synthetic case now. Output ONLY valid JSON, no explanation."""


QUALITY_CHECK_PROMPT = """You are a quality assurance expert for medical NLP training data.

Evaluate this synthetic training example for evidence identification:

Patient Question: {patient_question}
Clinician Question: {clinician_question}

Sentences and Labels:
{sentences_with_labels}

EVALUATION CRITERIA:
1. Medical plausibility: Is the clinical content realistic?
2. Question coherence: Does the question make sense? Is it specific?
3. Label accuracy: Are the relevance labels appropriate?
   - "essential" sentences should directly answer the question
   - "supplementary" sentences should provide context
   - "not-relevant" sentences should not help answer the question
4. Diversity: Is this different from typical training examples?

Rate the overall quality as: ACCEPT or REJECT
If REJECT, briefly explain why.

OUTPUT FORMAT:
DECISION: [ACCEPT/REJECT]
REASON: [brief explanation if rejected]"""


REPAIR_PROMPT = """You are a medical data quality assistant. Fix the following synthetic clinical case that was rejected.

ORIGINAL CASE:
Patient Question: {patient_question}
Clinician Question: {clinician_question}
Sentences:
{sentences_formatted}
Labels:
{labels_formatted}

REJECTION REASON: {rejection_reason}

FIX INSTRUCTIONS:
1. Address the specific rejection reason above
2. Keep the medical topic the same, but fix the issues
3. Write 10-20 sentences with proper relevance labels
4. Labels must be: "essential", "supplementary", or "not-relevant"
5. DISTRIBUTION: ~20-35% essential, ~5-10% supplementary, ~55-70% not-relevant
6. "not-relevant" should be the MAJORITY of sentences
7. "supplementary" should be the RAREST label

OUTPUT FORMAT (use exactly this JSON format):
{{
    "patient_question": "...",
    "clinician_question": "...",
    "sentences": [
        {{"id": "1", "text": "..."}},
        {{"id": "2", "text": "..."}}
    ],
    "relevance_labels": [
        {{"sentence_id": "1", "relevance": "essential"}},
        {{"sentence_id": "2", "relevance": "not-relevant"}}
    ]
}}

Output ONLY valid JSON, no explanation."""


def load_real_examples(data_dir: Path) -> List[Dict]:
    """Load real examples from the dev set."""
    import xml.etree.ElementTree as ET

    xml_path = data_dir / "dev" / "archehr-qa.xml"
    key_path = data_dir / "dev" / "archehr-qa_key.json"

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cases_data = {}
    for case in root.findall("case"):
        case_id = case.get("id")

        patient_narrative = case.find("patient_narrative")
        patient_text = (
            patient_narrative.text.strip()
            if patient_narrative is not None
            else ""
        )

        clinician_question = case.find("clinician_question")
        clinician_text = (
            clinician_question.text.strip()
            if clinician_question is not None
            else ""
        )

        sentences = []
        note_sentences = case.find("note_excerpt_sentences")
        if note_sentences is not None:
            for sent in note_sentences.findall("sentence"):
                sent_id = sent.get("id")
                sent_text = sent.text.strip() if sent.text else ""
                sentences.append({"id": sent_id, "text": sent_text})

        cases_data[case_id] = {
            "patient_question": patient_text,
            "clinician_question": clinician_text,
            "sentences": sentences,
        }

    # Parse labels
    with open(key_path, "r") as f:
        labels_data = json.load(f)

    # Combine
    examples = []
    for case in labels_data:
        case_id = case["case_id"]
        if case_id in cases_data:
            examples.append(
                {
                    "case_id": case_id,
                    "patient_question": cases_data[case_id][
                        "patient_question"
                    ],
                    "clinician_question": cases_data[case_id][
                        "clinician_question"
                    ],
                    "sentences": cases_data[case_id]["sentences"],
                    "relevance_labels": case["answers"],
                }
            )

    return examples


def format_example_for_prompt(example: Dict) -> Tuple[str, str]:
    """Format an example for the generation prompt."""
    sentences_formatted = "\n".join(
        f"{s['id']}: {s['text']}" for s in example["sentences"]
    )

    labels_formatted = "\n".join(
        f"Sentence {l['sentence_id']}: {l['relevance']}"
        for l in example["relevance_labels"]
    )

    return sentences_formatted, labels_formatted


def format_examples_block(
    source_example: Dict,
    all_examples: List[Dict],
    num_examples: int = NUM_FEW_SHOT_EXAMPLES,
) -> str:
    """Format multiple real examples into a block for the generation prompt.

    Always includes the source_example, then samples additional examples
    randomly from the rest to provide diverse few-shot demonstrations.
    """
    # Start with the source example, then sample others
    others = [
        ex for ex in all_examples if ex["case_id"] != source_example["case_id"]
    ]
    num_extra = min(num_examples - 1, len(others))
    sampled = random.sample(others, num_extra) if num_extra > 0 else []
    examples_to_show = [source_example] + sampled
    random.shuffle(examples_to_show)  # shuffle so source isn't always first

    blocks = []
    for i, ex in enumerate(examples_to_show, 1):
        sents_fmt, labels_fmt = format_example_for_prompt(ex)
        blocks.append(
            f"--- EXAMPLE {i} ---\n"
            f"Patient Question: {ex['patient_question']}\n"
            f"Clinician Question: {ex['clinician_question']}\n"
            f"Clinical Note Sentences:\n{sents_fmt}\n"
            f"Relevance Labels:\n{labels_fmt}"
        )
    return "\n\n".join(blocks)


def generate_synthetic_case(
    client: LLMClient,
    source_example: Dict,
    case_id: str,
    all_examples: List[Dict],
    max_retries: int = 2,
) -> Optional[SyntheticCase]:
    """Generate a single synthetic case based on multiple real examples."""

    examples_block = format_examples_block(source_example, all_examples)

    prompt = GENERATION_PROMPT.format(examples_block=examples_block)

    for attempt in range(max_retries + 1):
        try:
            result = client.generate(
                prompt=prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
            )

            # Parse JSON from response
            response_text = result.text.strip()

            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[
                    0
                ]
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1]

            response_text = response_text.strip()

            # First attempt: try direct parsing
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Second attempt: try to repair the JSON
                repaired = repair_json(response_text)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        continue  # Retry with fresh generation
                    raise e

            # Validate required fields
            if not all(
                key in data
                for key in [
                    "patient_question",
                    "clinician_question",
                    "sentences",
                    "relevance_labels",
                ]
            ):
                if attempt < max_retries:
                    continue
                raise KeyError("Missing required fields in response")

            # Validate sentences and labels are lists
            if not isinstance(data["sentences"], list) or not isinstance(
                data["relevance_labels"], list
            ):
                if attempt < max_retries:
                    continue
                raise ValueError(
                    "sentences and relevance_labels must be lists"
                )

            if len(data["sentences"]) == 0:
                if attempt < max_retries:
                    continue
                raise ValueError("No sentences generated")

            return SyntheticCase(
                case_id=case_id,
                patient_question=data["patient_question"],
                clinician_question=data["clinician_question"],
                sentences=data["sentences"],
                relevance_labels=data["relevance_labels"],
                source_case_id=source_example["case_id"],
                generation_method="llm_generation",
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == max_retries:
                print(
                    f"  Failed to parse LLM response after {max_retries + 1} attempts: {e}"
                )
            return None
        except Exception as e:
            print(f"  LLM generation error: {e}")
            return None

    return None


def heuristic_quality_filter(case: SyntheticCase) -> Tuple[bool, str]:
    """
    Apply heuristic quality checks to a synthetic case.

    Returns: (passed, reason)
    """
    # Check number of sentences
    num_sentences = len(case.sentences)
    if num_sentences < MIN_SENTENCES:
        return False, f"Too few sentences ({num_sentences} < {MIN_SENTENCES})"
    if num_sentences > MAX_SENTENCES:
        return False, f"Too many sentences ({num_sentences} > {MAX_SENTENCES})"

    # Check sentence lengths
    for sent in case.sentences:
        length = len(sent.get("text", ""))
        if length < MIN_SENTENCE_LENGTH:
            return False, f"Sentence {sent['id']} too short ({length} chars)"
        if length > MAX_SENTENCE_LENGTH:
            return False, f"Sentence {sent['id']} too long ({length} chars)"

    # Check relevance label distribution
    labels = [l["relevance"] for l in case.relevance_labels]
    valid_labels = {"essential", "supplementary", "not-relevant"}

    for label in labels:
        if label not in valid_labels:
            return False, f"Invalid label: {label}"

    # Check label counts match sentence counts
    if len(case.relevance_labels) != len(case.sentences):
        return False, "Label count doesn't match sentence count"

    # Check essential ratio
    essential_count = sum(1 for l in labels if l == "essential")
    essential_ratio = essential_count / len(labels) if labels else 0

    if essential_ratio < MIN_ESSENTIAL_RATIO:
        return False, f"Too few essential sentences ({essential_ratio:.1%})"
    if essential_ratio > MAX_ESSENTIAL_RATIO:
        return False, f"Too many essential sentences ({essential_ratio:.1%})"

    # Check not-relevant ratio (should be the majority)
    nr_count = sum(1 for l in labels if l == "not-relevant")
    nr_ratio = nr_count / len(labels) if labels else 0
    if nr_ratio < MIN_NOT_RELEVANT_RATIO:
        return (
            False,
            f"Too few not-relevant sentences ({nr_ratio:.1%}, need ≥{MIN_NOT_RELEVANT_RATIO:.0%})",
        )

    # Check supplementary ratio (should be rare)
    sup_count = sum(1 for l in labels if l == "supplementary")
    sup_ratio = sup_count / len(labels) if labels else 0
    if sup_ratio > MAX_SUPPLEMENTARY_RATIO:
        return (
            False,
            f"Too many supplementary sentences ({sup_ratio:.1%}, max {MAX_SUPPLEMENTARY_RATIO:.0%})",
        )

    # Check there's at least one not-relevant
    if "not-relevant" not in labels:
        return False, "No not-relevant sentences (unrealistic)"

    # Check question lengths
    if len(case.patient_question) < 20:
        return False, "Patient question too short"
    if len(case.clinician_question) < 10:
        return False, "Clinician question too short"
    if len(case.clinician_question) > 150:
        return False, "Clinician question too long (should be concise)"

    return True, "Passed all heuristic checks"


def llm_quality_filter(
    client: LLMClient, case: SyntheticCase
) -> Tuple[bool, str]:
    """
    Use LLM to evaluate the quality of a synthetic case.

    Returns: (passed, reason)
    """
    # Format sentences with labels
    sentences_with_labels = "\n".join(
        f"[{label['relevance'].upper()}] {sent['text']}"
        for sent, label in zip(case.sentences, case.relevance_labels)
    )

    prompt = QUALITY_CHECK_PROMPT.format(
        patient_question=case.patient_question,
        clinician_question=case.clinician_question,
        sentences_with_labels=sentences_with_labels,
    )

    try:
        result = client.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,  # Low temperature for consistent evaluation
        )

        response = result.text.strip().upper()

        # Parse decision
        if (
            "DECISION: ACCEPT" in response
            or "ACCEPT" in response.split("\n")[0]
        ):
            return True, "LLM approved"
        else:
            # Try to extract reason
            reason = "LLM rejected"
            if "REASON:" in result.text:
                reason = result.text.split("REASON:")[1].strip()[:100]
            return False, reason

    except Exception as e:
        # If LLM check fails, be conservative and accept
        print(f"  LLM quality check failed: {e}")
        return True, "LLM check skipped (error)"


def repair_synthetic_case(
    client: LLMClient,
    case: SyntheticCase,
    rejection_reason: str,
    max_retries: int = 2,
) -> Optional[SyntheticCase]:
    """
    Attempt to repair a rejected synthetic case using the LLM.

    Args:
        client: LLM client
        case: The rejected case to repair
        rejection_reason: Why the case was rejected
        max_retries: Number of repair attempts

    Returns:
        Repaired SyntheticCase or None if repair failed
    """
    # Format the case for the repair prompt
    sentences_formatted = "\n".join(
        f"{s['id']}: {s.get('text', '')}" for s in case.sentences
    )
    labels_formatted = "\n".join(
        f"Sentence {l['sentence_id']}: {l['relevance']}"
        for l in case.relevance_labels
    )

    prompt = REPAIR_PROMPT.format(
        patient_question=case.patient_question,
        clinician_question=case.clinician_question,
        sentences_formatted=sentences_formatted,
        labels_formatted=labels_formatted,
        rejection_reason=rejection_reason,
    )

    for attempt in range(max_retries + 1):
        try:
            result = client.generate(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.5,  # Lower temp for repairs
            )

            response_text = result.text.strip()

            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[
                    0
                ]
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1]

            response_text = response_text.strip()

            # Try to parse JSON
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                repaired = repair_json(response_text)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    if attempt < max_retries:
                        continue
                    return None

            # Validate required fields
            if not all(
                key in data
                for key in [
                    "patient_question",
                    "clinician_question",
                    "sentences",
                    "relevance_labels",
                ]
            ):
                if attempt < max_retries:
                    continue
                return None

            if not isinstance(data["sentences"], list) or not isinstance(
                data["relevance_labels"], list
            ):
                if attempt < max_retries:
                    continue
                return None

            if len(data["sentences"]) == 0:
                if attempt < max_retries:
                    continue
                return None

            return SyntheticCase(
                case_id=case.case_id,
                patient_question=data["patient_question"],
                clinician_question=data["clinician_question"],
                sentences=data["sentences"],
                relevance_labels=data["relevance_labels"],
                source_case_id=case.source_case_id,
                generation_method="llm_repair",
            )

        except Exception as e:
            if attempt == max_retries:
                return None

    return None


def generate_synthetic_dataset(
    data_dir: Path,
    output_path: Path,
    num_per_case: int = NUM_SYNTHETIC_PER_CASE,
    use_llm_filter: bool = True,
    model: str = OLLAMA_MODEL,
    max_repair_attempts: int = 3,
) -> List[SyntheticCase]:
    """
    Generate a synthetic dataset for training.

    Args:
        data_dir: Path to data directory
        output_path: Path to save synthetic data
        num_per_case: Number of synthetic cases per real case
        use_llm_filter: Whether to use LLM quality filter
        model: Ollama model to use
        max_repair_attempts: Max attempts to repair rejected cases

    Returns:
        List of accepted synthetic cases
    """
    print("=" * 60)
    print("Synthetic Data Generation for Evidence Identification")
    print("=" * 60)

    # Initialize LLM client
    print(f"\nInitializing LLM client (model: {model})...")
    client = LLMClient(backend="ollama", model=model)

    # Load real examples
    print("\nLoading real examples...")
    real_examples = load_real_examples(data_dir)
    print(f"Loaded {len(real_examples)} real cases")

    target_total = len(real_examples) * num_per_case
    print(
        f"Target: {target_total} synthetic cases ({num_per_case} per real case)"
    )

    # Generate synthetic cases
    synthetic_cases = []
    stats = {
        "generated": 0,
        "heuristic_rejected": 0,
        "llm_rejected": 0,
        "repaired": 0,
        "repair_failed": 0,
        "accepted": 0,
    }

    print(f"\nGenerating synthetic cases...")

    for example in tqdm(real_examples, desc="Processing cases"):
        cases_for_this_example = 0
        attempt_num = 0

        while cases_for_this_example < num_per_case:
            attempt_num += 1
            case_id = f"syn_{example['case_id']}_{cases_for_this_example + 1}"

            # Generate
            synthetic = generate_synthetic_case(
                client, example, case_id, all_examples=real_examples
            )
            if synthetic is None:
                if attempt_num > num_per_case * 3:  # Avoid infinite loops
                    break
                continue

            stats["generated"] += 1
            rejection_reason = None

            # Heuristic filter
            passed, reason = heuristic_quality_filter(synthetic)
            if not passed:
                stats["heuristic_rejected"] += 1
                rejection_reason = reason

            # LLM filter (optional) - only if passed heuristic
            if passed and use_llm_filter:
                passed, reason = llm_quality_filter(client, synthetic)
                if not passed:
                    stats["llm_rejected"] += 1
                    rejection_reason = reason

            # If rejected, try to repair
            if not passed and rejection_reason:
                for repair_attempt in range(max_repair_attempts):
                    repaired = repair_synthetic_case(
                        client, synthetic, rejection_reason
                    )
                    if repaired is None:
                        continue

                    # Re-check the repaired case
                    passed, reason = heuristic_quality_filter(repaired)
                    if not passed:
                        rejection_reason = reason
                        synthetic = repaired  # Use repaired version for next repair attempt
                        continue

                    if use_llm_filter:
                        passed, reason = llm_quality_filter(client, repaired)
                        if not passed:
                            rejection_reason = reason
                            synthetic = repaired
                            continue

                    # Repair succeeded!
                    synthetic = repaired
                    stats["repaired"] += 1
                    break

                if not passed:
                    stats["repair_failed"] += 1
                    if attempt_num > num_per_case * 3:
                        break
                    continue

            stats["accepted"] += 1
            synthetic_cases.append(synthetic)
            cases_for_this_example += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    print(f"Target:             {target_total}")
    print(f"Generated:          {stats['generated']}")
    print(f"Heuristic rejected: {stats['heuristic_rejected']}")
    print(f"LLM rejected:       {stats['llm_rejected']}")
    print(f"Repaired:           {stats['repaired']}")
    print(f"Repair failed:      {stats['repair_failed']}")
    print(f"Accepted:           {stats['accepted']}")
    print(
        f"Acceptance rate:    {stats['accepted']/max(stats['generated'], 1):.1%}"
    )
    print(
        f"Target reached:     {stats['accepted']}/{target_total} ({stats['accepted']/max(target_total, 1):.1%})"
    )

    # Save to file
    print(
        f"\nSaving {len(synthetic_cases)} synthetic cases to {output_path}..."
    )

    output_data = [asdict(case) for case in synthetic_cases]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")

    return synthetic_cases


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for evidence identification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./synthetic_data/synthetic_train.json",
        help="Path to save synthetic data",
    )
    parser.add_argument(
        "--num_per_case",
        type=int,
        default=3,
        help="Number of synthetic cases per real case",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2:3b", help="Ollama model to use"
    )
    parser.add_argument(
        "--no_llm_filter",
        action="store_true",
        help="Disable LLM-based quality filter",
    )
    parser.add_argument(
        "--max_repairs",
        type=int,
        default=3,
        help="Maximum repair attempts per rejected case",
    )

    args = parser.parse_args()

    generate_synthetic_dataset(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        num_per_case=args.num_per_case,
        use_llm_filter=not args.no_llm_filter,
        model=args.model,
        max_repair_attempts=args.max_repairs,
    )


if __name__ == "__main__":
    main()
