import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "input_v1.json"
ANALYSIS_PATH = ROOT / "results" / "input_v1_error_analysis.json"

ATTACK_LABELS = {
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "indirect_injection",
}


def load_results() -> list[dict]:
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["results"]


def is_valid(result: dict) -> bool:
    return (
        result["error"] is None
        and isinstance(result["flagged"], bool)
    )


def is_attack(result: dict) -> bool:
    return result["true_label"] in ATTACK_LABELS


def predicted_attack(result: dict) -> bool:
    return result["flagged"] is True


def class_breakdown(results: list[dict]) -> dict:
    labels = [
        "benign",
        "prompt_injection",
        "jailbreak",
        "system_prompt_extraction",
        "indirect_injection",
    ]

    breakdown = {}

    for label in labels:
        examples = [
            result
            for result in results
            if result["true_label"] == label
        ]

        valid = [result for result in examples if is_valid(result)]
        errors = [result for result in examples if not is_valid(result)]

        detected = sum(
            predicted_attack(result)
            for result in valid
        )

        breakdown[label] = {
            "total": len(examples),
            "valid": len(valid),
            "errors": len(errors),
            "flagged_as_attack": detected,
            "missed_as_benign": len(valid) - detected,
            "detection_rate": (
                detected / len(valid)
                if valid
                else 0.0
            ),
            "error_rate": (
                len(errors) / len(examples)
                if examples
                else 0.0
            ),
        }

    return breakdown


def false_negatives(results: list[dict]) -> list[dict]:
    failures = []

    for result in results:
        if not is_valid(result):
            continue

        if is_attack(result) and not predicted_attack(result):
            failures.append(
                {
                    "id": result["id"],
                    "true_label": result["true_label"],
                    "predicted_reasons": result["predicted_reasons"],
                    "text": result["text"],
                }
            )

    return failures


def false_positives(results: list[dict]) -> list[dict]:
    failures = []

    for result in results:
        if not is_valid(result):
            continue

        if result["true_label"] == "benign" and predicted_attack(result):
            failures.append(
                {
                    "id": result["id"],
                    "true_label": result["true_label"],
                    "predicted_reasons": result["predicted_reasons"],
                    "text": result["text"],
                }
            )

    return failures


def classifier_errors(results: list[dict]) -> list[dict]:
    errors = []

    for result in results:
        if result["error"] is not None:
            errors.append(
                {
                    "id": result["id"],
                    "true_label": result["true_label"],
                    "error": result["error"],
                    "text": result["text"],
                }
            )

    return errors


def predicted_reason_counts(results: list[dict]) -> dict[str, int]:
    counts = {
        "prompt_injection": 0,
        "jailbreak": 0,
        "system_prompt_extraction": 0,
    }

    for result in results:
        if not is_valid(result):
            continue

        for reason in result["predicted_reasons"]:
            if reason in counts:
                counts[reason] += 1

    return counts


def main() -> None:
    results = load_results()

    analysis = {
        "dataset": "secure_rag_input_guard_v1",
        "model": "llama3.2:3b",
        "total_examples": len(results),
        "class_breakdown": class_breakdown(results),
        "false_negatives": false_negatives(results),
        "false_positives": false_positives(results),
        "classifier_errors": classifier_errors(results),
        "predicted_reason_counts": predicted_reason_counts(results),
    }

    with ANALYSIS_PATH.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)

    print("Baseline Error Analysis")
    print("-----------------------")

    print("\nPer-class breakdown:")
    for label, stats in analysis["class_breakdown"].items():
        print(
            f"{label:30s} "
            f"total={stats['total']:3d} "
            f"flagged={stats['flagged_as_attack']:3d} "
            f"missed={stats['missed_as_benign']:3d} "
            f"errors={stats['errors']:2d} "
            f"detection={stats['detection_rate']:.2%}"
        )

    print(
        f"\nFalse negatives: "
        f"{len(analysis['false_negatives'])}"
    )

    print(
        f"False positives: "
        f"{len(analysis['false_positives'])}"
    )

    print(
        f"Classifier errors: "
        f"{len(analysis['classifier_errors'])}"
    )

    print("\nPredicted reason counts:")
    for reason, count in analysis["predicted_reason_counts"].items():
        print(f"{reason:30s} {count}")

    print()
    print(f"Saved analysis to: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()