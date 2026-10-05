import json
import time
from pathlib import Path

from secure_rag.classifiers.input import InputClassifier


ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"

DATASET_PATH = ROOT / "dataset" / "input.json"
RESULTS_PATH = RESULTS_DIR / "input_v1.json"
LABELS = [
    "benign",
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "indirect_injection",
]


def load_dataset() -> list[dict[str, str]]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    examples = data["examples"]

    if len(examples) != 200:
        raise ValueError(f"Expected 200 examples, found {len(examples)}")

    return examples


def main() -> None:
    examples = load_dataset()
    classifier = InputClassifier()

    results = []

    for index, example in enumerate(examples, start=1):
        text = example["text"]
        true_label = example["label"]

        start = time.perf_counter()

        try:
            prediction = classifier.classify(text)

            predicted_reasons = prediction.reasons
            flagged = prediction.flagged
            error = None

        except Exception as exc:
            predicted_reasons = []
            flagged = None
            error = str(exc)

        latency = time.perf_counter() - start

        result = {
            "id": example["id"],
            "text": text,
            "true_label": true_label,
            "flagged": flagged,
            "predicted_reasons": predicted_reasons,
            "latency_seconds": round(latency, 4),
            "error": error,
        }

        results.append(result)

        print(
            f"[{index:03d}/200] "
            f"{example['id']} -> "
            f"{predicted_reasons if flagged else 'benign'}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": "secure_rag_input_guard_v1",
        "model": classifier.model,
        "total_examples": len(results),
        "results": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()