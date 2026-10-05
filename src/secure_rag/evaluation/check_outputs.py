# pyrefly: ignore [missing-import]
import json
from pathlib import Path

from secure_rag.classifiers.input import InputClassifier

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "input_baseline.json"
OUTPUT_PATH = ROOT / "results" / "input_output_diagnostics.json"


def load_results() -> list[dict]:
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["results"]


def main() -> None:
    baseline_results = load_results()

    error_results = [
        result
        for result in baseline_results
        if result["error"] is not None
    ]

    classifier = InputClassifier()
    diagnostics = []

    print(f"Diagnosing {len(error_results)} classifier errors...")
    print()

    for index, result in enumerate(error_results, start=1):
        raw_response = None
        parse_error = None

        try:
            prediction = classifier.classify(result["text"])

            raw_response = classifier.last_raw_response

            diagnostics.append(
                {
                    "id": result["id"],
                    "true_label": result["true_label"],
                    "text": result["text"],
                    "raw_response": raw_response,
                    "parsed_flagged": prediction.flagged,
                    "parsed_reasons": prediction.reasons,
                    "parse_error": None,
                }
            )

        except Exception as exc:
            raw_response = classifier.last_raw_response
            parse_error = str(exc)

            diagnostics.append(
                {
                    "id": result["id"],
                    "true_label": result["true_label"],
                    "text": result["text"],
                    "raw_response": raw_response,
                    "parsed_flagged": None,
                    "parsed_reasons": [],
                    "parse_error": parse_error,
                }
            )

        print(
            f"[{index:02d}/{len(error_results):02d}] "
            f"{result['id']} -> "
            f"{'OK' if parse_error is None else 'ERROR'}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": "secure_rag_input_guard_v1",
        "model": classifier.model,
        "source_baseline": "input_baseline.json",
        "diagnosed_examples": len(diagnostics),
        "results": diagnostics,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print(f"Saved diagnostics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()