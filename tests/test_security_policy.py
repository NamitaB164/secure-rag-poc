# pyrefly: ignore [missing-import]
import pytest

from secure_rag.classifiers.models import ClassificationResult
from secure_rag.security.policy import (
    SecurityDecision,
    evaluate_classification,
)


def test_clean_result_is_allowed():
    result = ClassificationResult(
        flagged=False,
        reasons=[],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.ALLOW


@pytest.mark.parametrize(
    "reason",
    [
        "pii",
        "policy_violation",
        "injected_instruction_following",
        "prompt_injection",
        "jailbreak",
        "system_prompt_extraction",
    ],
)
def test_security_reasons_are_blocked(reason):
    result = ClassificationResult(
        flagged=True,
        reasons=[reason],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.BLOCK
    assert reason in action.message


def test_ungrounded_claim_generates_warning():
    result = ClassificationResult(
        flagged=True,
        reasons=["ungrounded_claim"],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.WARN
    assert "grounded" in action.message


def test_multiple_reasons_with_blocking_reason_are_blocked():
    result = ClassificationResult(
        flagged=True,
        reasons=[
            "ungrounded_claim",
            "pii",
        ],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.BLOCK


def test_unknown_reason_fails_closed():
    result = ClassificationResult(
        flagged=True,
        reasons=["unknown_security_reason"],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.BLOCK


def test_flagged_without_reasons_is_allowed():
    result = ClassificationResult(
        flagged=True,
        reasons=[],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.ALLOW


def test_not_flagged_with_reason_is_allowed():
    result = ClassificationResult(
        flagged=False,
        reasons=["ungrounded_claim"],
    )

    action = evaluate_classification(result)

    assert action.decision == SecurityDecision.ALLOW