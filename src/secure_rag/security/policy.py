from dataclasses import dataclass
from enum import Enum

from secure_rag.classifiers.models import ClassificationResult


class SecurityDecision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class SecurityAction:
    decision: SecurityDecision
    message: str


_BLOCKING_REASONS = {
    "pii",
    "policy_violation",
    "injected_instruction_following",
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
}


def evaluate_classification(
    result: ClassificationResult,
) -> SecurityAction:
    if not result.flagged or not result.reasons:
        return SecurityAction(
            decision=SecurityDecision.ALLOW,
            message="Content passed security checks.",
        )

    blocking_reasons = [
        reason
        for reason in result.reasons
        if reason in _BLOCKING_REASONS
    ]

    if blocking_reasons:
        return SecurityAction(
            decision=SecurityDecision.BLOCK,
            message=(
                "Content was blocked because it triggered security "
                "controls: "
                + ", ".join(blocking_reasons)
            ),
        )

    if "ungrounded_claim" in result.reasons:
        return SecurityAction(
            decision=SecurityDecision.WARN,
            message=(
                "The generated answer contains claims that could not "
                "be adequately grounded in the retrieved context."
            ),
        )

    return SecurityAction(
        decision=SecurityDecision.BLOCK,
        message="Content was blocked because it triggered a security control.",
    )