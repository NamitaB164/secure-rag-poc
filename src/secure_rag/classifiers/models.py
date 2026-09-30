from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)