
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from secure_rag.classifiers.models import ClassificationResult
from secure_rag.security.policy import SecurityAction


@dataclass
class AuditRecord:
    timestamp: str
    query: str
    user_id: str
    user_clearance: int
    retrieved_chunk_ids: list[str]
    input_classification: ClassificationResult
    output_classification: ClassificationResult
    security_decision: str
    security_message: str
    final_response: str


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(
        self,
        *,
        query: str,
        user_id: str,
        user_clearance: int,
        retrieved_chunk_ids: list[str],
        input_classification: ClassificationResult,
        output_classification: ClassificationResult,
        security_action: SecurityAction,
        final_response: str,
    ) -> AuditRecord:
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            user_id=user_id,
            user_clearance=user_clearance,
            retrieved_chunk_ids=retrieved_chunk_ids,
            input_classification=input_classification,
            output_classification=output_classification,
            security_decision=security_action.decision.value,
            security_message=security_action.message,
            final_response=final_response,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(record),
                file,
                ensure_ascii=False,
            )
            file.write("\n")

        return record
'''
Example:
{
  "timestamp": "2026-09-30T04:20:15.123456+00:00",
  "query": "What is the company's password policy?",
  "user_id": "user-001",
  "user_clearance": 2,
  "retrieved_chunk_ids": [
    "security-policy-chunk-0",
    "security-policy-chunk-1"
  ],
  "input_classification": {
    "flagged": false,
    "reasons": []
  },
  "output_classification": {
    "flagged": false,
    "reasons": []
  },
  "security_decision": "allow",
  "security_message": "Content passed security checks.",
  "final_response": "The password policy requires..."
}
'''