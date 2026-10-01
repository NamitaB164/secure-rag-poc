from dataclasses import dataclass

from secure_rag.classifiers.input import InputClassifier
from secure_rag.classifiers.models import ClassificationResult
from secure_rag.classifiers.output import OutputClassifier
from secure_rag.generation.generator import GroundedGenerator
from secure_rag.logging.audit import AuditLogger
from secure_rag.models import User
from secure_rag.retrieval.retriever import SecureRetriever
from secure_rag.security.policy import (
    SecurityAction,
    SecurityDecision,
    evaluate_classification,
)


@dataclass
class PipelineResult:
    response: str
    security_action: SecurityAction
    retrieved_chunk_ids: list[str]
    input_classification: ClassificationResult
    output_classification: ClassificationResult


class SecureRAGPipeline:
    def __init__(
        self,
        retriever: SecureRetriever,
        input_classifier: InputClassifier,
        generator: GroundedGenerator,
        output_classifier: OutputClassifier,
        audit_logger: AuditLogger,
        n_results: int = 5,
        security_enabled: bool = True,
    ) -> None:
        self.retriever = retriever
        self.input_classifier = input_classifier
        self.generator = generator
        self.output_classifier = output_classifier
        self.audit_logger = audit_logger
        self.n_results = n_results
        self.security_enabled = security_enabled

    def run(
        self,
        *,
        query: str,
        user: User,
    ) -> PipelineResult:

        # Security layer ON
        if self.security_enabled:
            input_classification = self.input_classifier.classify(query)
            input_action = evaluate_classification(input_classification)

            if input_action.decision == SecurityDecision.BLOCK:
                response = "Request blocked by security policy."

                self.audit_logger.log(
                    query=query,
                    user_id=user.id,
                    user_clearance=user.clearance,
                    retrieved_chunk_ids=[],
                    input_classification=input_classification,
                    output_classification=ClassificationResult(
                        flagged=False,
                    ),
                    security_action=input_action,
                    final_response=response,
                )

                return PipelineResult(
                    response=response,
                    security_action=input_action,
                    retrieved_chunk_ids=[],
                    input_classification=input_classification,
                    output_classification=ClassificationResult(
                        flagged=False,
                    ),
                )

        else:
            input_classification = ClassificationResult(flagged=False)

        # ACL retrieval ALWAYS happens.
        chunks = self.retriever.retrieve(
            query=query,
            user=user,
            n_results=self.n_results,
        )

        answer = self.generator.generate(
            query=query,
            chunks=chunks,
        )

        # Security layer ON
        if self.security_enabled:
            context = _build_context(chunks)

            output_classification = self.output_classifier.classify(
                answer=answer,
                context=context,
            )

            output_action = evaluate_classification(
                output_classification,
            )

            if output_action.decision == SecurityDecision.BLOCK:
                response = "Generated response blocked by security policy."

            elif output_action.decision == SecurityDecision.WARN:
                response = (
                    f"{answer}\n\n"
                    f"Security warning: {output_action.message}"
                )

            else:
                response = answer

        else:
            output_classification = ClassificationResult(
                flagged=False,
            )

            output_action = SecurityAction(
                decision=SecurityDecision.ALLOW,
                message="Security layer disabled.",
            )

            response = answer

        self.audit_logger.log(
            query=query,
            user_id=user.id,
            user_clearance=user.clearance,
            retrieved_chunk_ids=[chunk.id for chunk in chunks],
            input_classification=input_classification,
            output_classification=output_classification,
            security_action=output_action,
            final_response=response,
        )

        return PipelineResult(
            response=response,
            security_action=output_action,
            retrieved_chunk_ids=[chunk.id for chunk in chunks],
            input_classification=input_classification,
            output_classification=output_classification,
        )


def _build_context(chunks: list) -> str:
    if not chunks:
        return ""

    return "\n\n".join(
        f"[Chunk ID: {chunk.id}]\n{chunk.content}"
        for chunk in chunks
    )