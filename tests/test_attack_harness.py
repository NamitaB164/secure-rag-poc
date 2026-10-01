# pyrefly: ignore [missing-import]
import pytest

from secure_rag.classifiers.models import ClassificationResult
from secure_rag.logging.audit import AuditLogger
from secure_rag.models import User
from secure_rag.pipeline import SecureRAGPipeline
from secure_rag.security.policy import SecurityDecision
# pyrefly: ignore [missing-import]
from tests.fixtures.attack import INPUT_ATTACKS, OUTPUT_ATTACKS


class FakeInputClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        return self.result


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def retrieve(self, query, user, n_results):
        self.calls.append((query, user, n_results))
        return self.chunks


class FakeGenerator:
    def __init__(self, answer="Generated answer."):
        self.answer = answer
        self.calls = []

    def generate(self, query, chunks):
        self.calls.append((query, chunks))
        return self.answer


class FakeOutputClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify(self, answer, context):
        self.calls.append((answer, context))
        return self.result


def make_pipeline(
    tmp_path,
    *,
    input_result,
    output_result,
):
    input_classifier = FakeInputClassifier(input_result)
    retriever = FakeRetriever()
    generator = FakeGenerator()
    output_classifier = FakeOutputClassifier(output_result)

    pipeline = SecureRAGPipeline(
        retriever=retriever,
        input_classifier=input_classifier,
        generator=generator,
        output_classifier=output_classifier,
        audit_logger=AuditLogger(
            tmp_path / "audit.jsonl",
        ),
    )

    return (
        pipeline,
        input_classifier,
        retriever,
        generator,
        output_classifier,
    )


@pytest.mark.parametrize(
    "attack",
    INPUT_ATTACKS,
    ids=lambda attack: attack.name,
)
def test_input_attack_is_blocked_before_retrieval(
    tmp_path,
    attack,
):
    pipeline, input_classifier, retriever, generator, output_classifier = (
        make_pipeline(
            tmp_path,
            input_result=ClassificationResult(
                flagged=True,
                reasons=list(attack.expected_reasons),
            ),
            output_result=ClassificationResult(
                flagged=False,
            ),
        )
    )

    result = pipeline.run(
        query=attack.text,
        user=User(
            id="user-001",
            clearance=1,
        ),
    )

    assert result.security_action.decision == SecurityDecision.BLOCK
    assert result.response == "Request blocked by security policy."

    assert input_classifier.calls == [attack.text]
    assert retriever.calls == []
    assert generator.calls == []
    assert output_classifier.calls == []

    assert result.retrieved_chunk_ids == []


@pytest.mark.parametrize(
    "attack",
    OUTPUT_ATTACKS,
    ids=lambda attack: attack.name,
)
def test_output_attack_is_handled_after_generation(
    tmp_path,
    attack,
):
    pipeline, input_classifier, retriever, generator, output_classifier = (
        make_pipeline(
            tmp_path,
            input_result=ClassificationResult(
                flagged=False,
            ),
            output_result=ClassificationResult(
                flagged=True,
                reasons=list(attack.expected_reasons),
            ),
        )
    )

    result = pipeline.run(
        query="What information is available?",
        user=User(
            id="user-001",
            clearance=1,
        ),
    )

    assert input_classifier.calls == [
        "What information is available?",
    ]

    assert len(retriever.calls) == 1
    assert len(generator.calls) == 1
    assert len(output_classifier.calls) == 1

    if "ungrounded_claim" in attack.expected_reasons:
        assert result.security_action.decision == SecurityDecision.WARN
        assert "Security warning:" in result.response
    else:
        assert result.security_action.decision == SecurityDecision.BLOCK
        assert result.response == (
            "Generated response blocked by security policy."
        )