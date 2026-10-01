# Secure RAG POC

A local, self-hosted RAG proof of concept with security controls for prompt injection, jailbreaks, system-prompt extraction, PII, policy violations, injected instructions, and ungrounded responses.

**Requirements**

- Python 3.11 or later
- [Ollama](https://ollama.com) running locally
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDF support)

**Constitutional Safety**

The security layer uses complementary few-shot in-context classification prompts for input and output classification. The prompts define security rules, harmless cases, and structured JSON outputs. This project uses prompt-based classification and does not implement fine-tuned constitutional classifier models.

**Input Classifier**

The input classifier evaluates user queries for:

- Prompt injection
- Jailbreaks
- System-prompt extraction

Flagged inputs can be blocked before retrieval and generation.

**Output Classifier**

The output classifier evaluates generated responses for:

- PII
- Policy violations
- Injected instruction following
- Ungrounded claims

Its structured output is used by the security policy to allow, warn, or block responses.

**Instruction Isolation**

Retrieved content is treated as untrusted data. The generation prompt separates retrieved context from model instructions to reduce the risk of indirect prompt injection.

**Retrieval Security**

Retrieval applies clearance-based access control and trust filtering before content reaches generation. Documents support text, Markdown, and PDF ingestion, including PDF tables, embedded images, and OCR for scanned pages.

**Audit Logging**

Security-relevant events are recorded in JSONL format, including the query, retrieved chunks, classifier results, security decision, and final response.

**Limitations**

This is a proof of concept. The classifiers are few-shot in-context prompts running on a general-purpose local LLM and may produce false positives or false negatives.

**Installation**

```
pip install -e ".[dev]"
```

**Running**

```
ollama pull llama3.2:3b
py -m secure_rag.cli
```

**Testing**

```
ruff check .
ruff format --check .
py -m pytest
```

To automatically format all files:

```
ruff format .
```

**References**

- [1] [Anthropic: Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers)
- [2] [Stanford Online — XACS134: AI Security](https://online.stanford.edu/)
