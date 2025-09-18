"""Basic PII redaction for indexing."""

from __future__ import annotations

import re

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")


def redact_pii(text: str) -> str:
    """Mask common PII patterns before indexing."""

    text = EMAIL_PATTERN.sub("<EMAIL>", text)
    text = PHONE_PATTERN.sub("<PHONE>", text)
    text = DATE_PATTERN.sub("<DATE>", text)
    return text


__all__ = ["redact_pii"]
