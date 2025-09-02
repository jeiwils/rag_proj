"""Utility to enforce a simple grammar on LLM outputs."""

from __future__ import annotations

import re
from typing import List


def _extract_literals(grammar: str) -> List[str]:
    """Return all literal strings from a simple grammar definition.

    The grammar is expected to contain quoted literals separated by ``|``.  Only
    very small grammars are supported – this helper is intentionally minimal and
    is sufficient for the unit tests in this repository.
    """

    return re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', grammar)


def enforce_grammar(text: str, grammar: str) -> str:
    """Validate or rewrite ``text`` to satisfy ``grammar``.

    Parameters
    ----------
    text:
        The model response to validate.
    grammar:
        Simple grammar description containing quoted alternatives.

    Returns
    -------
    str
        A string that matches one of the grammar's literals.  If ``text`` does
        not match any literal exactly, the first literal appearing as a
        substring is returned.  As a final fallback the first literal in the
        grammar is returned.
    """

    literals = _extract_literals(grammar)
    if not literals:
        return text

    stripped = text.strip()
    for lit in literals:
        if stripped == lit:
            return lit
    for lit in literals:
        if lit in stripped:
            return lit
        num = re.search(r"\d+\.\d+", lit)
        if num and num.group(0) in stripped:
            return lit
    return literals[0]