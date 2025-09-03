from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"

CS_GRAMMAR = r'''
root ::= "CS: 0.00" | "CS: 0.25" | "CS: 0.50" | "CS: 0.75" | "CS: 1.00"
'''

GRAMMAR_TRAVERSAL_INT_OR_NULL = (
    'root  ::= INT | NULL\n'
    'INT   ::= [0-9]+\n'
    'NULL  ::= "null"\n'
)

CS_PROMPT = (PROMPTS_DIR / "cs_prompt.txt").read_text(encoding="utf-8")
HOPRAG_IQ_PROMPT = (PROMPTS_DIR / "hoprag_iq_prompt.txt").read_text(encoding="utf-8")
HOPRAG_OQ_PROMPT = (PROMPTS_DIR / "hoprag_oq_prompt.txt").read_text(encoding="utf-8")

__all__ = [
    "CS_GRAMMAR",
    "GRAMMAR_TRAVERSAL_INT_OR_NULL",
    "CS_PROMPT",
    "HOPRAG_IQ_PROMPT",
    "HOPRAG_OQ_PROMPT",
]