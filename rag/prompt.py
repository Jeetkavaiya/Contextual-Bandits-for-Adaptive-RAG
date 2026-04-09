from __future__ import annotations

from typing import List, Tuple

from .types import Chunk


def format_chat_history(chat_history: List[Tuple[str, str]], max_chars: int = 6000) -> str:
    """
    chat_history: list of (role, text), role in {"user","assistant","system"}.
    Keeps the most recent content within max_chars.
    """
    lines: List[str] = []
    for role, text in chat_history:
        role_cap = role.strip().lower().capitalize()
        lines.append(f"{role_cap}: {text.strip()}")

    joined = "\n".join(lines).strip()
    if len(joined) <= max_chars:
        return joined

    # keep the end (most recent)
    return joined[-max_chars:]


def build_answer_prompt(raw_query: str, memory_text: str, context_chunks: List[Chunk]) -> str:
    """
    IMPORTANT RULE:
    - raw_query is shown to the answer model
    - refined query is retrieval-only (not shown here)
    """
    ctx_lines: List[str] = []
    for c in context_chunks:
        page = c.meta.get("page", None)
        header = f"[{c.chunk_id}]"
        if page is not None:
            header += f" (page {page})"
        ctx_lines.append(f"{header}\n{c.text}")

    context_block = "\n\n".join(ctx_lines).strip()

    prompt = f"""You are a helpful assistant. Use the provided context when it is relevant.
If the context is not relevant to the question, answer using general knowledge.

Conversation Memory:
{memory_text if memory_text.strip() else "(none)"}

User Question:
{raw_query}

Context:
{context_block if context_block.strip() else "(no retrieved context)"}

Answer:
- Be concise but complete.
- If you use any context, cite chunk ids like [c000123].
"""
    return prompt.strip()