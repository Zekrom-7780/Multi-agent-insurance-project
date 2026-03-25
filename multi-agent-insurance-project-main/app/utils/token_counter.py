"""Token counting and truncation utilities using tiktoken."""

import tiktoken

# phi-3-mini doesn't have a dedicated tiktoken encoding,
# but cl100k_base provides a reasonable approximation.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*."""
    return len(_ENCODING.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens])
