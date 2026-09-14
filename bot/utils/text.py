class TextSplitter:
    """
    Utility class for safely splitting long text into chunks that satisfy
    Discord's maximum message length constraints.
    """

    def __init__(self, default_limit: int = 1950) -> None:
        self.default_limit = default_limit

    def split(self, text: str, limit: int | None = None) -> list[str]:
        """
        Splits long text into chunks below the specified character limit,
        preferring clean breaks at newlines, then whitespace, and finally hard slices.
        """
        max_len = limit if limit is not None else self.default_limit

        if not text:
            return []

        if len(text) <= max_len:
            return [text]

        chunks: list[str] = []
        remaining = text

        while len(remaining) > max_len:
            # Look for the last newline within the allowed length
            split_idx = remaining.rfind("\n", 0, max_len)
            if split_idx == -1:
                # If no newline exists, look for the last space
                split_idx = remaining.rfind(" ", 0, max_len)
            if split_idx == -1:
                # Fallback: hard slice if there are no spaces
                split_idx = max_len

            chunk = remaining[:split_idx].strip()
            if chunk:
                chunks.append(chunk)

            remaining = remaining[split_idx:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    @staticmethod
    def _strip_dialogue_quotes(text: str) -> str:
        """
        Strips outer matching quotation marks or Japanese brackets often generated
        by LLMs mimicking script or dialogue examples.
        """
        s = text.strip()
        if len(s) >= 2:
            quote_pairs = [('"', '"'), ("'", "'"), ('“', '”'), ('「', '」')]
            for q_open, q_close in quote_pairs:
                if s.startswith(q_open) and s.endswith(q_close):
                    inner = s[len(q_open) : -len(q_close)].strip()
                    if inner:
                        return inner
        return s

    def split_chat_messages(self, text: str, max_chunk_limit: int | None = None) -> list[str]:
        """
        Splits LLM response text into distinct, continuous Discord messages.
        Preserves multiline code blocks, cleans dialogue quotation marks, ignores empty lines,
        and guarantees all messages remain within the Discord character limit.
        """
        limit = max_chunk_limit if max_chunk_limit is not None else self.default_limit
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        raw_messages: list[str] = []
        current_code_block: list[str] = []
        in_code_block = False

        for line in lines:
            stripped_line = line.strip()

            # Handle code blocks
            if stripped_line.startswith("```"):
                if not in_code_block:
                    # Single-line code block: ```code```
                    if stripped_line.count("```") >= 2 and len(stripped_line) > 3:
                        raw_messages.append(stripped_line)
                    else:
                        in_code_block = True
                        current_code_block = [line]
                    continue
                else:
                    current_code_block.append(line)
                    in_code_block = False
                    raw_messages.append("\n".join(current_code_block).strip())
                    current_code_block = []
                    continue

            if in_code_block:
                current_code_block.append(line)
                continue

            if not stripped_line:
                continue

            cleaned = self._strip_dialogue_quotes(stripped_line)
            if cleaned:
                raw_messages.append(cleaned)

        if current_code_block:
            raw_messages.append("\n".join(current_code_block).strip())

        final_messages: list[str] = []
        for msg in raw_messages:
            msg = msg.strip()
            if not msg:
                continue
            if len(msg) > limit:
                final_messages.extend(self.split(msg, limit=limit))
            else:
                final_messages.append(msg)

        return final_messages
