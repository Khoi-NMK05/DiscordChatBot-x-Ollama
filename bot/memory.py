class MemoryManager:
    """
    Manages in-memory multi-turn conversation history per Discord guild and channel.
    Caps history size to prevent context window overflow.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        # Key: (guild_id, channel_id) -> list of {"role": "user"|"assistant", "content": "..."}
        self._memories: dict[tuple[int | None, int], list[dict[str, str]]] = {}

    def _get_key(self, guild_id: int | None, channel_id: int) -> tuple[int | None, int]:
        return (guild_id, channel_id)

    def get_history(self, guild_id: int | None, channel_id: int) -> list[dict[str, str]]:
        """Returns a copy of the stored history for the given channel."""
        key = self._get_key(guild_id, channel_id)
        return list(self._memories.get(key, []))

    def add_turn(
        self,
        guild_id: int | None,
        channel_id: int,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """
        Adds a user-assistant conversation turn and caps history
        to the maximum allowed turns.
        """
        key = self._get_key(guild_id, channel_id)
        if key not in self._memories:
            self._memories[key] = []

        history = self._memories[key]
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})

        max_entries = self.max_turns * 2
        if len(history) > max_entries:
            self._memories[key] = history[-max_entries:]

    def clear(self, guild_id: int | None, channel_id: int) -> bool:
        """
        Clears the conversation history for the given channel.
        Returns True if history was found and cleared, False if already empty.
        """
        key = self._get_key(guild_id, channel_id)
        if self._memories.get(key):
            del self._memories[key]
            return True
        elif key in self._memories:
            del self._memories[key]
        return False

    def has_memory(self, guild_id: int | None, channel_id: int) -> bool:
        """Checks whether any history exists for the given channel."""
        key = self._get_key(guild_id, channel_id)
        return bool(self._memories.get(key))

    def clear_all(self) -> None:
        """Clears memory across all channels."""
        self._memories.clear()
