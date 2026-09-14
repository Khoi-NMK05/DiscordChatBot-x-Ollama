import unittest

from bot.config import BotConfig
from bot.memory import MemoryManager
from bot.services.ollama_service import OllamaService
from bot.utils.text import TextSplitter


class TestTextSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = TextSplitter(default_limit=50)

    def test_empty_string(self):
        self.assertEqual(self.splitter.split(""), [])

    def test_short_message(self):
        text = "Hello world!"
        chunks = self.splitter.split(text)
        self.assertEqual(chunks, ["Hello world!"])

    def test_split_on_newline(self):
        line1 = "A" * 30
        line2 = "B" * 30
        text = f"{line1}\n{line2}"
        chunks = self.splitter.split(text, limit=40)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], line1)
        self.assertEqual(chunks[1], line2)

    def test_split_on_space(self):
        text = "word " * 15  # 75 chars
        chunks = self.splitter.split(text, limit=40)
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 40)

    def test_hard_slice_fallback(self):
        text = "A" * 100
        chunks = self.splitter.split(text, limit=30)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], "A" * 30)
        self.assertEqual(chunks[1], "A" * 30)
        self.assertEqual(chunks[2], "A" * 30)
        self.assertEqual(chunks[3], "A" * 10)

    def test_split_chat_messages_basic(self):
        text = "who said that\n....\nちょっとだけ\ndon't make it a whole thing 😭"
        messages = self.splitter.split_chat_messages(text)
        self.assertEqual(
            messages,
            ["who said that", "....", "ちょっとだけ", "don't make it a whole thing 😭"],
        )

    def test_split_chat_messages_empty_and_blanks(self):
        self.assertEqual(self.splitter.split_chat_messages(""), [])
        self.assertEqual(self.splitter.split_chat_messages("   \n\n  \n"), [])
        text = "line1\n\n\nline2\n  \nline3"
        self.assertEqual(self.splitter.split_chat_messages(text), ["line1", "line2", "line3"])

    def test_split_chat_messages_strip_dialogue_quotes(self):
        text = '"who said that"\n\'...maybe a little.\'\n“ちょっとだけ”\n「べつに」'
        messages = self.splitter.split_chat_messages(text)
        self.assertEqual(
            messages,
            ["who said that", "...maybe a little.", "ちょっとだけ", "べつに"],
        )
        # Inner quotes should not be stripped
        self.assertEqual(
            self.splitter.split_chat_messages('She said "no way" to me'),
            ['She said "no way" to me'],
        )

    def test_split_chat_messages_preserves_code_blocks(self):
        text = "Check this out:\n```python\nprint('hello')\nprint('world')\n```\nCool right?"
        messages = self.splitter.split_chat_messages(text)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], "Check this out:")
        self.assertEqual(messages[1], "```python\nprint('hello')\nprint('world')\n```")
        self.assertEqual(messages[2], "Cool right?")

    def test_split_chat_messages_exceeds_limit(self):
        long_line = "A" * 80
        text = f"short line\n{long_line}"
        messages = self.splitter.split_chat_messages(text, max_chunk_limit=50)
        self.assertEqual(messages[0], "short line")
        self.assertEqual(messages[1], "A" * 50)
        self.assertEqual(messages[2], "A" * 30)


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryManager(max_turns=3)

    def test_initial_state(self):
        self.assertFalse(self.memory.has_memory(1, 100))
        self.assertEqual(self.memory.get_history(1, 100), [])

    def test_add_turn(self):
        self.memory.add_turn(1, 100, "Alice: Hello", "Baka!")
        history = self.memory.get_history(1, 100)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "Alice: Hello"})
        self.assertEqual(history[1], {"role": "assistant", "content": "Baka!"})
        self.assertTrue(self.memory.has_memory(1, 100))

    def test_channel_segregation(self):
        self.memory.add_turn(1, 100, "User 1", "Bot 1")
        self.memory.add_turn(1, 200, "User 2", "Bot 2")
        self.memory.add_turn(None, 300, "DM User", "DM Bot")

        self.assertEqual(len(self.memory.get_history(1, 100)), 2)
        self.assertEqual(len(self.memory.get_history(1, 200)), 2)
        self.assertEqual(len(self.memory.get_history(None, 300)), 2)
        self.assertEqual(self.memory.get_history(1, 100)[0]["content"], "User 1")
        self.assertEqual(self.memory.get_history(1, 200)[0]["content"], "User 2")
        self.assertEqual(self.memory.get_history(None, 300)[0]["content"], "DM User")

    def test_history_capping(self):
        # max_turns = 3 means at most 6 messages
        for i in range(5):
            self.memory.add_turn(1, 100, f"User {i}", f"Bot {i}")

        history = self.memory.get_history(1, 100)
        self.assertEqual(len(history), 6)
        # Should keep turns 2, 3, 4
        self.assertEqual(history[0]["content"], "User 2")
        self.assertEqual(history[-1]["content"], "Bot 4")

    def test_clear_memory(self):
        self.memory.add_turn(1, 100, "User", "Bot")
        self.assertTrue(self.memory.clear(1, 100))
        self.assertFalse(self.memory.has_memory(1, 100))
        # Clearing already empty channel returns False
        self.assertFalse(self.memory.clear(1, 100))


class TestOllamaService(unittest.IsolatedAsyncioTestCase):
    def test_clean_think_tags(self):

        raw = "<think>\nThinking process...\nShould I be mean?\n</think>\nBaka! What do you want?"
        cleaned = OllamaService.clean_think_tags(raw)
        self.assertEqual(cleaned, "Baka! What do you want?")

    def test_extract_reaction_valid(self):
        text = "[REACTION: 💢]\nHmph, why are you asking me?!"
        reaction, clean_text = OllamaService.extract_reaction(text)
        self.assertEqual(reaction, "💢")
        self.assertEqual(clean_text, "Hmph, why are you asking me?!")

    def test_extract_reaction_none(self):
        text = "[REACTION: NONE]\nI don't care."
        reaction, clean_text = OllamaService.extract_reaction(text)
        self.assertIsNone(reaction)
        self.assertEqual(clean_text, "I don't care.")

    def test_extract_reaction_missing(self):
        text = "Just a normal response without reaction tags."
        reaction, clean_text = OllamaService.extract_reaction(text)
        self.assertIsNone(reaction)
        self.assertEqual(clean_text, text)

    def test_build_messages_payload(self):
        config = BotConfig(discord_token="fake_token", system_prompt="Test System Prompt")
        service = OllamaService(config=config)
        history = [
            {"role": "user", "content": "Prev user"},
            {"role": "assistant", "content": "Prev assistant"},
        ]
        payload = service.build_messages_payload(history, "Current prompt")
        self.assertEqual(len(payload), 4)
        self.assertEqual(payload[0], {"role": "system", "content": "Test System Prompt"})
        self.assertEqual(payload[1], {"role": "user", "content": "Prev user"})
        self.assertEqual(payload[2], {"role": "assistant", "content": "Prev assistant"})
        self.assertEqual(payload[3], {"role": "user", "content": "Current prompt"})


    async def test_generate_response_options(self):
        from unittest.mock import AsyncMock
        config = BotConfig(
            discord_token="fake_token",
            num_predict=2048,
            num_ctx=8192,
        )
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "Response"}
        }
        service = OllamaService(config=config, client=mock_client)
        await service.generate_response([], "Hello")

        call_kwargs = mock_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["options"]["num_predict"], 2048)
        self.assertEqual(call_kwargs["options"]["num_ctx"], 8192)
        self.assertEqual(call_kwargs["options"]["temperature"], 0.80)
        self.assertEqual(call_kwargs["options"]["top_p"], 0.90)
        self.assertEqual(call_kwargs["options"]["min_p"], 0.07)
        self.assertEqual(call_kwargs["options"]["repeat_penalty"], 1.08)


class TestBotConfig(unittest.TestCase):
    def test_default_values(self):
        config = BotConfig(discord_token="fake_token")
        self.assertTrue(config.separate_messages)
        self.assertEqual(config.message_delay, 0.8)
        self.assertEqual(config.temperature, 0.80)
        self.assertEqual(config.top_p, 0.90)
        self.assertEqual(config.min_p, 0.07)
        self.assertEqual(config.repeat_penalty, 1.08)
        self.assertEqual(config.max_history_turns, 15)

    def test_from_env(self):
        import os
        from unittest.mock import patch

        env_vars = {
            "DISCORD_BOT_TOKEN": "test_token",
            "SEPARATE_MESSAGES": "false",
            "MESSAGE_DELAY": "1.5",
            "MIN_P": "0.10",
            "REPEAT_PENALTY": "1.15",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = BotConfig.from_env()
            self.assertFalse(config.separate_messages)
            self.assertEqual(config.message_delay, 1.5)
            self.assertEqual(config.min_p, 0.10)
            self.assertEqual(config.repeat_penalty, 1.15)


class TestChatCog(unittest.IsolatedAsyncioTestCase):
    async def test_handle_bot_mention_separate_messages(self):
        from unittest.mock import AsyncMock, MagicMock
        from bot.cogs.chat import ChatCog
        from bot.services.ollama_service import LLMResponse

        config = BotConfig(discord_token="fake", separate_messages=True, message_delay=0.0)
        bot_mock = MagicMock()
        bot_mock.user.display_name = "B-chan"

        ollama_mock = AsyncMock()
        ollama_mock.config = config
        ollama_mock.generate_response.return_value = LLMResponse(
            content="who said that\n...\nちょっとだけ\ndon't make it a whole thing 😭",
            reaction=None,
        )

        memory_mock = MagicMock()
        memory_mock.get_history.return_value = []

        cog = ChatCog(
            bot=bot_mock,
            ollama_service=ollama_mock,
            memory_manager=memory_mock,
            text_splitter=TextSplitter(),
        )

        message_mock = AsyncMock()
        message_mock.clean_content = "@B-chan you didn't miss me?"
        message_mock.attachments = []
        message_mock.guild.id = 123
        message_mock.channel.id = 456
        message_mock.author.display_name = "Khoi"

        class DummyTypingCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        message_mock.channel.typing = MagicMock(return_value=DummyTypingCM())

        await cog._handle_bot_mention(message_mock)

        # Message 1 replied
        message_mock.reply.assert_called_once_with("who said that", mention_author=False)
        # Remaining 3 messages sent to channel
        self.assertEqual(message_mock.channel.send.call_count, 3)
        message_mock.channel.send.assert_any_call("...")
        message_mock.channel.send.assert_any_call("ちょっとだけ")
        message_mock.channel.send.assert_any_call("don't make it a whole thing 😭")

    async def test_handle_bot_mention_single_message_when_disabled(self):
        from unittest.mock import AsyncMock, MagicMock
        from bot.cogs.chat import ChatCog
        from bot.services.ollama_service import LLMResponse

        config = BotConfig(discord_token="fake", separate_messages=False, message_delay=0.0)
        bot_mock = MagicMock()
        bot_mock.user.display_name = "B-chan"

        full_content = "who said that\n...\nちょっとだけ\ndon't make it a whole thing 😭"
        ollama_mock = AsyncMock()
        ollama_mock.config = config
        ollama_mock.generate_response.return_value = LLMResponse(
            content=full_content,
            reaction=None,
        )

        memory_mock = MagicMock()
        memory_mock.get_history.return_value = []

        cog = ChatCog(
            bot=bot_mock,
            ollama_service=ollama_mock,
            memory_manager=memory_mock,
            text_splitter=TextSplitter(),
        )

        message_mock = AsyncMock()
        message_mock.clean_content = "@B-chan you didn't miss me?"
        message_mock.attachments = []
        message_mock.guild.id = 123
        message_mock.channel.id = 456
        message_mock.author.display_name = "Khoi"

        class DummyTypingCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        message_mock.channel.typing = MagicMock(return_value=DummyTypingCM())

        await cog._handle_bot_mention(message_mock)

        # Only 1 reply sent as a single block
        message_mock.reply.assert_called_once_with(full_content, mention_author=False)
        message_mock.channel.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()

