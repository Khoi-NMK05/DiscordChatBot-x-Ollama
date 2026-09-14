# DiscordChatBotxOllama

A Discord AI chatbot powered by local LLMs via [Ollama](https://ollama.com/), featuring an entertaining Tsundere anime personality, multi-turn conversation memory, dynamic Discord reaction emojis, and a clean Object-Oriented (OOP) architecture.

---

## 🌟 Features

- **Object-Oriented Design**: Clean modular structure adhering to SOLID principles (encapsulated configuration, memory management, LLM service, Discord cogs, and bot client).
- **Local AI with Ollama**: Fully private and local inference using models like `qwen3.5:4b`, `llama3.1`, `mistral`, etc.
- **Contextual Memory**: Remembers conversation turns per server/channel, automatically capping history to prevent context overflow.
- **Decoupled Computer Vision**:
  - Automatically perceives Discord image attachments using a lightweight decoupled vision model (e.g. `moondream`).
  - **Ephemeral Images**: Raw images are downscaled on-the-fly and immediately discarded after description.
  - **Text-Converted History**: Image descriptions are stored as pure text in chat memory, completely preventing KV-cache bloat and multi-turn VRAM spikes.
- **Personality & Expression**:
  - Unhinged tsundere persona replying in natural slang (Vietnamese/English).
  - Automatically adds Discord reactions (e.g., 💢, 🙄, 😳) to messages based on model output.
  - Automatically filters internal `<think>...</think>` reasoning tokens from reasoning/thinking models.
- **Realistic Continuous Messaging**: Replies with natural, separate Discord messages instead of a single wall of text, displaying real-time typing indicators between short messages.
- **Message Chunking**: Safely splits long AI answers into Discord-compliant chunks (<2000 characters) without breaking words or code blocks.
- **Backwards Compatibility**: Can be launched via `main.py` or the legacy `DcChatbot.py`.

---

## 📁 Project Structure

```text
DiscordChatBotxOllama/
├── bot/
│   ├── __init__.py          # Public package exports
│   ├── client.py            # TsundereBot (subclassing discord.ext.commands.Bot)
│   ├── config.py            # BotConfig dataclass (environment loader & settings)
│   ├── memory.py            # MemoryManager (multi-channel conversation history)
│   ├── cogs/
│   │   ├── __init__.py
│   │   ├── chat.py          # ChatCog (handles mentions, typing, vision, response flow)
│   │   └── commands.py      # GeneralCog (commands such as !clear_memory)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ollama_service.py# OllamaService & LLMResponse
│   │   └── vision_service.py# VisionService (decoupled micro-VLM image captioning)
│   └── utils/
│       ├── __init__.py
│       ├── image.py         # ImageProcessor (downscaling, format conversion, non-blocking)
│       └── text.py          # TextSplitter (splits long text safely for Discord)
├── tests/
│   ├── __init__.py
│   ├── test_components.py   # Unit test suite for core text modules
│   └── test_vision.py       # Unit test suite for vision & image pipeline
├── main.py                  # Primary application entry point
├── DcChatbot.py             # Legacy entry point wrapper
├── requirements.txt         # Project dependencies
├── .env.example             # Example environment variables template
├── .gitignore
└── README.md                # Documentation
```

---

## 🏗️ Architecture Overview

The codebase is structured around single-responsibility classes:

| Class | Location | Responsibility |
| :--- | :--- | :--- |
| **`BotConfig`** | `bot/config.py` | Encapsulates bot tokens, hyperparameters, prompts, vision settings, and loads from `.env`. |
| **`MemoryManager`** | `bot/memory.py` | Tracks and caps multi-turn conversation history keyed by `(guild_id, channel_id)`. |
| **`OllamaService`** | `bot/services/ollama_service.py` | Communicates with Ollama asynchronously, strips `<think>` tags, and extracts emoji reactions. |
| **`VisionService`** | `bot/services/vision_service.py` | Interfaces with a lightweight vision model (e.g., `moondream`) to generate concise image descriptions. |
| **`ImageProcessor`** | `bot/utils/image.py` | Validates MIME types, extracts animated GIF first frames, converts to RGB, and downscales images non-blockingly. |
| **`TextSplitter`** | `bot/utils/text.py` | Slices text into chunks obeying Discord's character limit using newline and word boundaries. |
| **`TsundereBot`** | `bot/client.py` | Custom `commands.Bot` subclass managing dependency injection and async cog loading. |
| **`ChatCog`** | `bot/cogs/chat.py` | Discord listener cog handling `@bot` pings, vision attachment analysis, typing indicator, and reply dispatch. |
| **`GeneralCog`** | `bot/cogs/commands.py` | Discord commands cog handling commands like `!clear_memory`. |


---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** (Tested on Python 3.14)
- **Ollama**: Installed and running locally ([Download Ollama](https://ollama.com/download))
  - Pull the chat and vision models:
    ```bash
    ollama pull qwen3.5:9b
    ollama pull qwen2.5vl:3b
    ```
- **Discord Bot Token**:
  - Create an application in the [Discord Developer Portal](https://discord.com/developers/applications).
  - Enable **Message Content Intent** under Bot > Privileged Gateway Intents.
  - Invite your bot to your server with permissions to read messages, send messages, and add reactions.

### 2. Installation

1. Clone or download this repository.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your Discord Bot Token:

```bash
cp .env.example .env
```

Edit `.env`:
```env
DISCORD_BOT_TOKEN=your_token_here
OLLAMA_MODEL=qwen3.5:9b
NUM_PREDICT=2048
NUM_CTX=8192

VISION_ENABLED=true
VISION_MODEL=qwen2.5vl:3b
VISION_MAX_DIMENSION=1536
VISION_NUM_PREDICT=1024
```


---

## 💬 Usage

### Run the Bot

Start the bot using the main entry point:
```bash
python main.py
```

*(Or use the legacy command: `python DcChatbot.py`)*

### Chatting with the Bot

- Mention the bot in any Discord channel it has access to:
  ```text
  @TsundereBot What is recursion?
  ```
- Clear channel memory:
  ```text
  !clear_memory
  ```

---

## 🧪 Running Tests

A unit test suite is included to verify the core components independently of Discord or Ollama:

```bash
python -m unittest discover -s tests -v
```
