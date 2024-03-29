from __future__ import annotations

from xiaogpt.bot.base_bot import BaseBot
from xiaogpt.bot.chatgptapi_bot import ChatGPTBot
from xiaogpt.bot.gpt3_bot import GPT3Bot
from xiaogpt.bot.newbing_bot import NewBingBot
from xiaogpt.bot.glm_bot import GLMBot
from xiaogpt.bot.bard_bot import BardBot
from xiaogpt.bot.gemini_bot import GeminiBot
from xiaogpt.bot.qwen_bot import QwenBot
from xiaogpt.bot.langchain_bot import LangChainBot
from xiaogpt.bot.light_bot import LightBot
from xiaogpt.bot.rag_bot import RagBot
from xiaogpt.bot.chinese_med_bot import ChineseMedBot
from xiaogpt.config import Config

BOTS: dict[str, type[BaseBot]] = {
    "gpt3": GPT3Bot,
    "newbing": NewBingBot,
    "chatgptapi": ChatGPTBot,
    "glm": GLMBot,
    "bard": BardBot,
    "gemini": GeminiBot,
    "qwen": QwenBot,
    "langchain": LangChainBot,
    "GPT4": LightBot,
    "RAG": RagBot,
    "ChineseMed": ChineseMedBot
}


def get_bot(config: Config) -> BaseBot:
    try:
        return BOTS[config.bot].from_config(config)
    except KeyError:
        raise ValueError(f"Unsupported bot {config.bot}, must be one of {list(BOTS)}")


__all__ = [
    "GPT3Bot",
    "ChatGPTBot",
    "NewBingBot",
    "GLMBot",
    "BardBot",
    "GeminiBot",
    "QwenBot",
    "get_bot",
    "LangChainBot",
    "GPT4",
    "RAG",
    "ChineseMed"
]
