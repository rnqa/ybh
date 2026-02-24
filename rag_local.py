#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from dotenv import load_dotenv

from rag30 import OtusStyleRAG


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("tg_rag30")


def load_env() -> None:
    load_dotenv("config.env")
    load_dotenv(".env")


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or not str(value).strip():
        return ""
    return str(value).strip()


def get_admin_ids() -> set[int]:
    raw = get_env("TELEGRAM_ADMIN_IDS", "")
    ids: set[int] = set()
    for part in re.split(r"[,;\\s]+", raw):
        if part and part.isdigit():
            ids.add(int(part))
    single = get_env("TELEGRAM_ADMIN_ID", "")
    if single.isdigit():
        ids.add(int(single))
    if not ids:
        ids.add(5166459333)
    return ids


def is_admin(message: types.Message, admin_ids: set[int]) -> bool:
    return bool(message.from_user and message.from_user.id in admin_ids)


def build_rag() -> OtusStyleRAG:
    index_dir = Path(get_env("RAG_INDEX_DIR", "faiss_indexes"))
    index_name = get_env("RAG_INDEX_NAME", "law_db")
    data_dir = Path(get_env("RAG_DATA_DIR", "NPA3001"))
    prompt_path = Path(get_env("RAG_PROMPT_PATH", "prompts/legal_student_prompt_v2.txt"))
    llm_query_expansion = get_env("RAG_LLM_QUERY_EXPANSION", "0").lower() in {"1", "true", "yes"}

    return OtusStyleRAG(
        index_dir=index_dir,
        index_name=index_name,
        data_dir=data_dir,
        use_llm_query_expansion=llm_query_expansion,
        prompt_path=prompt_path,
    )


def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    current = ""
    for line in text.splitlines():
        if len(current) + len(line) + 1 <= limit:
            current += (line + "\n")
        else:
            parts.append(current.strip())
            current = line + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts


async def main() -> None:
    load_env()

    token = get_env("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN не найден в config.env/.env")

    admin_ids = get_admin_ids()
    rag = build_rag()
    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start_cmd(message: types.Message) -> None:
        await message.answer(
            "Привет! Я юридический RAG-ассистент по праву РФ.\n"
            "Задайте вопрос, я отвечу строго по базе документов."
        )

    @dp.message(Command("restart"))
    async def restart_cmd(message: types.Message) -> None:
        if not is_admin(message, admin_ids):
            await message.answer("Недостаточно прав.")
            return
        await message.answer("Перезапуск...")
        await asyncio.sleep(0.2)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    @dp.message()
    async def handle_question(message: types.Message) -> None:
        user_question = (message.text or "").strip()
        if not user_question:
            await message.answer("Пожалуйста, задайте вопрос.")
            return

        await message.answer("Ищу в базе и формирую ответ...")
        try:
            result = await asyncio.to_thread(rag.ask, user_question)
            answer = str(result.get("answer", "")).strip()
        except Exception as exc:
            LOGGER.exception("Ошибка при обработке вопроса: %s", exc)
            await message.answer("Произошла ошибка при обработке запроса.")
            return

        for idx, part in enumerate(split_message(answer), start=1):
            if idx == 1:
                await message.answer(part)
            else:
                await message.answer(f"(продолжение {idx})\n{part}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
