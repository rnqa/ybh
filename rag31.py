#!/usr/bin/env python3
"""
ЮРИДИЧЕСКИЙ ENSEMBLE RAG-АССИСТЕНТ
Продвинутая версия на основе принципов Ensemble RAG
"""

from __future__ import annotations

import os
import json
import re
import sys
import logging
import hashlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Для корректного вывода Unicode в Windows-консолях
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _prefer_local_venv():
    """Перезапускает скрипт через локальный .venv, если он есть и текущий Python другой."""
    if os.environ.get("RAG31_VENV_REEXEC") == "1":
        return

    script_path = Path(__file__).resolve()
    venv_python = script_path.parent / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        # Если не удалось корректно сравнить пути — не блокируем запуск.
        return

    os.environ["RAG31_VENV_REEXEC"] = "1"
    print(f"ℹ Перезапуск через локальный интерпретатор: {venv_python}")
    os.execv(str(venv_python), [str(venv_python), str(script_path), *sys.argv[1:]])


_prefer_local_venv()

# Базовые импорты
from dotenv import load_dotenv

# Эмбеддинги

# LangChain импорты
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

FAISS = None
HAS_FAISS = False
FAISS_IMPORT_ERROR = ""
try:
    from langchain_community.vectorstores import FAISS
    HAS_FAISS = True
except ImportError as e:
    FAISS_IMPORT_ERROR = str(e)

# Для keyword поиска
TfidfVectorizer = None
cosine_similarity = None
HAS_SKLEARN = False
SKLEARN_IMPORT_ERROR = ""
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception as e:
    SKLEARN_IMPORT_ERROR = str(e)

# ==================== КОНФИГУРАЦИЯ ====================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
load_dotenv("config.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    logger.info("✅ Конфигурация загружена")
else:
    logger.warning("OPENAI_API_KEY не найден в config.env (проверка будет во время инициализации)")


# ==================== МОДЕЛИ ====================

class AdvancedEmbeddings(Embeddings):
    """Продвинутые эмбеддинги с кэшированием"""

    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru"):
        self.backend = "sentence_transformers"
        self.model = None
        self.model_name = model_name
        self.cache = {}
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Загружена модель эмбеддингов: {model_name}")
        except KeyboardInterrupt:
            logger.warning("Импорт sentence-transformers прерван. Переход на OpenAIEmbeddings (fallback).")
            self.backend = "openai"
            self.model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Не удалось загрузить sentence-transformers ({e}). Переход на OpenAIEmbeddings (fallback).")
            self.backend = "openai"
            self.model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embedding_dimension(self) -> int:
        """Возвращает размерность вектора для текущего backend."""
        if self.backend == "openai":
            # Используется модель text-embedding-3-large
            return 3072

        if self.model is not None and hasattr(self.model, "get_sentence_embedding_dimension"):
            try:
                return int(self.model.get_sentence_embedding_dimension())
            except Exception:
                pass

        # fallback (может быть дорогим, используем только если не удалось иначе)
        return len(self.embed_query("dim_check"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинг для документов с кэшированием"""
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self.cache:
                embeddings[i] = self.cache[key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            if self.backend == "openai":
                new_embeddings = self.model.embed_documents(uncached_texts)
            else:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )

            for text, idx, emb in zip(uncached_texts, uncached_indices, new_embeddings):
                key = self._cache_key(text)
                emb_list = emb if isinstance(emb, list) else emb.tolist()
                self.cache[key] = emb_list
                embeddings[idx] = emb_list

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг для запроса"""
        key = self._cache_key(text)
        if key in self.cache:
            return self.cache[key]

        if self.backend == "openai":
            embedding = self.model.embed_query(text)
        else:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].tolist()

        self.cache[key] = embedding
        return embedding


class LegalReranker:
    """Re-ранкер для юридических документов"""

    def __init__(self, model_name: str = "cointegrated/ce-ru-msmarco"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"Загружен cross-encoder: {model_name}")
        except KeyboardInterrupt:
            logger.warning("Импорт CrossEncoder прерван, используется fallback ранжирование")
            self.model = None
        except Exception:
            logger.warning(f"Не удалось загрузить {model_name}, используется fallback ранжирование")
            self.model = None

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Переранжирование документов"""
        if not documents or not self.model:
            return documents[:top_k]

        # Подготавливаем пары запрос-документ
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            # Получаем scores от cross-encoder
            scores = self.model.predict(pairs)

            # Сортируем документы по scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
            logger.info(f"Re-ранжирование завершено, выбрано {len(reranked_docs)} документов")

            # Добавляем scores в метаданные для отладки
            for i, (doc, score) in enumerate(scored_docs[:top_k]):
                if i < len(reranked_docs):
                    reranked_docs[i].metadata["rerank_score"] = float(score)

            return reranked_docs
        except Exception as e:
            logger.error(f"Ошибка re-ранжирования: {e}")
            return documents[:top_k]


# ==================== АНСАМБЛЬ РЕТРИВЕРОВ ====================

class EnsembleRetriever:
    """Ансамбль ретриверов для юридического поиска"""

    def __init__(self, vector_store: FAISS, documents: List[Document]):
        self.vector_store = vector_store
        self.documents = documents

        # Инициализация различных ретриверов
        self.semantic_retriever = self._init_semantic_retriever()
        self.keyword_retriever = self._init_keyword_retriever()
        self.metadata_retriever = self._init_metadata_retriever()

        # Инициализация re-ранкера
        self.reranker = LegalReranker()

        logger.info("✅ Ансамбль ретриверов инициализирован")

    def _init_semantic_retriever(self):
        """Семантический ретривер на основе FAISS"""
        # Держим дефолтный ретривер для совместимости, основной поиск идет через _semantic_search
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )

    def _semantic_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Устойчивый семантический поиск с fallback."""
        k = max(15, top_k * 2)
        threshold = 0.2

        # 1) Пытаемся получить релевантность и применить порог
        try:
            scored = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            filtered_docs = []
            for doc, score in scored:
                cloned_doc = self._clone_document(doc)
                try:
                    cloned_doc.metadata["semantic_score"] = float(score)
                except Exception:
                    pass
                if score is None or float(score) >= threshold:
                    filtered_docs.append(cloned_doc)

            if filtered_docs:
                return filtered_docs[:k]
        except Exception as e:
            logger.warning(f"Семантический поиск по relevance scores не сработал: {type(e).__name__}: {e!r}")

        # 2) Надежный fallback без score_threshold
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [self._clone_document(doc) for doc in docs]
        except Exception as e:
            logger.error(f"Ошибка fallback семантического поиска: {type(e).__name__}: {e!r}")
            return []

    @staticmethod
    def _clone_document(doc: Document) -> Document:
        return Document(page_content=doc.page_content, metadata=dict(doc.metadata))

    @staticmethod
    def _append_retriever_type(doc: Document, retriever_type: str):
        types = doc.metadata.get("retriever_types", [])
        if not isinstance(types, list):
            types = [str(types)]
        if retriever_type not in types:
            types.append(retriever_type)
        doc.metadata["retriever_types"] = types
        doc.metadata["retriever_type"] = ",".join(types)

    def _init_keyword_retriever(self):
        """Ключевой ретривер на основе TF-IDF"""
        if not HAS_SKLEARN:
            logger.warning(
                "Keyword-ретривер отключен: недоступен scikit-learn/scipy. "
                f"Причина: {SKLEARN_IMPORT_ERROR or 'неизвестно'}"
            )
            return False

        # Собираем все тексты
        texts = [doc.page_content for doc in self.documents]

        # Создаем TF-IDF векторизатор
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None  # Для юридических текстов лучше не удалять стоп-слова
        )

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF матрица создана: {self.tfidf_matrix.shape}")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации TF-IDF: {e}")
            return False

    def _init_metadata_retriever(self):
        """Ретривер на основе метаданных"""
        # Создаем индекс метаданных
        self.metadata_index = {}
        for idx, doc in enumerate(self.documents):
            metadata = doc.metadata

            # Индексируем по различным полям метаданных
            source_title = metadata.get("source_title", "").lower()
            law_id = metadata.get("law_id", "").lower()
            hierarchy = metadata.get("hierarchy_str", "").lower()

            for field in [source_title, law_id, hierarchy]:
                if field:
                    if field not in self.metadata_index:
                        self.metadata_index[field] = []
                    self.metadata_index[field].append(idx)

        logger.info(f"Создан индекс метаданных для {len(self.metadata_index)} уникальных ключей")
        return True

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Поиск по ключевым словам через TF-IDF"""
        if not hasattr(self, 'vectorizer') or not hasattr(self, 'tfidf_matrix'):
            return []

        try:
            # Векторизуем запрос
            query_vec = self.vectorizer.transform([query])

            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Получаем топ-K индексов
            top_indices = similarities.argsort()[-top_k:][::-1]

            # Возвращаем документы
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self._clone_document(self.documents[idx])
                    # Добавляем score в метаданные
                    doc.metadata["keyword_score"] = float(similarities[idx])
                    results.append(doc)

            logger.info(f"Keyword поиск: найдено {len(results)} документов")
            return results
        except Exception as e:
            logger.error(f"Ошибка keyword поиска: {e}")
            return []

    def _metadata_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Поиск по метаданным"""
        query_lower = query.lower()
        results = []
        seen_indices = set()

        # Разбиваем запрос на слова
        query_words = re.findall(r'\b\w+\b', query_lower)

        # Поиск по индексу метаданных
        for word in query_words:
            if len(word) > 3:  # Игнорируем слишком короткие слова
                for key, indices in self.metadata_index.items():
                    if word in key:
                        for idx in indices:
                            if idx not in seen_indices and idx < len(self.documents):
                                doc = self._clone_document(self.documents[idx])
                                doc.metadata["metadata_match"] = key
                                results.append(doc)
                                seen_indices.add(idx)

        # Ограничиваем количество результатов
        results = results[:top_k * 2]  # Берем немного больше для последующего ранжирования

        logger.info(f"Metadata поиск: найдено {len(results)} документов")
        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Основной метод поиска через ансамбль ретриверов"""
        all_results = []

        logger.info(f"🔍 Начало поиска по запросу: {query[:100]}...")

        # 1. Семантический поиск (FAISS)
        try:
            semantic_results = self._semantic_search(query, top_k=top_k)
            for doc in semantic_results:
                self._append_retriever_type(doc, "semantic")
                all_results.append(doc)
            logger.info(f"  ✓ Семантический поиск: {len(semantic_results)} документов")
        except Exception as e:
            logger.error(f"  ✗ Ошибка семантического поиска: {type(e).__name__}: {e!r}")

        # 2. Keyword поиск (TF-IDF)
        try:
            keyword_results = self._keyword_search(query, top_k=top_k)
            for doc in keyword_results:
                self._append_retriever_type(doc, "keyword")
            all_results.extend(keyword_results)
            logger.info(f"  ✓ Keyword поиск: {len(keyword_results)} документов")
        except Exception as e:
            logger.error(f"  ✗ Ошибка keyword поиска: {e}")

        # 3. Metadata поиск
        try:
            metadata_results = self._metadata_search(query, top_k=top_k)
            for doc in metadata_results:
                self._append_retriever_type(doc, "metadata")
            all_results.extend(metadata_results)
            logger.info(f"  ✓ Metadata поиск: {len(metadata_results)} документов")
        except Exception as e:
            logger.error(f"  ✗ Ошибка metadata поиска: {e}")

        # Убираем дубликаты по содержимому
        unique_results = self._deduplicate_documents(all_results)
        logger.info(f"📊 После дедупликации: {len(unique_results)} уникальных документов")

        # 4. Re-ранжирование
        if len(unique_results) > 1:
            reranked_results = self.reranker.rerank(query, unique_results, top_k=top_k * 2)
        else:
            reranked_results = unique_results

        return reranked_results[:top_k]

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Удаление дубликатов документов"""
        deduplicated: Dict[str, Document] = {}
        unique_docs = []

        for doc in documents:
            # Создаем хэш по первым 500 символам и заголовку
            hash_input = f"{doc.page_content[:500]}|{doc.metadata.get('source_title', '')}"
            content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

            if content_hash not in deduplicated:
                deduplicated[content_hash] = doc
                unique_docs.append(doc)
                continue

            existing_doc = deduplicated[content_hash]
            existing_types = existing_doc.metadata.get("retriever_types", [])
            incoming_types = doc.metadata.get("retriever_types", [])
            if not isinstance(existing_types, list):
                existing_types = [str(existing_types)]
            if not isinstance(incoming_types, list):
                incoming_types = [str(incoming_types)]
            for retriever_type in incoming_types:
                if retriever_type not in existing_types:
                    existing_types.append(retriever_type)
            existing_doc.metadata["retriever_types"] = existing_types
            existing_doc.metadata["retriever_type"] = ",".join(existing_types)

            if "keyword_score" in doc.metadata:
                existing_doc.metadata["keyword_score"] = max(
                    float(existing_doc.metadata.get("keyword_score", 0)),
                    float(doc.metadata["keyword_score"])
                )
            if "rerank_score" in doc.metadata:
                existing_doc.metadata["rerank_score"] = max(
                    float(existing_doc.metadata.get("rerank_score", float("-inf"))),
                    float(doc.metadata["rerank_score"])
                )

        return unique_docs


# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_law_documents(data_path: Path = Path("faiss_indexes")) -> List[Document]:
    """Загрузка юридических документов из JSONL файлов"""
    documents = []

    if not data_path.exists():
        logger.error(f"Папка с данными не найдена: {data_path}")
        return documents

    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"Не найдены JSONL файлы в папке: {data_path}")
        return documents

    logger.info(f"Найдено {len(jsonl_files)} JSONL файлов")

    total_loaded = 0
    for file_path in jsonl_files:
        logger.info(f"Загрузка файла: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())

                        # Извлекаем текст (поле может называться по-разному)
                        text = data.get("text", "") or data.get("content", "") or data.get("статья", "")

                        if text and len(text.strip()) > 50:  # Минимальная длина текста
                            # Создаем документ
                            doc = Document(
                                page_content=text.strip(),
                                metadata={
                                    "source_title": data.get("source_title", "Неизвестный источник"),
                                    "hierarchy_str": data.get("hierarchy_str", ""),
                                    "law_id": data.get("law_id", ""),
                                    "source_type": data.get("source_type", "закон"),
                                    "source_url": data.get("source_url", ""),
                                    "chunk_id": data.get("chunk_id", ""),
                                    "file_source": file_path.name,
                                    "line_number": line_num
                                }
                            )
                            documents.append(doc)
                            total_loaded += 1

                            if total_loaded % 10000 == 0:
                                logger.info(f"Загружено {total_loaded} документов...")

                    except json.JSONDecodeError:
                        logger.warning(f"Ошибка JSON в строке {line_num} файла {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка обработки строки {line_num}: {e}")

        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path.name}: {e}")

    logger.info(f"✅ Загружено всего документов: {len(documents)}")
    return documents


def load_or_create_index(documents: List[Document], embeddings: Embeddings, index_path: Path = Path("faiss_indexes")) -> \
Tuple[FAISS, bool]:
    """Загружает существующий индекс или создает новый"""
    if not HAS_FAISS:
        raise RuntimeError(
            "FAISS недоступен. Установите зависимости командой:\n"
            f"  {sys.executable} -m pip install langchain-community faiss-cpu\n"
            f"Причина импорта: {FAISS_IMPORT_ERROR or 'неизвестно'}"
        )

    # Проверяем существование индекса
    faiss_file = index_path / "law_db.faiss"
    metadata_file = index_path / "law_db_metadata.json"

    if faiss_file.exists() and metadata_file.exists():
        logger.info(f"🔄 Попытка загрузки существующего индекса из {index_path}")

        try:
            # Пробуем загрузить индекс
            vector_store = FAISS.load_local(
                folder_path=str(index_path),
                embeddings=embeddings,
                index_name="law_db",
                allow_dangerous_deserialization=True
            )

            # Проверяем совместимость размерности эмбеддингов и индекса
            index_dim = getattr(getattr(vector_store, "index", None), "d", None)
            expected_dim = None
            try:
                if hasattr(embeddings, "embedding_dimension"):
                    expected_dim = int(embeddings.embedding_dimension())
                else:
                    expected_dim = len(embeddings.embed_query("размерность"))
            except Exception as dim_err:
                logger.warning(f"Не удалось определить размерность эмбеддингов: {dim_err}")

            if index_dim is not None and expected_dim is not None and int(index_dim) != int(expected_dim):
                rebuild_on_mismatch = os.getenv("RAG31_REBUILD_INDEX_ON_MISMATCH", "0") == "1"
                msg = (
                    f"Несовместимая размерность эмбеддингов и индекса: index_dim={index_dim}, "
                    f"embedding_dim={expected_dim}."
                )
                if not rebuild_on_mismatch:
                    raise RuntimeError(
                        msg + "\nЧтобы автоматически пересоздать индекс, задайте "
                        "RAG31_REBUILD_INDEX_ON_MISMATCH=1 и перезапустите."
                    )
                logger.warning(msg + " Пересоздаем индекс...")
            else:
                logger.info("✅ Существующий индекс успешно загружен")
                return vector_store, False

        except Exception as e:
            logger.warning(f"Не удалось загрузить существующий индекс: {e}")
            logger.info("Создаем новый индекс...")

    # Если индекс не найден или не загрузился, создаем новый
    logger.info(f"Создание нового FAISS индекса для {len(documents)} документов...")

    # Создаем векторное хранилище
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Сохраняем индекс
    index_path.mkdir(exist_ok=True)
    vector_store.save_local(
        folder_path=str(index_path),
        index_name="law_db"
    )

    # Также сохраняем метаданные в JSON для совместимости
    metadatas = [doc.metadata for doc in documents]
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Новый индекс создан и сохранен в {index_path}")
    return vector_store, True


# Удалите старую функцию create_vector_store и замените её использование в _initialize

# ==================== RAG СИСТЕМА ====================

class LegalEnsembleRAG:
    """Основной класс Ensemble RAG системы"""

    def __init__(self, data_path: Path = Path("NPA3001")):
        self.data_path = data_path
        self.embeddings = None
        self.documents = []
        self.vector_store = None
        self.ensemble_retriever = None
        self.llm = None

        self._initialize()

    def _initialize(self):
        """Инициализация всех компонентов системы"""
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY не найден в config.env")
        if not HAS_FAISS:
            raise RuntimeError(
                "Отсутствует зависимость для FAISS. Установите пакеты командой:\n"
                f"  {sys.executable} -m pip install langchain-community faiss-cpu\n"
                f"Причина импорта: {FAISS_IMPORT_ERROR or 'неизвестно'}"
            )

        logger.info("=" * 60)
        logger.info("ИНИЦИАЛИЗАЦИЯ ENSEMBLE RAG СИСТЕМЫ")
        logger.info("=" * 60)

        # 1. Загрузка модели эмбеддингов
        logger.info("1. Загрузка моделей...")
        try:
            self.embeddings = AdvancedEmbeddings("cointegrated/LaBSE-en-ru")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            sys.exit(1)

        # 2. Загрузка документов (только если нужен новый индекс)
        logger.info("2. Загрузка юридических документов...")
        self.documents = load_law_documents(self.data_path)

        if not self.documents:
            logger.error("Не удалось загрузить документы")
            sys.exit(1)

        # 3. Загрузка или создание векторного хранилища
        logger.info("3. Загрузка/создание векторного хранилища...")
        self.vector_store, created_new = load_or_create_index(
            self.documents,
            self.embeddings,
            Path("faiss_indexes")
        )

        if created_new:
            logger.info("⚠️ Создан новый индекс. Процесс может занять время...")

        # 4. Инициализация ансамбля ретриверов
        logger.info("4. Инициализация ансамбля ретриверов...")
        self.ensemble_retriever = EnsembleRetriever(self.vector_store, self.documents)

        # 5. Инициализация LLM
        logger.info("5. Инициализация LLM...")
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=OPENAI_API_KEY,
                timeout=60,
                max_retries=2
            )

            # Тест подключения
            test_response = self.llm.invoke([HumanMessage(content="Тест. Ответь 'OK'")])
            logger.info(f"✅ LLM инициализирована: {test_response.content}")

        except Exception as e:
            logger.error(f"Ошибка инициализации LLM: {e}")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
        logger.info("=" * 60)

    def _create_legal_prompt(self, query: str, context: str) -> str:
        """Создание строгого юридического промпта"""
        return f"""Ты — профессиональный юридический ассистент, специализирующийся на законодательстве РФ.

ТВОЯ ЗАДАЧА:
Ответить на юридический вопрос, используя ТОЛЬКО предоставленные ниже фрагменты законодательства РФ.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай СТРОГО на основе предоставленных фрагментов законов
2. Если в предоставленных фрагментах недостаточно информации для полного ответа — так и скажи
3. НИКОГДА не придумывай информацию, которой нет в предоставленных фрагментах
4. Всегда цитируй конкретные законы и статьи, на которые ты ссылаешься
5. Будь точным, структурированным и профессиональным

КОНТЕКСТ (фрагменты законодательства РФ):
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

СТРУКТУРА ОТВЕТА:
1. Краткий ответ (основной вывод)
2. Правовое обоснование (со ссылками на конкретные статьи)
3. Если применимо — практические рекомендации
4. Предупреждение об ограничениях (если информация неполная)

ОТВЕТ:"""

    def query(self, question: str, top_k: int = 7) -> Dict[str, Any]:
        """Выполнение запроса к RAG системе"""
        start_time = datetime.now()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"ВОПРОС: {question}")
        logger.info(f"{'=' * 60}")

        try:
            # 1. Поиск через ансамбль ретриверов
            logger.info("🔍 Поиск через ансамбль ретриверов...")
            retrieved_docs = self.ensemble_retriever.retrieve(question, top_k=top_k * 2)

            if not retrieved_docs:
                return {
                    "success": False,
                    "answer": "❌ Не найдено релевантных документов в базе законодательства РФ.",
                    "sources": [],
                    "error": "No documents found"
                }

            logger.info(f"📊 Найдено документов: {len(retrieved_docs)}")

            # 2. Формирование контекста
            context_parts = []
            source_details = []

            for i, doc in enumerate(retrieved_docs[:top_k]):
                # Получаем метаданные
                source_title = doc.metadata.get("source_title", "Неизвестный источник")
                hierarchy = doc.metadata.get("hierarchy_str", "")
                law_id = doc.metadata.get("law_id", "")

                # Форматируем источник
                source_info = f"{source_title}"
                if law_id:
                    source_info += f" ({law_id})"
                if hierarchy:
                    source_info += f", {hierarchy}"

                # Добавляем в контекст
                context_parts.append(f"【Источник {i + 1}: {source_info}】\n{doc.page_content}")

                # Сохраняем детали для вывода
                source_details.append({
                    "id": i + 1,
                    "title": source_title,
                    "hierarchy": hierarchy,
                    "law_id": law_id,
                    "content_preview": doc.page_content[:200] + "..." if len(
                        doc.page_content) > 200 else doc.page_content,
                    "retriever_type": doc.metadata.get("retriever_type", "unknown"),
                    "score": doc.metadata.get("rerank_score", 0) if doc.metadata.get(
                        "rerank_score") else doc.metadata.get("keyword_score", 0)
                })

            context_text = "\n\n".join(context_parts)

            # 3. Создание промпта
            prompt = self._create_legal_prompt(question, context_text)

            # 4. Генерация ответа
            logger.info("💭 Генерация ответа...")
            messages = [
                SystemMessage(content="Ты — точный и надежный юридический ассистент по законодательству РФ."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content

            logger.info("✅ Ответ сгенерирован")

            # 5. Подготовка результата
            exec_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": source_details,
                "documents_used": len(retrieved_docs[:top_k]),
                "total_documents_found": len(retrieved_docs),
                "execution_time": round(exec_time, 2),
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке запроса: {e}")
            return {
                "success": False,
                "answer": f"❌ Ошибка системы: {str(e)[:200]}",
                "sources": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def save_result(self, result: Dict[str, Any], output_dir: str = "rag_results") -> str:
        """Сохранение результата в файл"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.json"
            filepath = output_path / filename

            # Форматируем для сохранения
            save_data = {
                "question": result.get("question", ""),
                "answer": result.get("answer", ""),
                "success": result.get("success", False),
                "metadata": {
                    "documents_used": result.get("documents_used", 0),
                    "total_documents_found": result.get("total_documents_found", 0),
                    "execution_time": result.get("execution_time", 0),
                    "timestamp": result.get("timestamp", "")
                },
                "sources": result.get("sources", [])
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Результат сохранен: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            return ""


# ==================== ИНТЕРФЕЙС ====================

def main():
    """Основная функция"""
    print("\n" + "=" * 60)
    print("⚖️  ЮРИДИЧЕСКИЙ ENSEMBLE RAG АССИСТЕНТ")
    print("=" * 60)
    print("Архитектура: Ансамбль ретриверов + Re-ранкинг")
    print("=" * 60)

    try:
        # Инициализация системы
        print("\n🔄 Инициализация системы...")
        rag_system = LegalEnsembleRAG(Path("NPA3001"))

        # Примеры вопросов
        example_questions = [
            "Какие основания для увольнения работника по инициативе работодателя согласно ТК РФ?",
            "Что такое неустойка и как она рассчитывается по ГК РФ?",
            "Какие налоги должен платить ИП на упрощенной системе налогообложения?",
            "Как происходит наследование по закону согласно ГК РФ?",
            "Какая ответственность предусмотрена за нарушение правил дорожного движения?"
        ]

        # Интерактивный цикл
        while True:
            print("\n" + "=" * 60)
            print("МЕНЮ:")
            print("1. 💬 Задать вопрос")
            print("2. 📋 Выбрать пример")
            print("3. 🧪 Тест системы")
            print("4. 🚪 Выход")

            try:
                choice = input("\nВыберите действие (1-4): ").strip()

                if choice == "1":
                    question = input("\nВведите ваш юридический вопрос:\n> ").strip()
                    if question:
                        process_question(rag_system, question)

                elif choice == "2":
                    print("\nПримеры вопросов:")
                    for i, q in enumerate(example_questions, 1):
                        print(f"{i}. {q}")

                    try:
                        q_num = input(f"\nВыберите номер (1-{len(example_questions)}): ").strip()
                        idx = int(q_num) - 1
                        if 0 <= idx < len(example_questions):
                            process_question(rag_system, example_questions[idx])
                        else:
                            print("❌ Неверный номер")
                    except Exception:
                        print("❌ Неверный ввод")

                elif choice == "3":
                    print("\n🧪 Запуск теста системы...")
                    test_results = []

                    for i, q in enumerate(example_questions[:2], 1):
                        print(f"\n[{i}/2] Тестирование: {q[:50]}...")
                        start = datetime.now()
                        result = rag_system.query(q, top_k=5)
                        exec_time = (datetime.now() - start).total_seconds()

                        status = "✅" if result["success"] else "❌"
                        print(f"   {status} Время: {exec_time:.1f}с, Документов: {result.get('documents_used', 0)}")
                        test_results.append(result["success"])

                    success_rate = sum(test_results) / len(test_results) * 100
                    print(f"\n📊 Результат теста: {success_rate:.0f}% успешных запросов")

                elif choice == "4":
                    print("\n👋 Завершение работы")
                    break

                else:
                    print("❌ Неверный выбор")

            except KeyboardInterrupt:
                print("\n\n⚠ Прервано пользователем")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

    except Exception as e:
        print(f"❌ Критическая ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()


def process_question(rag_system: LegalEnsembleRAG, question: str):
    """Обработка одного вопроса"""
    result = rag_system.query(question)

    print("\n" + "=" * 60)
    print("📋 РЕЗУЛЬТАТ:")
    print("=" * 60)

    if result["success"]:
        print(f"\n💬 ОТВЕТ:\n{result['answer']}")

        if result.get("sources"):
            print(f"\n📚 ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:")
            for source in result["sources"]:
                print(f"\n• {source['title']}")
                if source['hierarchy']:
                    print(f"  📍 {source['hierarchy']}")
                if source['law_id']:
                    print(f"  🏛️  {source['law_id']}")
                print(f"  🔍 Тип поиска: {source.get('retriever_type', 'unknown')}")
                if source.get('score', 0) > 0:
                    print(f"  📊 Score: {source['score']:.3f}")

        print(f"\n📊 СТАТИСТИКА:")
        print(f"   • Время выполнения: {result['execution_time']}с")
        print(f"   • Использовано документов: {result['documents_used']}")
        print(f"   • Всего найдено: {result['total_documents_found']}")

        # Предложение сохранить
        try:
            save = input("\n💾 Сохранить результат? (y/n): ").strip().lower()
            if save == 'y':
                rag_system.save_result(result)
                print("✅ Результат сохранен")
        except Exception:
            pass

    else:
        print(f"\n❌ ОШИБКА:\n{result['answer']}")


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    # Проверка зависимостей
    required_packages = [
        ("langchain-community", "langchain_community"),
        ("langchain-openai", "langchain_openai"),
        ("faiss-cpu", "faiss")
    ]

    missing_packages = []
    for package_name, import_name in required_packages:
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        print("❌ Отсутствуют необходимые пакеты:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\nPython интерпретатор: {sys.executable}")
        print("\nУстановите их командой:")
        print(f"{sys.executable} -m pip install {' '.join(missing_packages)}")
        if FAISS_IMPORT_ERROR:
            print(f"\nПричина ошибки импорта FAISS: {FAISS_IMPORT_ERROR}")
        sys.exit(1)

    if not HAS_SKLEARN:
        print("⚠ scikit-learn/scipy недоступен: keyword-поиск будет отключен.")
        if SKLEARN_IMPORT_ERROR:
            print(f"Причина: {SKLEARN_IMPORT_ERROR}")

    try:
        __import__("sentence_transformers")
    except Exception:
        print("⚠ sentence-transformers недоступен: будет использован OpenAIEmbeddings, а re-ranker отключен.")

    # Запуск основной программы
    main()
