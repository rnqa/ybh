import os
import json
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from tqdm import tqdm
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticLawRAG:
    """
    СЕМАНТИЧЕСКАЯ система поиска по законодательству
    """

    def __init__(self, qdrant_path: str = "./semantic_laws_db"):
        self.qdrant_path = qdrant_path
        self.model = None
        self.client = None

    def initialize_system(self):
        """Инициализация с оптимизированной моделью"""
        logger.info("🔄 Загрузка оптимизированной модели для семантического поиска...")

        # 🎯 ИСПОЛЬЗУЕМ МОДЕЛЬ ДЛЯ СЕМАНТИЧЕСКОГО ПОИСКА
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        self.client = QdrantClient(path=self.qdrant_path)
        logger.info("✅ Система инициализирована")

    def extract_semantic_chunks(self, law_item: dict) -> list:
        """
        РАЗБИВАЕМ ТЕКСТ НА СЕМАНТИЧЕСКИЕ ЧАНКИ
        Это ключевое улучшение!
        """
        original_text = law_item.get("text", "")
        if not original_text or len(original_text.strip()) < 10:
            return []

        # 🎯 ОСНОВНАЯ ИДЕЯ: создаем семантические фрагменты
        semantic_chunks = []

        # 1. Извлекаем ключевые фразы из текста
        sentences = self._split_into_sentences(original_text)

        # 2. Создаем семантические группы
        chunks = self._create_semantic_chunks(sentences, law_item)

        return chunks

    def _split_into_sentences(self, text: str) -> list:
        """Разбиваем текст на предложения"""
        # Простое разбиение по точкам, восклицательным и вопросительным знакам
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _create_semantic_chunks(self, sentences: list, law_item: dict) -> list:
        """Создаем семантические чанки"""
        chunks = []

        # 🎯 СТРАТЕГИЯ 1: Создаем чанки по смыслу
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Если текущий чанк стал слишком большим или начинается новая тема
            if current_length + sentence_length > 200 or self._is_new_topic(sentence):
                if current_chunk:
                    # Сохраняем текущий чанк
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) > 30:  # Минимальная длина
                        chunks.append(self._create_chunk_data(chunk_text, law_item))

                    # Начинаем новый чанк
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Добавляем последний чанк
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) > 30:
                chunks.append(self._create_chunk_data(chunk_text, law_item))

        # 🎯 СТРАТЕГИЯ 2: Если мало чанков, создаем ключевые фразы
        if len(chunks) < 2 and sentences:
            key_phrases = self._extract_key_phrases(sentences, law_item)
            chunks.extend(key_phrases)

        return chunks

    def _is_new_topic(self, sentence: str) -> bool:
        """Определяем, начинается ли новая тема"""
        new_topic_indicators = [
            'также', 'кроме того', 'при этом', 'в то же время',
            'с другой стороны', 'в соответствии', 'на основании',
            'статья', 'пункт', 'часть'
        ]

        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in new_topic_indicators)

    def _create_chunk_data(self, chunk_text: str, law_item: dict) -> dict:
        """Создаем данные для чанка"""
        return {
            "text": chunk_text,
            "source_type": law_item.get("source_type", ""),
            "source_title": law_item.get("source_title", ""),
            "hierarchy": law_item.get("hierarchy", []),
            "hierarchy_str": law_item.get("hierarchy_str", ""),
            "law_id": law_item.get("law_id", ""),
            "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:16],
            "source_url": law_item.get("source_url", ""),
            "original_id": law_item.get("id", ""),
            "is_semantic_chunk": True  # Помечаем как семантический чанк
        }

    def _extract_key_phrases(self, sentences: list, law_item: dict) -> list:
        """Извлекаем ключевые фразы для улучшения поиска"""
        key_chunks = []

        # Простая стратегия: берем первые 2-3 предложения и ключевые фразы
        if len(sentences) >= 2:
            # Основной смысл
            main_idea = " ".join(sentences[:2])
            key_chunks.append(self._create_chunk_data(main_idea, law_item))

        # Ищем определения и ключевые понятия
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in
                   ['означает', 'определение', 'является', 'полномочия', 'обязан']):
                if len(sentence) > 20:
                    key_chunks.append(self._create_chunk_data(sentence, law_item))

        return key_chunks

    def rebuild_semantic_database(self, laws_folder: str):
        """
        ПЕРЕСТРОЙКА БАЗЫ ДАННЫХ с семантическими чанками
        """
        print("🚀 ПЕРЕСТРОЙКА БАЗЫ ДАННЫХ С СЕМАНТИЧЕСКИМ ПОИСКОМ")
        print("=" * 60)

        self.initialize_system()

        # Загружаем исходные данные
        laws_data = self._load_original_data(laws_folder)
        if not laws_data:
            return

        # Создаем семантические чанки
        semantic_data = []
        for law_item in tqdm(laws_data, desc="🎯 Создание семантических чанков"):
            chunks = self.extract_semantic_chunks(law_item)
            semantic_data.extend(chunks)

        print(f"📊 Создано {len(semantic_data)} семантических чанков")

        # Создаем коллекцию
        self._create_semantic_collection()

        # Генерируем эмбеддинги и загружаем
        points = self._generate_semantic_embeddings(semantic_data)
        self._upload_semantic_data(points)

        print("🎉 СЕМАНТИЧЕСКАЯ БАЗА ДАННЫХ ГОТОВА!")

    def _load_original_data(self, folder_path: str) -> list:
        """Загрузка исходных данных"""
        laws_data = []

        if not os.path.exists(folder_path):
            print(f"❌ Папка {folder_path} не существует!")
            return []

        json_files = [f for f in os.listdir(folder_path) if f.endswith(('.json', '.jsonl'))]

        for file_name in tqdm(json_files, desc="📖 Загрузка файлов"):
            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            law_item = json.loads(line)
                            laws_data.append(law_item)
            except Exception as e:
                continue

        print(f"✅ Загружено {len(laws_data)} исходных записей")
        return laws_data

    def _create_semantic_collection(self):
        """Создание семантической коллекции"""
        try:
            if self.client.collection_exists("semantic_laws"):
                self.client.delete_collection("semantic_laws")

            vector_size = self.model.get_sentence_embedding_dimension()

            self.client.create_collection(
                collection_name="semantic_laws",
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Семантическая коллекция создана (размер векторов: {vector_size})")
        except Exception as e:
            print(f"❌ Ошибка создания коллекции: {e}")

    def _generate_semantic_embeddings(self, semantic_data: list):
        """Генерация эмбеддингов для семантических чанков"""
        texts = [item["text"] for item in semantic_data]

        print("🔢 Создание семантических эмбеддингов...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=128,  # Увеличиваем для скорости
            normalize_embeddings=True
        )

        points = []
        for i, (item, embedding) in enumerate(zip(semantic_data, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "semantic_text": item["text"],
                    "original_text": item.get("text", ""),
                    "source_title": item.get("source_title", ""),
                    "hierarchy_str": item.get("hierarchy_str", ""),
                    "law_id": item.get("law_id", ""),
                    "is_semantic": True
                }
            )
            points.append(point)

        return points

    def _upload_semantic_data(self, points: list):
        """Загрузка семантических данных"""
        print(f"📤 Загрузка {len(points)} семантических векторов...")

        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="⬆️ Загрузка"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name="semantic_laws",
                points=batch
            )

    def semantic_search(self, query: str, limit: int = 10):
        """
        СЕМАНТИЧЕСКИЙ ПОИСК - полностью переработанный
        """
        if self.model is None:
            self.initialize_system()

        # 🎯 УЛУЧШЕНИЕ: расширяем запрос для лучшего поиска
        enhanced_query = self._enhance_search_query(query)

        print(f"🔍 Семантический поиск: '{query}'")
        print(f"🎯 Улучшенный запрос: '{enhanced_query}'")

        query_embedding = self.model.encode([enhanced_query]).tolist()[0]

        results = self.client.search(
            collection_name="semantic_laws",
            query_vector=query_embedding,
            limit=limit
        )

        return results

    def _enhance_search_query(self, query: str) -> str:
        """Улучшаем поисковый запрос для лучшего семантического поиска"""
        query_lower = query.lower()

        # Добавляем синонимы и связанные понятия
        enhancements = []

        # Юридические синонимы
        legal_synonyms = {
            'полномочия': ['компетенция', 'права', 'функции', 'обязанности'],
            'верховный суд': ['вс рф', 'верховный суд российской федерации'],
            'обязан': ['должен', 'обязанности'],
            'права': ['полномочия', 'компетенция'],
            'суд': ['судья', 'судьи', 'судебный']
        }

        for term, synonyms in legal_synonyms.items():
            if term in query_lower:
                enhancements.extend(synonyms)

        enhanced_query = query
        if enhancements:
            enhanced_query += " " + " ".join(enhancements)

        return enhanced_query

    def test_semantic_search(self):
        """Тестирование семантического поиска"""
        test_queries = [
            "полномочия верховного суда",
            "права судей",
            "обжалование судебных решений",
            "уголовное дело",
            "гражданский процесс",
            "административные правонарушения"
        ]

        print("\n🧪 ТЕСТИРОВАНИЕ СЕМАНТИЧЕСКОГО ПОИСКА")
        print("=" * 60)

        for query in test_queries:
            print(f"\n🔍 Запрос: '{query}'")
            results = self.semantic_search(query, limit=3)

            if results:
                for i, result in enumerate(results):
                    print(f"   {i + 1}. 📚 {result.payload.get('source_title', '')}")
                    print(f"      🏛️ {result.payload.get('hierarchy_str', '')}")
                    print(f"      📝 {result.payload.get('semantic_text', '')[:100]}...")
                    print(f"      🎯 Схожесть: {result.score:.4f}")
            else:
                print("   ❌ Ничего не найдено")


def main():
    """
    Запуск семантической системы
    """
    print("🤖 СЕМАНТИЧЕСКАЯ СИСТЕМА ПОИСКА ПО ЗАКОНОДАТЕЛЬСТВУ")
    print("=" * 50)

    semantic_rag = SemanticLawRAG()

    laws_folder = "D:\HelperYoristBot\Laws"  # Ваша папка

    # 🏗️ ПЕРЕСТРОЙКА БАЗЫ (запустите один раз)
    print("1. 🏗️ Перестроить семантическую базу данных")
    print("2. 🔍 Тестировать поиск по существующей базе")

    choice = input("Выберите действие (1 или 2): ").strip()

    if choice == "1":
        semantic_rag.rebuild_semantic_database(laws_folder)

    # Тестируем поиск
    semantic_rag.test_semantic_search()


if __name__ == "__main__":
    main()