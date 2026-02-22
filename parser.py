import requests
from bs4 import BeautifulSoup
import re
import json
import time
import hashlib
import uuid

BASE_URL = 'http://www.consultant.ru'
START_URL = 'https://www.consultant.ru/document/cons_doc_LAW_34683'  # пример — УК РФ
HEADERS = {'User-Agent': 'Mozilla/5.0'}
OUTPUT_FILE = 'LABOR CODE_rf_structured.jsonl'

LAW_ID = 'LABOR CODE'
SOURCE_TITLE = 'ТРУДОВОЙ КОДЕКС'
SOURCE_TYPE = 'кодекс'  # или ФЗ, указ, приказ и т.п.

def get_soup(url):
    time.sleep(1)  # задержка, чтобы не перегружать сервер
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def clean(text):
    # Очистка от лишних пробелов, переносов
    return re.sub(r'\s+', ' ', text.strip())

def make_chunk_id(law_id, hierarchy):
    # Хешируем строку из law_id и иерархии для получения уникального chunk_id
    base_str = law_id + "__" + "__".join(filter(None, hierarchy))
    return hashlib.md5(base_str.encode('utf-8')).hexdigest()

def parse_article_page(url, hierarchy):
    soup = get_soup(url)
    content = soup.find('div', class_='document-page__content')
    if not content:
        print(f"[!] Не найден основной контент на странице {url}")
        return []

    results = []
    current_part = None
    current_section = None
    current_chapter = None
    current_article = hierarchy[-1] if hierarchy else None
    current_point = None

    for el in content.find_all(['h1', 'h2', 'h3', 'p', 'strong']):
        text = clean(el.get_text())

        if not text:
            continue

        # Определяем уровень иерархии по тексту

        # Часть: "Часть 1", "Часть 2" и т.п.
        if re.match(r'^Часть\s+\d+', text, re.I):
            current_part = text
            current_section = None
            current_chapter = None
            current_article = None
            current_point = None
            continue

        # Раздел: "Раздел I", "Раздел II"
        if re.match(r'^Раздел\s+[IVXLCDM]+', text, re.I):
            current_section = text
            current_chapter = None
            current_article = None
            current_point = None
            continue

        # Глава: "Глава 1", "Глава 2"
        if re.match(r'^Глава\s+\d+', text, re.I):
            current_chapter = text
            current_article = None
            current_point = None
            continue

        # Статья: "Статья 12", "Статья 12.1"
        m_article = re.match(r'^Статья\s+\d+(\.\d+)?', text)
        if m_article:
            current_article = text
            current_point = None
            continue

        # Пункт: "1.", "2." и т.п.
        if re.match(r'^\d+\.', text):
            current_point = f"Пункт {text.split('.')[0]}"
            # Строим иерархию для пункта
            hierarchy_extended = list(filter(None, [current_part, current_section, current_chapter, current_article, current_point]))
            hierarchy_str = " > ".join(hierarchy_extended)
            chunk_id = make_chunk_id(LAW_ID, hierarchy_extended)

            results.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": SOURCE_TYPE,
                "source_title": SOURCE_TITLE,
                "hierarchy": hierarchy_extended,
                "hierarchy_str": hierarchy_str,
                "law_id": LAW_ID,
                "chunk_id": chunk_id,
                "source_url": url
            })
            continue

        # Подпункт: "а)", "б)" и т.п.
        if re.match(r'^[а-яё]\)', text, re.I):
            sub_point = f"Подпункт {text.split(')')[0]})"
            hierarchy_extended = list(filter(None, [current_part, current_section, current_chapter, current_article, current_point, sub_point]))
            hierarchy_str = " > ".join(hierarchy_extended)
            chunk_id = make_chunk_id(LAW_ID, hierarchy_extended)

            results.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": SOURCE_TYPE,
                "source_title": SOURCE_TITLE,
                "hierarchy": hierarchy_extended,
                "hierarchy_str": hierarchy_str,
                "law_id": LAW_ID,
                "chunk_id": chunk_id,
                "source_url": url
            })
            continue

        # Если просто текст в рамках статьи, добавляем как абзац
        if current_article:
            hierarchy_extended = list(filter(None, [current_part, current_section, current_chapter, current_article]))
            hierarchy_str = " > ".join(hierarchy_extended)
            chunk_id = make_chunk_id(LAW_ID, hierarchy_extended)

            results.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_type": SOURCE_TYPE,
                "source_title": SOURCE_TITLE,
                "hierarchy": hierarchy_extended,
                "hierarchy_str": hierarchy_str,
                "law_id": LAW_ID,
                "chunk_id": chunk_id,
                "source_url": url
            })

    return results

def parse_main():
    soup = get_soup(START_URL)
    toc_links = soup.select('div.document-page__toc a[href]')
    all_data = []

    for link in toc_links:
        href = link['href']
        if not href.startswith('/document/cons_doc_LAW'):
            continue
        article_name = clean(link.text)
        full_url = BASE_URL + href
        print(f"[+] Обрабатываем: {article_name} → {full_url}")

        hierarchy = [article_name]
        try:
            entries = parse_article_page(full_url, hierarchy)
            all_data.extend(entries)
        except Exception as e:
            print(f"[!] Ошибка при обработке {full_url}: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"[+] Готово! Сохранено {len(all_data)} записей в {OUTPUT_FILE}")

if __name__ == '__main__':
    parse_main()
