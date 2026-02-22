import requests
from bs4 import BeautifulSoup
import re
import json
import time
import hashlib
import uuid

BASE_URL = 'http://www.consultant.ru'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# ------------------------
# Список законов для парсинга
# ------------------------
LAWS = [
    {
        "law_id": "АПK",
        "source_title": "Арбитражно-процессуальный кодекс РФ",
        "source_type": "кодекс",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_37800/"
    },

    {
        "law_id": "ФЗ О банкротстве",
        "source_title": "Федеральный закон О несостоятельности (банкротстве) от 26.10.2002 N 127-ФЗ",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_39331/"
    },

    {
        "law_id": "ФЗ О полиции",
        "source_title": "Федеральный закон О полиции от 07.02.2011 N 3-ФЗ",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_110165/"
    },

    {
        "law_id": "ФКЗ О Верховном суде",
        "source_title": "Федеральный конституционный закон от 05.02.2014 N 3-ФКЗ (ред. от 14.07.2022) О Верховном Суде Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_158641/"
    },

    {
        "law_id": "ФКЗ О Конституционном суде",
        "source_title": "Федеральный конституционный закон от 21.07.1994 N 1-ФКЗ (ред. от 31.07.2023) О Конституционном Суде Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_4172/"
    },

    {
        "law_id": "ФКЗ о судах общей юрисдикции",
        "source_title": "Федеральный конституционный закон от 07.02.2011 N 1-ФКЗ (ред. от 23.07.2025) О судах общей юрисдикции в Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_110271/"
    },

    {
        "law_id": "ФКЗ О военных судах",
        "source_title": "Федеральный конституционный закон от 23.06.1999 N 1-ФКЗ (ред. от 29.12.2025) О военных судах Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_23479/"
    },

    {
        "law_id": "ФКЗ ОБ арбитражных судах",
        "source_title": "Федеральный конституционный закон от 28.04.1995 N 1-ФКЗ (ред. от 31.07.2023) Об арбитражных судах в Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_6510/"
    },

    {
        "law_id": "ФЗ О мировых судьях",
        "source_title": "Федеральный закон О мировых судьях в Российской Федерации от 17.12.1998 N 188-ФЗ (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_21335/"
    },

    {
        "law_id": "ФЗ О прокуратуре",
        "source_title": "Федеральный закон О прокуратуре Российской Федерации от 17.01.1992 N 2202-1 (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_19671/"
    },

    {
        "law_id": "ФЗ О прокуратуре",
        "source_title": "Федеральный закон О прокуратуре Российской Федерации от 17.01.1992 N 2202-1 (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_262/"
    },

    {
        "law_id": "ФЗ О СК РФ",
        "source_title": "Федеральный закон О Следственном комитете Российской Федерации от 28.12.2010 N 403-ФЗ (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_108565/"
    },

    {
        "law_id": "ФЗ О ФСБ",
        "source_title": "Федеральный закон О федеральной службе безопасности от 03.04.1995 N 40-ФЗ (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_6300/"
    },

    {
        "law_id": "ФЗ Об органах судебного сообщетсва",
        "source_title": "Федеральный закон Об органах судейского сообщества в Российской Федерации от 14.03.2002 N 30-ФЗ (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_35868/"
    },

    {
        "law_id": "ФКЗ О судебной системе РФ",
        "source_title": "Федеральный конституционный закон от 31.12.1996 N 1-ФКЗ (ред. от 29.12.2025) О судебной системе Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_12834/"
    },

    {
        "law_id": "ФКЗ О Правительстве",
        "source_title": "Федеральный конституционный закон от 06.11.2020 N 4-ФКЗ (ред. от 28.12.2025) О Правительстве Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_366950/"
    },

    {
        "law_id": "ФЗ Об уполномоченном по правам человека",
        "source_title": "Федеральный конституционный закон от 26.02.1997 N 1-ФКЗ (ред. от 29.05.2023) Об Уполномоченном по правам человека в Российской Федерации",
        "source_type": "Федеральный конституционный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_13440/"
    },

    {
        "law_id": "ФЗ О внешней разведке",
        "source_title": "Федеральный закон О внешней разведке от 10.01.1996 N 5-ФЗ (последняя редакция)",
        "source_type": "Федеральный закон",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_8842/"
    },

    {
        "law_id": "Постановление Пленума Верховного Суда РФ от 23.06.2015 N 25",
        "source_title": "Постановление Пленума Верховного Суда РФ от 23.06.2015 N 25 О применении судами некоторых положений раздела I части первой Гражданского кодекса Российской Федерации",
        "source_type": "Постановление Пленума Верховного Суда РФ",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_181602/"
    },

    {
        "law_id": "Постановление Пленума Верховного Суда РФ от 29.09.2015 N 43",
        "source_title": "Постановление Пленума Верховного Суда РФ от 29.09.2015 N 43 (ред. от 22.06.2021) О некоторых вопросах, связанных с применением норм Гражданского кодекса Российской Федерации об исковой давности",
        "source_type": "Постановление Пленума Верховного Суда РФ",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_186662/"
    },

    {
        "law_id": "Постановление Пленума Верховного Суда РФ от 30.06.2015 N 28",
        "source_title": "Постановление Пленума Верховного Суда РФ от 30.06.2015 N 28 О некоторых вопросах, возникающих при рассмотрении судами дел об оспаривании результатов определения кадастровой стоимости объектов недвижимости",
        "source_type": "Постановление Пленума Верховного Суда РФ",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_181899/"
    },

    {
        "law_id": "Постановление Пленума Верховного Суда РФ от 23.06.2015 N 25",
        "source_title": "Постановление Пленума Верховного Суда РФ от 23.06.2015 N 25 О применении судами некоторых положений раздела I части первой Гражданского кодекса Российской Федерации",
        "source_type": "Постановление Пленума Верховного Суда РФ",
        "start_url": "https://www.consultant.ru/document/cons_doc_LAW_181602/"
    },


    # Добавляйте остальные законы
]

# ------------------------
# Функции
# ------------------------
def clean(text):
    """
    Очистка текста закона для RAG:
    - убирает лишние пробелы, табы и переносы строк
    - заменяет неразрывные пробелы на обычные
    - удаляет сноски вида [1], [2], [10]
    - удаляет строки типа "Документ обновлён" и исторические пометки
    - удаляет остатки HTML-тегов
    """
    if not text:
        return ""

    # Лишние пробелы и табы
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = text.replace('\xa0', ' ')

    # Убираем сноски
    text = re.sub(r'\[\d+\]', '', text)

    # Игнорируем исторические пометки
    if re.match(r'^\(в ред\.|^\(см\. текст в предыдущей редакции\)', text, re.I):
        return ""

    # Строки "Документ обновлён …"
    text = re.sub(r'Документ.*обновлён.*', '', text, flags=re.I)

    # HTML-теги
    text = re.sub(r'<.*?>', '', text)

    # Лишние переносы строк
    text = re.sub(r'\n+', '\n', text)

    return text.strip()

def make_chunk_id(law_id, hierarchy):
    base_str = law_id + "__" + "__".join(filter(None, hierarchy))
    return hashlib.md5(base_str.encode('utf-8')).hexdigest()

def split_text_chunks(text, chunk_size=1000, overlap=200):
    """Разбивает текст на чанки фиксированного размера с перекрытием"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_soup(url):
    time.sleep(1)  # задержка для защиты сервера
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def parse_article_page(url, hierarchy, law_id, source_type, source_title):
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

        # ----------------------
        # Определяем иерархию (но не создаём отдельные чанки)
        # ----------------------
        if re.match(r'^Часть\s+\d+', text, re.I):
            current_part = text
            current_section = None
            current_chapter = None
            current_article = None
            current_point = None
            continue
        if re.match(r'^Раздел\s+[IVXLCDM]+', text, re.I):
            current_section = text
            current_chapter = None
            current_article = None
            current_point = None
            continue
        if re.match(r'^Подраздел\s+\d+', text, re.I):
            current_subsection = text
            continue
        if re.match(r'^Глава\s+\d+', text, re.I):
            current_chapter = text
            current_article = None
            current_point = None
            continue
        m_article = re.match(r'^Статья\s+\d+(\.\d+)?', text)
        if m_article:
            current_article = text
            current_point = None
            continue

        # ----------------------
        # Определяем пункты и подпункты
        # ----------------------
        if re.match(r'^\d+\.', text):
            current_point = f"Пункт {text.split('.')[0]}"
        elif re.match(r'^[а-яё]\)', text, re.I):
            current_subpoint = f"Подпункт {text.split(')')[0]})"
        else:
            current_subpoint = None

        # ----------------------
        # Формируем иерархию
        # ----------------------
        hierarchy_extended = list(filter(None, [
            current_part,
            current_section,
            current_chapter,
            current_article,
            current_point,
            current_subpoint
        ]))

        if not hierarchy_extended:
            continue  # пропускаем заголовки без текста

        hierarchy_str = " > ".join(hierarchy_extended)
        chunk_id = make_chunk_id(law_id, hierarchy_extended)

        # ----------------------
        # Разбиваем текст статьи / пункта на чанки, только если это значимый текст
        # ----------------------
        if len(text) > 30:  # короткие заголовки игнорируем
            chunks = split_text_chunks(text)
            for chunk_text in chunks:
                results.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "source_type": source_type,
                    "source_title": source_title,
                    "hierarchy": hierarchy_extended,
                    "hierarchy_str": hierarchy_str,
                    "law_id": law_id,
                    "chunk_id": chunk_id,
                    "source_url": url
                })

    return results

def parse_law(law):
    soup = get_soup(law["start_url"])
    toc_links = soup.select('div.document-page__toc a[href]')
    all_data = []
    for link in toc_links:
        href = link['href']
        if not href.startswith('/document/cons_doc_LAW'):
            continue
        article_name = clean(link.text)
        full_url = BASE_URL + href
        print(f"[+] Обрабатываем: {article_name} → {full_url}")
        try:
            entries = parse_article_page(full_url, [article_name], law["law_id"], law["source_type"], law["source_title"])
            all_data.extend(entries)
        except Exception as e:
            print(f"[!] Ошибка при обработке {full_url}: {e}")
    return all_data

# ------------------------
# Главный запуск
# ------------------------
if __name__ == "__main__":
    OUTPUT_FILE = "apkdoplen.jsonl"
    all_chunks = []
    for law in LAWS:
        law_chunks = parse_law(law)
        all_chunks.extend(law_chunks)

    # Сохраняем JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_chunks:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"[+] Готово! Всего чанков: {len(all_chunks)}. Сохранено в {OUTPUT_FILE}")
