import requests
from bs4 import BeautifulSoup
import uuid
import re
import json

HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- Функции ---
def fetch_soup(url):
    resp = requests.get(url, headers=HEADERS)
    resp.encoding = 'utf-8'
    return BeautifulSoup(resp.text, 'html.parser')

def detect_source_type(title):
    title_lower = title.lower()
    if "федеральный конституционный закон" in title_lower:
        return "ФКЗ"
    elif "федеральный закон" in title_lower:
        return "ФЗ"
    elif "кодекс" in title_lower:
        return "кодекс"
    elif "указ президента" in title_lower:
        return "указ"
    elif "постановление пленума" in title_lower:
        return "постановление пленума"
    return "документ"

def extract_law_id(title, source_type):
    if source_type == "ФКЗ":
        match = re.search(r'N\s*(\d+)-ФКЗ', title)
        if match:
            return f"FKZ_{match.group(1)}"
    elif source_type == "ФЗ":
        match = re.search(r'N\s*(\d+)-ФЗ', title)
        if match:
            return f"FZ_{match.group(1)}"
    else:
        cleaned = re.sub(r'\W+', "_", title)
        return cleaned[:30]
    return "UNKNOWN"

def is_valid_chunk(text):
    text = text.strip().lower()
    if not text:
        return False
    garbage_patterns = [
        r'содержание', r'введение', r'примечания', r'общие положения'
    ]
    for pat in garbage_patterns:
        if re.search(pat, text):
            return False
    return True

def clean_text(text):
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def parse_page(url, source_type, law_id):
    soup = fetch_soup(url)
    law_title_tag = soup.select_one(".document-page__title-link a")
    law_title = law_title_tag.text.strip() if law_title_tag else "Unknown Title"

    chunks = []
    hierarchy_stack = []

    content_divs = soup.select(".document-page__content .document__style, .full-text__wrapper")
    for div in content_divs:
        for elem in div.descendants:
            if elem.name in ['h1','h2','h3','h4','h5','h6']:
                text = elem.get_text(strip=True)
                if not text:
                    continue

                # Определяем уровень заголовка
                level = int(elem.name[1])
                while len(hierarchy_stack) >= level:
                    hierarchy_stack.pop()
                hierarchy_stack.append(text)

            elif elem.name == 'p' or elem.name == 'div':
                text = elem.get_text(strip=True)
                if text and is_valid_chunk(text):
                    norm_text = clean_text(text)
                    article_short = hierarchy_stack[-1].replace(" ", "_") if hierarchy_stack else "chunk"
                    chunk = {
                        "id": str(uuid.uuid4()),
                        "text": norm_text,
                        "source_type": source_type,
                        "source_title": law_title,
                        "hierarchy": hierarchy_stack.copy(),
                        "hierarchy_str": " → ".join(hierarchy_stack),
                        "law_id": law_id,
                        "chunk_id": f"{law_id}__{article_short}",
                        "source_url": url
                    }
                    chunks.append(chunk)

    # --- Ищем подстраницы ---
    subpage_links = []
    for a in soup.select(".document-page__menu a"):
        href = a.get('href')
        if href and href.startswith("/document/"):
            full_url = "https://www.consultant.ru" + href
            if full_url != url:
                subpage_links.append(full_url)

    return chunks, subpage_links

def parse_law(url):
    soup = fetch_soup(url)
    law_title_tag = soup.select_one(".document-page__title-link a")
    law_title = law_title_tag.text.strip() if law_title_tag else "Unknown Title"
    source_type = detect_source_type(law_title)
    law_id = extract_law_id(law_title, source_type)

    all_chunks = []
    to_visit = [url]
    visited = set()

    while to_visit:
        page_url = to_visit.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)
        try:
            chunks, subpages = parse_page(page_url, source_type, law_id)
            all_chunks.extend(chunks)
            for link in subpages:
                if link not in visited:
                    to_visit.append(link)
        except Exception as e:
            print(f"Ошибка при парсинге {page_url}: {e}")

    return all_chunks

def parse_multiple_laws(url_list):
    all_chunks = []
    for url in url_list:
        print(f"Парсим закон: {url}")
        chunks = parse_law(url)
        all_chunks.extend(chunks)
        print(f"  Чанков получено: {len(chunks)}")
    return all_chunks

# --- Использование ---
if __name__ == "__main__":
    law_links = [
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        "https://www.consultant.ru/document/cons_doc_LAW_4172/",  # ФКЗ пример
        "https://www.consultant.ru/document/cons_doc_LAW_37800/",
        # АПК РФ пример
        # Добавь свои ссылки на ФЗ, указы, постановления пленума
    ]

    all_chunks = parse_multiple_laws(law_links)

    with open("laws_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Парсинг завершен. Всего чанков: {len(all_chunks)}")
