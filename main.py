import requests
from bs4 import BeautifulSoup
import re
import json
import time
import hashlib

BASE_URL = 'http://www.consultant.ru'
START_URL = 'https://www.consultant.ru/document/cons_doc_LAW_34481'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
OUTPUT_FILE = 'upk_rf.jsonl'
LAW_ID = 'upk_rf'

def get_soup(url):
    time.sleep(1)
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def clean(text):
    return re.sub(r'\s+', ' ', text.strip())

def make_chunk_id(hierarchy):
    return hashlib.md5('_'.join(filter(None, hierarchy)).encode()).hexdigest()

def parse_article_page(url, hierarchy):
    soup = get_soup(url)
    content = soup.find('div', class_='document-page__content')
    if not content:
        return []

    results = []
    current_article = hierarchy[-1] if hierarchy else ''
    current_part = None
    current_point = None

    for el in content.find_all(['p', 'h3', 'h4']):
        text = clean(el.get_text())

        # Пункт
        if re.match(r'^\d+\.', text):
            current_part = f"Часть {text.split('.')[0]}"
            current_point = None
            hierarchy_extended = hierarchy + [current_part]
        elif re.match(r'^[а-яё]\)', text):
            current_point = f"Пункт {text.split(')')[0]})"
            hierarchy_extended = hierarchy + [current_part, current_point]
        else:
            hierarchy_extended = hierarchy + [current_part] if current_part else hierarchy

        entry = {
            'law_id': LAW_ID,
            'hierarchy': hierarchy_extended,
            'hierarchy_str': ' > '.join(filter(None, hierarchy_extended)),
            'chunk_id': make_chunk_id(hierarchy_extended),
            'text': text,
            'source_url': url
        }
        results.append(entry)

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
        print(f"[+] {article_name}: {full_url}")

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
