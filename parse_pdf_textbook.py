import uuid
import fitz  # PyMuPDF
import re
import json

# ======== Конфигурация =========
pdf_path = "E:\\HelperYoristBot\\teor_gos_prava.pdf"  # Укажи путь к своему PDF
output_jsonl = "teor_gos_prava_UCH.jsonl"
chunk_size_words = 500

source_type = "учебник"
source_title = "Учебник по ТГП"
law_id = "teor_gos_prava_UCH"
source_url = None
# ===============================

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((i + 1, text))
    return pages

def chunk_text(text, size_words):
    words = text.split()
    for i in range(0, len(words), size_words):
        yield " ".join(words[i:i + size_words])

def extract_hierarchy(text):
    """
    Ищет заголовки типа: Глава 1, Раздел 2, Параграф 3
    """
    patterns = [
        r"(Глава\s+\d+[а-яА-Я]*)",
        r"(Раздел\s+\d+[а-яА-Я]*)",
        r"(Параграф\s+\d+[а-яА-Я]*)",
        r"(Тема\s+\d+[а-яА-Я]*)"
    ]

    hierarchy = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hierarchy.append(match.group(1).strip())

    return hierarchy

def create_fragments(pages, chunk_size_words):
    fragments = []
    chunk_idx = 0

    for page_number, page_text in pages:
        page_hierarchy = extract_hierarchy(page_text)

        for chunk in chunk_text(page_text, chunk_size_words):
            hierarchy = extract_hierarchy(chunk)
            final_hierarchy = hierarchy or page_hierarchy
            hierarchy_str = " > ".join(final_hierarchy)

            fragment = {
                "id": str(uuid.uuid4()),
                "text": chunk.strip(),
                "source_type": source_type,
                "source_title": source_title,
                "hierarchy": final_hierarchy,
                "hierarchy_str": hierarchy_str,
                "law_id": law_id,
                "chunk_id": f"{law_id}__{chunk_idx}",
                "source_url": source_url,
                "page_number": page_number
            }
            fragments.append(fragment)
            chunk_idx += 1

    return fragments

def save_to_jsonl(fragments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for frag in fragments:
            f.write(json.dumps(frag, ensure_ascii=False) + "\n")

# ========== Запуск ==========
pages = extract_text_by_page(pdf_path)
fragments = create_fragments(pages, chunk_size_words)
save_to_jsonl(fragments, output_jsonl)
print(f"✅ Успешно. Фрагментов: {len(fragments)}. Сохранено в: {output_jsonl}")
