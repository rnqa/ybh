#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASELINE_ADDITIONS = {
    "codes": [
        "Конституция РФ",
        "КоАП РФ",
        "КАС РФ",
        "Градостроительный кодекс РФ",
        "Водный кодекс РФ (опционально)",
        "Воздушный кодекс РФ (опционально)",
    ],
    "fkz": [
        "ФКЗ о референдуме РФ",
        "ФКЗ о чрезвычайном положении",
        "ФКЗ о военном положении",
    ],
    "laws_orgs": [
        "ФЗ о войсках национальной гвардии РФ (Росгвардия)",
        "ФЗ о внешней разведке (СВР)",
        "ФЗ о ФСИН / учреждениях и органах, исполняющих уголовные наказания",
        "Закон РФ о статусе судей в РФ",
        "ФЗ об адвокатской деятельности и адвокатуре",
        "Основы законодательства РФ о нотариате",
        "ФЗ о судебных приставах",
        "ФЗ об оперативно-розыскной деятельности",
        "ФЗ о государственной гражданской службе",
        "ФЗ о противодействии коррупции",
        "ФЗ о персональных данных",
    ],
    "plenum": [
        "Постановления Пленума ВС РФ по ГПК/АПК/УПК/КАС/КоАП (базовые разъяснения)",
    ],
    "prosecutor": [
        "Разъяснения/методические письма прокуратуры по ключевым вопросам",
    ],
    "textbooks": [
        "Конституционное право",
        "Административное право",
        "Уголовное право",
        "Гражданский процесс",
        "Арбитражный процесс",
        "Трудовое право",
        "Налоговое право",
        "Финансовое право",
        "Семейное право",
        "Земельное право",
        "Экологическое право",
        "Международное право (по необходимости)",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def collect_folder(folder: Path) -> Dict[str, Any]:
    files = sorted(folder.glob("*.jsonl"))
    file_stats: Dict[str, int] = {}
    law_id_counts: Counter[str] = Counter()
    source_title_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    unique_sources: Dict[Tuple[str, str, str], int] = {}

    total_docs = 0
    for fp in files:
        count = 0
        for item in read_jsonl(fp):
            count += 1
            total_docs += 1
            law_id = (item.get("law_id") or "").strip()
            title = (item.get("source_title") or "").strip()
            stype = (item.get("source_type") or "").strip()
            if law_id:
                law_id_counts[law_id] += 1
            if title:
                source_title_counts[title] += 1
            if stype:
                source_type_counts[stype] += 1
            if law_id or title or stype:
                unique_sources[(law_id, title, stype)] = 1
        file_stats[fp.name] = count

    sources_list = [
        {"law_id": k[0], "source_title": k[1], "source_type": k[2]}
        for k in sorted(unique_sources.keys())
    ]

    return {
        "files": [f.name for f in files],
        "file_counts": file_stats,
        "total_documents": total_docs,
        "law_id_counts": dict(law_id_counts.most_common()),
        "source_title_counts": dict(source_title_counts.most_common()),
        "source_type_counts": dict(source_type_counts.most_common()),
        "sources": sources_list,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Data Inventory")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append("")

    for section_key in ("npa", "books"):
        section = report.get(section_key, {})
        lines.append(f"## {section_key.upper()}")
        lines.append(f"Total documents: {section.get('total_documents', 0)}")
        lines.append(f"Files: {len(section.get('files', []))}")
        lines.append("")
        lines.append("### Source Types")
        for k, v in section.get("source_type_counts", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Law IDs")
        for k, v in section.get("law_id_counts", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Source Titles")
        for k, v in section.get("source_title_counts", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Baseline Additions")
    for key, items in BASELINE_ADDITIONS.items():
        lines.append(f"### {key}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    npa_dir = repo_root / "NPA3001"
    books_dir = repo_root / "books"
    diagnostics_dir = repo_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": now_iso(),
        "npa": collect_folder(npa_dir) if npa_dir.exists() else {},
        "books": collect_folder(books_dir) if books_dir.exists() else {},
        "baseline_additions": BASELINE_ADDITIONS,
    }

    (diagnostics_dir / "data_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (repo_root / "data_inventory.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
