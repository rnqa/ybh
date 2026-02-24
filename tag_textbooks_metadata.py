#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def is_textbook_file(path: Path) -> bool:
    name = path.name.upper()
    return "_UCH" in name or "UCH" in name


def is_textbook_item(item: Dict[str, Any], path: Path) -> bool:
    if is_textbook_file(path):
        return True
    law_id = str(item.get("law_id", "")).upper()
    source_title = str(item.get("source_title", "")).upper()
    if "UCH" in law_id or "UCH" in source_title:
        return True
    return False


def source_type_is_textbook(value: str) -> bool:
    val = (value or "").strip().lower()
    return "учеб" in val


def process_file(path: Path, dry_run: bool) -> Tuple[int, int]:
    updated = 0
    total = 0
    out_lines = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total += 1
            try:
                item = json.loads(raw)
            except Exception:
                out_lines.append(raw)
                continue

            if is_textbook_item(item, path):
                current = str(item.get("source_type", "")).strip()
                if not source_type_is_textbook(current):
                    item["source_type"] = "учебник"
                    updated += 1

            out_lines.append(json.dumps(item, ensure_ascii=False))

    if not dry_run and updated > 0:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return total, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure textbook metadata in JSONL files.")
    parser.add_argument("--data-dir", default="NPA3001")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit("No JSONL files found.")

    total_rows = 0
    total_updated = 0
    for fp in files:
        rows, updated = process_file(fp, dry_run=args.dry_run)
        total_rows += rows
        total_updated += updated

    print(f"Files: {len(files)}")
    print(f"Rows: {total_rows}")
    print(f"Updated: {total_updated}")
    if args.dry_run:
        print("Dry run only. No changes were written.")


if __name__ == "__main__":
    main()
