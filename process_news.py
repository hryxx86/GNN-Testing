"""
Convert all Factiva RTF files in Newstitle_20210129_20260128_1491 to a clean CSV.
Uses macOS textutil to convert RTF -> plaintext, then splits by 'Document <ID>'.
"""

import csv
import pathlib
import re
import subprocess
from typing import List, Tuple

ROOT = pathlib.Path("Newstitle_20210129_20260128_1491")
OUT_PATH = pathlib.Path("news_clean.csv")

DATE_LINE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r" \\d{1,2}, \\d{4}|\\d{1,2}:\\d{2} (AM|PM), \\d{1,2} \\w+ \\d{4}",
    re.IGNORECASE,
)


def run_textutil(path: pathlib.Path) -> str:
    """Call textutil to get plaintext for one RTF file."""
    result = subprocess.run(
        ["textutil", "-convert", "txt", str(path), "-stdout"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def split_articles(text: str) -> List[Tuple[str, str]]:
    """
    Split the plaintext into articles by 'Document <ID>'.
    Returns list of (doc_id, article_text).
    """
    articles: List[Tuple[str, str]] = []
    current_lines: List[str] = []
    doc_id: str = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Document "):
            parts = stripped.split()
            doc_id = parts[1] if len(parts) > 1 else ""
            article_text = "\n".join(l for l in current_lines if l.strip())
            if article_text:
                articles.append((doc_id, article_text))
            current_lines = []
            doc_id = ""
        else:
            current_lines.append(stripped)
    # last article
    article_text = "\n".join(l for l in current_lines if l.strip())
    if article_text:
        articles.append((doc_id, article_text))
    return articles


def extract_fields(article_text: str) -> dict:
    """
    Heuristic: first non-empty line = title, second non-empty line = meta (source+date),
    rest = body.
    """
    lines = [l for l in (ln.strip() for ln in article_text.splitlines()) if l]
    if not lines:
        return {}
    title = lines[0]
    meta = lines[1] if len(lines) > 1 else ""
    body = "\n".join(lines[2:]) if len(lines) > 2 else ""
    source = meta.split(",")[0].strip() if "," in meta else meta
    return {"title": title, "meta": meta, "source": source, "body": body}


def main():
    rows = []
    for rtf_file in sorted(ROOT.glob("*.rtf")):
        txt = run_textutil(rtf_file)
        for doc_id, article_text in split_articles(txt):
            fields = extract_fields(article_text)
            if not fields:
                continue
            fields.update({"filename": rtf_file.name, "doc_id": doc_id})
            rows.append(fields)

    # Deduplicate by (title, meta)
    seen = set()
    unique_rows = []
    for row in rows:
        key = (row["title"], row["meta"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["filename", "doc_id", "title", "meta", "source", "body"]
        )
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"written {len(unique_rows)} articles -> {OUT_PATH}")


if __name__ == "__main__":
    main()
