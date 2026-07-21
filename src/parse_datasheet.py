"""
Parses a datasheet PDF into section-aware chunks with page-level metadata.

This is deliberately NOT a naive "extract all text, split every 500 chars"
parser. Datasheets have structure (Absolute Maximum Ratings, Electrical
Characteristics, Pin Description, etc.) and dense tables that a flat text
split destroys. This script:

  1. Extracts text per page with layout awareness (PyMuPDF)
  2. Detects section headers using common datasheet heading patterns
  3. Extracts tables separately (pdfplumber) and serializes each row into
     a readable sentence instead of a flattened grid
  4. Groups everything into chunks keyed by (section, page) so each chunk
     stays semantically coherent and small-enough to embed cleanly

Usage:
    python src/parse_datasheet.py data/raw_pdfs/TPS7A4700.pdf

Output:
    data/processed_chunks/TPS7A4700.json
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

# Common datasheet section headers - extend this list as you see more
# real datasheets; manufacturers are fairly consistent about these.
SECTION_PATTERNS = [
    r"^\s*Absolute Maximum Ratings",
    r"^\s*Recommended Operating Conditions",
    r"^\s*Electrical Characteristics",
    r"^\s*Thermal (Information|Characteristics|Resistance)",
    r"^\s*Pin (Configuration|Description|Functions)",
    r"^\s*Application(s)? Information",
    r"^\s*Typical Application",
    r"^\s*Detailed Description",
    r"^\s*Device Comparison Table",
    r"^\s*Ordering Information",
    r"^\s*Package (Information|Option)",
    r"^\s*Functional Block Diagram",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)

# Standalone fragments like pin numbers on a pinout diagram ("1", "2", "3.3V"
# axis ticks on a graph, lone "+"/"-" symbols). These carry no retrievable
# meaning on their own and just inflate the chunk count with noise.
NOISE_PATTERN = re.compile(r"^[\d\.\-–+/,%]{1,3}$")


def is_noise(text: str) -> bool:
    return bool(NOISE_PATTERN.match(text.strip()))


def extract_text_chunks(pdf_path: Path, part_number: str):
    """Extract text per page, tagging each block with the current section."""
    chunks = []
    current_section = "General / Overview"
    noise_filtered = 0

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
            blocks.sort(key=lambda b: (b[1], b[0]))  # reading order: top-to-bottom, left-to-right

            for block in blocks:
                raw = block[4].strip()
                if not raw:
                    continue

                lines = raw.splitlines()
                first_line = lines[0].strip()

                # Only test the FIRST LINE against the header patterns, not
                # the whole block - PyMuPDF frequently merges a heading with
                # the note/line that follows it into a single block, and
                # checking the whole block's length was causing real section
                # boundaries to be missed entirely.
                match = SECTION_REGEX.search(first_line)
                if match and len(first_line) < 80:
                    current_section = first_line
                    # If the block had more content after the heading line,
                    # keep it as body text under the NEW section rather than
                    # discarding it.
                    remainder = "\n".join(lines[1:]).strip()
                    if remainder and not is_noise(remainder):
                        chunks.append({
                            "part_number": part_number,
                            "page_number": page_num,
                            "section": current_section,
                            "type": "text",
                            "content": remainder,
                        })
                    continue

                if is_noise(raw):
                    noise_filtered += 1
                    continue

                chunks.append({
                    "part_number": part_number,
                    "page_number": page_num,
                    "section": current_section,
                    "type": "text",
                    "content": raw,
                })

    if noise_filtered:
        print(f"  filtered {noise_filtered} noise fragments (pin/graph labels)")

    return chunks


def extract_table_chunks(pdf_path: Path, part_number: str):
    """Extract tables and serialize each row as a readable sentence."""
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                header = [str(h).strip() if h else "" for h in table[0]]
                for row in table[1:]:
                    row = [str(c).strip() if c else "" for c in row]
                    if not any(row):
                        continue

                    # Serialize as "Header1: val1, Header2: val2, ..." instead
                    # of a flattened grid - this reads far better for both
                    # embedding and for the LLM at generation time.
                    pairs = [
                        f"{h}: {v}" for h, v in zip(header, row) if h and v
                    ]
                    if not pairs:
                        continue

                    sentence = f"[{part_number}, table on page {page_num}] " + "; ".join(pairs)
                    chunks.append({
                        "part_number": part_number,
                        "page_number": page_num,
                        "section": f"Table {table_idx + 1} (page {page_num})",
                        "type": "table_row",
                        "content": sentence,
                    })

    return chunks


def parse_datasheet(pdf_path: Path):
    part_number = pdf_path.stem
    print(f"Parsing {part_number}...")

    text_chunks = extract_text_chunks(pdf_path, part_number)
    table_chunks = extract_table_chunks(pdf_path, part_number)

    print(f"  {len(text_chunks)} text chunks, {len(table_chunks)} table-row chunks")

    all_chunks = text_chunks + table_chunks

    output_dir = pdf_path.parent.parent / "processed_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{part_number}.json"

    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"  saved -> {output_path}")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/parse_datasheet.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    parse_datasheet(pdf_path)