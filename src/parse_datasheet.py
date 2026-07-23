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
# real datasheets; manufacturers are fairly consistent about these, but TI
# numbers every heading (e.g. "6.1 Absolute Maximum Ratings", "4 Device
# Comparison Table"), so we allow an optional leading number/decimal prefix.
SECTION_TITLES = [
    "Absolute Maximum Ratings",
    "ESD Ratings",
    "Recommended Operating Conditions",
    "Thermal Information",
    "Thermal Characteristics",
    "Thermal Resistance",
    "Electrical Characteristics",
    "Typical Characteristics",
    "Pin Configuration and Functions",
    "Pin Functions",
    "Pin Description",
    "Device Comparison Table",
    "Detailed Description",
    "Application and Implementation",
    "Application Information",
    "Typical Application",
    "Power Supply Recommendations",
    "Layout",
    "Device and Documentation Support",
    "Ordering Information",
    "Package Information",
    "Package Option Addendum",
    "Functional Block Diagram",
    "Specifications",
]

NUMERIC_PREFIX = r"^\s*(?:\d+(?:\.\d+)*\s+)?"
SECTION_REGEX = re.compile(
    NUMERIC_PREFIX + "(?:" + "|".join(re.escape(t) for t in SECTION_TITLES) + ")",
    re.IGNORECASE,
)

# Standalone fragments like pin numbers on a pinout diagram ("1", "2", "3.3V"
# axis ticks on a graph, lone "+"/"-" symbols), plus control characters and
# stray glyph artifacts (checkbox icons, table border remnants) that some
# PDF renderers leave behind as their own tiny text blocks.
NOISE_PATTERN = re.compile(r"^[\d\.\-–+/,%]{1,3}$")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Short alphabetic-only fragments: dimension callouts on mechanical drawings
# (W, H, L) and graph axis unit labels (dB, VI). Real text, but a standalone
# block of just "dB" has no retrievable meaning on its own.
SHORT_LABEL_PATTERN = re.compile(r"^[A-Za-z°ΩµμΔ]{1,3}$")


def is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if CONTROL_CHAR_PATTERN.search(stripped):
        return True
    if NOISE_PATTERN.match(stripped) or SHORT_LABEL_PATTERN.match(stripped):
        return True
    # Single stray symbol with no alphanumeric content at all (e.g. "_", "|")
    if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


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
                    trailing = first_line[match.end():].strip()
                    # Reject false positives where the matched title is just
                    # the start of a sentence, not an actual heading - e.g.
                    # "Specifications are for TJ=25C (unless otherwise noted)"
                    # matches "Specifications" but isn't a section header.
                    # Real headers only have symbols/parens/dashes/numbers
                    # trailing (e.g. "(continued)", "- All Output Versions"),
                    # never a continuing lowercase word.
                    looks_like_sentence = trailing and trailing[0].islower()
                    if not looks_like_sentence:
                        current_section = first_line
                        # If the block had more content after the heading
                        # line, keep it as body text under the NEW section
                        # rather than discarding it.
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