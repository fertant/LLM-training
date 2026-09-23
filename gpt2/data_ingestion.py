"""
Extracts plain text out of Excel (.xlsx), PDF (.pdf), and Word (.docx) files
so it can be fed into the same tokenizer/dataset pipeline as any other text.

GPTModel (and every LLM like it) only ever consumes plain text -> token ids.
It has no notion of spreadsheet cells, PDF pages, or Word paragraphs. So the
job of this module is entirely "flatten whatever format you gave me into one
long string", after which dataset.py takes over exactly as it would for a
.txt file.

Requires (not part of the repo's base requirements.txt, only needed if you
use these formats):
    pip install openpyxl pypdf python-docx
"""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".xlsx", ".xls", ".pdf", ".docx"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_xlsx(path: Path) -> str:
    import openpyxl  # local import: only required if you actually use .xlsx files

    workbook = openpyxl.load_workbook(path, data_only=True)
    lines = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            # Join non-empty cells in a row with a space; skip fully empty rows.
            cells = [str(cell) for cell in row if cell is not None]
            if cells:
                lines.append(" ".join(cells))
    return "\n".join(lines)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # local import: only required if you actually use .pdf files

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx  # python-docx; local import: only required if you actually use .docx files

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


_READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".csv": _read_txt,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def extract_text(file_path) -> str:
    """Extract plain text from a single supported file."""
    path = Path(file_path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}' for {path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return reader(path)


def load_training_corpus(folder) -> str:
    """
    Read every supported file in `folder` (non-recursively excluded files are
    skipped) and concatenate them into one training corpus, separated by the
    GPT-2 end-of-text marker so the model learns that one document has ended
    and an unrelated one is starting.
    """
    folder = Path(folder)
    texts = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            texts.append(extract_text(path))

    if not texts:
        raise FileNotFoundError(
            f"No supported training files found in {folder}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return "<|endoftext|>".join(texts)
