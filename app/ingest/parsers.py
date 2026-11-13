"""Document parsers for the five supported source formats.

    .pdf   datasheets and Technical Reference Manuals
    .rst   ESP-IDF Programming Guide sources
    .md    ESP-IDF / README style documentation
    .html  pages saved from docs.espressif.com (rendered ESP-IDF Programming Guide)
    .json  GitHub issue exports and forum thread dumps
    .txt   plain-text forum threads

Every parser returns `ParsedDocument` objects made of titled sections. The
chunker then splits the sections; the tagger assigns the chip.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

SUPPORTED_EXTENSIONS = (".pdf", ".rst", ".md", ".html", ".htm", ".json", ".txt")
MANIFEST_NAME = "manifest.json"


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class ParsedDocument:
    source_path: str
    title: str
    source_type: str
    sections: list[Section] = field(default_factory=list)
    chip_hint: Optional[str] = None  # explicit chip from the source itself
    url: str = ""
    doc_id: Optional[str] = None  # stable id supplied by the source, if any

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class UnsupportedFormatError(ValueError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _title_from_path(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


def infer_source_type(path: Path) -> str:
    name = path.as_posix().lower()
    ext = path.suffix.lower()
    if ext == ".pdf":
        if "technical_reference" in name or "technical-reference" in name or "trm" in name:
            return "trm"
        return "datasheet"
    if ext in (".rst", ".md", ".html", ".htm"):
        return "esp-idf-docs"
    if "issue" in name or "github" in name:
        return "github_issue"
    return "forum"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_PDF_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][A-Za-z][^\n]{1,70})$")
_PDF_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\s+")
_PDF_BOILERPLATE = re.compile(
    r"^(?:espressif systems|submit documentation feedback|www\.espressif\.com|contents|list of (?:tables|figures)|"
    r".{0,60}(?:datasheet|technical reference manual)\s+v?\d[\w.]*)$",
    re.I,
)
_PDF_SECTION_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,4}$")
_PDF_BARE_NUMBER = re.compile(r"^\d{1,4}$")
_PDF_TOC_LINE = re.compile(r"(?:\.\s?){4,}")


def _pdf_lines(text: str) -> list[str]:
    """Drop running headers/footers, page numbers and table-of-contents leader lines."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _PDF_BOILERPLATE.match(stripped) or _PDF_TOC_LINE.search(stripped) or _PDF_SECTION_NUMBER.match(stripped):
            continue
        kept.append(line)
    # A bare number is a page number only at the very start or end of a page.
    # Anywhere else it is a table cell (a strap value, a pin number) and must stay.
    filled = [i for i, line in enumerate(kept) if line.strip()]
    for index in filled[:1] + filled[-2:]:
        if _PDF_BARE_NUMBER.match(kept[index].strip()):
            kept[index] = ""
    return kept


def _parse_pdf_pymupdf(path: Path) -> tuple[str, list[Section]]:
    """PyMuPDF: clean text, real section headings from the PDF outline, and tables
    emitted row by row (`cell | cell | cell`) so pin and electrical tables survive."""
    import pymupdf

    document = pymupdf.open(str(path))
    title = (document.metadata or {}).get("title") or ""

    # page number -> {heading text as it appears on the page: full outline title}
    headings: dict[int, dict[str, str]] = {}
    for _level, entry, page_no in document.get_toc():
        entry = " ".join(entry.split())
        bare = _PDF_NUMBERING.sub("", entry)
        for key in {entry.lower(), bare.lower()}:
            headings.setdefault(page_no, {})[key] = entry

    sections: list[Section] = []
    heading = "Front matter"
    heading_page = 1
    buffer: list[str] = []

    def flush() -> None:
        body = clean_text("\n".join(buffer))
        if len(body.split()) >= 8:
            sections.append(Section(heading=f"{heading} (p. {heading_page})", text=body))
        buffer.clear()

    for page_no, page in enumerate(document, start=1):
        page_headings = headings.get(page_no, {})
        for line in _pdf_lines(page.get_text()):
            key = " ".join(line.split()).lower()
            if key in page_headings:
                if page_headings[key] != heading:
                    flush()
                    heading, heading_page = page_headings[key], page_no
                continue
            buffer.append(line)
        try:
            tables = page.find_tables().tables
        except Exception:  # table detection is best-effort
            tables = []
        for table in tables:
            rows = []
            for row in table.extract():
                cells = [" ".join(str(cell).split()) for cell in row if cell not in (None, "")]
                if len(cells) >= 2:
                    rows.append(" | ".join(cells))
            if len(rows) >= 2:
                buffer.append("\nTable:\n" + "\n".join(rows) + "\n")
        buffer.append("")
    flush()
    document.close()
    return title.strip(), sections


def _parse_pdf_pypdf(path: Path) -> tuple[str, list[Section]]:
    """Fallback when PyMuPDF is not installed: headings are guessed from numbered lines."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (getattr(reader.metadata, "title", None) or "") if reader.metadata else ""
    sections: list[Section] = []
    heading = "Front matter"
    buffer: list[str] = []

    def flush() -> None:
        body = clean_text("\n".join(buffer))
        if len(body.split()) >= 8:
            sections.append(Section(heading=heading, text=body))
        buffer.clear()

    for page_no, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # a damaged page must not abort the whole document
            continue
        for line in _pdf_lines(page_text):
            stripped = line.strip()
            match = _PDF_HEADING.match(stripped)
            # Numbered table rows and footnotes look like headings; real ones have
            # a small chapter number, lower-case words and no trailing full stop.
            if (
                match
                and int(match.group(1).split(".")[0]) <= 15
                and not stripped.endswith(".")
                and re.search(r"[a-z]{3}", match.group(2))
                and not re.match(r"(?:[GMk]?Hz|[mµu]?[AV]|[KMG]B)\b", match.group(2))
            ):
                flush()
                heading = f"{match.group(1)} {match.group(2).strip()} (p. {page_no})"
            else:
                buffer.append(line)
        buffer.append("")
    flush()
    return title.strip(), sections


def parse_pdf(path: Path) -> list[ParsedDocument]:
    try:
        title, sections = _parse_pdf_pymupdf(path)
    except ImportError:
        title, sections = _parse_pdf_pypdf(path)
    if len(title) <= 3:
        title = _title_from_path(path)
    return [ParsedDocument(str(path), title, infer_source_type(path), sections)]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")


def _clean_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.M)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    return clean_text(text)


def parse_markdown(path: Path) -> list[ParsedDocument]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = _title_from_path(path)
    sections: list[Section] = []
    heading = "Introduction"
    buffer: list[str] = []
    in_fence = False
    saw_title = False

    def flush() -> None:
        body = _clean_markdown("\n".join(buffer))
        if body:
            sections.append(Section(heading=heading, text=body))
        buffer.clear()

    for line in raw.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        match = None if in_fence else _MD_HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            if not saw_title and len(match.group(1)) == 1:
                title, saw_title = heading, True
        else:
            buffer.append(line)
    flush()
    return [ParsedDocument(str(path), title, infer_source_type(path), sections)]


# ---------------------------------------------------------------------------
# reStructuredText (ESP-IDF docs)
# ---------------------------------------------------------------------------

_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_RST_ROLE_TARGET = re.compile(r":[a-zA-Z:_\-]+:`([^`<]+?)\s*<[^>]+>`")
_RST_ROLE = re.compile(r":[a-zA-Z:_\-]+:`~?([^`]+)`")
_RST_LINK = re.compile(r"`([^`<]+?)\s*<[^>]+>`_+")
_RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+[\w\-:]+::.*$")
_RST_OPTION = re.compile(r"^\s+:[\w\-]+:.*$")
_RST_COMMENT = re.compile(r"^\s*\.\.\s*(_[^:]+:.*)?$")
_RST_ONLY = re.compile(r"^\s*\.\.\s+only::\s*(.+)$")

# ESP-IDF docs use substitutions that resolve per build target.
_IDF_SUBSTITUTIONS = {
    "esp32": {"IDF_TARGET_NAME": "ESP32", "IDF_TARGET_PATH_NAME": "esp32"},
    "esp32-s2": {"IDF_TARGET_NAME": "ESP32-S2", "IDF_TARGET_PATH_NAME": "esp32s2"},
    "esp32-s3": {"IDF_TARGET_NAME": "ESP32-S3", "IDF_TARGET_PATH_NAME": "esp32s3"},
    "esp32-c3": {"IDF_TARGET_NAME": "ESP32-C3", "IDF_TARGET_PATH_NAME": "esp32c3"},
    "esp32-c6": {"IDF_TARGET_NAME": "ESP32-C6", "IDF_TARGET_PATH_NAME": "esp32c6"},
}


def _clean_rst_line(line: str) -> str:
    line = _RST_ROLE_TARGET.sub(r"\1", line)
    line = _RST_ROLE.sub(r"\1", line)
    line = _RST_LINK.sub(r"\1", line)
    line = line.replace("``", "")
    return re.sub(r"\|([^|\s][^|]*)\|", r"\1", line)


def parse_rst(path: Path, target_chip: Optional[str] = None) -> list[ParsedDocument]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    for name, value in _IDF_SUBSTITUTIONS.get(target_chip or "", {}).items():
        raw = raw.replace("{" + name + "}", value)
    raw = re.sub(r"\{IDF_TARGET_[A-Z_]+(?::[^}]*)?\}", "the target chip", raw)

    lines = raw.split("\n")
    title = _title_from_path(path)
    sections: list[Section] = []
    heading = "Introduction"
    buffer: list[str] = []
    saw_title = False

    def flush() -> None:
        body = clean_text("\n".join(buffer))
        if body:
            sections.append(Section(heading=heading, text=body))
        buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # overline + title + underline
        if _RST_UNDERLINE.match(line) and nxt.strip() and i + 2 < len(lines) and _RST_UNDERLINE.match(lines[i + 2]):
            flush()
            heading = _clean_rst_line(nxt.strip())
            if not saw_title:
                title, saw_title = heading, True
            i += 3
            continue
        # title + underline
        if line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 1:
            flush()
            heading = _clean_rst_line(line.strip())
            if not saw_title:
                title, saw_title = heading, True
            i += 2
            continue
        if _RST_ONLY.match(line) or _RST_COMMENT.match(line) or _RST_OPTION.match(line):
            i += 1
            continue
        if _RST_DIRECTIVE.match(line):
            # Keep admonition / code bodies (they follow, indented); drop the marker.
            i += 1
            continue
        buffer.append(_clean_rst_line(line))
        i += 1
    flush()
    return [ParsedDocument(str(path), title, infer_source_type(path), sections)]


# ---------------------------------------------------------------------------
# HTML (rendered documentation pages)
# ---------------------------------------------------------------------------

_HTML_SKIP = ("script", "style", "nav", "footer", "header", "form", "button", "svg", "noscript")
_HTML_BLOCKS = ("p", "li", "pre", "dt", "dd", "tr", "caption", "blockquote")
# Auto-generated API listings (every function, struct and macro of a header).
# They are reference dumps, not explanations, and they crowd out the prose
# sections in retrieval, so they are not indexed.
API_DUMP_HEADINGS = frozenset({
    "api reference", "header file", "functions", "structures", "macros", "enumerations",
    "type definitions", "unions", "classes",
})


def parse_html(path: Path) -> list[ParsedDocument]:
    """Parse a saved documentation page. Handles the Sphinx layout used by
    docs.espressif.com and falls back to <main>/<body> for other sites."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    root = (
        soup.find(attrs={"itemprop": "articleBody"})
        or soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )
    for tag in root.find_all(_HTML_SKIP):
        tag.decompose()
    for tag in root.find_all("a", class_="headerlink"):  # Sphinx's pilcrow anchors
        tag.decompose()

    title = _title_from_path(path)
    h1 = root.find("h1")
    if h1 and h1.get_text(strip=True):
        title = " ".join(h1.get_text(" ", strip=True).split())
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True).split(" - ")[0].strip()

    sections: list[Section] = []
    heading = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        body = clean_text("\n\n".join(buffer))
        if body:
            sections.append(Section(heading=heading, text=body))
        buffer.clear()

    for element in root.find_all(("h1", "h2", "h3", "h4") + _HTML_BLOCKS):
        name = element.name
        if name in ("h1", "h2", "h3", "h4"):
            flush()
            heading = " ".join(element.get_text(" ", strip=True).split()) or heading
            continue
        if heading.lower() in API_DUMP_HEADINGS:
            continue
        # Skip blocks nested in another block we already emit (li > p, dd > p, td > p ...).
        if element.find_parent(_HTML_BLOCKS) is not None:
            continue
        if name == "tr":
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in element.find_all(("th", "td"))]
            text = " | ".join(c for c in cells if c)
        elif name == "pre":
            text = element.get_text()
        else:
            text = " ".join(element.get_text(" ", strip=True).split())
        if text.strip():
            buffer.append(text)
    flush()
    return [ParsedDocument(str(path), title, infer_source_type(path), sections)]


# ---------------------------------------------------------------------------
# JSON (GitHub issues, forum dumps, pre-chunked corpora)
# ---------------------------------------------------------------------------

_TEXT_KEYS = ("text", "body", "content", "question", "post", "message")
_REPLY_KEYS = ("comments", "replies", "posts", "answers")
_LIST_KEYS = ("items", "records", "issues", "threads", "documents", "chunks", "data")


def _first(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _reply_text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    if isinstance(reply, dict):
        body = _first(reply, _TEXT_KEYS)
        if not isinstance(body, str):
            return ""
        author = reply.get("user") or reply.get("author") or reply.get("username")
        if isinstance(author, dict):
            author = author.get("login") or author.get("name")
        return f"{author} wrote: {body}" if author else body
    return ""


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
        return [payload]
    return []


def parse_json(path: Path) -> list[ParsedDocument]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    documents: list[ParsedDocument] = []
    for index, record in enumerate(_records(payload)):
        body = _first(record, _TEXT_KEYS)
        if not isinstance(body, str) or not body.strip():
            continue
        title = str(record.get("title") or record.get("subject") or f"{_title_from_path(path)} #{index + 1}")
        section = str(record.get("section") or "")

        replies = _first(record, _REPLY_KEYS)
        has_replies = isinstance(replies, list) and bool(replies)
        source_type = record.get("source_type")
        if not source_type:
            is_issue = "html_url" in record or "number" in record or "issue" in path.name.lower()
            source_type = "github_issue" if is_issue else infer_source_type(path)

        sections = [Section(heading=section or "Original post", text=clean_text(body))]
        if has_replies:
            joined = "\n\n".join(t for t in (_reply_text(r) for r in replies) if t.strip())
            if joined.strip():
                sections.append(Section(heading="Replies", text=clean_text(joined)))

        chip = record.get("chip")
        raw_id = record.get("id") or record.get("number")
        documents.append(
            ParsedDocument(
                source_path=f"{path}#{raw_id if raw_id is not None else index}",
                title=title,
                source_type=str(source_type),
                sections=sections,
                chip_hint=str(chip).lower() if chip else None,
                url=str(record.get("url") or record.get("html_url") or ""),
                doc_id=str(raw_id) if raw_id is not None else None,
            )
        )
    return documents


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def parse_txt(path: Path) -> list[ParsedDocument]:
    raw = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if not raw:
        return []
    lines = raw.split("\n")
    title = _title_from_path(path)
    # A short first line followed by a blank line is treated as the thread title.
    if len(lines) > 2 and len(lines[0]) <= 120 and not lines[1].strip():
        title = re.sub(r"^(title|subject)\s*:\s*", "", lines[0], flags=re.I).strip() or title
        raw = "\n".join(lines[2:]).strip()
    return [ParsedDocument(str(path), title, infer_source_type(path), [Section("Thread", raw)])]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def parse_file(path: Path, target_chip: Optional[str] = None) -> list[ParsedDocument]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".md":
        return parse_markdown(path)
    if ext == ".rst":
        return parse_rst(path, target_chip=target_chip)
    if ext in (".html", ".htm"):
        return parse_html(path)
    if ext == ".json":
        return parse_json(path)
    if ext == ".txt":
        return parse_txt(path)
    raise UnsupportedFormatError(f"Unsupported file type: {path.suffix} ({path})")


def discover_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
        and p.name != MANIFEST_NAME
    )
