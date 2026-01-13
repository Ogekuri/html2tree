"""Core pipeline for html2tree."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import mimetypes
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LOG = logging.getLogger("html2tree")

EXIT_INVALID_ARGS = 6
EXIT_OUTPUT_DIR = 7
EXIT_POSTPROC_ARTIFACT = 9
EXIT_POSTPROC_DEP = 10

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
TEST_MODE_ENV = "HTML2TREE_TEST_MODE"
TEST_PIX2TEX_FORMULA_ENV = "HTML2TREE_TEST_PIX2TEX_FORMULA"
TEST_GEMINI_IMAGE_ENV = "HTML2TREE_TEST_GEMINI_IMAGE_ANNOTATION"
TEST_GEMINI_EQUATION_ENV = "HTML2TREE_TEST_GEMINI_EQUATION_ANNOTATION"
TEST_PIX2TEX_DEFAULT_FORMULA = r"\int_{0}^{1} x^2 \, dx = \frac{1}{3}"
TEST_GEMINI_IMAGE_DEFAULT = "Test image annotation generated during automated test execution."
TEST_GEMINI_EQUATION_DEFAULT = (
    "Test equation annotation generated during automated test execution.\n\n"
    "Representative LaTeX: $$ {formula} $$"
)

PROMPT_EQUATION_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

Return ENGLISH Markdown optimized for RAG with the following sections (keep headings exactly):

## Overview
One sentence describing what the image contains (e.g., "A single equation", "A system of equations with a diagram", etc.).

## Mathematical transcription
- Transcribe ALL mathematical expressions visible (even if multiple).
- Preserve symbols, subscripts/superscripts, Greek letters, limits, summations, matrices, piecewise definitions.
- If something is unreadable, write: [UNREADABLE] and do NOT guess.

## LaTeX (MathJax)
Provide the equation(s) as MathJax-ready LaTeX in Markdown:
- Use one or more display blocks:
  $$ ... $$
- If multiple equations, use separate $$ blocks, in reading order.
- Do NOT add extra equations not present in the image.

## Definitions / Variables (only if explicitly present)
List variable meanings ONLY if they are written in the image. Otherwise write "Not specified."

## Notes on layout (only if useful)
Mention important spatial structure: aligned equations, braces, arrows, numbered steps, boxed results, etc.

## Ambiguities / Unclear parts
Bullet list of any uncertain characters/symbols and where they appear.
"""
)
PROMPT_NON_EQUATION_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

Return ENGLISH Markdown optimized for RAG with the following sections (keep headings exactly):

## Overview
One sentence describing the image type (photo, diagram, chart, UI screenshot, table, flowchart, etc.).

## Visible text (verbatim)
Transcribe all readable text exactly as shown (line breaks if meaningful).
If text is unreadable, write: [UNREADABLE] and do NOT guess.

## Entities and layout
Describe the layout and key elements (objects, labels, axes, boxes, arrows, regions, legend, callouts).
Including all details useful to understand mathematical graphs, processes flows, flow charts, temporal diagrams, mind maps and graphs behaviors.
If there are flows/process steps, describe them in order.

## Tables (if any)
Recreate any table as Markdown table, preserving headers and cell values.
If a cell is unreadable, use [UNREADABLE].

## Quantities / Data (if any)
List numeric values, units, ranges, axis ticks, categories exactly as visible.

## Ambiguities / Unclear parts
Bullet list of any uncertain text/labels and where they appear.
"""
)
PROMPT_UNCERTAIN_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

First decide whether the image contains a mathematical equation/expression that should be transcribed as LaTeX.
Then produce ENGLISH Markdown optimized for RAG in ONE of the two formats below.

### If the image contains mathematical equation(s):
Use EXACTLY these sections:

## Overview
One sentence describing the content.

## Classification
Equation: YES (confidence 0-100)

## Mathematical transcription
Transcribe ALL mathematical expressions visible. Do NOT guess unreadable parts; use [UNREADABLE].

## LaTeX (MathJax)
Provide ONLY the equation(s) that appear in the image as display blocks:
$$ ... $$
Use multiple blocks if needed, in reading order.

## Non-math context (if present)
Briefly describe any accompanying diagram/text/table that changes how the equation is read (e.g., variable definitions, constraints, figure references).

## Ambiguities / Unclear parts
List uncertain symbols/text and location.

### If the image does NOT contain equations:
Use EXACTLY these sections:

## Overview
One sentence describing the content.

## Classification
Equation: NO (confidence 0-100)

## Visible text (verbatim)
Transcribe all readable text exactly; use [UNREADABLE] when needed.

## Entities and layout
Describe the layout, objects, labels, arrows/flows, charts, and tables.
Including all details useful to understand mathematical graphs, processes flows, flow charts, temporal diagrams, mind maps and graphs behaviors.
If there are flows/process steps, describe them in order.

## Tables (if any)
Recreate as Markdown table.

## Ambiguities / Unclear parts
List uncertain parts and location.

Important rules:
- Do NOT invent equations or text.
- If unsure, choose the most likely classification but reflect uncertainty via confidence and Ambiguities.
"""
)
DEFAULT_PROMPTS = {
    "prompt_equation": PROMPT_EQUATION_DEFAULT.strip(),
    "prompt_non_equation": PROMPT_NON_EQUATION_DEFAULT.strip(),
    "prompt_uncertain": PROMPT_UNCERTAIN_DEFAULT.strip(),
}

TABLE_PLACEHOLDER_PREFIX = "HTML2TREE_TABLE_PLACEHOLDER_"


@dataclass
class PostProcessingConfig:
    enable_pix2tex: bool
    disable_pix2tex: bool
    equation_min_len: int
    verbose: bool
    debug: bool
    annotate_images: bool
    annotate_equations: bool
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_module: str
    test_mode: bool
    disable_remove_small_images: bool
    disable_cleanup: bool
    disable_toc: bool
    enable_pdf_pages_ref: bool
    min_size_x: int
    min_size_y: int
    prompt_equation: str
    prompt_non_equation: str
    prompt_uncertain: str
    skip_toc_validation: bool = False


@dataclass
class TocNode:
    title: str
    level: int
    page: Optional[int]
    anchor: Optional[str]
    children: List["TocNode"]


@dataclass
class TocValidationResult:
    ok: bool
    source_titles: List[str]
    md_titles: List[str]
    mismatches: List[Tuple[int, str, str]]
    source_count: int
    md_count: int
    reason: str


@dataclass
class HtmlTable:
    index: int
    placeholder: str
    stem: str
    md: str
    rows: List[List[str]]


def _env_flag_enabled(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def is_test_mode() -> bool:
    env_flag = os.environ.get(TEST_MODE_ENV)
    if env_flag is not None:
        return _env_flag_enabled(env_flag)
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _get_test_pix2tex_formula() -> str:
    override = os.environ.get(TEST_PIX2TEX_FORMULA_ENV)
    if override is not None and override.strip():
        return override.strip()
    return TEST_PIX2TEX_DEFAULT_FORMULA


def _get_test_annotation_text(is_equation: bool, equation_text: Optional[str]) -> str:
    if is_equation:
        override = os.environ.get(TEST_GEMINI_EQUATION_ENV)
        if override is not None and override.strip():
            return override.strip()
        formula = equation_text or _get_test_pix2tex_formula() or TEST_PIX2TEX_DEFAULT_FORMULA
        return TEST_GEMINI_EQUATION_DEFAULT.format(formula=formula)
    override = os.environ.get(TEST_GEMINI_IMAGE_ENV)
    if override is not None and override.strip():
        return override.strip()
    return TEST_GEMINI_IMAGE_DEFAULT


def _write_prompts_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_PROMPTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_prompts_file(path: Path) -> Dict[str, str]:
    try:
        data_raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to read prompts file {path}: {exc}") from exc

    prompts: Dict[str, str] = {}
    for key in ("prompt_equation", "prompt_non_equation", "prompt_uncertain"):
        value = data_raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prompts file {path} missing non-empty key: {key}")
        prompts[key] = value.strip()
    return prompts


def select_annotation_prompt(is_equation: bool, pix2tex_executed: bool, config: PostProcessingConfig) -> str:
    if pix2tex_executed:
        return config.prompt_equation if is_equation else config.prompt_non_equation
    return config.prompt_uncertain


def _resolve_log_level(verbose: bool, debug: bool) -> int:
    return logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)


def _configure_html2tree_logger(level: int) -> None:
    LOG.setLevel(level)
    LOG.propagate = False
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handler.setLevel(level)
        LOG.addHandler(handler)
    else:
        for handler in LOG.handlers:
            handler.setLevel(level)
            if handler.formatter is None:
                handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))


def setup_logging(verbose: bool, debug: bool) -> None:
    level = _resolve_log_level(verbose, debug)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    _configure_html2tree_logger(level)


def _progress_bar_line(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[?]"
    clamped = max(0, min(current, total))
    filled = int((clamped / total) * width)
    filled = min(filled, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _log_verbose_progress(prefix: str, current: int, total: int, detail: Optional[str] = None) -> None:
    bar = _progress_bar_line(current, total)
    counter = f"[{current}/{total}]" if total > 0 else f"[{current}]"
    if total > 0:
        msg = f"{prefix} {bar} {counter} ({(current / total) * 100.0:.1f}%)"
    else:
        msg = f"{prefix} {bar} {counter}"
    if detail:
        msg = f"{msg} | {detail}"
    LOG.info(msg)


def slugify_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "document"


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def normalize_path_for_md(p: str) -> str:
    p = p.replace("\\", "/")
    if p.startswith("file://"):
        p = p[len("file://") :]
    return p


def relative_to_output(path: Path, out_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(out_dir.resolve()))
    except Exception:
        return path.as_posix()


def parse_html_toc(toc_path: Path) -> Tuple[List[List[Any]], TocNode]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"beautifulsoup4 not available: {exc}") from exc

    raw = toc_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    root_ul = soup.find("ul")
    entries: List[List[Any]] = []

    def walk_list(ul, level: int) -> None:
        for li in ul.find_all("li", recursive=False):
            anchor = None
            title = ""
            link = li.find("a", recursive=False)
            if link is not None:
                title = link.get_text(" ", strip=True)
                href = link.get("href") or ""
                if "#" in href:
                    anchor = href.split("#", 1)[1].strip() or None
            else:
                title = li.get_text(" ", strip=True)
            if title:
                entries.append([level, title, anchor])
            child_ul = li.find("ul", recursive=False)
            if child_ul is not None:
                walk_list(child_ul, level + 1)

    if root_ul is not None:
        walk_list(root_ul, 1)

    toc_root = build_toc_tree(entries)
    return entries, toc_root


def build_toc_tree(toc_list: List[List[Any]]) -> TocNode:
    root = TocNode(title="root", page=None, level=0, anchor=None, children=[])
    stack: List[TocNode] = [root]

    for entry in toc_list or []:
        if len(entry) < 2:
            continue
        try:
            level = int(entry[0])
        except Exception:
            continue
        title = str(entry[1]).strip()
        anchor = str(entry[2]).strip() if len(entry) >= 3 and entry[2] else None
        node = TocNode(title=title, page=None, level=level, anchor=anchor, children=[])

        while stack and level <= stack[-1].level:
            stack.pop()
        parent = stack[-1] if stack else root
        parent.children.append(node)
        stack.append(node)

    return root


def serialize_toc_tree(node: TocNode) -> List[Dict[str, Any]]:
    return [
        {
            "title": child.title,
            "pdf_source_page": child.page,
            "anchor": child.anchor,
            "children": serialize_toc_tree(child),
        }
        for child in node.children
    ]


def find_context_for_page(root: TocNode, page_no: Optional[int]) -> List[str]:
    if page_no is None:
        return []

    best_path: List[TocNode] = []
    best_page = -1
    fallback_path: List[TocNode] = []
    fallback_page: Optional[int] = None

    def dfs(node: TocNode, ancestors: List[TocNode]) -> None:
        nonlocal best_path, best_page, fallback_path, fallback_page
        current_path = ancestors + ([node] if node.level > 0 else [])
        if node.level > 0:
            if fallback_page is None or (node.page is not None and node.page < fallback_page):
                fallback_page = node.page
                fallback_path = current_path
            if node.page is not None and node.page <= page_no:
                if node.page > best_page:
                    best_page = node.page
                    best_path = current_path
        for child in node.children:
            dfs(child, current_path)

    dfs(root, [])
    if best_path:
        return [n.title for n in best_path]
    return [n.title for n in fallback_path]


def find_context(
    toc_path: Optional[Path],
    toc_root: Optional[TocNode],
    asset_names: Iterable[str],
    fallback_page: Optional[int],
) -> Tuple[List[str], str]:
    names = {n for n in asset_names if n}
    if toc_path and toc_path.exists() and names:
        try:
            lines = toc_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []

        heading_re = re.compile(r"^(?P<indent>\s*)-\s*\[(?P<title>[^\]]+)\]\([^)]*\)")
        stack: List[str] = []
        found: Optional[List[str]] = None

        for raw in lines:
            if not raw.strip():
                continue
            h_match = heading_re.match(raw)
            if h_match:
                indent = h_match.group("indent")
                level = int(len(indent) / 2) + 1
                title = h_match.group("title").strip()

                while len(stack) >= level:
                    stack.pop()
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(title)
                continue

            if any(name in raw for name in names):
                found = [item for item in stack if item]
                break

        if found is not None:
            context_path = list(found)
            context_str = " > ".join(context_path)
            return context_path, context_str

    context_titles = (
        find_context_for_page(toc_root or build_toc_tree([]), fallback_page)
        if fallback_page is not None
        else []
    )
    context_str = " > ".join(context_titles)
    return context_titles, context_str


def build_context_metadata(context_titles: List[str]) -> Tuple[str, List[str]]:
    context_path = list(context_titles)
    context_str = " > ".join(context_path)
    return context_str, context_path


PAGE_IN_NAME_RE = re.compile(r"-([0-9]{3,4})-")


def guess_page_from_filename(name: str) -> Optional[int]:
    match = PAGE_IN_NAME_RE.search(name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    alt = re.search(r"page[_-]?([0-9]{3,4})", name, re.IGNORECASE)
    if alt:
        try:
            return int(alt.group(1))
        except Exception:
            return None
    return None


IMG_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^\)]+)(\))")


CAPTION_RE = re.compile(r"(?m)^(Figura\s+\d+(\.\d+)?\s*:.*)$")
PJM_PLACEHOLDER_RE = re.compile(
    r"\*\*==>\s*picture\s*\[[^\]]+\]\s*intentionally\s*omitted\s*<==\*\*",
    re.IGNORECASE,
)
ANNOTATION_LINE_RE = re.compile(r"^>\s*\[annotation:([^\]]+)\]:", re.IGNORECASE)
_BR_TAG_RE = r"(?:<br\s*/?>\s*)?"
EQUATION_BLOCK_START_RE = re.compile(
    rf"^(?:\*\*)?-----\s*Start of equation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
EQUATION_BLOCK_END_RE = re.compile(
    rf"^(?:\*\*)?-----\s*End of equation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
ANNOTATION_BLOCK_START_RE = re.compile(
    rf"^(?:\*\*)?-----\s*Start of annotation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
ANNOTATION_BLOCK_END_RE = re.compile(
    rf"^(?:\*\*)?-----\s*End of annotation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
PAGE_END_MARKER_RE = re.compile(r"(?m)^\s*---\s*end of page\.page_number=\d+\s*---\s*$\n?")
PAGE_START_MARKER_CAPTURE_RE = re.compile(r"^\s*---\s*start of page\.page_number=(\d+)\s*---\s*$", re.IGNORECASE)
PAGE_END_MARKER_CAPTURE_RE = re.compile(r"^\s*---\s*end of page\.page_number=(\d+)\s*---\s*$", re.IGNORECASE)


def extract_image_basenames_from_markdown(md: str) -> Set[str]:
    out: Set[str] = set()
    for match in IMG_LINK_RE.finditer(md or ""):
        url = match.group(2).strip().strip('"').strip("'")
        url = normalize_path_for_md(url)
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1].strip()
        if base:
            out.add(base)
    return out


def extract_image_basenames_in_order(md: str) -> List[str]:
    ordered: List[str] = []
    for match in IMG_LINK_RE.finditer(md or ""):
        url = match.group(2).strip().strip('"').strip("'")
        url = normalize_path_for_md(url)
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1].strip()
        if base:
            ordered.append(base)
    return ordered


def rewrite_image_links_to_assets_subdir(md: str, subdir: str = "assets") -> str:
    def repl(match: re.Match) -> str:
        before, url, after = match.group(1), match.group(2), match.group(3)
        url_clean = normalize_path_for_md(url.strip().strip('"').strip("'"))
        url_clean = url_clean.split("?", 1)[0].split("#", 1)[0]
        rel = ""
        if "/assets/" in f"/{url_clean}":
            idx = f"/{url_clean}".rfind("/assets/")
            rel = f"/{url_clean}"[idx + len("/assets/") :]
        elif url_clean.startswith("assets/"):
            rel = url_clean[len("assets/") :]
        else:
            rel = url_clean.split("/")[-1]
        rel = rel.lstrip("/")
        return f"{before}{subdir}/{rel}{after}"

    return IMG_LINK_RE.sub(repl, md)


def _sanitize_table_cell(text: str) -> str:
    cleaned = text.replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("|", "\\|")
    return cleaned


def html_table_to_markdown(table) -> Tuple[str, List[List[str]]]:
    rows: List[List[str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        row_text = [cell.get_text(" ", strip=True) for cell in cells]
        rows.append(row_text)

    if not rows:
        return "", []

    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    header = padded[0]
    body = padded[1:]

    md_lines = [
        "| " + " | ".join(_sanitize_table_cell(c) for c in header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in body:
        md_lines.append("| " + " | ".join(_sanitize_table_cell(c) for c in row) + " |")

    return "\n".join(md_lines), padded


def extract_tables_from_html(soup) -> List[HtmlTable]:
    tables = soup.find_all("table")
    outputs: List[HtmlTable] = []
    used_stems: Set[str] = set()

    for idx, table in enumerate(tables, start=1):
        table_id = table.get("id") or f"table-{idx:03d}"
        base_stem = slugify_filename(str(table_id))
        stem = base_stem
        suffix = 1
        while stem in used_stems:
            stem = f"{base_stem}__{suffix}"
            suffix += 1
        used_stems.add(stem)

        md, rows = html_table_to_markdown(table)
        placeholder = f"{TABLE_PLACEHOLDER_PREFIX}{idx:03d}"
        placeholder_tag = soup.new_tag("p")
        placeholder_tag.string = placeholder
        table.replace_with(placeholder_tag)

        outputs.append(HtmlTable(index=idx, placeholder=placeholder, stem=stem, md=md, rows=rows))

    return outputs


def replace_table_placeholders(md_text: str, tables: List[HtmlTable]) -> str:
    if not tables:
        return md_text

    table_by_placeholder = {table.placeholder: table for table in tables}
    lines = md_text.splitlines()
    output: List[str] = []

    for line in lines:
        stripped = line.strip()
        table = table_by_placeholder.get(stripped)
        if table is None:
            output.append(line)
            continue
        if table.md:
            output.append(table.md)
        ref_lines = [f"[Markdown](tables/{table.stem}.md)", "", f"[CSV](tables/{table.stem}.csv)"]
        output.append("")
        output.append("\n".join(ref_lines).strip())
        output.append("")

    result = "\n".join(output)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def export_tables_files(tables_dir: Path, tables: List[HtmlTable]) -> List[List[Path]]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    exported: List[List[Path]] = []
    for table in tables:
        files: List[Path] = []
        if table.md.strip():
            path_md = tables_dir / f"{table.stem}.md"
            safe_write_text(path_md, table.md.strip() + "\n")
            files.append(path_md)
        if table.rows:
            path_csv = tables_dir / f"{table.stem}.csv"
            write_csv(path_csv, table.rows)
            files.append(path_csv)
        if files:
            exported.append(files)
    return exported


def format_table_references(exported: List[List[Path]], out_dir: Path) -> List[str]:
    blocks: List[str] = []
    for files in exported:
        rel_md: Optional[str] = None
        rel_csv: Optional[str] = None
        for path in files:
            rel = relative_to_output(path, out_dir)
            if path.suffix.lower() == ".md":
                rel_md = rel
            elif path.suffix.lower() == ".csv":
                rel_csv = rel

        lines: List[str] = []
        if rel_md:
            lines.append(f"[Markdown]({rel_md})")
        if rel_csv:
            if lines:
                lines.append("")
            lines.append(f"[CSV]({rel_csv})")

        blocks.append("\n".join(lines) + ("\n" if lines else ""))

    return blocks


def normalize_markdown_format(md_text: str) -> str:
    if not md_text:
        return md_text
    return re.sub(r"(?i)<br\s*/?>", "\n", md_text)


def _slugify_markdown_heading(title: str) -> str:
    slug = unicodedata.normalize("NFKD", title or "").lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug or "section"


def _normalize_title_for_toc(title: str) -> str:
    norm = unicodedata.normalize("NFKD", title or "")
    norm = norm.replace("\u2019", "'").replace("\u2018", "'")
    norm = norm.replace("\u201c", '"').replace("\u201d", '"')
    norm = norm.replace("\u00a0", " ")
    norm = re.sub(r"[*_`]+", "", norm)
    norm = re.sub(r"^\d+(?:\.\d+)*\s+", "", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip()


def _scan_markdown_for_toc_entries(md_text: str) -> List[Tuple[str, int, str, str]]:
    lines = md_text.splitlines()
    toc_sequence: List[Tuple[str, int, str, str]] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    image_re = re.compile(r"!\[[^\]]*\]\(\s*assets/[^)]+\)")
    table_re = re.compile(r"\(\s*tables/[^)]+\)")

    for raw in lines:
        line = raw.rstrip()
        heading = heading_re.match(line)
        if heading:
            level = len(heading.group(1))
            title_raw = heading.group(2).strip()
            title_display = re.sub(r"[*_`]+", "", title_raw).strip()
            title_clean = re.sub(r"[*_`]+", "", title_raw).strip()
            title_clean = re.sub(r"^\d+(?:\.\d+)*\s+", "", title_clean).strip()
            if not title_display:
                continue
            if title_clean.lower() in {"indice", "toc", "html toc"}:
                continue
            toc_sequence.append(("heading", level, title_display, ""))
            continue
        if image_re.search(line):
            toc_sequence.append(("raw", 0, line.strip(), ""))
            continue
        if table_re.search(line):
            toc_sequence.append(("raw", 0, line.strip(), ""))
            continue

    return toc_sequence


def generate_markdown_toc_file(md_text: str, md_path: Path, out_dir: Path) -> Tuple[Path, List[Tuple[int, str]]]:
    toc_sequence = _scan_markdown_for_toc_entries(md_text)
    toc_lines: List[str] = []
    heading_entries: List[Tuple[int, str]] = []

    for item in toc_sequence:
        kind, level, text, _ = item
        if kind == "heading":
            indent = "  " * max(0, level - 1)
            anchor = _slugify_markdown_heading(text)
            toc_lines.append(f"{indent}- [{text}](#{anchor})")
            heading_entries.append((level, text))
        else:
            toc_lines.append(text)

    toc_path = md_path.with_suffix(".toc")
    content = "\n".join([ln for ln in toc_lines if ln.strip()])
    safe_write_text(toc_path, (content + "\n") if content else "")
    return toc_path, heading_entries


def normalize_markdown_headings(md_text: str, toc_headings: List[Tuple[Any, ...]]) -> str:
    lines = md_text.splitlines()

    normalized_toc: List[Tuple[int, str, Optional[str]]] = []
    for entry in toc_headings:
        if len(entry) < 2:
            continue
        level = int(entry[0])
        title = str(entry[1])
        anchor = str(entry[2]) if len(entry) >= 3 and entry[2] else None
        normalized_toc.append((level, title, anchor))

    search_pos = 0
    for level, title, _ in normalized_toc:
        target = _normalize_title_for_toc(title)
        for idx in range(search_pos, len(lines)):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("<!--"):
                continue
            candidate = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            candidate = re.sub(r"[*_`]+", "", candidate).strip()
            candidate = re.sub(r"^\d+(?:\.\d+)*\s+", "", candidate).strip()
            if not candidate:
                continue
            if _normalize_title_for_toc(candidate) == target:
                heading_line = f"{'#' * max(1, min(level, 6))} {title}"
                lines[idx] = heading_line
                search_pos = idx + 1
                break

    result = "\n".join(lines)
    if md_text.endswith("\n"):
        result += "\n"
    return result


def clean_markdown_headings(md_text: str, toc_headings: List[Tuple[int, str, Optional[str]]]) -> str:
    normalized_titles = {
        _normalize_title_for_toc(str(title))
        for _, title, *rest in toc_headings
        if str(title).strip()
    }

    heading_re = re.compile(r"^(?P<prefix>\s*)(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

    def _strip_bold(text: str) -> str:
        t = text.strip()
        if (t.startswith("**") and t.endswith("**")) or (t.startswith("__") and t.endswith("__")):
            return t[2:-2].strip()
        return t

    lines = md_text.splitlines()
    output: List[str] = []
    for line in lines:
        match = heading_re.match(line)
        if not match:
            output.append(line)
            continue

        title_raw = match.group("title").strip()
        normalized = _normalize_title_for_toc(title_raw)

        if normalized in normalized_titles:
            output.append(line)
            continue

        clean_title = _strip_bold(title_raw).upper()
        bold_title = f"**{clean_title}**"
        output.append(f"{match.group('prefix')}{bold_title}")

    result = "\n".join(output)
    if md_text.endswith("\n"):
        result += "\n"
    return result


def add_html_toc_to_markdown(md_text: str, toc_headings: List[Tuple[int, str, Optional[str]]]) -> str:
    if not md_text or not toc_headings:
        return md_text

    def _normalize_toc_heading_variants(content: str) -> str:
        lines = content.splitlines()
        pattern = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*{0,2}\s*)?(?:html\s+toc|toc)\s*(?:\*{0,2}\s*)?$", re.IGNORECASE)
        normalized: List[str] = []
        for line in lines:
            if pattern.match(line.strip()):
                normalized.append("** HTML TOC **")
            else:
                normalized.append(line)
        result = "\n".join(normalized)
        if content.endswith("\n") and not result.endswith("\n"):
            result += "\n"
        return result

    md_text = _normalize_toc_heading_variants(md_text)

    toc_lines: List[str] = ["** HTML TOC **", ""]
    for level, title, _ in toc_headings:
        indent = "  " * max(0, int(level) - 1)
        anchor = _slugify_markdown_heading(title)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")
    toc_lines.append("")

    lines = md_text.splitlines()
    insert_at = 0
    for idx, raw in enumerate(lines):
        if PAGE_START_MARKER_CAPTURE_RE.match(raw.strip()):
            insert_at = idx + 1
            break

    new_lines = lines[:insert_at] + toc_lines + lines[insert_at:]
    result = "\n".join(new_lines)
    if md_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def remove_markdown_index(md_text: str, toc_headings: List[Tuple[int, str, Optional[str]]]) -> str:
    if not md_text or not toc_headings:
        return md_text

    first_title = _normalize_title_for_toc(str(toc_headings[0][1]).strip()) if len(toc_headings[0]) >= 2 else ""
    if not first_title:
        return md_text

    heading_re = re.compile(r"^(?P<prefix>\s*)(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

    lines = md_text.splitlines()
    output: List[str] = []
    keep_content = False
    matched_heading = False

    idx = 0
    if lines and lines[0].strip() == "---":
        output.append(lines[0])
        idx = 1
        while idx < len(lines):
            output.append(lines[idx])
            if lines[idx].strip() == "---":
                idx += 1
                break
            idx += 1
    else:
        idx = 0

    for line in lines[idx:]:
        stripped = line.strip()

        if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
            output.append(line)
            continue

        if not keep_content:
            match = heading_re.match(line)
            if match:
                candidate = _normalize_title_for_toc(match.group("title"))
                if candidate == first_title:
                    keep_content = True
                    matched_heading = True
                    output.append(line)
                    continue
            continue

        output.append(line)

    if not matched_heading:
        return md_text

    result = "\n".join(output)
    if md_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def cleanup_markdown(md_text: str, toc_headings: Optional[List[Tuple[int, str, Optional[str]]]] = None) -> str:
    if not md_text:
        return md_text

    base_text = clean_markdown_headings(md_text, toc_headings or []) if toc_headings else md_text

    cleaned_lines: List[str] = []
    for raw in base_text.splitlines():
        stripped = raw.strip()
        if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
            continue
        cleaned_lines.append(raw)

    result = "\n".join(cleaned_lines)
    if base_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def normalize_markdown_file(
    toc_path: Path, md_path: Path, out_dir: Path, *, add_toc: bool = True
) -> Tuple[str, List[Tuple[int, str]], List[List[Any]], Path, List[Tuple[int, str, Optional[str]]]]:
    toc_raw, toc_root = parse_html_toc(toc_path)

    toc_headings: List[Tuple[int, str, Optional[str]]] = []
    for entry in toc_raw:
        if len(entry) < 2:
            continue
        level = int(entry[0])
        title = str(entry[1]).strip()
        anchor = str(entry[2]).strip() if len(entry) >= 3 and entry[2] else None
        if not title or title.lower() in {"indice", "toc", "html toc"}:
            continue
        toc_headings.append((level, title, anchor))

    md_text = md_path.read_text(encoding="utf-8")
    md_text = normalize_markdown_format(md_text)
    md_text = remove_markdown_index(md_text, toc_headings)
    normalized_md = normalize_markdown_headings(md_text, toc_raw)
    cleaned_md = clean_markdown_headings(normalized_md, toc_headings)
    final_md = add_html_toc_to_markdown(cleaned_md, toc_headings) if add_toc else cleaned_md
    safe_write_text(md_path, final_md)
    toc_md_path, toc_headings_simple = generate_markdown_toc_file(final_md, md_path, out_dir)

    return final_md, toc_headings_simple, toc_raw, toc_md_path, toc_headings


def validate_markdown_toc_against_source(
    toc_entries: List[List[Any]], headings_md: List[Tuple[int, str]]
) -> TocValidationResult:
    source_headings: List[Tuple[int, str]] = []
    for entry in toc_entries:
        if len(entry) < 2:
            continue
        try:
            level = int(entry[0])
        except Exception:
            continue
        title = str(entry[1]).strip()
        if title.lower() in {"indice", "toc", "html toc"}:
            continue
        source_headings.append((level, title))

    titles_source = [_normalize_title_for_toc(title) for _, title in source_headings]
    titles_md = [_normalize_title_for_toc(title) for _, title in headings_md]

    mismatches: List[Tuple[int, str, str]] = []
    max_len = max(len(titles_source), len(titles_md))
    for idx in range(max_len):
        source_title = titles_source[idx] if idx < len(titles_source) else "<missing>"
        md_title = titles_md[idx] if idx < len(titles_md) else "<missing>"
        if source_title != md_title:
            mismatches.append((idx, source_title, md_title))

    ok = len(mismatches) == 0
    if ok:
        reason = ""
    elif len(titles_source) != len(titles_md):
        reason = f"TOC length differs (source={len(titles_source)}, md={len(titles_md)})"
    else:
        first = mismatches[0]
        reason = f"TOC content differs at position {first[0] + 1}: source='{first[1]}' vs md='{first[2]}'"

    return TocValidationResult(
        ok=ok,
        source_titles=titles_source,
        md_titles=titles_md,
        mismatches=mismatches,
        source_count=len(titles_source),
        md_count=len(titles_md),
        reason=reason,
    )


def log_toc_validation_result(result: TocValidationResult, *, verbose: bool, debug: bool) -> None:
    if result.ok:
        if verbose:
            LOG.info("TOC validation passed (%d entries)", result.source_count)
        return

    base_msg = f"TOC mismatch between source and Markdown .toc (source={result.source_count}, md={result.md_count})"
    if result.mismatches:
        first = result.mismatches[0]
        base_msg += f"; first difference at position {first[0] + 1}: source='{first[1]}' vs md='{first[2]}'"
    if result.reason:
        base_msg = f"{base_msg} | {result.reason}"
    LOG.error(base_msg)

    if verbose:
        LOG.info("TOC comparison (source vs Markdown):")
        mismatch_idx = {idx for idx, _, _ in result.mismatches}
        max_len = max(result.source_count, result.md_count)
        for idx in range(max_len):
            src_title = result.source_titles[idx] if idx < result.source_count else "<missing>"
            md_title = result.md_titles[idx] if idx < result.md_count else "<missing>"
            status = "OK" if idx not in mismatch_idx else "FAIL"
            LOG.info("[%d] %s | SRC=\"%s\" | MD=\"%s\"", idx + 1, status, src_title or "<empty>", md_title or "<empty>")

    if debug:
        LOG.debug("TOC normalized source titles (%d): %s", result.source_count, result.source_titles)
        LOG.debug("TOC normalized Markdown titles (%d): %s", result.md_count, result.md_titles)
        LOG.debug("TOC mismatches indexes: %s", [idx for idx, _, _ in result.mismatches])


def build_manifest_from_outputs(
    *,
    source_path: Path,
    md_path: Path,
    out_dir: Path,
    assets_dir: Path,
    tables_dir: Path,
    toc_path: Optional[Path] = None,
    toc_root: Optional[TocNode] = None,
    toc_raw: Optional[List[List[Any]]] = None,
    manifest_tables: Optional[List[Dict[str, Any]]] = None,
    image_source: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    current_manifest: Dict[str, Any] = {}
    tables: List[Dict[str, Any]] = list(manifest_tables or current_manifest.get("tables") or [])
    toc_root_local = toc_root
    toc_path_local = toc_path or md_path.with_suffix(".toc")

    try:
        md_text_for_tables = md_path.read_text(encoding="utf-8")
    except Exception:
        md_text_for_tables = ""
    table_ref_re = re.compile(r"tables/([^)\s]+?)(?:\\.|\)|\s)")
    referenced_table_stems: Set[str] = set()
    for match in table_ref_re.finditer(md_text_for_tables):
        stem = Path(match.group(1)).stem
        if stem:
            referenced_table_stems.add(stem)

    if toc_root_local is None:
        if toc_raw is not None:
            toc_root_local = build_toc_tree(toc_raw)
        else:
            toc_root_local = build_toc_tree([])

    manifest_markdown = {
        "file": relative_to_output(md_path, out_dir),
        "toc_tree": serialize_toc_tree(toc_root_local),
    }

    existing_table_stems: Set[str] = set()
    for entry in tables:
        for f in entry.get("files", []):
            existing_table_stems.add(Path(f).stem)

    if tables_dir.exists():
        grouped_tables: Dict[str, List[Path]] = {}
        for path in tables_dir.iterdir():
            if not path.is_file():
                continue
            grouped_tables.setdefault(path.stem, []).append(path)

        for stem, files in grouped_tables.items():
            if stem in existing_table_stems:
                continue
            if referenced_table_stems and stem not in referenced_table_stems:
                continue
            page_for_table = guess_page_from_filename(stem)
            table_names = [p.name for p in files]
            context_path, context_str = find_context(toc_path_local, toc_root_local, table_names, page_for_table)
            title = context_path[-1] if context_path else ""
            tables.append(
                {
                    "pdf_source_page": page_for_table,
                    "title": title,
                    "context": context_str,
                    "context_path": context_path,
                    "files": [relative_to_output(p, out_dir) for p in sorted(files)],
                }
            )

    manifest_images: List[Dict[str, Any]] = []
    old_images_by_base: Dict[str, Dict[str, Any]] = {}
    for entry in current_manifest.get("images", []) or []:
        base = Path(str(entry.get("file", ""))).name
        if base:
            old_images_by_base[base] = entry

    if assets_dir.exists():
        for path in sorted(p for p in assets_dir.rglob("*") if p.is_file()):
            base_name = path.name
            previous = old_images_by_base.get(base_name, {})
            page_for_img: Optional[int] = None
            if page_for_img is None:
                page_for_img = previous.get("pdf_source_page")
            if page_for_img is None:
                guessed = guess_page_from_filename(base_name)
                page_for_img = guessed if guessed is not None else None

            context_path, context_str = find_context(toc_path_local, toc_root_local, [base_name], page_for_img)
            title = context_path[-1] if context_path else ""

            manifest_images.append(
                {
                    "file": relative_to_output(path, out_dir),
                    "pdf_source_page": page_for_img,
                    "title": title,
                    "context": context_str,
                    "context_path": context_path,
                    "source": previous.get("source") or (image_source or {}).get(base_name) or "html",
                    "type": previous.get("type", "image"),
                    **({"equation": previous["equation"]} if "equation" in previous else {}),
                    **({"annotation": previous["annotation"]} if "annotation" in previous else {}),
                }
            )

    manifest = {
        "source_html": source_path.name,
        "markdown": manifest_markdown,
        "tables": tables,
        "images": manifest_images,
    }

    return manifest, tables, manifest_images


def generate_item_ids(manifest: Dict[str, Any]) -> Dict[str, Any]:
    toc_counter = 1

    def _assign_toc_ids(nodes: List[Dict[str, Any]]) -> None:
        nonlocal toc_counter
        for node in nodes:
            node["id"] = node.get("id") or f"toc-{toc_counter}"
            toc_counter += 1
            children = node.get("children") or []
            node["children"] = children
            _assign_toc_ids(children)

    markdown_section = manifest.get("markdown") or {}
    toc_tree = markdown_section.get("toc_tree") or []
    _assign_toc_ids(toc_tree)

    table_counter = 1
    for table in manifest.get("tables") or []:
        table["id"] = table.get("id") or f"table-{table_counter}"
        table_counter += 1

    image_counter = 1
    for image in manifest.get("images") or []:
        image["id"] = image.get("id") or f"img-{image_counter}"
        image_counter += 1

    return manifest


def referring_toc(manifest: Dict[str, Any]) -> Dict[str, Any]:
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    ordered_nodes: List[Tuple[Dict[str, Any], Optional[str]]] = []

    def _collect(nodes: List[Dict[str, Any]], parent_id: Optional[str]) -> None:
        for node in nodes:
            node_id = node.get("id")
            children = node.get("children") or []
            if node_id:
                ordered_nodes.append((node, parent_id))
                _collect(children, node_id)
            else:
                _collect(children, parent_id)

    _collect(toc_tree, None)

    for idx, (node, parent_id) in enumerate(ordered_nodes):
        if parent_id:
            node["parent_id"] = parent_id
        else:
            node.pop("parent_id", None)

        if idx > 0:
            node["prev_id"] = ordered_nodes[idx - 1][0].get("id")
        else:
            node.pop("prev_id", None)

        if idx + 1 < len(ordered_nodes):
            node["next_id"] = ordered_nodes[idx + 1][0].get("id")
        else:
            node.pop("next_id", None)

    return manifest


def _find_toc_parent_id_from_context(toc_tree: List[Dict[str, Any]], context_path: List[str]) -> Optional[str]:
    if not context_path:
        return None

    current_nodes = toc_tree
    parent_id: Optional[str] = None
    for title in context_path:
        target = _normalize_title_for_toc(title)
        match: Optional[Dict[str, Any]] = None
        for candidate in current_nodes:
            cand_title = _normalize_title_for_toc(str(candidate.get("title", "")))
            if cand_title == target:
                match = candidate
                break
        if match is None:
            parent_id = None
            break
        parent_id = match.get("id")
        current_nodes = match.get("children") or []

    if parent_id:
        return parent_id

    last_title = _normalize_title_for_toc(context_path[-1])
    stack = list(toc_tree)
    while stack:
        node = stack.pop(0)
        if _normalize_title_for_toc(str(node.get("title", ""))) == last_title:
            return node.get("id")
        stack.extend(node.get("children") or [])

    return None


def referring_tables(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tables = manifest.get("tables") or []
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    parent_groups: Dict[str, List[Dict[str, Any]]] = {}

    for table in tables:
        context_path = table.get("context_path") or []
        parent_id = _find_toc_parent_id_from_context(toc_tree, context_path)
        if parent_id:
            table["parent_id"] = parent_id
            parent_groups.setdefault(parent_id, []).append(table)
        else:
            table.pop("parent_id", None)

    for siblings in parent_groups.values():
        if len(siblings) <= 1:
            if siblings:
                siblings[0].pop("prev_id", None)
                siblings[0].pop("next_id", None)
            continue
        for idx, table in enumerate(siblings):
            if idx > 0:
                table["prev_id"] = siblings[idx - 1].get("id")
            else:
                table.pop("prev_id", None)
            if idx + 1 < len(siblings):
                table["next_id"] = siblings[idx + 1].get("id")
            else:
                table.pop("next_id", None)

    return manifest


def referring_images(manifest: Dict[str, Any]) -> Dict[str, Any]:
    images = manifest.get("images") or []
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    parent_groups: Dict[str, List[Dict[str, Any]]] = {}

    for image in images:
        context_path = image.get("context_path") or []
        parent_id = _find_toc_parent_id_from_context(toc_tree, context_path)
        if parent_id:
            image["parent_id"] = parent_id
            parent_groups.setdefault(parent_id, []).append(image)
        else:
            image.pop("parent_id", None)

    for siblings in parent_groups.values():
        if len(siblings) <= 1:
            if siblings:
                siblings[0].pop("prev_id", None)
                siblings[0].pop("next_id", None)
            continue
        for idx, image in enumerate(siblings):
            if idx > 0:
                image["prev_id"] = siblings[idx - 1].get("id")
            else:
                image.pop("prev_id", None)
            if idx + 1 < len(siblings):
                image["next_id"] = siblings[idx + 1].get("id")
            else:
                image.pop("next_id", None)

    return manifest


def populate_tables(manifest: Dict[str, Any]) -> Dict[str, Any]:
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []
    tables = manifest.get("tables") or []

    node_by_id: Dict[str, Dict[str, Any]] = {}

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                node_by_id[node_id] = node
                node.setdefault("tables", [])
                node.setdefault("images", [])
            _collect(node.get("children") or [])

    _collect(toc_tree)

    for table in tables:
        parent_id = table.get("parent_id")
        if not parent_id:
            continue
        node = node_by_id.get(parent_id)
        if node is None:
            continue
        node.setdefault("tables", [])
        if table.get("id") and table["id"] not in node["tables"]:
            node["tables"].append(table["id"])

    return manifest


def populate_images(manifest: Dict[str, Any]) -> Dict[str, Any]:
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []
    images = manifest.get("images") or []

    node_by_id: Dict[str, Dict[str, Any]] = {}

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                node_by_id[node_id] = node
                node.setdefault("tables", [])
                node.setdefault("images", [])
            _collect(node.get("children") or [])

    _collect(toc_tree)

    for image in images:
        parent_id = image.get("parent_id")
        if not parent_id:
            continue
        node = node_by_id.get(parent_id)
        if node is None:
            continue
        node.setdefault("images", [])
        if image.get("id") and image["id"] not in node["images"]:
            node["images"].append(image["id"])

    return manifest


def cleanup_manifest(manifest: Dict[str, Any], keep_pdf_pages_ref: bool) -> Dict[str, Any]:
    if keep_pdf_pages_ref:
        return manifest

    markdown = manifest.get("markdown") or {}
    toc_nodes = markdown.get("toc_tree") or []

    def _strip_pdf_page(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node.pop("pdf_source_page", None)
            _strip_pdf_page(node.get("children") or [])

    _strip_pdf_page(toc_nodes)

    for table in manifest.get("tables") or []:
        table.pop("pdf_source_page", None)

    for image in manifest.get("images") or []:
        image.pop("pdf_source_page", None)

    return manifest


def _build_line_index(md_text: str) -> Tuple[List[str], List[int], List[int]]:
    lines = md_text.splitlines()
    line_starts: List[int] = []
    newline_lens: List[int] = []
    offset = 0
    trailing_newline = md_text.endswith("\n")
    for idx, line in enumerate(lines):
        line_starts.append(offset)
        has_newline = idx < len(lines) - 1 or trailing_newline
        newline_len = 1 if has_newline else 0
        newline_lens.append(newline_len)
        offset += len(line) + newline_len
    return lines, line_starts, newline_lens


def _line_char_range(
    line_starts: List[int], newline_lens: List[int], lines: List[str], start_idx: int, end_idx: int
) -> Tuple[int, int]:
    start_char = line_starts[start_idx]
    end_offset = line_starts[end_idx] + len(lines[end_idx]) + newline_lens[end_idx]
    end_char = end_offset - 1 if end_offset > start_char else start_char
    return start_char, end_char


def set_toc_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    toc_nodes = (manifest.get("markdown") or {}).get("toc_tree") or []
    if not toc_nodes:
        return manifest

    heading_re = re.compile(r"^#{1,6}\s+(.+?)\s*$")
    toc_link_re = re.compile(r"^\s*-\s*\[(.+?)\]\([^)]*\)\s*$")

    start_map: Dict[int, int] = {}
    end_map: Dict[int, int] = {}
    total_lines = len(lines)

    flattened: List[Dict[str, Any]] = []

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            flattened.append(node)
            _collect(node.get("children") or [])

    _collect(toc_nodes)
    if not flattened:
        return manifest

    positions: List[Tuple[Dict[str, Any], int]] = []

    def _find_line_for_title(target_norm: str, start_from: int) -> Optional[int]:
        for idx in range(start_from, len(lines)):
            match = heading_re.match(lines[idx])
            if not match:
                continue
            candidate = _normalize_title_for_toc(match.group(1))
            if candidate == target_norm:
                return idx

        for idx in range(start_from, len(lines)):
            match = toc_link_re.match(lines[idx])
            if not match:
                continue
            candidate = _normalize_title_for_toc(match.group(1))
            if candidate == target_norm:
                return idx

        return None

    search_start = 0
    for node in flattened:
        title = str(node.get("title") or "").strip()
        if not title:
            continue
        target_norm = _normalize_title_for_toc(title)
        found_idx = _find_line_for_title(target_norm, search_start)
        if found_idx is not None:
            positions.append((node, found_idx))
            search_start = found_idx + 1

    if not positions:
        return manifest

    positions.sort(key=lambda item: item[1])

    for index, (node, start_idx) in enumerate(positions):
        start_map[id(node)] = start_idx
        next_start_idx = positions[index + 1][1] if index + 1 < len(positions) else total_lines
        end_idx = next_start_idx - 1 if next_start_idx > 0 else total_lines - 1
        if end_idx < start_idx:
            end_idx = start_idx
        end_map[id(node)] = end_idx

    for node, start_idx in positions:
        end_idx = end_map.get(id(node), start_idx)
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        node["start_line"] = start_idx + 1
        node["end_line"] = end_idx + 1
        node["start_char"] = start_char
        node["end_char"] = end_char

    return manifest


def set_tables_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    tables = manifest.get("tables") or []
    for table in tables:
        files = table.get("files") or []
        hit_lines: List[int] = []
        for idx, line in enumerate(lines):
            for file_ref in files:
                if file_ref and file_ref in line:
                    hit_lines.append(idx)
        if not hit_lines:
            continue
        start_idx = min(hit_lines)
        end_idx = max(hit_lines)

        table_heading_re = re.compile(r"^#{3,6}\s+tabella", re.IGNORECASE)
        table_block_marker_re = re.compile(r"^###\s+tabelle", re.IGNORECASE)

        expanded_start = start_idx
        seen_block = False
        for i in range(start_idx - 1, -1, -1):
            stripped = lines[i].strip()

            if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
                break

            is_heading = bool(table_heading_re.match(stripped) or table_block_marker_re.match(stripped))
            is_table_row = "|" in stripped and not stripped.startswith("---")
            is_fallback_marker = stripped.lower().startswith("<!-- extracted_tables_fallback")

            if is_heading or is_table_row or is_fallback_marker:
                expanded_start = i
                seen_block = True
                continue

            if stripped == "":
                if seen_block:
                    expanded_start = i
                    continue
                break

            if seen_block:
                break
            break

        start_idx = expanded_start
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        table["start_line"] = start_idx + 1
        table["end_line"] = end_idx + 1
        table["start_char"] = start_char
        table["end_char"] = end_char

    return manifest


def set_images_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    for image in manifest.get("images") or []:
        file_rel = image.get("file")
        if not file_rel:
            continue
        base = Path(str(file_rel)).name
        if not base:
            continue
        candidates: List[int] = []
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            start_eq = EQUATION_BLOCK_START_RE.match(stripped)
            if start_eq and start_eq.group(1).strip() == base:
                candidates.append(idx)
                continue
            end_eq = EQUATION_BLOCK_END_RE.match(stripped)
            if end_eq and end_eq.group(1).strip() == base:
                candidates.append(idx)
                continue
            start_ann = ANNOTATION_BLOCK_START_RE.match(stripped)
            if start_ann and start_ann.group(1).strip() == base:
                candidates.append(idx)
                continue
            end_ann = ANNOTATION_BLOCK_END_RE.match(stripped)
            if end_ann and end_ann.group(1).strip() == base:
                candidates.append(idx)
                continue
            for match in IMG_LINK_RE.finditer(raw):
                url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
                if url.split("/")[-1] == base:
                    candidates.append(idx)
                    break
        if not candidates:
            continue
        start_idx = min(candidates)
        end_idx = max(candidates)
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        image["start_line"] = start_idx + 1
        image["end_line"] = end_idx + 1
        image["start_char"] = start_char
        image["end_char"] = end_char

    return manifest


def embed_equations_in_markdown(md_text: str, formulas_by_base: Dict[str, str]) -> str:
    if not formulas_by_base:
        return md_text

    lines = (md_text or "").splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        start_match = EQUATION_BLOCK_START_RE.match(stripped)
        if start_match:
            block_base = start_match.group(1).strip()
            if block_base in formulas_by_base:
                i += 1
                while i < len(lines):
                    if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                        i += 1
                        break
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines):
                output.append(lines[i])
                if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            continue

        matches = list(IMG_LINK_RE.finditer(line))
        if not matches:
            output.append(line)
            i += 1
            continue

        formula_match = None
        formula_base = ""
        for match in matches:
            url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
            base = url.split("/")[-1]
            if base in formulas_by_base:
                formula_match = match
                formula_base = base
                break

        if not formula_match:
            output.append(line)
            i += 1
            continue

        before = line[: formula_match.start()]
        after = line[formula_match.end() :]
        if before.strip():
            output.append(before.rstrip())

        if output and re.match(r"^\s*\$\$.*\$\$\s*$", output[-1]):
            output.pop()

        if output and output[-1].strip():
            output.append("")

        start_line = f"**----- Start of equation: {formula_base} -----**\n\n"
        output.append(start_line)
        output.append(f"$${formulas_by_base[formula_base]}$$")
        output.append(f"\n**----- End of equation: {formula_base} -----**\n")
        if output and output[-1].strip():
            output.append("")
        output.append(formula_match.group(0))
        if after.strip():
            output.append(after.lstrip())
        i += 1

    return "\n".join(output)


def embed_annotations_in_markdown(md_text: str, annotations: Dict[str, str]) -> str:
    if not annotations:
        return md_text

    lines = (md_text or "").splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        ann_old = ANNOTATION_LINE_RE.match(stripped)
        if ann_old:
            base_existing = ann_old.group(1)
            if base_existing in annotations:
                i += 1
                while i < len(lines) and lines[i].startswith(">"):
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                output.append(lines[i])
                i += 1
            continue

        ann_start = ANNOTATION_BLOCK_START_RE.match(stripped)
        if ann_start:
            base_block = ann_start.group(1).strip()
            if base_block in annotations:
                i += 1
                while i < len(lines):
                    if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                        i += 1
                        break
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines):
                output.append(lines[i])
                if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            continue

        output.append(line)
        matches = list(IMG_LINK_RE.finditer(line))
        for match in matches:
            url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
            base = url.split("/")[-1]
            if base not in annotations:
                continue
            text = annotations[base].strip()
            if not text:
                continue
            if output and output[-1].strip():
                output.append("")
            start_line = f"**----- Start of annotation: {base} -----**\n\n"
            output.append(start_line)
            ann_lines = text.splitlines() or [text]
            for extra in ann_lines:
                output.append(extra.strip())
            output.append(f"\n**----- End of annotation: {base} -----**\n")
            output.append("")
        i += 1

    return "\n".join(output)


def _strip_image_links_from_line(line: str, basenames: Set[str]) -> Tuple[str, bool]:
    if not basenames:
        return line, False
    result_parts: List[str] = []
    last_index = 0
    removed = False
    for match in IMG_LINK_RE.finditer(line):
        result_parts.append(line[last_index : match.start()])
        url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1]
        if base in basenames:
            removed = True
        else:
            result_parts.append(match.group(0))
        last_index = match.end()
    if not removed:
        return line, False
    result_parts.append(line[last_index:])
    new_line = "".join(result_parts)
    if not new_line.strip():
        return "", True
    return new_line, True


def strip_image_references_from_markdown(md_text: str, basenames: Set[str]) -> str:
    if not basenames:
        return md_text

    lines = md_text.splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        eq_start = EQUATION_BLOCK_START_RE.match(stripped)
        if eq_start and eq_start.group(1).strip() in basenames:
            i += 1
            while i < len(lines):
                if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        ann_start = ANNOTATION_BLOCK_START_RE.match(stripped)
        if ann_start and ann_start.group(1).strip() in basenames:
            i += 1
            while i < len(lines):
                if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        new_line, removed = _strip_image_links_from_line(line, basenames)
        if removed and not new_line.strip():
            if output and not output[-1].strip():
                pass
            else:
                output.append("")
        else:
            output.append(new_line)
        i += 1

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def remove_small_images_phase(
    manifest: Dict[str, Any],
    md_text: str,
    out_dir: Path,
    config: PostProcessingConfig,
) -> Tuple[str, Dict[str, Any]]:
    images = list(manifest.get("images") or [])
    if not images:
        return md_text, manifest

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow not available: {exc}") from exc

    kept: List[Dict[str, Any]] = []
    removed_basenames: Set[str] = set()
    min_width = max(1, int(config.min_size_x))
    min_height = max(1, int(config.min_size_y))
    total = len(images)

    for index, entry in enumerate(images, start=1):
        rel_file = entry.get("file")
        if not rel_file:
            kept.append(entry)
            continue
        rel_path = normalize_path_for_md(str(rel_file))
        path = out_dir / rel_path
        base_name = Path(rel_path).name
        width: Optional[int] = None
        height: Optional[int] = None
        if path.exists():
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception as exc:
                if config.debug:
                    LOG.debug("remove-small-images unable to read %s: %s", path, exc)
        else:
            if config.debug:
                LOG.debug("remove-small-images missing file: %s", path)

        should_remove = bool(width is not None and height is not None and width < min_width and height < min_height)
        if config.verbose:
            size_label = f"{width}x{height}px" if width is not None and height is not None else "unknown size"
            status = "REMOVE" if should_remove else "KEEP"
            _log_verbose_progress(
                "remove-small-images",
                index,
                total,
                detail=f"{rel_file} -> {status} ({size_label})",
            )

        if should_remove:
            removed_basenames.add(base_name)
            if config.debug:
                LOG.debug("remove-small-images flagged %s for removal but retains disk file", path)
            continue

        kept.append(entry)

    manifest["images"] = kept
    if not removed_basenames:
        return md_text, manifest

    cleaned_md = strip_image_references_from_markdown(md_text, removed_basenames)
    return cleaned_md, manifest


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _build_gemini_parts(prompt: str, mime_type: str, image_bytes: bytes) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
            ],
        }
    ]


def _init_gemini_model(api_key: str, model_name: str, module_path: str = "google.genai") -> Any:
    try:
        genai = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(f"Unable to import module {module_path}: {exc}") from exc

    if hasattr(genai, "configure"):
        genai.configure(api_key=api_key)

    if hasattr(genai, "GenerativeModel"):
        return genai.GenerativeModel(model_name)

    if hasattr(genai, "Client"):
        client = genai.Client(api_key=api_key)

        class _ClientWrapper:
            def __init__(self, client_obj: Any, name: str) -> None:
                self._client = client_obj
                self._model = name

            def generate_content(self, parts: Any) -> Any:
                return self._client.models.generate_content(model=self._model, contents=parts)

        return _ClientWrapper(client, model_name)

    raise RuntimeError(f"Module {module_path} does not provide GenerativeModel or Client APIs")


def _run_pix2tex_test_mode(
    manifest: Dict[str, Any], md_text: str, config: PostProcessingConfig
) -> Tuple[str, Dict[str, Any]]:
    from html2tree.latex import validate_latex_formula

    formula_raw = _get_test_pix2tex_formula()
    formulas: Dict[str, str] = {}
    if not formula_raw:
        if config.debug:
            LOG.debug("Pix2Tex test mode skipped because canned formula is empty")
        return md_text, manifest

    threshold = config.equation_min_len
    length = len(formula_raw)
    if length < threshold:
        if config.verbose:
            LOG.info(
                "Pix2Tex test mode skipped because canned formula length %d < threshold %d",
                length,
                threshold,
            )
        return md_text, manifest

    is_valid = validate_latex_formula(formula_raw)
    if not is_valid:
        if config.verbose:
            LOG.info("Pix2Tex test mode skipped because canned formula failed validation")
        return md_text, manifest

    images = manifest.get("images") or []
    total_images = len(images)
    if config.verbose:
        LOG.info("Pix2Tex test mode active: using canned formula for %d images", total_images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        if not file_rel:
            continue
        base_name = Path(str(file_rel)).name
        formulas[base_name] = formula_raw
        entry["type"] = "equation"
        entry["equation"] = formula_raw
        if config.verbose:
            LOG.info("Pix2Tex images[%d/%d] %s validation result: PASSED (test mode)", index, total_images, file_rel)

    if not formulas:
        return md_text, manifest

    if config.debug:
        LOG.debug("Pix2Tex test mode formula applied: %s", formula_raw)

    md_text = embed_equations_in_markdown(md_text, formulas)
    return md_text, manifest


def run_pix2tex_phase(
    manifest: Dict[str, Any], md_text: str, out_dir: Path, config: PostProcessingConfig
) -> Tuple[str, Dict[str, Any]]:
    from html2tree.latex import validate_latex_formula

    if config.test_mode:
        return _run_pix2tex_test_mode(manifest, md_text, config)

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow not available: {exc}") from exc

    try:
        from pix2tex.cli import LatexOCR  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pix2tex not available: {exc}") from exc

    model = LatexOCR()
    formulas: Dict[str, str] = {}
    images = manifest.get("images") or []
    total_images = len(images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        if not file_rel:
            continue
        path = out_dir / normalize_path_for_md(str(file_rel))
        pos_ref = f"images[{index}/{total_images}]" if total_images else "images[?]"
        if config.verbose:
            _log_verbose_progress("Pix2Tex", index, total_images, detail=str(file_rel))
        try:
            img = Image.open(path)
        except Exception as exc:
            LOG.debug("Pix2Tex skipped %s (cannot open image): %s", path, exc)
            continue

        try:
            formula_raw = str(model(img)).strip()
        except Exception as exc:
            LOG.debug("Pix2Tex failed on %s: %s", path, exc)
            try:
                img.close()
            except Exception:
                pass
            continue
        finally:
            try:
                img.close()
            except Exception:
                pass

        if config.debug:
            LOG.debug("Pix2Tex %s raw output: %s", pos_ref, formula_raw)

        length = len(formula_raw)
        threshold = config.equation_min_len
        if length >= threshold:
            is_valid = validate_latex_formula(formula_raw)
            if config.verbose:
                status = "PASSED" if is_valid else "FAILED"
                LOG.info("Pix2Tex %s %s validation result: %s", pos_ref, file_rel, status)
            if is_valid:
                base_name = Path(file_rel).name
                formulas[base_name] = formula_raw
                entry["type"] = "equation"
                entry["equation"] = formula_raw
        else:
            if config.verbose:
                LOG.info(
                    "Pix2Tex %s %s validation result: SKIPPED (len=%d < threshold=%d)",
                    pos_ref,
                    file_rel,
                    length,
                    threshold,
                )

    if formulas:
        md_text = embed_equations_in_markdown(md_text, formulas)
    return md_text, manifest


def run_annotation_phase(
    manifest: Dict[str, Any],
    md_text: str,
    out_dir: Path,
    config: PostProcessingConfig,
    pix2tex_executed: bool,
) -> Tuple[str, Dict[str, Any]]:
    if not (config.annotate_images or config.annotate_equations):
        return md_text, manifest

    if not config.gemini_api_key:
        raise RuntimeError("Gemini API key missing for annotation")

    model = None
    if not config.test_mode:
        module_path = config.gemini_module or "google.genai"
        try:
            model = _init_gemini_model(
                config.gemini_api_key,
                config.gemini_model or GEMINI_DEFAULT_MODEL,
                module_path=module_path,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to initialize Gemini model: {exc}") from exc
    elif config.verbose:
        LOG.info("Gemini annotation test mode active: using canned responses")

    annotations: Dict[str, str] = {}
    images = manifest.get("images") or []
    total_images = len(images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        entry_type = str(entry.get("type") or "").lower()
        if not file_rel:
            continue
        is_equation = entry_type == "equation"
        if is_equation and not config.annotate_equations:
            continue
        if not is_equation and not config.annotate_images:
            continue
        if config.verbose:
            _log_verbose_progress("Annotation", index, total_images, detail=str(file_rel))

        path = out_dir / normalize_path_for_md(str(file_rel))
        if config.verbose:
            LOG.info("Annotating %s: %s", "equation" if is_equation else "image", file_rel)
        try:
            if config.test_mode:
                annotation_final = _get_test_annotation_text(is_equation, entry.get("equation"))
            else:
                prompt = select_annotation_prompt(is_equation, pix2tex_executed, config)
                try:
                    image_bytes = path.read_bytes()
                except Exception as exc:
                    raise RuntimeError(f"Unable to read image {path}: {exc}") from exc

                mime_type = _guess_mime_type(path)
                parts = _build_gemini_parts(prompt, mime_type, image_bytes)
                try:
                    model_response = model.generate_content(parts)
                except Exception as exc:
                    if config.debug:
                        LOG.debug("Gemini request failed for %s", path.name, exc_info=exc)
                    raise RuntimeError(f"Gemini request failed for {path.name}: {exc}") from exc

                if config.debug:
                    LOG.debug("Gemini raw response for %s: %r", path.name, model_response)

                annotation_text = getattr(model_response, "text", "") if model_response is not None else ""
                if not annotation_text and hasattr(model_response, "candidates"):
                    candidates = getattr(model_response, "candidates", []) or []
                    if candidates:
                        annotation_text = (
                            getattr(candidates[0], "text", "")
                            or getattr(candidates[0], "content", "")
                            or ""
                        )

                if not annotation_text:
                    raise RuntimeError(f"Empty annotation from Gemini for {path.name}")

                annotation_final = str(annotation_text).strip()

            annotations[Path(file_rel).name] = annotation_final
            entry["annotation"] = annotation_final
            if config.verbose:
                LOG.info("Annotation completed: %s", file_rel)
            if config.debug:
                LOG.debug("Annotation content for %s: %s", file_rel, annotation_final)
        except Exception as exc:
            LOG.error("Annotation failed for %s: %s", file_rel, exc)
            if config.debug:
                LOG.debug("Annotation failure details for %s", file_rel, exc_info=exc)
            continue

    if annotations:
        md_text = embed_annotations_in_markdown(md_text, annotations)

    return md_text, manifest


def processing_prepare_output_dirs(out_dir: Path, debug_enabled: bool) -> Tuple[Path, Path, Optional[Path]]:
    images_dir = out_dir / "assets"
    tables_dir = out_dir / "tables"
    debug_dir = out_dir / "debug" if debug_enabled else None
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, tables_dir, debug_dir


def copy_assets_dir(source_assets: Path, dest_assets: Path, verbose: bool = False) -> None:
    if dest_assets.exists():
        if dest_assets.is_dir():
            shutil.rmtree(dest_assets)
        else:
            dest_assets.unlink()
    shutil.copytree(source_assets, dest_assets)
    if verbose:
        LOG.info("Copied assets: %s -> %s", source_assets, dest_assets)


def convert_html_to_markdown(document_path: Path, verbose: bool = False) -> Tuple[str, List[HtmlTable]]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"beautifulsoup4 not available: {exc}") from exc

    try:
        from markdownify import markdownify as md_convert  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"markdownify not available: {exc}") from exc

    raw_html = document_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    tables = extract_tables_from_html(soup)
    content = soup.body if soup.body is not None else soup
    html_str = str(content)

    if verbose:
        LOG.info("Converting HTML to Markdown via markdownify")

    md_text = md_convert(html_str, heading_style="ATX")
    md_text = replace_table_placeholders(md_text, tables)
    md_text = rewrite_image_links_to_assets_subdir(md_text, subdir="assets")
    return md_text, tables


def find_existing_markdown(out_dir: Path, stem: str) -> Optional[Path]:
    candidate = out_dir / f"{slugify_filename(stem)}.md"
    if candidate.exists():
        return candidate
    md_files = sorted(out_dir.glob("*.md"))
    return md_files[0] if md_files else None


def run_processing_pipeline(
    *,
    from_dir: Path,
    out_dir: Path,
    post_processing_cfg: PostProcessingConfig,
) -> Tuple[str, Path, Path]:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)

    assets_dir, tables_dir, _ = processing_prepare_output_dirs(out_dir, bool(post_processing_cfg.debug))

    document_path = from_dir / "document.html"
    toc_path = from_dir / "toc.html"
    assets_source = from_dir / "assets"

    if not document_path.exists():
        raise RuntimeError(f"document.html not found in {from_dir}")
    if not toc_path.exists():
        raise RuntimeError(f"toc.html not found in {from_dir}")
    if not assets_source.exists():
        raise RuntimeError(f"assets directory not found in {from_dir}")

    md_text, tables = convert_html_to_markdown(document_path, verbose=post_processing_cfg.verbose)

    if tables:
        export_tables_files(tables_dir, tables)
        if post_processing_cfg.verbose:
            LOG.info("Exported %d table(s) to %s", len(tables), tables_dir)

    md_text = md_text.strip() + "\n"

    base = slugify_filename(document_path.stem)
    md_out = out_dir / f"{base}.md"
    manifest_out = out_dir / f"{document_path.stem}.json"

    safe_write_text(md_out, md_text)

    copy_assets_dir(assets_source, assets_dir, verbose=post_processing_cfg.verbose)

    toc_entries, toc_root = parse_html_toc(toc_path)
    toc_md_path, _ = generate_markdown_toc_file(md_text, md_out, out_dir)
    image_bases = extract_image_basenames_from_markdown(md_text)
    image_source = {name: "html" for name in image_bases}

    manifest, _, _ = build_manifest_from_outputs(
        source_path=document_path,
        md_path=md_out,
        out_dir=out_dir,
        assets_dir=assets_dir,
        tables_dir=tables_dir,
        toc_path=toc_md_path,
        toc_root=toc_root,
        toc_raw=toc_entries,
        image_source=image_source,
    )
    safe_write_text(manifest_out, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    backup_path = md_out.with_suffix(md_out.suffix + ".processing.md")
    shutil.copyfile(md_out, backup_path)
    if post_processing_cfg.verbose:
        LOG.info("Backup created: %s", backup_path)

    return md_text, md_out, manifest_out


def run_post_processing_pipeline(
    *,
    out_dir: Path,
    from_dir: Path,
    md_path: Path,
    manifest_path: Path,
    config: PostProcessingConfig,
) -> Tuple[str, Dict[str, Any], bool]:
    _configure_html2tree_logger(_resolve_log_level(config.verbose, config.debug))

    def _save_markdown(md_content: str) -> str:
        final_text = md_content if md_content.endswith("\n") else f"{md_content}\n"
        safe_write_text(md_path, final_text)
        return final_text

    backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
    if not backup_path.exists():
        raise RuntimeError(f"Backup Markdown (.processing.md) not found: {backup_path}")

    try:
        shutil.copyfile(backup_path, md_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to restore Markdown from backup {backup_path}: {exc}") from exc

    toc_path_source = from_dir / "toc.html"
    if not toc_path_source.exists():
        raise RuntimeError(f"toc.html not found in {from_dir}")

    normalized_md_text, toc_headings, toc_raw, toc_md_path, toc_headings_full = normalize_markdown_file(
        toc_path_source, md_path, out_dir, add_toc=not config.disable_toc
    )
    normalized_md_text = _save_markdown(normalized_md_text)

    toc_mismatch = False
    if config.skip_toc_validation:
        LOG.info("TOC validation skipped in test mode.")
    else:
        toc_result = validate_markdown_toc_against_source(toc_raw, toc_headings)
        log_toc_validation_result(toc_result, verbose=config.verbose, debug=config.debug)
        toc_mismatch = not toc_result.ok

    assets_dir = out_dir / "assets"
    manifest_built, _, _ = build_manifest_from_outputs(
        source_path=from_dir / "document.html",
        md_path=md_path,
        out_dir=out_dir,
        assets_dir=assets_dir,
        tables_dir=out_dir / "tables",
        toc_path=toc_md_path,
        toc_root=build_toc_tree(toc_raw),
        toc_raw=toc_raw,
    )

    safe_write_text(manifest_path, json.dumps(manifest_built, ensure_ascii=False, indent=2) + "\n")

    updated_md = normalized_md_text
    updated_manifest = manifest_built
    pix2tex_executed = False

    if not config.disable_remove_small_images:
        updated_md, updated_manifest = remove_small_images_phase(
            manifest=updated_manifest, md_text=updated_md, out_dir=out_dir, config=config
        )
        updated_md = _save_markdown(updated_md)

    if config.enable_pix2tex and not config.disable_pix2tex:
        updated_md, updated_manifest = run_pix2tex_phase(updated_manifest, updated_md, out_dir, config)
        pix2tex_executed = True
        updated_md = _save_markdown(updated_md)
    elif config.verbose:
        if config.disable_pix2tex:
            LOG.info("Pix2Tex disabled via --disable-pic2tex flag.")
        else:
            LOG.info("Pix2Tex disabled by default (use --enable-pic2tex to activate)")

    if config.annotate_images or config.annotate_equations:
        updated_md, updated_manifest = run_annotation_phase(
            updated_manifest, updated_md, out_dir, config, pix2tex_executed=pix2tex_executed
        )
        updated_md = _save_markdown(updated_md)

    if not config.disable_cleanup:
        updated_md = cleanup_markdown(updated_md, toc_headings_full)
        updated_md = _save_markdown(updated_md)
    elif config.verbose:
        LOG.info("Cleanup disabled via --disable-cleanup flag; Markdown markers preserved.")

    updated_manifest = generate_item_ids(updated_manifest)
    updated_manifest = referring_toc(updated_manifest)
    updated_manifest = referring_tables(updated_manifest)
    updated_manifest = referring_images(updated_manifest)
    updated_manifest = populate_tables(updated_manifest)
    updated_manifest = populate_images(updated_manifest)

    updated_manifest = set_toc_lines(updated_manifest, updated_md)
    updated_manifest = set_tables_lines(updated_manifest, updated_md)
    updated_manifest = set_images_lines(updated_manifest, updated_md)

    updated_manifest = cleanup_manifest(updated_manifest, config.enable_pdf_pages_ref)

    safe_write_text(manifest_path, json.dumps(updated_manifest, ensure_ascii=False, indent=2) + "\n")

    updated_md = _save_markdown(updated_md)

    return updated_md, updated_manifest, toc_mismatch
