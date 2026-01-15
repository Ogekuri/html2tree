from pathlib import Path

import pytest

import html2tree.cli as cli
import html2tree.core as core


def _suppress_online_check(monkeypatch):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: None)


def _create_source_dir(tmp_path: Path, *, toc_html: str, doc_html: str) -> Path:
    source_dir = tmp_path / "src"
    (source_dir / "assets").mkdir(parents=True, exist_ok=True)
    (source_dir / "toc.html").write_text(toc_html, encoding="utf-8")
    (source_dir / "document.html").write_text(doc_html, encoding="utf-8")
    return source_dir


def test_normalize_markdown_headings_does_not_promote_list_items_to_headings():
    md_text = (
        "# 3 Developer Guides\n\n"
        "* 3.1 PRU-ICSSG Input/Output Modes\n\n"
        "## 3.1 PRU-ICSSG Input/Output Modes\n\n"
        "Content\n"
    )
    toc_entries = [
        [1, "3 Developer Guides", "page-1"],
        [2, "3.1 PRU-ICSSG Input/Output Modes", "page-2"],
    ]

    normalized = core.normalize_markdown_headings(md_text, toc_entries)

    assert "* 3.1 PRU-ICSSG Input/Output Modes" in normalized
    assert normalized.count("## 3.1 PRU-ICSSG Input/Output Modes") == 1


def test_normalize_markdown_headings_unescapes_markdownify_escapes():
    md_text = "# 1 Intro\n\n## PRU\\_ICSSG Overview\n"
    toc_entries = [
        [1, "1 Intro", "page-1"],
        [2, "PRU_ICSSG Overview", "page-2"],
    ]

    normalized = core.normalize_markdown_headings(md_text, toc_entries)

    assert "## PRU_ICSSG Overview" in normalized
    assert "\\_" not in normalized


def test_post_processing_does_not_duplicate_toc_entries_when_document_contains_links_list(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "0")

    source_dir = _create_source_dir(
        tmp_path,
        toc_html=(
            "<html><body><ul>"
            '<li><a href="#intro">1 Intro</a></li>'
            '<li><a href="#sub">2 Subsection</a></li>'
            "</ul></body></html>\n"
        ),
        doc_html=(
            "<html><body>\n"
            '<h1 id="intro">1 Intro</h1>\n'
            "<p>Intro text.</p>\n"
            "<ul><li><a>2 Subsection</a></li></ul>\n"
            '<h1 id="sub">2 Subsection</h1>\n'
            "<p>Content.</p>\n"
            "</body></html>\n"
        ),
    )
    out_dir = tmp_path / "out"

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing",
                "--disable-annotate-images",
            ]
        )
        == 0
    )

    toc_path = out_dir / "document.toc"
    toc_text = toc_path.read_text(encoding="utf-8")
    assert toc_text.count("2 Subsection") == 1


def test_cli_returns_exit_code_10_on_toc_mismatch_when_validation_enabled(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "0")

    source_dir = _create_source_dir(
        tmp_path,
        toc_html=(
            "<html><body><ul>"
            '<li><a href="#section-1">Section 1</a></li>'
            '<li><a href="#section-2">Section 2</a></li>'
            "</ul></body></html>\n"
        ),
        doc_html=(
            "<html><body>\n"
            "<h1>Section 1</h1>\n"
            "<p>Ok</p>\n"
            "<h1>Section 3</h1>\n"
            "<p>Extra heading not in toc</p>\n"
            "</body></html>\n"
        ),
    )
    out_dir = tmp_path / "out"

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing",
                "--disable-annotate-images",
            ]
        )
        == 10
    )

