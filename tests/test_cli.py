import json
import re
from pathlib import Path

import pytest
import html2tree.cli as cli


def _suppress_online_check(monkeypatch):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: None)


def _write_test_png(path: Path, *, width: int, height: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(200, 10, 10)).save(path)


def _create_minimal_html_source(tmp_path: Path) -> Path:
    source_dir = tmp_path / "src"
    assets_dir = source_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    _write_test_png(assets_dir / "tiny.png", width=10, height=10)
    _write_test_png(assets_dir / "big.png", width=200, height=200)

    (source_dir / "toc.html").write_text(
        """<html><body><ul><li><a href="#section-1">Section 1</a></li></ul></body></html>\n""",
        encoding="utf-8",
    )
    (source_dir / "document.html").write_text(
        """<html><body>
<h1>Section 1</h1>
<p>Intro</p>
<p><img src="assets/tiny.png" alt="tiny"/></p>
<p><img src="assets/big.png" alt="big"/></p>
</body></html>\n""",
        encoding="utf-8",
    )
    return source_dir


def _create_html_source_with_table(tmp_path: Path) -> Path:
    source_dir = tmp_path / "src_table"
    assets_dir = source_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "toc.html").write_text(
        """<html><body><ul><li><a href=\"#section-1\">Section 1</a></li></ul></body></html>\n""",
        encoding="utf-8",
    )

    (source_dir / "document.html").write_text(
        """<html><body>
<h1>Section 1</h1>
<p>Before table.</p>
<table id=\"sample-table\">
  <tr><th>Col1</th><th>Col2</th></tr>
  <tr><td>A</td><td>B</td></tr>
</table>
<p>After table.</p>
</body></html>\n""",
        encoding="utf-8",
    )

    return source_dir


def _create_html_source_with_nested_toc(tmp_path: Path) -> Path:
    source_dir = tmp_path / "src_nested"
    assets_dir = source_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    (source_dir / "toc.html").write_text(
        """<html><body>
<ul>
  <li><a href="#section-1">Section 1</a>
    <ul><li><a href="#subsection-1-1">Subsection 1.1</a></li></ul>
  </li>
</ul>
</body></html>\n""",
        encoding="utf-8",
    )

    (source_dir / "document.html").write_text(
        """<html><body>
<h1>Section 1</h1>
<p>Intro</p>
<h2>Subsection 1.1</h2>
<p>Details</p>
</body></html>\n""",
        encoding="utf-8",
    )

    return source_dir


def test_version_flags_print_version(monkeypatch, capsys):
    _suppress_online_check(monkeypatch)

    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == cli.__version__

    assert cli.main(["--ver"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == cli.__version__


def test_help_and_no_args_show_usage(monkeypatch, capsys):
    _suppress_online_check(monkeypatch)

    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "html2tree" in out
    assert cli.__version__ in out

    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "html2tree" in out
    assert cli.__version__ in out


def test_upgrade_calls_uv_tool_install(monkeypatch, capsys):
    _suppress_online_check(monkeypatch)
    calls = []

    def fake_run(args, check=False):
        calls.append((args, check))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["--upgrade"]) == 0
    capsys.readouterr()

    assert calls == [
        (
            [
                "uv",
                "tool",
                "install",
                "usereq",
                "--force",
                "--from",
                "git+https://github.com/Ogekuri/html2tree.git",
            ],
            False,
        )
    ]


def test_uninstall_calls_uv_tool_uninstall(monkeypatch, capsys):
    _suppress_online_check(monkeypatch)
    calls = []

    def fake_run(args, check=False):
        calls.append((args, check))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["--uninstall"]) == 0
    capsys.readouterr()

    assert calls == [(["uv", "tool", "uninstall", "html2tree"], False)]


def test_online_check_no_message_on_equal_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: cli.__version__)

    cli._check_latest_version()
    out = capsys.readouterr().out
    assert out == ""


def test_online_check_message_on_newer_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: "0.1.3")

    cli._check_latest_version()
    out = capsys.readouterr().out.strip()
    expected = (
        "A new version of html2tree is available: current "
        f"{cli.__version__}, latest 0.1.3. To upgrade, run: html2tree --upgrade"
    )
    assert out == expected


def test_online_check_silent_on_api_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: None)

    cli._check_latest_version()
    out = capsys.readouterr().out
    assert out == ""


def test_convert_html_creates_outputs(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    source_dir = Path(__file__).resolve().parents[1] / "html_sample"
    out_dir = tmp_path / "out"

    result = cli.main([
        "--from-dir",
        str(source_dir),
        "--to-dir",
        str(out_dir),
        "--post-processing",
        "--disable-annotate-images",
    ])
    assert result == 0

    md_path = out_dir / "document.md"
    assert md_path.exists()
    backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
    assert backup_path.exists()

    manifest_path = out_dir / "document.json"
    assert manifest_path.exists()
    toc_path = out_dir / "document.toc"
    assert toc_path.exists()

    assets_dir = out_dir / "assets"
    assert assets_dir.exists()
    assert any(p.is_file() for p in assets_dir.rglob("*"))

    tables_dir = out_dir / "tables"
    assert any(tables_dir.glob("*.md"))
    assert any(tables_dir.glob("*.csv"))


def test_convert_without_post_processing_skips_manifest_and_toc(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    source_dir = Path(__file__).resolve().parents[1] / "html_sample"
    out_dir = tmp_path / "out"

    result = cli.main([
        "--from-dir",
        str(source_dir),
        "--to-dir",
        str(out_dir),
    ])
    assert result == 0

    md_path = out_dir / "document.md"
    assert md_path.exists()
    backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
    assert backup_path.exists()

    toc_html_path = out_dir / "toc.html"
    assert toc_html_path.exists()

    assert not (out_dir / "document.json").exists()
    assert not (out_dir / "document.toc").exists()


def test_post_processing_uses_output_toc_html(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_html_source_with_nested_toc(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0

    out_toc_path = out_dir / "toc.html"
    assert out_toc_path.exists()

    out_toc_path.write_text(
        """<html><body>
<ul>
  <li><a href="#section-1">Section 1</a></li>
  <li><a href="#subsection-1-1">Subsection 1.1</a></li>
</ul>
</body></html>\n""",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-annotate-images",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "# Subsection 1.1" in md_text
    assert "## Subsection 1.1" not in md_text


def test_tables_are_embedded_and_linked(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    source_dir = _create_html_source_with_table(tmp_path)
    out_dir = tmp_path / "out"

    result = cli.main([
        "--from-dir",
        str(source_dir),
        "--to-dir",
        str(out_dir),
        "--post-processing",
        "--disable-annotate-images",
    ])
    assert result == 0

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "HTML2TREE" not in md_text
    assert "sample-table" in md_text
    assert "[Markdown](tables/sample-table.md)" in md_text
    assert "[CSV](tables/sample-table.csv)" in md_text

    before_idx = md_text.index("Before table.")
    table_idx = md_text.index("| Col1 | Col2 |")
    after_idx = md_text.index("After table.")
    assert before_idx < table_idx < after_idx

    tables_dir = out_dir / "tables"
    assert (tables_dir / "sample-table.md").exists()
    assert (tables_dir / "sample-table.csv").exists()

    toc_text = (out_dir / "document.toc").read_text(encoding="utf-8")
    assert "tables/sample-table.md" in toc_text
    assert "tables/sample-table.csv" in toc_text


def test_post_processing_only_runs(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = Path(__file__).resolve().parents[1] / "html_sample"
    out_dir = tmp_path / "out"

    result = cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)])
    assert result == 0

    result = cli.main(
        [
            "--from-dir",
            str(source_dir),
            "--to-dir",
            str(out_dir),
            "--post-processing-only",
            "--disable-annotate-images",
        ]
    )
    assert result == 0

    md_path = out_dir / "document.md"
    toc_path = md_path.with_suffix(".toc")
    assert toc_path.exists()
    assert "** HTML TOC **" in md_path.read_text(encoding="utf-8")

    manifest_path = out_dir / "document.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []
    assert toc_tree
    assert "id" in toc_tree[0]
    assert "start_line" in toc_tree[0]
    assert "end_line" in toc_tree[0]

    assert manifest.get("tables")
    assert "id" in manifest["tables"][0]

    assert manifest.get("images")
    assert "id" in manifest["images"][0]


def test_post_processing_builds_manifest_from_markdown_images(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_minimal_html_source(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0

    backup_path = out_dir / "document.md.processing.md"
    original = backup_path.read_text(encoding="utf-8")
    filtered_lines = [line for line in original.splitlines() if "tiny.png" not in line]
    backup_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-annotate-images",
                "--disable-remove-small-images",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "tiny.png" not in md_text
    assert "big.png" in md_text

    manifest = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    image_files = [Path(entry["file"]).name for entry in manifest.get("images") or [] if entry.get("file")]
    assert image_files == ["big.png"]


def test_post_processing_builds_manifest_from_markdown_tables(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_html_source_with_table(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0

    backup_path = out_dir / "document.md.processing.md"
    original = backup_path.read_text(encoding="utf-8")
    filtered_lines = [line for line in original.splitlines() if "sample-table" not in line and "tables/" not in line]
    backup_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-annotate-images",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "sample-table" not in md_text
    assert "tables/sample-table" not in md_text

    manifest = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    assert manifest.get("tables") == []


def test_write_prompts_file(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    prompts_path = tmp_path / "prompts.json"

    result = cli.main(["--write-prompts", str(prompts_path)])
    assert result == 0

    data = json.loads(prompts_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"prompt_equation", "prompt_non_equation", "prompt_uncertain"}


def test_prompts_file_validation_and_selection(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    source_dir = _create_minimal_html_source(tmp_path)
    out_dir = tmp_path / "out"

    missing_prompts = tmp_path / "missing_prompts.json"

    called = {"run": 0}

    def fail_run_processing_pipeline(**kwargs):
        called["run"] += 1
        raise AssertionError("run_processing_pipeline must not be called when prompts are invalid")

    import html2tree.core as core

    monkeypatch.setattr(core, "run_processing_pipeline", fail_run_processing_pipeline)

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--prompts",
                str(missing_prompts),
            ]
        )
        == 6
    )
    assert called["run"] == 0
    assert not out_dir.exists()

    invalid_prompts = tmp_path / "invalid_prompts.json"
    invalid_prompts.write_text("{", encoding="utf-8")

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--prompts",
                str(invalid_prompts),
            ]
        )
        == 6
    )
    assert called["run"] == 0
    assert not out_dir.exists()

    valid_prompts = tmp_path / "valid_prompts.json"
    valid_prompts.write_text(
        json.dumps(
            {
                "prompt_equation": "EQUATION_PROMPT",
                "prompt_non_equation": "NON_EQUATION_PROMPT",
                "prompt_uncertain": "UNCERTAIN_PROMPT",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def capture_run_processing_pipeline(*, from_dir, out_dir, post_processing_cfg, generate_post_artifacts=True):
        captured["prompt_equation"] = post_processing_cfg.prompt_equation
        captured["prompt_non_equation"] = post_processing_cfg.prompt_non_equation
        captured["prompt_uncertain"] = post_processing_cfg.prompt_uncertain
        return "", out_dir / "document.md", out_dir / "document.json"

    monkeypatch.setattr(core, "run_processing_pipeline", capture_run_processing_pipeline)

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--prompts",
                str(valid_prompts),
            ]
        )
        == 0
    )
    assert captured == {
        "prompt_equation": "EQUATION_PROMPT",
        "prompt_non_equation": "NON_EQUATION_PROMPT",
        "prompt_uncertain": "UNCERTAIN_PROMPT",
    }

    cfg = core.PostProcessingConfig(
        enable_pix2tex=True,
        disable_pix2tex=False,
        equation_min_len=5,
        verbose=False,
        debug=False,
        annotate_images=True,
        annotate_equations=True,
        gemini_api_key="dummy",
        gemini_model=core.GEMINI_DEFAULT_MODEL,
        gemini_module="google.genai",
        test_mode=True,
        disable_remove_small_images=False,
        disable_cleanup=False,
        disable_toc=False,
        # enable_pdf_pages_ref removed; always False
        enable_pdf_pages_ref=False,
        min_size_x=100,
        min_size_y=100,
        prompt_equation="EQUATION_PROMPT",
        prompt_non_equation="NON_EQUATION_PROMPT",
        prompt_uncertain="UNCERTAIN_PROMPT",
    )
    assert core.select_annotation_prompt(is_equation=False, pix2tex_executed=True, config=cfg) == "NON_EQUATION_PROMPT"
    assert core.select_annotation_prompt(is_equation=True, pix2tex_executed=True, config=cfg) == "EQUATION_PROMPT"
    assert core.select_annotation_prompt(is_equation=False, pix2tex_executed=False, config=cfg) == "UNCERTAIN_PROMPT"


def test_remove_small_images_phase(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_minimal_html_source(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0
    assert (out_dir / "assets" / "tiny.png").exists()

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-annotate-images",
                "--min-size-x",
                "100",
                "--min-size-y",
                "100",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "tiny.png" not in md_text
    assert "big.png" in md_text

    manifest = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    image_files = {Path(entry["file"]).name for entry in (manifest.get("images") or []) if entry.get("file")}
    assert "tiny.png" not in image_files
    assert "big.png" in image_files

    assert (out_dir / "assets" / "tiny.png").exists()


def test_pix2tex_test_mode_inserts_equations(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_minimal_html_source(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-annotate-images",
                "--disable-remove-small-images",
                "--enable-pic2tex",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "Start of equation: big.png" in md_text
    assert "End of equation: big.png" in md_text

    manifest = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    entry_by_name = {Path(entry["file"]).name: entry for entry in (manifest.get("images") or []) if entry.get("file")}
    assert entry_by_name["big.png"]["type"] == "equation"
    assert entry_by_name["big.png"]["equation"]


def test_annotation_test_mode_inserts_annotations(monkeypatch, tmp_path):
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "1")
    source_dir = _create_minimal_html_source(tmp_path)
    out_dir = tmp_path / "out"

    assert cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)]) == 0

    assert (
        cli.main(
            [
                "--from-dir",
                str(source_dir),
                "--to-dir",
                str(out_dir),
                "--post-processing-only",
                "--disable-remove-small-images",
                "--gemini-api-key",
                "dummy-key",
            ]
        )
        == 0
    )

    md_text = (out_dir / "document.md").read_text(encoding="utf-8")
    assert "Start of annotation: big.png" in md_text
    assert "End of annotation: big.png" in md_text

    manifest = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    entry_by_name = {Path(entry["file"]).name: entry for entry in (manifest.get("images") or []) if entry.get("file")}
    assert entry_by_name["big.png"]["annotation"]


def test_annotation_converts_svg_to_png(monkeypatch, tmp_path):
    import html2tree.core as core

    assets_dir = tmp_path / "out" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_path = assets_dir / "diagram.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>", encoding="utf-8")

    manifest = {"images": [{"file": "assets/diagram.svg", "type": "image"}]}
    md_text = "![](assets/diagram.svg)\n"

    captured = {}

    class FakeModel:
        def generate_content(self, parts):
            captured["parts"] = parts
            return type("Resp", (), {"text": "annot text"})

    monkeypatch.setattr(core, "_init_gemini_model", lambda api_key, model_name, module_path="google.genai": FakeModel())
    monkeypatch.setattr(
        core,
        "_convert_svg_to_png",
        lambda svg, png: (png.parent.mkdir(parents=True, exist_ok=True), png.write_bytes(b"PNGDATA")),
    )

    def fake_build(prompt, mime_type, image_bytes):
        captured["prompt"] = prompt
        captured["mime"] = mime_type
        captured["bytes"] = image_bytes
        return [{"data": "ignored"}]

    monkeypatch.setattr(core, "_build_gemini_parts", fake_build)

    config = core.PostProcessingConfig(
        enable_pix2tex=False,
        disable_pix2tex=True,
        equation_min_len=5,
        verbose=False,
        debug=False,
        annotate_images=True,
        annotate_equations=False,
        gemini_api_key="dummy-key",
        gemini_model=core.GEMINI_DEFAULT_MODEL,
        gemini_module="google.genai",
        test_mode=False,
        disable_remove_small_images=False,
        disable_cleanup=False,
        disable_toc=False,
        enable_pdf_pages_ref=False,
        min_size_x=100,
        min_size_y=100,
        prompt_equation="PEQ",
        prompt_non_equation="PNON",
        prompt_uncertain="PUNC",
    )

    updated_md, updated_manifest = core.run_annotation_phase(
        manifest, md_text, tmp_path / "out", config, pix2tex_executed=False
    )

    png_path = assets_dir / "diagram.png"
    assert png_path.exists()
    assert captured.get("mime") == "image/png"
    assert captured.get("bytes") == b"PNGDATA"

    entry = (updated_manifest.get("images") or [])[0]
    assert entry.get("annotation") == "annot text"
    assert "Start of annotation: diagram.svg" in updated_md

def test_html_sample_with_verbose_debug(monkeypatch, capsys):
    """Test processing html_sample/ directory with --verbose and --debug flags."""
    import shutil

    _suppress_online_check(monkeypatch)

    # Setup paths
    source_dir = Path(__file__).resolve().parents[1] / "html_sample"
    out_dir = Path(__file__).resolve().parents[1] / "temp" / "html_sample"

    # Ensure output directory is clean
    if out_dir.exists():
        shutil.rmtree(out_dir)

    try:
        # Run conversion with verbose and debug flags
        result = cli.main([
            "--from-dir", str(source_dir),
            "--to-dir", str(out_dir),
            "--verbose",
            "--debug",
            "--post-processing",
            "--disable-annotate-images",
        ])

        # Verify successful completion
        assert result == 0

        # Verify output artifacts exist
        assert (out_dir / "document.md").exists()
        assert (out_dir / "document.md.processing.md").exists()
        assert (out_dir / "document.json").exists()
        assert (out_dir / "assets").exists()
        assert (out_dir / "tables").exists()

    finally:
        # Cleanup: remove temporary directory
        if out_dir.exists():
            shutil.rmtree(out_dir)


def test_toc_normalization_handles_escaped_characters(monkeypatch, tmp_path):
    """Test that TOC validation handles escaped characters like underscores from markdownify."""
    _suppress_online_check(monkeypatch)
    monkeypatch.setenv("HTML2TREE_TEST_MODE", "0")

    source_dir = tmp_path / "src_escaped"
    assets_dir = source_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Create HTML with titles containing special characters (underscores, asterisks)
    (source_dir / "toc.html").write_text(
        """<html><body>
<ul>
  <li><a href="#section-1">PRU_ICSSG Overview</a></li>
  <li><a href="#section-2">Test*Name Section</a></li>
  <li><a href="#section-3">Normal Section</a></li>
</ul>
</body></html>\n""",
        encoding="utf-8",
    )

    (source_dir / "document.html").write_text(
        """<html><body>
<h1>PRU_ICSSG Overview</h1>
<p>Content about PRU_ICSSG</p>
<h1>Test*Name Section</h1>
<p>Content with special chars</p>
<h1>Normal Section</h1>
<p>Normal content</p>
</body></html>\n""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"

    # First run: process only
    result = cli.main(["--from-dir", str(source_dir), "--to-dir", str(out_dir)])
    assert result == 0

    # Verify markdown was created
    md_path = out_dir / "document.md"
    assert md_path.exists()

    # Check that markdownify escaped the underscore and asterisk
    md_content = md_path.read_text(encoding="utf-8")
    # markdownify should have escaped the special characters
    assert "PRU" in md_content
    assert "ICSSG" in md_content

    # Second run: post-processing only with TOC validation
    result = cli.main([
        "--from-dir", str(source_dir),
        "--to-dir", str(out_dir),
        "--post-processing-only",
        "--disable-annotate-images",
    ])

    # Should succeed (exit 0) because normalization now handles escaped characters
    assert result == 0

    # Verify the TOC was properly validated and processed
    toc_path = out_dir / "document.toc"
    assert toc_path.exists()

    toc_content = toc_path.read_text(encoding="utf-8")
    # The .toc file may contain escaped versions (single backslash in file = double in Python string literal)
    assert ("PRU_ICSSG Overview" in toc_content or
            "PRU" in toc_content and "ICSSG Overview" in toc_content)
    assert ("Test*Name Section" in toc_content or
            "Test" in toc_content and "Name Section" in toc_content)
    assert "Normal Section" in toc_content


def test_normalize_title_for_toc_removes_backslash_escapes():
    """Test that _normalize_title_for_toc correctly removes backslash escapes."""
    import html2tree.core as core

    # Test underscore escape
    assert core._normalize_title_for_toc("PRU\\_ICSSG") == core._normalize_title_for_toc("PRU_ICSSG")

    # Test asterisk escape
    assert core._normalize_title_for_toc("Test\\*Name") == core._normalize_title_for_toc("Test*Name")

    # Test backtick escape
    assert core._normalize_title_for_toc("\\`code\\`") == core._normalize_title_for_toc("`code`")

    # Test multiple escapes
    assert core._normalize_title_for_toc("A\\_B\\*C") == core._normalize_title_for_toc("A_B*C")

    # Test that non-escaped characters remain unchanged
    assert core._normalize_title_for_toc("Normal Title") == "Normal Title"

    # Test combination with other normalizations (quotes, spaces, etc.)
    assert core._normalize_title_for_toc("Title\\_with\\_underscores") == "Titlewithunderscores"
