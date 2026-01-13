"""Command-line interface for html2tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .version import __version__

LATEST_RELEASE_URL = "https://api.github.com/repos/Ogekuri/html2tree/releases/latest"


def _parse_version(value: str) -> tuple[int, int, int] | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.startswith("v"):
        cleaned = cleaned[1:]
    parts = cleaned.split(".")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _get_usage() -> str:
    return (
        f"html2tree {__version__}\n"
        "Usage:\n"
        "  html2tree [--help] [--version|--ver] [--upgrade] [--uninstall]\n"
        "  html2tree --from-dir FROM_DIR --to-dir TO_DIR [options]\n\n"
        "Options:\n"
        "  --post-processing            Run post-processing after conversion\n"
        "  --post-processing-only       Run post-processing on existing output\n"
        "  --enable-pic2tex             Enable Pix2Tex equation extraction\n"
        "  --equation-min-len N          Minimum Pix2Tex formula length (default: 5)\n"
        "  --disable-cleanup            Preserve markers during cleanup\n"
        "  --disable-toc                Skip insertion of TOC into Markdown\n"
        "  --disable-annotate-images    Disable Gemini image annotation\n"
        "  --enable-annotate-equations  Enable Gemini equation annotation\n"
        "  --gemini-api-key KEY         Gemini API key\n"
        "  --gemini-model MODEL         Gemini model name\n"
        "  --prompts PATH               Use prompts JSON (equation/non-equation/uncertain)\n"
        "  --write-prompts PATH         Write default prompts JSON and exit\n"
        "  --verbose                    Verbose progress logs\n"
        "  --debug                      Debug logs + extra artifacts"
    )


def _get_latest_version() -> str | None:
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            data = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    try:
        payload = json.loads(data)
    except Exception:
        return None
    tag_name = payload.get("tag_name") if isinstance(payload, dict) else None
    if isinstance(tag_name, str):
        return tag_name
    return None


def _check_latest_version() -> None:
    latest = _get_latest_version()
    latest_version = _parse_version(latest or "")
    current_version = _parse_version(__version__)
    if latest_version is None or current_version is None:
        return
    if latest_version > current_version:
        latest_str = ".".join(str(part) for part in latest_version)
        message = (
            "A new version of html2tree is available: current "
            f"{__version__}, latest {latest_str}. To upgrade, run: html2tree --upgrade"
        )
        print(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--ver", action="store_true")
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--from-dir", help="Source project directory containing document.html, toc.html, assets/")
    parser.add_argument("--to-dir", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress logs")
    parser.add_argument("--debug", action="store_true", help="Debug logs + extra artifacts")
    parser.add_argument("--post-processing", action="store_true", help="Run post-processing after conversion")
    parser.add_argument(
        "--post-processing-only",
        action="store_true",
        help="Skip conversion and run post-processing on existing output",
    )
    parser.add_argument(
        "--enable-pic2tex",
        "-enable-pic2tex",
        action="store_true",
        help="Enable Pix2Tex phase within post-processing (disabled by default)",
    )
    parser.add_argument(
        "--disable-pic2tex",
        action="store_true",
        help="Disable Pix2Tex phase within post-processing even when --enable-pic2tex is provided",
    )
    parser.add_argument(
        "--disable-remove-small-images",
        action="store_true",
        help="Disable the remove-small-images post-processing phase",
    )
    parser.add_argument(
        "--enable-pdf-pages-ref",
        action="store_true",
        help="Keep pdf_source_page fields in the final manifest instead of removing them during cleanup",
    )
    parser.add_argument(
        "--disable-cleanup",
        action="store_true",
        help="Disable the cleanup step that removes page markers before manifest enrichment",
    )
    parser.add_argument(
        "--disable-toc",
        action="store_true",
        help="Disable insertion of the Markdown TOC rebuilt from toc.html during post-processing",
    )
    parser.add_argument(
        "--equation-min-len",
        type=int,
        default=5,
        help="Minimum length of Pix2Tex output to classify an image as equation (default: 5)",
    )
    parser.add_argument(
        "--min-size-x",
        type=int,
        default=100,
        help="Minimum width in pixels for remove-small-images (default: 100)",
    )
    parser.add_argument(
        "--min-size-y",
        type=int,
        default=100,
        help="Minimum height in pixels for remove-small-images (default: 100)",
    )
    parser.add_argument(
        "--disable-annotate-images",
        action="store_true",
        help="Disable Gemini-based annotation for images during post-processing (enabled by default)",
    )
    parser.add_argument(
        "--enable-annotate-equations",
        action="store_true",
        help="Enable Gemini-based annotation for equations during post-processing",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="API key for Gemini used by annotation phase (fallback: GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name for annotation",
    )
    parser.add_argument(
        "--prompts",
        help="Path to a JSON file containing prompt_equation, prompt_non_equation, prompt_uncertain",
    )
    parser.add_argument(
        "--write-prompts",
        help="Write the default prompts JSON to the given path and exit",
    )
    return parser


def _validate_numeric_args(args: argparse.Namespace) -> str | None:
    if args.equation_min_len is None or args.equation_min_len <= 0:
        return "Invalid value for --equation-min-len: must be > 0"
    if args.min_size_x is None or args.min_size_x <= 0:
        return "Invalid value for --min-size-x: must be > 0"
    if args.min_size_y is None or args.min_size_y <= 0:
        return "Invalid value for --min-size-y: must be > 0"
    return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(_get_usage())
        return 2

    if args.post_processing and args.post_processing_only:
        print("Options --post-processing and --post-processing-only are mutually exclusive", file=sys.stderr)
        return 6

    numeric_error = _validate_numeric_args(args)
    if numeric_error:
        print(numeric_error, file=sys.stderr)
        return 6

    if not argv or args.help:
        _check_latest_version()
        print(_get_usage())
        return 0

    if args.version or args.ver:
        _check_latest_version()
        print(__version__)
        return 0

    if args.upgrade:
        _check_latest_version()
        subprocess.run(
            [
                "uv",
                "tool",
                "install",
                "usereq",
                "--force",
                "--from",
                "git+https://github.com/Ogekuri/html2tree.git",
            ],
            check=False,
        )
        return 0

    if args.uninstall:
        _check_latest_version()
        subprocess.run(["uv", "tool", "uninstall", "html2tree"], check=False)
        return 0

    if args.write_prompts:
        _check_latest_version()
        try:
            from html2tree import core
        except Exception as exc:
            print(f"Unable to import html2tree core: {exc}", file=sys.stderr)
            return 6
        core.setup_logging(args.verbose, args.debug)
        target = Path(args.write_prompts).expanduser().resolve()
        try:
            core._write_prompts_file(target)
        except Exception as exc:
            print(f"Unable to write prompts file {target}: {exc}", file=sys.stderr)
            return 6
        if args.verbose:
            print(f"Default prompts written to {target}")
        return 0

    if not args.from_dir or not args.to_dir:
        print(_get_usage())
        print("Options --from-dir and --to-dir are required unless --write-prompts or --version/--ver is used", file=sys.stderr)
        return 6

    from_dir = Path(args.from_dir).expanduser().resolve()
    to_dir = Path(args.to_dir).expanduser().resolve()

    if not from_dir.exists() or not from_dir.is_dir():
        print(f"Source directory not found: {from_dir}", file=sys.stderr)
        return 6

    document_path = from_dir / "document.html"
    toc_path = from_dir / "toc.html"
    assets_dir = from_dir / "assets"
    if not document_path.exists():
        print(f"document.html not found in {from_dir}", file=sys.stderr)
        return 6
    if not toc_path.exists():
        print(f"toc.html not found in {from_dir}", file=sys.stderr)
        return 6
    if not assets_dir.exists() or not assets_dir.is_dir():
        print(f"assets directory not found in {from_dir}", file=sys.stderr)
        return 6

    if to_dir.exists():
        if not to_dir.is_dir():
            print(f"Output path is not a directory: {to_dir}", file=sys.stderr)
            return 7
        if not args.post_processing_only and any(to_dir.iterdir()):
            print(f"Output directory must be empty: {to_dir}", file=sys.stderr)
            return 7

    _check_latest_version()

    try:
        from html2tree import core
    except Exception as exc:
        print(f"Unable to import html2tree core: {exc}", file=sys.stderr)
        return 6

    core.setup_logging(args.verbose, args.debug)

    prompts_cfg = dict(core.DEFAULT_PROMPTS)
    if args.prompts:
        prompts_path = Path(args.prompts).expanduser().resolve()
        if not prompts_path.exists() or not prompts_path.is_file():
            print(f"Prompts file not found: {prompts_path}", file=sys.stderr)
            return 6
        try:
            prompts_cfg = core.load_prompts_file(prompts_path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 6

    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    gemini_module = os.environ.get("HTML2TREE_GEMINI_MODULE", "google.genai")

    post_processing_active = bool(args.post_processing or args.post_processing_only)
    annotate_images_enabled = not args.disable_annotate_images
    annotate_equations_enabled = bool(args.enable_annotate_equations)
    effective_annotate_images = annotate_images_enabled if post_processing_active else False
    effective_annotate_equations = annotate_equations_enabled if post_processing_active else False

    post_processing_cfg = core.PostProcessingConfig(
        enable_pix2tex=bool(args.enable_pic2tex) and post_processing_active,
        disable_pix2tex=bool(args.disable_pic2tex),
        equation_min_len=int(args.equation_min_len),
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        annotate_images=effective_annotate_images,
        annotate_equations=effective_annotate_equations,
        gemini_api_key=gemini_api_key,
        gemini_model=str(args.gemini_model or core.GEMINI_DEFAULT_MODEL),
        gemini_module=str(gemini_module or "google.genai"),
        test_mode=core.is_test_mode(),
        disable_remove_small_images=bool(args.disable_remove_small_images),
        disable_cleanup=bool(args.disable_cleanup),
        disable_toc=bool(args.disable_toc),
        enable_pdf_pages_ref=bool(args.enable_pdf_pages_ref),
        min_size_x=int(args.min_size_x),
        min_size_y=int(args.min_size_y),
        prompt_equation=prompts_cfg["prompt_equation"],
        prompt_non_equation=prompts_cfg["prompt_non_equation"],
        prompt_uncertain=prompts_cfg["prompt_uncertain"],
        skip_toc_validation=core.is_test_mode(),
    )

    annotation_active = post_processing_cfg.annotate_images or post_processing_cfg.annotate_equations
    if annotation_active and not post_processing_cfg.gemini_api_key:
        print(
            "Gemini API key is required when annotation is enabled (image annotation enabled by default or --enable-annotate-equations requested)",
            file=sys.stderr,
        )
        return 6

    manifest_path = to_dir / f"{document_path.stem}.json"

    if args.post_processing_only:
        if not to_dir.exists():
            print(f"Output directory not found for post-processing: {to_dir}", file=sys.stderr)
            return 9
        md_path = core.find_existing_markdown(to_dir, document_path.stem)
        if not md_path or not md_path.exists():
            print(f"Markdown file not found in output directory: {to_dir}", file=sys.stderr)
            return 9
        backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
        if not backup_path.exists():
            print(f"Backup Markdown (.processing.md) not found: {backup_path}", file=sys.stderr)
            return 9
        try:
            _, _, toc_mismatch = core.run_post_processing_pipeline(
                out_dir=to_dir,
                from_dir=from_dir,
                md_path=md_path,
                manifest_path=manifest_path,
                config=post_processing_cfg,
            )
        except RuntimeError as exc:
            print(f"Post-processing failed: {exc}", file=sys.stderr)
            return 10
        return 10 if toc_mismatch else 0

    try:
        _, md_path, manifest_path = core.run_processing_pipeline(
            from_dir=from_dir,
            out_dir=to_dir,
            post_processing_cfg=post_processing_cfg,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 6

    if not args.post_processing:
        return 0

    try:
        _, _, toc_mismatch = core.run_post_processing_pipeline(
            out_dir=to_dir,
            from_dir=from_dir,
            md_path=md_path,
            manifest_path=manifest_path,
            config=post_processing_cfg,
        )
    except RuntimeError as exc:
        print(f"Post-processing failed: {exc}", file=sys.stderr)
        return 10

    return 10 if toc_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
