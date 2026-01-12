"""Command-line interface for html2tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

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
        "Usage: html2tree [--help] [--version|--ver] [--upgrade] [--uninstall]"
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
    except json.JSONDecodeError:
        return None
    tag_name = payload.get("tag_name")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(_get_usage())
        return 2

    _check_latest_version()

    if not argv or args.help:
        print(_get_usage())
        return 0

    if args.version or args.ver:
        print(__version__)
        return 0

    if args.upgrade:
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
        subprocess.run(["uv", "tool", "uninstall", "html2tree"], check=False)
        return 0

    print(_get_usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
