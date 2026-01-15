import os
import re
import subprocess
from pathlib import Path

import pytest


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _extract_examples_log_errors(log_text: str) -> list[str]:
    errors: list[str] = []
    for raw_line in (log_text or "").splitlines():
        line = _strip_ansi(raw_line).strip()
        if not line:
            continue

        if re.search(r"\[(ERROR|FATAL)\]", line, flags=re.IGNORECASE):
            errors.append(line)
            continue

        if re.match(r"(ERROR|FATAL)\b", line, flags=re.IGNORECASE):
            errors.append(line)
            continue

        if re.search(r"\bFAIL\b", line) and ("SRC=" in line or "MD=" in line):
            errors.append(line)
            continue

        rc_match = re.search(r"\(rc=(\d+)\)", line)
        if rc_match:
            try:
                if int(rc_match.group(1)) != 0:
                    errors.append(line)
                    continue
            except ValueError:
                errors.append(line)
                continue

    return errors


def test_examples_script_log_has_no_errors():
    if os.environ.get("RUN_EXAMPLES_TEST") != "1":
        pytest.skip("RUN_EXAMPLES_TEST not enabled")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "examples.sh"
    log_path = repo_root / "examples.log"

    if not script_path.exists():
        pytest.fail(f"examples.sh not found at {script_path}")

    env = os.environ.copy()
    env["HTML2TREE_TEST_MODE"] = "0"
    env.pop("PYTEST_CURRENT_TEST", None)

    result = subprocess.run(["bash", str(script_path)], cwd=str(repo_root), env=env, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(
            "examples.sh exited with non-zero code "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    if not log_path.exists():
        pytest.fail("examples.log not found after running examples.sh")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    error_lines = _extract_examples_log_errors(log_text)

    if error_lines:
        snippet = "\n".join(error_lines[:20])
        pytest.fail(f"examples.log contains errors:\n{snippet}")


def test_examples_log_error_parser_detects_fail_and_rc():
    log_text = """
INFO: [10] FAIL | SRC="New in this Release" | MD="Release Notes 09.01.00"
[ERROR] (rc=10) on dir examples/out_software-dl.ti.com_ind_comms_sdk_am64x_latest_docs_api_guide_am64x
ERROR: TOC mismatch between source and Markdown .toc (source=29, md=30)
"""
    errors = _extract_examples_log_errors(log_text)
    assert any("FAIL" in line for line in errors)
    assert any("rc=10" in line for line in errors)
    assert any("TOC mismatch" in line for line in errors)


def test_examples_log_error_parser_ignores_debug_failed_import():
    log_text = """
DEBUG: Image: failed to import FpxImagePlugin: No module named 'olefile'
DEBUG: Image: failed to import MicImagePlugin: No module named 'olefile'
"""
    errors = _extract_examples_log_errors(log_text)
    assert errors == []


def test_examples_log_error_parser_strips_ansi():
    log_text = "\x1b[31mERROR: TOC mismatch between source and Markdown .toc\x1b[0m\n"
    errors = _extract_examples_log_errors(log_text)
    assert errors == ["ERROR: TOC mismatch between source and Markdown .toc"]

