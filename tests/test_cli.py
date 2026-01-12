import html2tree.cli as cli


def _suppress_online_check(monkeypatch):
    monkeypatch.setattr(cli, "_get_latest_version", lambda: None)


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
