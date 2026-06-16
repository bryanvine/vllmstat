import json

from vllmstat.cli import main


def test_run_once_json_mock(capsys):
    rc = main(["--mock", "--once", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "gen_tps" in out
    assert out["kv"]["dtype"] is not None
    assert "running" in out


def test_once_json_fleet_emits_array(capsys):
    import json as _json

    from vllmstat.cli import main

    rc = main(
        [
            "--once",
            "--json",
            "--mock",
            "--url",
            "http://localhost:8000",
            "--url",
            "http://localhost:8001",
        ]
    )
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert isinstance(out, list) and len(out) == 2
    assert all("name" in e and "running" in e["snapshot"] for e in out)


def test_resolve_instances_applies_config_globals(tmp_path):
    from vllmstat.cli import resolve_instances
    from vllmstat.config import Config

    p = tmp_path / "vllmstat.toml"
    p.write_text(
        'interval = 2.5\ngpu = false\n[[instance]]\nname = "a"\nurl = "http://localhost:8000"\n'
    )
    cfg = Config(config_path=str(p))
    resolve_instances(cfg, {})
    assert cfg.interval == 2.5  # config global applied (flag left at default)
    assert cfg.gpu is False
    assert len(cfg.instances) == 1 and cfg.instances[0].name == "a"


def test_resolve_instances_applies_cli_logs_default():
    from vllmstat.cli import resolve_instances
    from vllmstat.config import Config

    cfg = Config.from_sources(["--logs", "docker:vllm-xpu"], {})
    resolve_instances(cfg, {})
    assert len(cfg.instances) == 1
    assert cfg.instances[0].logs == "docker:vllm-xpu"


def test_resolve_instances_cli_logs_do_not_override_per_instance(tmp_path):
    from vllmstat.cli import resolve_instances
    from vllmstat.config import Config

    p = tmp_path / "vllmstat.toml"
    p.write_text(
        '[[instance]]\nname = "a"\nurl = "http://localhost:8000"\nlogs = "docker:from-config"\n'
    )
    cfg = Config.from_sources(["--config", str(p), "--logs", "docker:cli-default"], {})
    resolve_instances(cfg, {})
    assert cfg.instances[0].logs == "docker:from-config"


def test_auto_discover_skipped_when_default_up(monkeypatch):
    from vllmstat import cli
    from vllmstat.config import Config

    monkeypatch.setattr(cli, "port_responding", lambda url, timeout=0.4: True)
    called = {"n": 0}
    monkeypatch.setattr(
        cli,
        "discover_docker",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [],
    )
    cfg = Config.from_sources([], {})
    cli.resolve_instances(cfg, {})
    assert [i.url for i in cfg.instances] == ["http://localhost:8000"]
    assert called["n"] == 0  # discovery not consulted when default is up


def test_auto_discover_used_when_default_down(monkeypatch):
    from vllmstat import cli
    from vllmstat.config import Config
    from vllmstat.core.state import Instance

    monkeypatch.setattr(cli, "port_responding", lambda url, timeout=0.4: False)
    monkeypatch.setattr(
        cli,
        "discover_docker",
        lambda *a, **k: [Instance(name="c", url="http://localhost:9001", locality="local")],
    )
    cfg = Config.from_sources([], {})
    cli.resolve_instances(cfg, {})
    assert [i.url for i in cfg.instances] == ["http://localhost:9001"]


def test_auto_discover_falls_back_to_default_when_none(monkeypatch):
    from vllmstat import cli
    from vllmstat.config import Config

    monkeypatch.setattr(cli, "port_responding", lambda url, timeout=0.4: False)
    monkeypatch.setattr(cli, "discover_docker", lambda *a, **k: [])
    cfg = Config.from_sources([], {})
    cli.resolve_instances(cfg, {})
    assert [i.url for i in cfg.instances] == ["http://localhost:8000"]
