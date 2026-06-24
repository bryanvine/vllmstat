from vllmstat.config import Config


def test_defaults():
    c = Config.from_sources(argv=[], env={})
    assert c.url == "http://localhost:8000"
    assert c.interval == 1.0
    assert c.gpu is True
    assert c.mock is False


def test_flags_override_env():
    c = Config.from_sources(
        argv=["--url", "http://h:9", "-i", "2.5", "--no-gpu", "--mock"],
        env={"VLLM_API_KEY": "secret"},
    )
    assert c.url == "http://h:9"
    assert c.interval == 2.5
    assert c.gpu is False
    assert c.mock is True
    assert c.api_key == "secret"


def test_once_json_flags():
    c = Config.from_sources(argv=["--once", "--json"], env={})
    assert c.once is True and c.json is True


def test_urls_repeatable():
    c = Config.from_sources(["--url", "http://a:8000", "-u", "http://b:8001"], {})
    assert c.urls == ["http://a:8000", "http://b:8001"]
    assert c.url == "http://a:8000"  # back-compat: first


def test_url_default_when_none():
    c = Config.from_sources([], {})
    assert c.urls == [] and c.url == "http://localhost:8000"


def test_config_and_discover_flags():
    c = Config.from_sources(["--config", "/tmp/x.toml", "--discover-docker"], {})
    assert c.config_path == "/tmp/x.toml" and c.discover_docker is True


def test_logs_flag():
    c = Config.from_sources(["--logs", "docker:vllm-xpu"], {})
    assert c.logs == "docker:vllm-xpu"


def test_proxy_flag():
    c = Config.from_sources(["--proxy", "9000"], {})
    assert c.proxy == "9000"


def test_main_rejects_invalid_proxy(capsys):
    from vllmstat.cli import main

    rc = main(["--mock", "--proxy", "not-a-port"])
    assert rc == 2
    assert "invalid proxy port" in capsys.readouterr().err


def test_config_has_energy_default():
    from vllmstat.core.energy import EnergyConfig

    cfg = Config.from_sources([], {})
    assert isinstance(cfg.energy, EnergyConfig)
    assert cfg.energy.currency == "$"


def test_energy_config_loaded_from_file(tmp_path):
    from vllmstat.cli import resolve_instances

    p = tmp_path / "vllmstat.toml"
    p.write_text(
        '[energy]\ncurrency = "£"\n'
        '[[energy.tou]]\ndefault = true\nrate = 0.15\n'
    )
    cfg = Config.from_sources(["--config", str(p)], {})
    resolve_instances(cfg, {})
    assert cfg.energy.currency == "£"
    assert cfg.energy.tou[0].rate == 0.15


def test_invalid_energy_config_is_ignored(tmp_path, capsys):
    # bad config (tou rules but no default) must not crash; energy stays at defaults
    from vllmstat.cli import resolve_instances

    p = tmp_path / "vllmstat.toml"
    p.write_text(
        '[[energy.tou]]\ndays = "mon-fri"\nfrom = "9:00"\nto = "17:00"\nrate = 0.3\n'
    )
    cfg = Config.from_sources(["--config", str(p)], {})
    resolve_instances(cfg, {})
    assert cfg.energy.currency == "$"  # default, parse was rejected
    err = capsys.readouterr().err
    assert "energy" in err.lower()
