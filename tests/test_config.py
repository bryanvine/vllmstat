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
