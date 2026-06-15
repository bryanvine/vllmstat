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
