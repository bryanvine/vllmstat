import json

from vllmtop.cli import run_once_json


def test_run_once_json_mock(capsys):
    rc = run_once_json(["--mock", "--once", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "gen_tps" in out
    assert out["kv"]["dtype"] is not None
    assert "running" in out
