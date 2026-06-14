import vllmtop


def test_version_present():
    assert isinstance(vllmtop.__version__, str)
    assert vllmtop.__version__.count(".") >= 2
