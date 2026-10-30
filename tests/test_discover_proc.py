from vllmstat.providers.discover_proc import discover_procs, instance_from_argv, serve_args


def _mkproc(root, pid, argv, cgroup="0::/user.slice/user-1001.slice/app.slice/vllm.service\n"):
    d = root / str(pid)
    d.mkdir()
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (d / "cgroup").write_text(cgroup)


def test_serve_args_console_script_and_module():
    assert serve_args(["/v/bin/python", "/v/bin/vllm", "serve", "m", "--port", "1"]) == [
        "m",
        "--port",
        "1",
    ]
    assert serve_args(["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "m"]) == [
        "--model",
        "m",
    ]
    assert serve_args(["VLLM::EngineCore"]) is None
    assert serve_args(["python", "train_vllm.py"]) is None


def test_instance_uses_bound_host_port_and_served_name():
    inst = instance_from_argv(
        [
            "/v/bin/python",
            "/v/bin/vllm",
            "serve",
            "/models/qwen3-8b",
            "--served-model-name",
            "qwen3-8b-v4",
            "qwen3-8b-v3",
            "--host",
            "100.77.28.107",
            "--port",
            "8003",
            "--max-model-len",
            "16384",
        ]
    )
    assert inst is not None
    assert inst.url == "http://100.77.28.107:8003"
    assert inst.name == "qwen3-8b-v4"
    assert inst.locality == "local"


def test_instance_defaults_wildcard_host_and_port_8000():
    inst = instance_from_argv(["vllm", "serve", "/models/llama/", "--host=0.0.0.0"])
    assert inst is not None and inst.url == "http://localhost:8000"
    assert inst.name == "llama"
    v6 = instance_from_argv(["vllm", "serve", "m", "--host", "::1", "--port=9000"])
    assert v6 is not None and v6.url == "http://[::1]:9000"


def test_instance_skips_unix_socket_and_bad_port():
    assert instance_from_argv(["vllm", "serve", "m", "--uds", "/tmp/v.sock"]) is None
    assert instance_from_argv(["vllm", "serve", "m", "--port", "abc"]) is None


def test_discover_procs_scans_dedupes_and_skips_containers(tmp_path):
    _mkproc(tmp_path, 10, ["/v/bin/python", "/v/bin/vllm", "serve", "a", "--port", "8001"])
    _mkproc(tmp_path, 11, ["uv", "run", "vllm", "serve", "a", "--port", "8001"])  # same server
    _mkproc(tmp_path, 12, ["VLLM::EngineCore"])
    _mkproc(tmp_path, 13, ["bash"])
    _mkproc(
        tmp_path,
        14,
        ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "b"],
        cgroup="0::/system.slice/docker-abc123.scope\n",
    )
    (tmp_path / "self").mkdir()
    got = discover_procs(proc_root=str(tmp_path))
    assert [(i.name, i.url) for i in got] == [("a", "http://localhost:8001")]


def test_discover_procs_missing_procfs(tmp_path):
    assert discover_procs(proc_root=str(tmp_path / "nope")) == []
