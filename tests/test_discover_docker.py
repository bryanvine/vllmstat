from vllmstat.providers.discover_docker import (
    build_instances,
    discover_docker,
    gpus_from_inspect,
    host_port_from_inspect,
    is_vllm_container,
    parse_ps,
)


def test_is_vllm_container():
    assert is_vllm_container({"Image": "vllm/vllm-openai:latest", "Command": "python -m x"})
    assert not is_vllm_container({"Image": "nginx", "Command": "nginx -g"})


def test_parse_ps_jsonl():
    rows = parse_ps('{"ID":"abc","Image":"vllm/vllm-openai"}\n\n{"ID":"def","Image":"nginx"}\n')
    assert [r["ID"] for r in rows] == ["abc", "def"]


def test_host_port_prefers_8000():
    insp = {
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostPort": "9001"}], "22/tcp": [{"HostPort": "2222"}]}
        }
    }
    assert host_port_from_inspect(insp) == 9001


def test_gpus_device_requests():
    insp = {
        "HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]], "DeviceIDs": ["0", "1"]}]}
    }
    assert gpus_from_inspect(insp) == (0, 1)


def test_gpus_visible_devices_env():
    insp = {"Config": {"Env": ["FOO=1", "NVIDIA_VISIBLE_DEVICES=0,2"]}}
    assert gpus_from_inspect(insp) == (0, 2)


def test_gpus_all_uses_host_count():
    insp = {"HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]], "Count": -1}]}}
    assert gpus_from_inspect(insp, host_gpu_count=2) == (0, 1)


def test_build_instances_filters_and_maps():
    ps = [
        {"ID": "abc", "Image": "vllm/vllm-openai", "Command": "python -m vllm"},
        {"ID": "def", "Image": "nginx", "Command": "nginx"},
    ]
    insp = {
        "abc": {
            "Name": "/qwen",
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "9000"}]}},
            "HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]], "DeviceIDs": ["0"]}]},
        }
    }
    out = build_instances(ps, insp)
    assert len(out) == 1
    assert out[0].name == "qwen" and out[0].url == "http://localhost:9000"
    assert out[0].gpus == (0,) and out[0].locality == "local"
    assert out[0].logs == "docker:qwen"


def test_discover_docker_with_stub_run():
    def run(cmd):
        if cmd[1] == "ps":
            return '{"ID":"abc","Image":"vllm/vllm-openai","Command":"python -m vllm"}\n'
        return (
            '[{"Name":"/qwen","NetworkSettings":{"Ports":{"8000/tcp":[{"HostPort":"9000"}]}},'
            '"HostConfig":{"DeviceRequests":[{"Capabilities":[["gpu"]],"DeviceIDs":["0"]}]}}]'
        )

    out = discover_docker(run=run)
    assert len(out) == 1 and out[0].url == "http://localhost:9000"


def test_discover_docker_failure_returns_empty():
    def run(cmd):
        raise FileNotFoundError("docker missing")

    assert discover_docker(run=run) == []


def test_discover_docker_failure_prints_stderr(capsys):
    def run(cmd):
        raise FileNotFoundError("docker missing")

    assert discover_docker(run=run) == []
    assert "docker discovery failed" in capsys.readouterr().err
