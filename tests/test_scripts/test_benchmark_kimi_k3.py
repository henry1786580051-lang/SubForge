import json
from argparse import Namespace

import pytest

from scripts import benchmark_kimi_k3 as benchmark


def test_kimi_benchmark_rejects_concurrency_above_free_limit(monkeypatch):
    monkeypatch.setattr(benchmark, "_nvidia_api_key", lambda: "test-key")

    with pytest.raises(ValueError, match="between 1 and 5"):
        benchmark._build_config(threads=6, batch_size=20, optimize=True)


def test_kimi_benchmark_is_locked_to_nvidia_kimi_k3(monkeypatch):
    monkeypatch.setattr(benchmark, "_nvidia_api_key", lambda: "test-key")

    config = benchmark._build_config(threads=5, batch_size=20, optimize=False)

    assert config["llm"] == {
        "api_key": "test-key",
        "api_base": benchmark.NVIDIA_BASE_URL,
        "model": benchmark.KIMI_K3_MODEL,
    }
    assert config["subtitle"]["thread_num"] == 5
    assert config["subtitle"]["batch_size"] == 20
    assert config["subtitle"]["optimize"] is False


def test_kimi_benchmark_main_forwards_isolated_runtime(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    output = tmp_path / "output.srt"
    captured = {}

    monkeypatch.setattr(benchmark, "_nvidia_api_key", lambda: "test-key")

    def fake_run(args, config):
        captured["args"] = args
        captured["config"] = config
        return 0

    monkeypatch.setattr(benchmark.subtitle_cmd, "run", fake_run)
    monkeypatch.setattr(
        benchmark.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(
            input=source,
            output=output,
            threads=5,
            batch_size=20,
            translate_only=True,
        ),
    )

    assert benchmark.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["model"] == benchmark.KIMI_K3_MODEL
    assert captured["args"].input == str(source.resolve())
    assert captured["config"]["llm"]["model"] == benchmark.KIMI_K3_MODEL
    assert captured["config"]["subtitle"]["thread_num"] == 5
