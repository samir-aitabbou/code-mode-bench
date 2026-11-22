from pathlib import Path

from typer.testing import CliRunner

from cmbench.cli import app

runner = CliRunner()


def test_data_command_generates_db(tmp_path: Path) -> None:
    out = tmp_path / "crm.db"
    result = runner.invoke(app, ["data", "--out", str(out), "--seed", "42"])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_serve_cmd_gpt_oss() -> None:
    result = runner.invoke(app, ["serve-cmd", "--model", "gpt-oss-20b"])
    assert result.exit_code == 0, result.output
    assert "vllm serve openai/gpt-oss-20b" in result.output
    assert "--tool-call-parser openai" in result.output
    assert "--no-enable-prefix-caching" in result.output


def test_serve_cmd_qwen35_includes_tokenizer_override() -> None:
    # Qwen3.5-9B-FP8-dynamic's tokenizer_config.json sets an unrecognized
    # tokenizer_class -- see configs/models.yaml. --tokenizer points at the
    # base repo instead, which uses a standard tokenizer class.
    result = runner.invoke(app, ["serve-cmd", "--model", "qwen3.5-9b"])
    assert result.exit_code == 0, result.output
    assert "--tokenizer Qwen/Qwen3.5-9B" in result.output


def test_serve_cmd_unknown_model_fails() -> None:
    result = runner.invoke(app, ["serve-cmd", "--model", "does-not-exist"])
    assert result.exit_code != 0
