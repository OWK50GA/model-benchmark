import json
import subprocess
import sys
from pathlib import Path

import pytest

from fashionista_vision_lab.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_validate_example_without_images(capsys):
    assert main(["dataset", "validate", str(ROOT / "datasets/annotations/example.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and not result["images_checked"]
    assert result["summary"]["synthetic"]


def test_image_checks_fail_for_example_files(capsys, tmp_path):
    assert main(["dataset", "validate", str(ROOT / "datasets/annotations/example.json"), "--check-images", "--image-root", str(tmp_path)]) == 1
    assert len(json.loads(capsys.readouterr().out)["errors"]) == 3


@pytest.mark.parametrize("command", [["neighbors"], ["run", "--model", "dinov3", "--image", "missing.jpg", "--task", "embedding"]])
def test_future_commands_never_claim_success(command, capsys):
    assert main(command) == 2
    captured = capsys.readouterr()
    assert not captured.out and "Phase" in captured.err


def test_capability_mismatch(capsys):
    assert main(["run", "--model", "dinov3", "--image", "missing.jpg", "--task", "detection"]) == 1
    assert "no planned detection" in capsys.readouterr().err


def test_config_does_not_echo_invalid_secret(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"models": {"gemini": {"api_key": "DO-NOT-ECHO"}}}')
    assert main(["config", "show", str(path)]) == 1
    captured = capsys.readouterr()
    assert "DO-NOT-ECHO" not in captured.err + captured.out


def test_missing_file_is_actionable_without_traceback(tmp_path, capsys):
    assert main(["dataset", "validate", str(tmp_path / "missing.json")]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["dataset", "config", "result"])
def test_schema_export(kind, capsys):
    assert main(["schema", kind]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "object"


def test_fresh_process_imports_no_ml_runtimes():
    code = "from fashionista_vision_lab.cli import main; import sys; main(['models','list']); assert not ({'torch','transformers','google.genai','numpy','sklearn'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert {model["name"] for model in json.loads(result.stdout) if model["implemented"]} == {"gemini", "florence"}
