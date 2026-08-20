from pathlib import Path

from darpan.experiment.recorder import ResultRecorder


def test_result_recorder_write_text_creates_nested_parent(tmp_path: Path) -> None:
    recorder = ResultRecorder(tmp_path / "artifact")
    path = recorder.write_text("rendered/service.txt", "hello\n")
    assert path.read_text(encoding="utf-8") == "hello\n"
