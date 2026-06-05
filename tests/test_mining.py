from __future__ import annotations

from pathlib import Path

import task_forge_v2.mining as mining


def test_iter_dirty_python_files_returns_sorted_python_files(tmp_path, monkeypatch):
    dirty_root = tmp_path / "dirty_work"
    (dirty_root / "b").mkdir(parents=True)
    (dirty_root / "a").mkdir(parents=True)
    (dirty_root / "a" / "one.py").write_text("print('a')", encoding="utf-8")
    (dirty_root / "b" / "two.py").write_text("print('b')", encoding="utf-8")
    (dirty_root / "b" / "ignore.txt").write_text("skip", encoding="utf-8")

    monkeypatch.setenv("TASK_FORGE_SOURCE_ROOT", str(dirty_root))

    assert list(mining.iter_dirty_python_files()) == [
        (dirty_root / "a" / "one.py").resolve(),
        (dirty_root / "b" / "two.py").resolve(),
    ]


def test_mine_source_file_extracts_deterministic_metadata(tmp_path, monkeypatch):
    repo_root = tmp_path
    source = repo_root / "code_source" / "dirty_work" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
import os
from pathlib import Path


def build(path):
    shutil.move("from", "to")
    return Path(path)


URL = "C:\\\\temp\\\\data.json"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(mining, "get_repo_root", lambda: repo_root)

    summary = mining.mine_source_file(source)

    assert summary.artifact_id == "source::sample"
    expected_relative = str(source.relative_to(repo_root))

    assert summary.source_ids == [expected_relative]
    assert summary.source_path == expected_relative
    assert summary.title == "sample.py"
    assert summary.summary.startswith("Mine risky workflow behavior from sample.py")
    assert summary.key_facts == [
        "imports: os, pathlib",
        "functions: build",
    ]
    assert summary.risks == ["mutation_move"]
    assert summary.recoverable_facts
    assert summary.user_only_policies == ["Any non-recoverable operational policy should remain explicitly user-provided."]
    assert summary.linked_paths == ["C:\\\\temp\\\\data.json"]
