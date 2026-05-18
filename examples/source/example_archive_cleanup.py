import os
import tarfile
from pathlib import Path


def compress_folder(source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def split_file(archive_path: Path, output_dir: Path, chunk_size: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    with archive_path.open("rb") as src:
        index = 0
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            chunk_path = output_dir / f"{archive_path.name}.part_{index:03d}"
            chunk_path.write_bytes(data)
            chunks.append(chunk_path)
            index += 1
    return chunks


def cleanup_original_archive(archive_path: Path, chunks: list[Path]) -> None:
    if chunks:
        os.remove(archive_path)
