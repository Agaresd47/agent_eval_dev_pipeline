from pathlib import Path
import shutil


def collect_complete_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_dir.glob("*.nii.gz"):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def promote_finished_pairs(image_dir: Path, mask_dir: Path, finished_image_dir: Path, finished_mask_dir: Path) -> None:
    pairs = collect_complete_pairs(image_dir, mask_dir)
    for image_path, mask_path in pairs:
        shutil.move(str(image_path), str(finished_image_dir / image_path.name))
        shutil.move(str(mask_path), str(finished_mask_dir / mask_path.name))
