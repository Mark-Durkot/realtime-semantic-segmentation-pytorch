from pathlib import Path
import shutil


SRC_ROOT = Path("/Users/md/Cityscapes/src/data")
GT_ROOT = Path("/Users/md/Cityscapes/gt/data")
DATASET_ROOT = Path("/Users/md/Cityscapes/dataset")

SPLITS = ("test", "train", "val")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_src_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_leftImg8bit"):
        return stem[: -len("_leftImg8bit")]
    return stem


def normalize_gt_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_color"):
        stem = stem[: -len("_color")]
    if stem.endswith("_gtFine"):
        stem = stem[: -len("_gtFine")]
    return stem


def gather_src_files(split_root: Path) -> dict[str, Path]:
    files_by_key: dict[str, Path] = {}
    if not split_root.exists():
        return files_by_key

    city_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for city_dir in city_dirs:
        for file_path in sorted(city_dir.iterdir(), key=lambda p: p.name):
            if not is_image_file(file_path):
                continue
            key = normalize_src_key(file_path)
            if key in files_by_key:
                raise ValueError(f"Duplicate src key '{key}' in split '{split_root.name}'.")
            files_by_key[key] = file_path
    return files_by_key


def gather_gt_files(split_root: Path) -> dict[str, Path]:
    files_by_key: dict[str, Path] = {}
    if not split_root.exists():
        return files_by_key

    city_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for city_dir in city_dirs:
        for file_path in sorted(city_dir.iterdir(), key=lambda p: p.name):
            if not is_image_file(file_path):
                continue
            if not file_path.stem.endswith("_color"):
                continue
            key = normalize_gt_key(file_path)
            if key in files_by_key:
                raise ValueError(f"Duplicate gt key '{key}' in split '{split_root.name}'.")
            files_by_key[key] = file_path
    return files_by_key


def prepare_split(split: str) -> None:
    split_src_root = SRC_ROOT / split
    split_gt_root = GT_ROOT / split

    out_src_dir = DATASET_ROOT / split / "src"
    out_gt_dir = DATASET_ROOT / split / "gt"
    out_src_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    src_files = gather_src_files(split_src_root)
    gt_files = gather_gt_files(split_gt_root)

    common_keys = sorted(set(src_files) & set(gt_files))
    only_src = sorted(set(src_files) - set(gt_files))
    only_gt = sorted(set(gt_files) - set(src_files))

    if only_src:
        print(f"[{split}] Warning: {len(only_src)} src files have no matching gt _color file.")
    if only_gt:
        print(f"[{split}] Warning: {len(only_gt)} gt _color files have no matching src file.")

    for index, key in enumerate(common_keys, start=1):
        src_path = src_files[key]
        gt_path = gt_files[key]

        src_target = out_src_dir / f"src_{index}{src_path.suffix.lower()}"
        gt_target = out_gt_dir / f"gt_{index}{gt_path.suffix.lower()}"

        shutil.copy2(src_path, src_target)
        shutil.copy2(gt_path, gt_target)

    print(f"[{split}] Copied {len(common_keys)} paired images.")


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        prepare_split(split)
    print(f"Done. Prepared dataset in: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
