"""用于模型资源和生成产物的仓库本地路径。"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = REPOSITORY_ROOT / "pretrained"


def pretrained_path(backend: str, filename: str) -> Path:
    """返回 ``pretrained/<backend>`` 下的官方预训练权重路径。"""
    return PRETRAINED_ROOT / backend / filename


def require_pretrained(path: Path, *, hint: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official pretrained checkpoint is missing: {path}. {hint}"
        )
    return str(path)


def ensure_pretrained(path: Path, *, url: str | None, hint: str) -> str:
    """当官方预训练权重缺失时，直接下载到本地仓库中。"""
    if path.is_file():
        return str(path)
    if not url:
        return require_pretrained(path, hint=hint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")

    import sys

    def _progress_hook(count: int, block_size: int, total_size: int):
        downloaded = count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading {path.name}: {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
            sys.stdout.flush()
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloading {path.name}: {mb_down:.1f} MB")
            sys.stdout.flush()

    try:
        print(f"Pretrained checkpoint not found at {path}. Automatically downloading from official source ({url})...")
        urlretrieve(url, temporary, reporthook=_progress_hook)
        print()
        temporary.replace(path)
        print(f"Successfully downloaded to {path}")
    except Exception as error:
        print()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download official pretrained model from {url} into {path}. {hint}"
        ) from error
    return str(path)
