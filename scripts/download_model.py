"""Download the exact MPNet inference snapshot without hosted inference APIs."""

from __future__ import annotations

import argparse
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

MODEL_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "1_Pooling/config.json",
)


class DownloadProgress:
    """File-like adapter that updates a tqdm byte counter."""

    def __init__(self, raw: object, progress: tqdm[None]) -> None:
        self.raw = raw
        self.progress = progress

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)  # type: ignore[attr-defined]
        self.progress.update(len(data))
        return data


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """Stream one file atomically with retry and partial-file cleanup."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "redrob-ai-ranker/1.0"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                total = int(response.headers.get("Content-Length", "0")) or None
                if destination.is_file() and total == destination.stat().st_size:
                    return
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                    dynamic_ncols=True,
                ) as progress, temporary.open("wb") as output:
                    shutil.copyfileobj(
                        DownloadProgress(response, progress), output, length=1024 * 1024
                    )
            os.replace(temporary, destination)
            return
        except (OSError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Unable to download {url}: {exc}") from exc
            time.sleep(2**attempt)


def main() -> int:
    """Download all files needed by SentenceTransformer local inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="https://hf-mirror.com",
        help="Hugging Face-compatible file endpoint.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "all-mpnet-base-v2",
    )
    arguments = parser.parse_args()
    base = (
        f"{arguments.endpoint.rstrip('/')}/sentence-transformers/"
        "all-mpnet-base-v2/resolve/main"
    )
    for relative_path in MODEL_FILES:
        download_file(
            f"{base}/{relative_path}", arguments.output / relative_path
        )
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
