"""Download the two public Qwen 4B model weights used by the reference recipe."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
}
ROOT = Path("downloads/models")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name, repo in MODELS.items():
        target = ROOT / name
        revision = api.model_info(repo).sha
        print(f"Downloading {repo}@{revision} to {target}", flush=True)
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=target,
            max_workers=4,
            ignore_patterns=["*.md", "*.png", "*.jpg", "*.pdf", "*.mp4"],
        )
        (target / "source_revision.txt").write_text(f"{repo}\n{revision}\n", encoding="utf-8")
        print(f"Ready: {target}", flush=True)


if __name__ == "__main__":
    main()
