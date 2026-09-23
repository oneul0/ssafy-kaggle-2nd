"""모델 사전 다운로드 (대회 당일 오프라인 대비).

저장 위치: 가이드 및 코드/downloads/models/<모델명>/   (기존 Qwen2.5-VL-3B 와 같은 구조)
중간에 끊겨도 같은 명령을 다시 실행하면 받은 부분은 건너뛰고 이어서 받는다.

  baseline\\Scripts\\python.exe src\\download_models.py
  baseline\\Scripts\\python.exe src\\download_models.py Qwen/Qwen3.5-4B      # 특정 모델만
"""
import os
import sys
import time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)  # 다운로드는 온라인이어야 함
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download

MODELS = Path(__file__).resolve().parent.parent / "가이드 및 코드" / "downloads" / "models"

DEFAULT = [
    "Qwen/Qwen3.5-4B",            # 약 9.3GB, transformers 5.x 필요
    "Qwen/Qwen3-VL-4B-Instruct",  # 약 8.9GB
    "Qwen/Qwen3-VL-8B-Instruct",  # 약 17.5GB
    "Qwen/Qwen2.5-VL-7B-Instruct",  # 약 16.6GB
]

for repo in (sys.argv[1:] or DEFAULT):
    dst = MODELS / repo.split("/")[-1]
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[시작] {repo} -> {dst}", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(dst),
                      ignore_patterns=["*.md", ".gitattributes"], max_workers=4)
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e9
    print(f"[완료] {repo}  {size:.1f}GB  {(time.time()-t0)/60:.1f}분", flush=True)
print("전체 완료")
