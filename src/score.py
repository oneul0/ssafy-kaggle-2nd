"""[2단계] 채점기 / 추론기.

모델이 a/b/c/d 중 무엇을 고르는지 '글로 생성'하지 않고, 4개 글자의 점수(로짓)를 비교한다.
  - 파싱 실패가 없다, 결과가 매번 같다(재현성), 확률을 저장할 수 있다(앙상블/TTA용)

사용 예
  # 홀드아웃(모의고사)으로 정확도 측정
  baseline\\Scripts\\python.exe src\\score.py --split holdout --max-tokens 768 --tag q25vl3b_t768
  # test 추론 + 제출 파일 생성
  baseline\\Scripts\\python.exe src\\score.py --split test --max-tokens 768 --tag q25vl3b_t768
  # 빠른 동작 확인 (20장만)
  baseline\\Scripts\\python.exe src\\score.py --split holdout --limit 20 --tag smoke

산출물
  outputs/probs/{tag}_{split}.npz      : ids, probs(N x 4)  <- 앙상블/분석의 원재료
  outputs/submissions/{tag}.csv        : (test 일 때) 제출 파일
  outputs/experiments.csv              : 실험 기록표에 한 줄 추가
중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행한다.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor

from common import (DATA, LETTERS, MODELS, OUT, SYSTEM, build_prompt, load_csv,
                    question_type, wilson_ci)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "test"], required=True)
    ap.add_argument("--model", default="Qwen2.5-VL-3B-Instruct",
                    help="downloads/models/ 아래 폴더명, 또는 절대 경로")
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="이미지 크기 예산. '토큰 x 28x28 픽셀'로 환산하므로 모델과 무관하게 같은 픽셀 크기가 된다 "
                         "(원본 720x960 = 약 880). Qwen3 계열은 조각이 32px 라 실제 visual 토큰 수는 더 적다")
    ap.add_argument("--min-tokens", type=int, default=256, help="이보다 작은 이미지는 이 크기까지 확대")
    ap.add_argument("--adapter", default="", help="LoRA 어댑터 폴더(예: outputs/adapters/<tag>/final). 합치지 않고 그대로 붙여서 추론")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none",
                    help="16GB 에 안 들어가는 큰 모델(7~8B)용 압축 로드")
    ap.add_argument("--tag", required=True, help="실험 이름. 산출물 파일명에 사용")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (동작 확인용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rot", type=int, choices=[0, 1, 2, 3], default=0,
                    help="TTA: 보기 순환 이동 칸 수. 화면에는 보기가 [k, k+1, ..] 순서로 보이고, "
                         "저장되는 확률은 원래 a~d 순서로 되돌린 것. 0 이면 기존과 동일. "
                         "1 이상이면 태그 뒤에 _rot{k} 가 자동으로 붙는다")
    return ap.parse_args()


def load_model(model_arg, quant="none"):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    path = Path(model_arg) if Path(model_arg).exists() else MODELS / model_arg
    kw = {}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map="cuda", local_files_only=True, **kw
    ).eval()
    return path, model


def main():
    args = parse_args()
    if args.rot:
        args.tag = f"{args.tag}_rot{args.rot}"
    # 화면 위치 j 에 원래 보기 (j+rot)%4 를 보여준다. 원래 보기 i 의 확률 = 화면 위치 (i-rot)%4 의 확률
    back = [(i - args.rot) % 4 for i in range(4)]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.split == "holdout":
        df = pd.read_csv(OUT / "split" / "holdout.csv")
    else:
        df = load_csv("test")
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    n = len(df)

    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(parents=True, exist_ok=True)
    npz_path = OUT / "probs" / f"{args.tag}_{args.split}.npz"
    part_path = OUT / "probs" / f"{args.tag}_{args.split}.partial.npy"

    model_path, model = load_model(args.model, args.quant)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"LoRA 어댑터 적용: {args.adapter}", flush=True)
    proc = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.min_tokens * 28 * 28,
        max_pixels=args.max_tokens * 28 * 28,
        local_files_only=True,
    )
    tok = proc.tokenizer
    ids = []
    for ch in LETTERS:
        e = tok.encode(ch, add_special_tokens=False)
        assert len(e) == 1, f"'{ch}' 가 토큰 1개가 아님: {e}"
        ids.append(e[0])
    assert len(set(ids)) == 4

    # 이어하기
    probs = np.full((n, 4), np.nan, dtype=np.float32)
    if part_path.exists():
        old = np.load(part_path)
        if old.shape == probs.shape:
            probs = old
    start = int(np.isnan(probs[:, 0]).argmax()) if np.isnan(probs[:, 0]).any() else n
    if start:
        print(f"이어하기: {start}/{n} 부터")

    merge = getattr(proc.image_processor, "merge_size", 2)
    vis_tokens = []
    t0 = time.time()
    for i in range(start, n):
        r = df.iloc[i]
        img = Image.open(DATA / r["path"]).convert("RGB")
        opts = [r[LETTERS[(j + args.rot) % 4]] for j in range(4)]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": build_prompt(r["question"], *opts)},
            ]},
        ]
        # enable_thinking=False: Qwen3.5 계열은 기본값이면 답 자리에 '<think>' 가 붙어 a/b/c/d 점수를
        # 읽는 위치가 틀어진다. 이 옵션을 쓰지 않는 템플릿(Qwen2.5-VL 등)에서는 무시된다.
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        vis_tokens.append(int(inp["image_grid_thw"][0].prod()) // merge ** 2)
        with torch.inference_mode():
            logits = model(**inp).logits[0, -1, ids].float()
        probs[i] = torch.softmax(logits, dim=-1).cpu().numpy()[back]

        if (i + 1) % 100 == 0 or i + 1 == n:
            np.save(part_path, probs)
            el = time.time() - t0
            done = i + 1 - start
            eta = el / done * (n - i - 1)
            print(f"[{i+1}/{n}] {el/60:.1f}분 경과, 남은 예상 {eta/60:.1f}분", flush=True)
    elapsed = time.time() - t0

    assert not np.isnan(probs).any()
    np.savez(npz_path, ids=df["id"].to_numpy(), probs=probs)
    if part_path.exists():
        part_path.unlink()
    pred = np.array(list(LETTERS))[probs.argmax(1)]
    dist = {k: round(float((pred == k).mean()) * 100, 1) for k in LETTERS}
    print(f"\n예측 분포(%): {dist}")

    acc, lo, hi = float("nan"), float("nan"), float("nan")
    if args.split == "holdout":
        gold = df["answer"].to_numpy()
        ok = pred == gold
        k = int(ok.sum())
        acc = k / n
        lo, hi = wilson_ci(k, n)
        print(f"\n=== 홀드아웃 정확도: {acc*100:.2f}%  ({k}/{n}, 95% 신뢰구간 {lo*100:.1f}~{hi*100:.1f}%) ===")
        qt = df["question"].map(question_type)
        rows = pd.DataFrame({"유형": qt, "정답": ok}).groupby("유형")["정답"].agg(["mean", "sum", "count"])
        rows["mean"] = (rows["mean"] * 100).round(1)
        rows.columns = ["정확도(%)", "맞힌 수", "문제 수"]
        print(rows.to_string())
        pd.DataFrame({"id": df["id"], "gold": gold, "pred": pred, "correct": ok,
                      **{f"p_{c}": probs[:, j] for j, c in enumerate(LETTERS)}}
                     ).to_csv(OUT / "probs" / f"{args.tag}_{args.split}_detail.csv", index=False)
    else:
        sample = load_csv("sample_submission")
        sub = pd.DataFrame({"id": df["id"], "answer": pred})
        if not args.limit:
            assert len(sub) == len(sample), f"행 수 불일치 {len(sub)} != {len(sample)}"
            assert (sub["id"].to_numpy() == sample["id"].to_numpy()).all(), "id 순서가 sample_submission 과 다름"
            assert sub["answer"].isin(list(LETTERS)).all()
        out_csv = OUT / "submissions" / f"{args.tag}.csv"
        sub.to_csv(out_csv, index=False)
        print(f"제출 파일 저장: {out_csv}  ({len(sub)}행, 형식 검사 통과)")

    exp = OUT / "experiments.csv"
    new = not exp.exists()
    with open(exp, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "split", "model", "max_tokens", "n", "acc", "ci_low", "ci_high",
                        "sec_per_item", "vram_gb", "pred_dist", "quant", "vis_tokens_mean", "adapter"])
        w.writerow([args.tag, args.split, model_path.name, args.max_tokens, n,
                    f"{acc:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{elapsed/max(1,n-start):.3f}",
                    f"{torch.cuda.max_memory_allocated()/1e9:.1f}", dist,
                    args.quant, f"{np.mean(vis_tokens):.0f}" if vis_tokens else "", args.adapter])
    print(f"소요 {elapsed/60:.1f}분, 장당 {elapsed/max(1,n-start):.2f}초, GPU 메모리 최대 {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
