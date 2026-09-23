"""LoRA 학습이 16GB GPU 에 들어가는지, 얼마나 빠른지 재는 시험 (정확도는 재지 않는다).

  baseline5\\Scripts\\python.exe src\\mem_test.py --model Qwen3.5-4B --batches 1,2,4
  baseline5\\Scripts\\python.exe src\\mem_test.py --model Qwen3.5-4B --quant 4bit --batches 1,2,4

- 정답 토큰 1개에만 loss (나머지 -100), 그래디언트 체크포인팅, LoRA r16, 언어 모델 부분만
- 배치 크기별로 몇 스텝 돌려 GPU 메모리 최대치와 스텝당 시간을 출력. OOM 이면 OOM 으로 표시.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common import DATA, MODELS, OUT, SYSTEM, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--quant", choices=["none", "4bit"], default="none")
ap.add_argument("--batches", default="1,2,4")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--rank", type=int, default=16)
args = ap.parse_args()

path = MODELS / args.model
kw = {}
if args.quant == "4bit":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForImageTextToText.from_pretrained(
    str(path), dtype=torch.bfloat16, device_map="cuda", local_files_only=True, **kw)
proc = AutoProcessor.from_pretrained(str(path), min_pixels=256 * 784, max_pixels=args.max_tokens * 784,
                                     local_files_only=True)
proc.tokenizer.padding_side = "right"
base_gb = torch.cuda.memory_allocated() / 1e9

if args.quant == "4bit":
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

# 언어 모델 부분의 선형층만 (비전 타워 제외)
cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg)
model.train()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
print(f"모델 {args.model} quant={args.quant}  가중치 적재 후 {base_gb:.1f}GB  LoRA 학습 파라미터 {n_train/1e6:.1f}M", flush=True)

df = pd.read_csv(OUT / "split" / "train_fit.csv").iloc[:64]


def make_batch(rows):
    texts, imgs, golds = [], [], []
    for _, r in rows.iterrows():
        img = Image.open(DATA / r["path"]).convert("RGB")
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": build_prompt(r["question"], r["a"], r["b"], r["c"], r["d"])}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        texts.append(t + r["answer"] + "<|im_end|>")
        imgs.append(img)
    enc = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], -100)
    for i in range(len(rows)):
        L = int(enc["attention_mask"][i].sum())
        labels[i, L - 2] = enc["input_ids"][i, L - 2]  # 정답 글자 위치만 채점
    enc["labels"] = labels
    return {k: v.to("cuda") for k, v in enc.items()}


for B in [int(x) for x in args.batches.split(",")]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    try:
        for s in range(args.steps):
            batch = make_batch(df.iloc[(s * B) % 60:(s * B) % 60 + B])
            torch.cuda.synchronize()
            t0 = time.time()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if s > 0:  # 첫 스텝은 예열이라 제외
                times.append(time.time() - t0)
            losses.append(float(loss))
        per = sum(times) / len(times)
        print(f"배치 {B}: 최대 메모리 {torch.cuda.max_memory_allocated()/1e9:.1f}GB  스텝당 {per:.2f}초  "
              f"문제당 {per/B:.2f}초  → 5,714장 1바퀴 약 {per/B*5714/60:.0f}분  (loss {losses[0]:.2f}→{losses[-1]:.2f})", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"배치 {B}: OOM (16GB 초과)", flush=True)
        break
