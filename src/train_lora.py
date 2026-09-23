"""LoRA 파인튜닝 (압축 없는 bf16). train_fit.csv(홀드아웃 제외 5,714장)로 학습한다.

  baseline5\\Scripts\\python.exe src\\train_lora.py --model Qwen3.5-4B --tag q35_4b_lora
  baseline\\Scripts\\python.exe  src\\train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_4b_lora

핵심 설계 (STRATEFY.MD §4.4, 선행 사례 참고)
  - 정답 글자 1개에만 loss (프롬프트·사진 부분은 -100)
  - 학습할 때마다 보기 순서를 무작위로 섞고 정답 글자도 같이 옮김 (위치 버릇 제거)
  - 압축 없이 bf16, LoRA는 언어 모델 부분만 (비전 타워는 동결)
  - 홀드아웃은 절대 사용하지 않음 (mid-eval 도 홀드아웃 앞 N장이므로 '학습 중 확인용'일 뿐 선택 기준으로 쓰지 말 것)
  - 중간 저장 + 이어하기(--resume): 끊겨도 같은 명령에 --resume 을 붙여 재실행

산출물: outputs/adapters/{tag}/ckpt (중간), outputs/adapters/{tag}/final (최종), outputs/logs/train_{tag}.csv
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# GPU 메모리 예약이 물리 메모리(16GB)를 넘으면 Windows 가 시스템 메모리로 밀어내 수십 배 느려진다.
# 실제 사용량은 12GB 안팎이므로 예약 상한을 두고, 조각 나는 것을 줄여 캐시를 재활용하게 한다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:512")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                   set_peft_model_state_dict)
from safetensors.torch import load_file
from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                           get_cosine_schedule_with_warmup)

from common import DATA, LETTERS, MODELS, OUT, SYSTEM, build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=1, help="한 번에 올리는 문제 수 (16GB: Qwen3.5=1~2, Qwen3-VL=1~4)")
ap.add_argument("--accum", type=int, default=8, help="모아서 업데이트할 횟수 (유효 배치 = batch x accum)")
ap.add_argument("--max-tokens", type=int, default=768, help="사진 크기 예산 (score.py 와 동일 의미)")
ap.add_argument("--min-tokens", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0, help="학습 데이터 앞 N장만 (동작 확인용)")
ap.add_argument("--save-every", type=int, default=50, help="N 스텝마다 중간 저장")
ap.add_argument("--eval-every", type=int, default=100, help="N 스텝마다 홀드아웃 앞 eval-n 장으로 확인 (0=끔)")
ap.add_argument("--eval-n", type=int, default=200)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--no-shuffle-options", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                 help="QLoRA: 압축 로드 후 그 위에 LoRA를 학습(base는 얼려져 있으므로 압축해도 LoRA 자체 학습엔 지장 없음). "
                      "16GB 로컬의 4B급은 미압축이 들어가므로 불필요(압축은 정확도를 깎음, WORKLOG_0922.MD 참고). "
                      "Colab 30B급처럼 미압축이 GPU에 안 들어갈 때만 사용")
ap.add_argument("--init-adapter", default="",
                 help="2단계 학습용: 이미 학습된 어댑터(예: outputs/adapters/q35_4b_lora/final)를 초기값으로 불러와서 "
                      "이어서 학습(보통 낮은 lr, 적은 epoch). --resume(같은 tag 중단 재개)과는 다름 — 새 tag로 새 스텝 카운트 0부터 시작")
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.92")))

adir = OUT / "adapters" / args.tag
adir.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)
log_path = OUT / "logs" / f"train_{args.tag}.csv"

# ---------- 데이터 ----------
df = pd.read_csv(OUT / "split" / "train_fit.csv")
if args.limit:
    df = df.iloc[: args.limit].reset_index(drop=True)
hold = pd.read_csv(OUT / "split" / "holdout.csv").iloc[: args.eval_n].reset_index(drop=True)
N = len(df)
total_samples = int(round(args.epochs * N))
eff = args.batch * args.accum
total_steps = math.ceil(total_samples / eff)
order = np.concatenate([np.random.default_rng(args.seed + e).permutation(N) for e in range(math.ceil(args.epochs))])[:total_samples]

# ---------- 모델 ----------
path = MODELS / args.model
quant_kw = {}
if args.quant == "4bit":
    quant_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
elif args.quant == "8bit":
    quant_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(path), dtype=torch.bfloat16, device_map="cuda", local_files_only=True, **quant_kw)
proc = AutoProcessor.from_pretrained(str(path), min_pixels=args.min_tokens * 784, max_pixels=args.max_tokens * 784, local_files_only=True)
proc.tokenizer.padding_side = "right"
ids = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
assert len(set(ids)) == 4

if args.quant != "none":
    # prepare_model_for_kbit_training 이 gradient checkpointing 활성화 + enable_input_require_grads +
    # 레이어놈 fp32 캐스팅을 한번에 처리한다 (양자화 모델은 이 순서를 지키지 않으면 grad 가 안 흐를 수 있음)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
# 언어 모델의 선형층: 일반 어텐션(q/k/v/o), MLP, 그리고 Qwen3.5 의 선형 어텐션(in_proj_qkv/in_proj_z/out_proj). 비전 타워 제외.
cfg = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)")
model = get_peft_model(model, cfg)
if args.init_adapter:
    set_peft_model_state_dict(model, load_file(str(Path(args.init_adapter) / "adapter_model.safetensors")))
    print(f"2단계 학습: {args.init_adapter} 에서 어댑터 가중치 로드", flush=True)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
print(f"모델 {args.model} | 학습 파라미터 {n_train/1e6:.1f}M | 학습 {N}장 x {args.epochs}epoch = {total_samples}샘플 | "
      f"유효배치 {eff} | 총 {total_steps}스텝 | lr {args.lr} r{args.rank}/a{args.alpha}", flush=True)

step0 = 0
ckpt = adir / "ckpt"
if args.resume and (ckpt / "state.pt").exists():
    set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
    st = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    step0 = st["step"]
    print(f"이어하기: 스텝 {step0}/{total_steps} 부터", flush=True)


# ---------- 배치 만들기 ----------
def load_img(rel):
    return Image.open(DATA / rel).convert("RGB")


def shuffled(row, sample_no):
    """보기 순서를 무작위로 섞고 정답 글자를 새 위치로 옮긴다."""
    opts = [str(row[c]) for c in LETTERS]
    gi = LETTERS.index(str(row["answer"]).strip().lower())
    if args.no_shuffle_options:
        return opts, LETTERS[gi]
    perm = np.random.default_rng([args.seed, sample_no]).permutation(4)
    return [opts[j] for j in perm], LETTERS[int(np.where(perm == gi)[0][0])]


def make_train_batch(idxs, sample_nos):
    texts, imgs, golds = [], [], []
    for i, sn in zip(idxs, sample_nos):
        r = df.iloc[i]
        opts, gold = shuffled(r, sn)
        img = load_img(r["path"])
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": build_prompt(r["question"], *opts)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        texts.append(t + gold + "<|im_end|>")
        imgs.append(img)
    enc = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], -100)
    for b in range(len(idxs)):
        L = int(enc["attention_mask"][b].sum())
        labels[b, L - 2] = enc["input_ids"][b, L - 2]  # 정답 글자 위치만 채점
    enc["labels"] = labels
    return {k: v.to("cuda") for k, v in enc.items()}


@torch.no_grad()
def quick_eval():
    model.eval()
    ok = 0
    for _, r in hold.iterrows():
        img = load_img(r["path"])
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": build_prompt(r["question"], r["a"], r["b"], r["c"], r["d"])}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=[t], images=[img], return_tensors="pt").to("cuda")
        lg = model(**inp).logits[0, -1, ids].float()
        ok += LETTERS[int(lg.argmax())] == r["answer"]
    model.train()
    return ok / len(hold)


def save(done_steps, final=False):
    d = adir / ("final" if final else "ckpt")
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))
    if not final:
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": done_steps}, d / "state.pt")


# ---------- 학습 ----------
new_log = not log_path.exists() or not args.resume
with open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8-sig") as lf:
    w = csv.writer(lf)
    if new_log:
        w.writerow(["step", "samples", "loss", "lr", "elapsed_min", "quick_eval_acc"])
    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for step in range(step0, total_steps):
        for a in range(args.accum):
            s = (step * args.accum + a) * args.batch
            sample_nos = list(range(s, min(s + args.batch, total_samples)))
            if not sample_nos:
                break
            batch = make_train_batch([order[k] for k in sample_nos], sample_nos)
            loss = model(**batch).loss
            (loss / args.accum).backward()
            run_loss += float(loss.detach())
            run_n += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1

        qe = ""
        if args.eval_every and done % args.eval_every == 0:
            qe = f"{quick_eval():.4f}"
        if done % args.log_every == 0 or done == total_steps or qe:
            el = (time.time() - t0) / 60
            eta = el / (done - step0) * (total_steps - done)
            avg = run_loss / max(1, run_n)
            w.writerow([done, done * eff, f"{avg:.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{el:.1f}", qe])
            lf.flush()
            print(f"[{done}/{total_steps}] loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{el:.1f}분 경과 / 남은 예상 {eta:.0f}분 GPU 사용최대 {torch.cuda.max_memory_allocated()/1e9:.1f}GB 예약최대 {torch.cuda.max_memory_reserved()/1e9:.1f}GB"
                  + (f"  ★홀드아웃 앞 {len(hold)}장 정확도 {qe}" if qe else ""), flush=True)
            run_loss, run_n = 0.0, 0
        if done % args.save_every == 0 and done < total_steps:
            save(done)

save(total_steps, final=True)
print(f"완료. 어댑터 저장: {adir/'final'}  (총 {(time.time()-t0)/60:.1f}분)", flush=True)
