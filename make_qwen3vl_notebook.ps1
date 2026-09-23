$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$script = Get-Content -LiteralPath (Join-Path $root 'qwen3vl_5060ti.py') -Raw -Encoding utf8
$pData = $script.IndexOf('def normalized_question(')
$pPrompt = $script.IndexOf('def prompt(')
$pScore = $script.IndexOf('@torch.inference_mode()')
$pMain = $script.IndexOf('def main():')
if ($pData -lt 0 -or $pPrompt -lt 0 -or $pScore -lt 0 -or $pMain -lt 0) {
    throw 'Could not find expected sections in qwen3vl_5060ti.py.'
}

$cells = [System.Collections.Generic.List[object]]::new()
function Add-Cell([string]$kind, [string]$source) {
    $cell = [ordered]@{cell_type=$kind; metadata=[ordered]@{}; source=$source}
    if ($kind -eq 'code') { $cell.execution_count=$null; $cell.outputs=@() }
    $cells.Add($cell)
}

Add-Cell 'markdown' @'
# Qwen3-VL-8B VQA 로컬 실험 노트북 — RTX 5060 Ti 16GB

목표: public **0.95382 → 0.98**. 이 노트북은 원래 코드가 없어 새로 만든 독립 실험이다. `valid`로 고정 검증을 먼저 수행하고, 개선이 확인된 설정만 `submit`으로 재학습한다. 모든 추론은 로컬 GPU에서 실행하며 외부 추론 API를 호출하지 않는다.

**준비:** VS Code에서 이 파일을 열고 `C:\SSAFY\AIChallenge\baseline\Scripts\python.exe`를 Jupyter 커널로 선택한다. 현재 `dataset` 폴더를 그대로 사용한다. Qwen3-VL-8B 가중치는 `downloads/models/Qwen3-VL-8B-Instruct`에 받았다. 다운로드는 가중치 확보용이며 추론 API가 아니다.

**GPU:** 현재 다른 Python 작업이 약 14.6GB를 사용 중이다. 8B 모델 로딩 전에 12GB 이상 여유가 있어야 하며, 이 노트북은 기존 작업을 종료하지 않는다. 학습 중 OOM이 나면 `MAX_VISUAL_TOKENS=192`로 낮춰 smoke test를 반복한다. 정확도 실험에서는 검증 점수를 보고 해상도를 다시 높인다.

공식 참고: [Qwen3-VL](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl), [TRL VLM 학습](https://huggingface.co/docs/trl/sft_trainer).
'@
Add-Cell 'code' @'
import sys
from importlib.metadata import version
print("QWEN3VL_5060TI_LOCAL_V2: warmup_steps=0.05")
print("Python:", sys.executable)
for package in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "pandas", "Pillow"):
    print(f"{package}: {version(package)}")
'@
Add-Cell 'markdown' @'
## 1. 설정

처음에는 `SMOKE=16`으로 데이터와 loss masking을 확인한다. 전체 실험은 `SMOKE=0`으로 다시 실행한다. `MODE="valid"`에서 채택할 설정을 결정한 뒤, **새 커널**에서 `MODE="submit"`으로 전체 train을 재학습한다.
'@
Add-Cell 'code' $script.Substring(0, $pData).TrimEnd()
Add-Cell 'code' @'
ROOT = Path(r"C:\SSAFY\AIChallenge")
DATA_DIR = ROOT / "dataset"
MODE = "valid"                    # "valid" or "submit"
SMOKE = 16                        # 0 for a full run
OUT = ROOT / "output" / ("qwen3vl8b_smoke" if SMOKE else f"qwen3vl8b_{MODE}")
LOCAL_MODEL_DIR = ROOT / "downloads" / "models" / "Qwen3-VL-8B-Instruct"
MODEL_ID = str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "config.json").exists() else "Qwen/Qwen3-VL-8B-Instruct"
MAX_VISUAL_TOKENS = 256
EPOCHS = 2
LEARNING_RATE = 5e-5
SEED = 42

assert torch.cuda.is_available(), "CUDA GPU가 필요합니다."
assert torch.cuda.is_bf16_supported(), "BF16을 지원하는 GPU가 필요합니다."
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU 여유 메모리: {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB")
assert free_bytes >= 12 * 2**30, "기존 GPU 작업을 마친 뒤 12 GiB 이상 여유가 있을 때 실행하세요."
for required in ("train.csv", "dev.csv", "test.csv", "sample_submission.csv"):
    assert (DATA_DIR / required).exists(), f"Missing {DATA_DIR / required}"
OUT.mkdir(parents=True, exist_ok=True)
print(torch.cuda.get_device_name(0), "data:", DATA_DIR, "output:", OUT)
'@
Add-Cell 'markdown' @'
## 2. 분할과 dev 정제

train은 질문 문자열 기준으로 묶고, 100개의 고정 시드 후보 중 정답 분포가 가장 비슷한 약 10%를 검증에 사용한다. validation과 같은 질문, 매우 가까운 이미지는 training fold와 dev 후보에서 제외한다. dev는 3표 이상·낮은 응답 엔트로피만 사용한다.
'@
Add-Cell 'code' $script.Substring($pData, $pPrompt - $pData).TrimEnd()
Add-Cell 'code' @'
random.seed(SEED)
torch.manual_seed(SEED)
train_df = pd.read_csv(DATA_DIR / "train.csv").fillna("")
test_df = pd.read_csv(DATA_DIR / "test.csv").fillna("")
if MODE == "valid":
    fit_df, valid_df = make_split(train_df)
    valid_df[["id", "path", "question", "answer"]].to_csv(OUT / "valid_split.csv", index=False)
    fit_df = avoid_validation_leakage(fit_df, valid_df, DATA_DIR)
else:
    fit_df, valid_df = train_df, None

dev_df = dev_consensus(pd.read_csv(DATA_DIR / "dev.csv").fillna(""))
if valid_df is not None:
    dev_df = avoid_validation_leakage(dev_df, valid_df, DATA_DIR)
fit_df = pd.concat([fit_df, dev_df], ignore_index=True)
if SMOKE:
    fit_df = fit_df.sample(n=min(SMOKE, len(fit_df)), random_state=SEED).reset_index(drop=True)
    if valid_df is not None:
        valid_df = valid_df.head(10)
print("train rows:", len(fit_df), "dev added:", len(dev_df), "validation rows:", 0 if valid_df is None else len(valid_df))
assert fit_df.answer.isin(list(CHOICES)).all()
'@
Add-Cell 'markdown' @'
## 3. 모델과 학습

Qwen3-VL-8B 4bit LoRA를 사용한다. language model 모듈만 대상으로 LoRA를 적용한다. 손실은 assistant의 정답 글자 한 토큰에만 계산하며, 첫 batch에서 토큰 수를 검사한다. 16GB VRAM에 맞춰 batch size 1과 gradient checkpointing을 사용한다.
'@
Add-Cell 'code' $script.Substring($pPrompt, $pScore - $pPrompt).TrimEnd()
Add-Cell 'code' @'
model, processor, token_ids, targets = load_model(MAX_VISUAL_TOKENS)
config = SFTConfig(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=LEARNING_RATE,
    warmup_steps=0.05,
    lr_scheduler_type="linear",
    bf16=True,
    max_length=None,
    assistant_only_loss=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(
    model=model, args=config,
    train_dataset=training_dataset(fit_df, DATA_DIR),
    processing_class=processor,
    peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=targets, task_type="CAUSAL_LM"),
)
trainer.data_collator = make_answer_only_collator(trainer.data_collator, token_ids)
probe = trainer.data_collator([trainer.train_dataset[0]])
assert (probe["labels"] != -100).sum().item() == 1
print("Answer-token loss mask verified.")
'@
Add-Cell 'code' @'
trainer.train()
trainer.model.save_pretrained(OUT / "adapter")
processor.save_pretrained(OUT / "adapter")
print("Adapter saved:", OUT / "adapter")
'@
Add-Cell 'markdown' @'
## 4. 검증 점수 또는 제출 파일

문항마다 a/b/c/d log probability를 저장한다. 검증 점수를 먼저 확인한다. `SMOKE` 실행에서는 제출 파일을 만들지 않는다.
'@
Add-Cell 'code' $script.Substring($pScore, $pMain - $pScore).TrimEnd()
Add-Cell 'code' @'
if MODE == "valid":
    score(valid_df, trainer.model, processor, token_ids, DATA_DIR, OUT / "valid_scores.csv")
elif not SMOKE:
    score(test_df, trainer.model, processor, token_ids, DATA_DIR, OUT / "test_scores.csv")
    scores = pd.read_csv(OUT / "test_scores.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert len(sample) == len(test_df) and sample.id.equals(test_df.id)
    sample["answer"] = scores.pred
    assert sample.answer.isin(list(CHOICES)).all()
    sample.to_csv(OUT / "submission.csv", index=False)
    print("Submission saved:", OUT / "submission.csv")
'@
Add-Cell 'markdown' @'
## 5. 검증 오답과 다음 실험

검증에서는 실제 오답을 CSV로 남긴다. `RUN_HIGHRES_ABLATION=True`로 바꾸면 가장 불확실한 15%만 높은 해상도로 다시 읽고, **같은 검증 행에서 정답 수가 늘었는지** 확인한다. 개선이 없으면 제출에 적용하지 않는다.
'@
Add-Cell 'code' @'
if MODE == "valid" and not SMOKE:
    scored = pd.read_csv(OUT / "valid_scores.csv")
    errors = scored.loc[scored.gold != scored.pred].copy()
    errors.to_csv(OUT / "valid_errors.csv", index=False)
    val_accuracy = (scored.gold == scored.pred).mean()
    pd.DataFrame([{
        "model": MODEL_ID, "seed": SEED, "train_rows": len(fit_df),
        "dev_added": len(dev_df), "valid_rows": len(scored),
        "max_visual_tokens": MAX_VISUAL_TOKENS, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "val_accuracy": val_accuracy,
    }]).to_csv(OUT / "experiment_summary.csv", index=False)
    print(f"Validation: {(scored.gold == scored.pred).sum()}/{len(scored)} = {val_accuracy:.5f}")
    print("Errors:", len(errors), "saved:", OUT / "valid_errors.csv")
    display(errors[["id", "question", "gold", "pred"]].head(30))
'@
Add-Cell 'code' @'
RUN_HIGHRES_ABLATION = False
if MODE == "valid" and not SMOKE and RUN_HIGHRES_ABLATION:
    scored = pd.read_csv(OUT / "valid_scores.csv")
    logp = scored[[f"logp_{c}" for c in CHOICES]].to_numpy()
    margins = pd.Series([sorted(row)[-1] - sorted(row)[-2] for row in logp], index=scored.id)
    selected_ids = set(margins.nsmallest(max(1, round(len(scored) * 0.15))).index)
    selected = valid_df.loc[valid_df.id.isin(selected_ids)].copy()
    old_budget = processor.image_processor.size["longest_edge"]
    processor.image_processor.size["longest_edge"] = 512 * 32 * 32
    try:
        score(selected, trainer.model, processor, token_ids, DATA_DIR, OUT / "valid_highres_subset.csv")
    finally:
        processor.image_processor.size["longest_edge"] = old_budget
    replacement = pd.read_csv(OUT / "valid_highres_subset.csv").set_index("id")
    combined = scored.set_index("id")
    combined.update(replacement)
    combined.reset_index().to_csv(OUT / "valid_scores_highres_selective.csv", index=False)
    old_correct = (scored.gold == scored.pred).sum()
    new_correct = (combined.gold == combined.pred).sum()
    print(f"Base {old_correct}/{len(scored)} → selective high-res {new_correct}/{len(scored)}; delta {new_correct-old_correct}")
'@

$notebook = [ordered]@{
    cells = $cells.ToArray()
    metadata = [ordered]@{
        kernelspec = [ordered]@{display_name='Python 3'; language='python'; name='python3'}
        language_info = [ordered]@{name='python'}
        colab = [ordered]@{provenance=@()}
    }
    nbformat = 4
    nbformat_minor = 4
}
$destination = Join-Path $root 'qwen3vl_5060ti_local.ipynb'
$notebook | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $destination -Encoding utf8
$newDestination = Join-Path $root 'qwen3vl_5060ti_local_v2.ipynb'
Copy-Item -LiteralPath $destination -Destination $newDestination -Force
Write-Output "Created $destination and $newDestination with $($cells.Count) cells"
