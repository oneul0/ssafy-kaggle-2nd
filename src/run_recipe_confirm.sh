#!/bin/bash
# 로컬 레시피 확정: (1) LoRA rank 16 vs 64 소규모 비교, (2) 2단계 저 lr 학습. 순서대로 실행, 약 2.5시간.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
mkdir -p outputs/logs
LOG=outputs/logs/recipe_confirm.log
: > "$LOG"

echo "== [1/4] rank 16 (부분학습 1500장, Qwen3-VL-4B) ==" | tee -a "$LOG"
$B4 src/train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_rank16_test --limit 1500 \
    --batch 2 --accum 4 --lr 1e-4 --rank 16 --alpha 32 --epochs 1 --resume >> "$LOG" 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 \
    --adapter outputs/adapters/q3vl_rank16_test/final --tag q3vl_rank16_test_t768 >> "$LOG" 2>&1

echo "== [2/4] rank 64 (부분학습 1500장, Qwen3-VL-4B, alpha 128) ==" | tee -a "$LOG"
$B4 src/train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_rank64_test --limit 1500 \
    --batch 2 --accum 4 --lr 1e-4 --rank 64 --alpha 128 --epochs 1 --resume >> "$LOG" 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 \
    --adapter outputs/adapters/q3vl_rank64_test/final --tag q3vl_rank64_test_t768 >> "$LOG" 2>&1

echo "== [3/4] 2단계 학습: Qwen3.5-4B (기존 final 어댑터에서 이어서, lr 2e-5, 0.3epoch) ==" | tee -a "$LOG"
$B5 src/train_lora.py --model Qwen3.5-4B --tag q35_4b_lora2 \
    --init-adapter outputs/adapters/q35_4b_lora/final \
    --batch 1 --accum 8 --lr 2e-5 --rank 16 --alpha 32 --epochs 0.3 --resume >> "$LOG" 2>&1
$B5 src/score.py --split holdout --model Qwen3.5-4B --max-tokens 768 \
    --adapter outputs/adapters/q35_4b_lora2/final --tag q35_4b_lora2_t768 >> "$LOG" 2>&1

echo "== [4/4] 2단계 학습: Qwen3-VL-4B ==" | tee -a "$LOG"
$B4 src/train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_4b_lora2 \
    --init-adapter outputs/adapters/q3vl_4b_lora/final \
    --batch 2 --accum 4 --lr 2e-5 --rank 16 --alpha 32 --epochs 0.3 --resume >> "$LOG" 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 \
    --adapter outputs/adapters/q3vl_4b_lora2/final --tag q3vl_4b_lora2_t768 >> "$LOG" 2>&1

echo recipe_confirm_done > outputs/logs/recipe_confirm_done.txt
