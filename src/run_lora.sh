#!/bin/bash
# Phase 2: 전체 train(홀드아웃 제외) LoRA 학습 -> 홀드아웃 평가. 두 후보를 차례로.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
mkdir -p outputs/logs

# 1) Qwen3.5-4B (baseline5). 배치1 x 누적8 = 유효배치 8
$B5 src/train_lora.py --model Qwen3.5-4B --tag q35_4b_lora --batch 1 --accum 8 --lr 1e-4 --rank 16 --alpha 32 --epochs 1 --resume > outputs/logs/train_q35_4b_lora.log 2>&1
$B5 src/score.py --split holdout --model Qwen3.5-4B --max-tokens 768 --adapter outputs/adapters/q35_4b_lora/final --tag q35_4b_lora_t768 > outputs/logs/eval_q35_4b_lora.log 2>&1
echo q35_done > outputs/logs/lora_q35_done.txt

# 2) Qwen3-VL-4B (baseline). 배치2 x 누적4 = 유효배치 8
$B4 src/train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_4b_lora --batch 2 --accum 4 --lr 1e-4 --rank 16 --alpha 32 --epochs 1 --resume > outputs/logs/train_q3vl_4b_lora.log 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 --adapter outputs/adapters/q3vl_4b_lora/final --tag q3vl_4b_lora_t768 > outputs/logs/eval_q3vl_4b_lora.log 2>&1
echo all_done > outputs/logs/lora_all_done.txt
