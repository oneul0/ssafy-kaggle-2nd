#!/bin/bash
# 학습된 LoRA 모델로 test 6,714장 추론 -> 확률 파일 + 제출 CSV 생성 (모델당 약 45~50분)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
mkdir -p outputs/logs
$B5 src/score.py --split test --model Qwen3.5-4B --max-tokens 768 --adapter outputs/adapters/q35_4b_lora/final --tag q35_4b_lora_t768 > outputs/logs/test_q35_4b_lora.log 2>&1
echo q35_test_done > outputs/logs/test_q35_done.txt
$B4 src/score.py --split test --model Qwen3-VL-4B-Instruct --max-tokens 768 --adapter outputs/adapters/q3vl_4b_lora/final --tag q3vl_4b_lora_t768 > outputs/logs/test_q3vl_4b_lora.log 2>&1
echo all_test_done > outputs/logs/test_all_done.txt
