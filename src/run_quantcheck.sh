#!/bin/bash
# 같은 모델을 압축 없이(bf16) vs 4bit 로 돌려 '압축이 정확도에 주는 영향'을 잰다 (홀드아웃 1,000장)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
$B5 src/score.py --split holdout --model Qwen3.5-4B --max-tokens 768 --quant 4bit --tag q35_4b_4bit_t768 > outputs/logs/q35_4b_4bit.log 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 --quant 4bit --tag q3vl_4b_4bit_t768 > outputs/logs/q3vl_4b_4bit.log 2>&1
echo done > outputs/logs/quantcheck_done.txt
