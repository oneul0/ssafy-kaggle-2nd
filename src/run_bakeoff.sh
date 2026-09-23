#!/bin/bash
# 모델 대결(bake-off): 같은 홀드아웃 1,000장, 같은 사진 크기(예산 768 => 672x896px)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe    # transformers 4.57.6
B5=./baseline5/Scripts/python.exe   # transformers 5.17.0 (Qwen3.5 용)
mkdir -p outputs/logs
$B5 src/score.py --split holdout --model Qwen3.5-4B          --max-tokens 768 --tag q35_4b_t768      > outputs/logs/q35_4b.log 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-4B-Instruct --max-tokens 768 --tag q3vl_4b_t768    > outputs/logs/q3vl_4b.log 2>&1
$B4 src/score.py --split holdout --model Qwen3-VL-8B-Instruct --max-tokens 768 --quant 4bit --tag q3vl_8b_4bit_t768 > outputs/logs/q3vl_8b.log 2>&1
$B4 src/score.py --split holdout --model Qwen2.5-VL-7B-Instruct --max-tokens 768 --quant 4bit --tag q25vl7b_4bit_t768 > outputs/logs/q25vl7b.log 2>&1
echo ALL_DONE > outputs/logs/bakeoff_done.txt
