#!/bin/bash
# 추론 전용 앙상블 재료: Qwen2.5-VL-7B(4bit) test 6,714장 (약 77분)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
./baseline/Scripts/python.exe src/score.py --split test --model Qwen2.5-VL-7B-Instruct --max-tokens 768 --quant 4bit --tag q25vl7b_4bit_t768 > outputs/logs/test_q25vl7b_4bit.log 2>&1
echo q25vl7b_test_done > outputs/logs/test_q25vl7b_done.txt
