#!/bin/bash
# 채택된 2단계 어댑터로 test 6,714장 추론 (약 48분)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
./baseline5/Scripts/python.exe src/score.py --split test --model Qwen3.5-4B --max-tokens 768 \
    --adapter outputs/adapters/q35_4b_lora2/final --tag q35_4b_lora2_t768 > outputs/logs/test_q35_4b_lora2.log 2>&1
echo q35_lora2_test_done > outputs/logs/test_q35_lora2_done.txt
