#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
$B5 src/mem_test.py --model Qwen3.5-4B --batches 1,2,4          > outputs/logs/mem_q35_bf16.log 2>&1
$B5 src/mem_test.py --model Qwen3.5-4B --quant 4bit --batches 1,2,4,8 > outputs/logs/mem_q35_4bit.log 2>&1
$B4 src/mem_test.py --model Qwen3-VL-4B-Instruct --batches 1,2,4 > outputs/logs/mem_q3vl_bf16.log 2>&1
$B4 src/mem_test.py --model Qwen3-VL-4B-Instruct --quant 4bit --batches 1,2,4,8 > outputs/logs/mem_q3vl_4bit.log 2>&1
echo done > outputs/logs/memtest_done.txt
