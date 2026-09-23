#!/bin/bash
# 7B test 추론이 끝나면(GPU 메모리 겹침 방지) 보기 회전 TTA 를 홀드아웃에서 측정. 총 약 45분.
#  1) rot0 일치 확인: 회전 코드가 기존 결과(20장)를 그대로 재현하는지. 다르면 중단
#  2) 두 LoRA 모델 x rot 1,2,3 (홀드아웃 1,000장씩)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
B4=./baseline/Scripts/python.exe; B5=./baseline5/Scripts/python.exe
A35="--model Qwen3.5-4B --max-tokens 768 --adapter outputs/adapters/q35_4b_lora/final"
AVL="--model Qwen3-VL-4B-Instruct --max-tokens 768 --adapter outputs/adapters/q3vl_4b_lora/final"
LOG=outputs/logs/tta_holdout.log
until [ -f outputs/logs/test_q25vl7b_done.txt ]; do sleep 60; done

$B5 src/score.py --split holdout $A35 --limit 20 --tag tmp_rot0check > $LOG 2>&1
$B5 - >> $LOG 2>&1 <<'PY'
import numpy as np, sys
a=np.load("outputs/probs/tmp_rot0check_holdout.npz",allow_pickle=True)["probs"]
b=np.load("outputs/probs/q35_4b_lora_t768_holdout.npz",allow_pickle=True)["probs"][:20]
d=np.abs(a-b).max(); print("rot0 최대 차이", d)
sys.exit(0 if d<1e-3 else 1)
PY
if [ $? -ne 0 ]; then echo tta_rot0_mismatch > outputs/logs/tta_holdout_done.txt; exit 1; fi
rm -f outputs/probs/tmp_rot0check_holdout*

for k in 1 2 3; do
  $B5 src/score.py --split holdout $A35 --rot $k --tag q35_4b_lora_t768 >> $LOG 2>&1
done
for k in 1 2 3; do
  $B4 src/score.py --split holdout $AVL --rot $k --tag q3vl_4b_lora_t768 >> $LOG 2>&1
done
echo tta_holdout_done > outputs/logs/tta_holdout_done.txt
