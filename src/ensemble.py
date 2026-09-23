"""LoRA 학습 모델 + 추론 전용 모델의 홀드아웃 앙상블 비교 (확률 균등 평균 / 로그확률 평균).

  baseline\Scripts\python.exe src\ensemble.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT

TAGS = {
    "Q3.5-4B-LoRA": "q35_4b_lora_t768",
    "Q3VL-4B-LoRA": "q3vl_4b_lora_t768",
    "Q3VL-8B(4bit)": "q3vl_8b_4bit_t768",
    "Q2.5-7B(4bit)": "q25vl7b_4bit_t768",
    "Q3.5-4B": "q35_4b_t768",
    "Q3VL-4B": "q3vl_4b_t768",
}
h = pd.read_csv(OUT / "split" / "holdout.csv")
gold = h.answer.to_numpy()
L = np.array(list("abcd"))
P = {}
for k, t in TAGS.items():
    z = np.load(OUT / "probs" / f"{t}_holdout.npz", allow_pickle=True)
    assert (z["ids"] == h.id.to_numpy()).all()
    P[k] = z["probs"]
ks = list(P)


def acc(keys, log=False):
    a = np.array([np.log(np.clip(P[k], 1e-6, 1)) if log else P[k] for k in keys])
    return (L[a.mean(0).argmax(1)] == gold).mean() * 100


print("== 단일")
for k in ks:
    print(f"{k:16s}{acc([k]):.1f}")
print("\n== 앙상블 (확률평균 / 로그평균), 상위 15")
rows = [(acc(c), acc(c, True), c) for r in range(2, len(ks) + 1) for c in itertools.combinations(ks, r)]
for a, b, c in sorted(rows, key=lambda x: -x[0])[:15]:
    print(f"{a:.1f} / {b:.1f}  {' + '.join(c)}")
print("\n== 핵심 조합")
for c in [ks[:2], ks[:3], ks[:4], ks[:2] + ks[2:4]]:
    print(f"{acc(c):.1f} / {acc(c, True):.1f}  {' + '.join(c)}")
