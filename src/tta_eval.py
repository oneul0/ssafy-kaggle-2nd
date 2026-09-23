"""보기 회전 TTA 홀드아웃 결과: 모델별 rot 개수에 따른 정확도, 위치 편향, 앙상블 조합.

  baseline\Scripts\python.exe src\tta_eval.py
"""
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT

h = pd.read_csv(OUT / "split" / "holdout.csv")
gold = h.answer.to_numpy()
L = np.array(list("abcd"))
ld = lambda t: np.load(OUT / "probs" / f"{t}_holdout.npz", allow_pickle=True)["probs"]
ok = lambda p: L[p.argmax(1)] == gold


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    n = x + y
    return x, y, 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


T = {}
for name, tag in [("Q3.5-LoRA", "q35_4b_lora_t768"), ("Q3VL-LoRA", "q3vl_4b_lora_t768")]:
    R = [ld(tag)] + [ld(f"{tag}_rot{k}") for k in (1, 2, 3)]
    print(f"== {name}")
    for k, p in enumerate(R):
        print(f"  rot{k} 단독 {ok(p).mean()*100:.1f}%")
    print(f"  TTA2(rot0+2) {ok(R[0]+R[2]).mean()*100:.1f}%  TTA4(평균) {ok(sum(R)).mean()*100:.1f}%  "
          f"TTA4(로그평균) {ok(sum(np.log(np.clip(r, 1e-6, 1)) for r in R)).mean()*100:.1f}%")
    x, y, p = mcnemar(ok(sum(R)), ok(R[0]))
    print(f"  TTA4 vs rot0: {x}:{y} p={p:.3f}")
    print(f"  회전간 예측 일치율(rot0 vs rot1/2/3): "
          + " ".join(f"{(R[0].argmax(1) == r.argmax(1)).mean()*100:.1f}%" for r in R[1:]))
    T[name] = sum(R) / 4

print("\n== 앙상블")
base = (ld("q35_4b_lora_t768") + ld("q3vl_4b_lora_t768")) / 2
tta = (T["Q3.5-LoRA"] + T["Q3VL-LoRA"]) / 2
print(f"  LoRA 2개 (TTA 없음) {ok(base).mean()*100:.1f}%")
print(f"  LoRA 2개 (TTA4)     {ok(tta).mean()*100:.1f}%   vs TTA 없음 {mcnemar(ok(tta), ok(base))}")
s7 = ld("q25vl7b_4bit_t768")
print(f"  LoRA 2개 (TTA4) + 7B {ok((T['Q3.5-LoRA'] + T['Q3VL-LoRA'] + s7) / 3).mean()*100:.1f}%")
