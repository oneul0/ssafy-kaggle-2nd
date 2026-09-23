"""모델 비교 분석: outputs/probs/*_holdout.npz 를 읽어 짝 비교·오답 겹침·앙상블·편향·확신도를 출력한다.

  baseline\\Scripts\\python.exe src\\compare_models.py > outputs\\model_comparison.txt

TAGS 에 비교할 실험을 추가하면 된다. (앙상블은 확률 균등 평균이며 가중치 튜닝은 하지 않는다)
"""
import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT

TAGS = {
    "Q2.5-3B": "q25vl3b_t768",
    "Q3.5-4B": "q35_4b_t768",
    "Q3-VL-4B": "q3vl_4b_t768",
    "Q3-VL-8B(4bit)": "q3vl_8b_4bit_t768",
    "Q2.5-7B(4bit)": "q25vl7b_4bit_t768",
}
BASE = "Q2.5-3B"  # 기준선. 나머지를 '신형'으로 취급

h = pd.read_csv(OUT / "split" / "holdout.csv")
gold = h.answer.to_numpy()
L = np.array(list("abcd"))
P = {}
for k, t in TAGS.items():
    z = np.load(OUT / "probs" / f"{t}_holdout.npz", allow_pickle=True)
    assert (z["ids"] == h.id.to_numpy()).all()
    P[k] = z["probs"]
C = {k: (L[p.argmax(1)] == gold) for k, p in P.items()}
ks = list(P)
new = [k for k in ks if k != BASE]


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    n = x + y
    p = 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n)
    return x, y, p


print("== 정확도")
for k in ks:
    print(f"{k:16s}{C[k].mean()*100:.1f}")

print("\n== 짝 비교 (앞 모델만 정답 : 뒤 모델만 정답, 정확 이항검정 p)")
for a, b in itertools.combinations(ks, 2):
    x, y, p = mcnemar(C[a], C[b])
    print(f"{a:16s} vs {b:16s} {x:3d}:{y:3d}  p={p:.3f}")

M = np.array([C[k] for k in ks])
Mn = np.array([C[k] for k in new])
print(f"\n전부 오답 {int((~M).all(0).sum())}문제(5개) / 신형 {int((~Mn).all(0).sum())}문제({len(new)}개) / 하나라도 정답 {int(M.any(0).sum())}")
print("문제당 정답 모델 수 분포(신형):", np.bincount(Mn.sum(0), minlength=len(new) + 1).tolist())

print("\n== 오답 겹침 (Jaccard)")
for a, b in itertools.combinations(new, 2):
    ea, eb = ~C[a], ~C[b]
    print(f"{a:16s} {b:16s} 오답 {int(ea.sum())}/{int(eb.sum())}, 공통 {int((ea&eb).sum())}, Jaccard {(ea&eb).sum()/(ea|eb).sum():.2f}")


def acc(keys):
    return (L[np.mean([P[k] for k in keys], 0).argmax(1)] == gold).mean() * 100


print("\n== 앙상블 (확률 균등 평균)")
res = sorted(((acc(c), c) for r in range(2, len(new) + 1) for c in itertools.combinations(new, r)), reverse=True)
for a, c in res[:8]:
    print(f"{a:.1f}  {' + '.join(c)}")
print(f"신형 전부 평균 {acc(new):.1f} / 최악 조합 {res[-1][0]:.1f} {res[-1][1]}")

print("\n== 위치 편향: d 예측 비율 / 정답이 d일 때 정확도 / 그 외 정확도")
for k in ks:
    pr = L[P[k].argmax(1)]
    print(f"{k:16s} d예측 {np.mean(pr=='d')*100:4.1f}%  gold=d {C[k][gold=='d'].mean()*100:.1f}%  gold!=d {C[k][gold!='d'].mean()*100:.1f}%")

print("\n== 확신도 0.9 이상 비율 / 그때 정확도 / 0.9 미만 정확도 / 오답 중 0.9 미만 비율")
for k in ks:
    cf = P[k].max(1)
    hi = cf >= 0.9
    print(f"{k:16s} {hi.mean()*100:4.1f}%  {C[k][hi].mean()*100:.1f}%  {C[k][~hi].mean()*100:.1f}%  {((~C[k])&~hi).sum()/(~C[k]).sum()*100:.0f}%")

print("\n== 유형별 정확도(%)")
t = pd.DataFrame({k: pd.Series(C[k]).groupby(h.qtype).mean() * 100 for k in ks}).round(1)
t["n"] = h.qtype.value_counts()
print(t.to_string())

idx = np.where((~Mn).all(0))[0]
bad = h.iloc[idx][["id", "path", "qtype", "question", "a", "b", "c", "d", "answer"]].copy()
bad["preds"] = ["".join(L[P[k][i].argmax()] for k in new) for i in idx]
bad.to_csv(OUT / "probs" / "holdout_all_new_models_wrong.csv", index=False)
print(f"\n신형 전부 오답 {len(bad)}문제 -> outputs/probs/holdout_all_new_models_wrong.csv")
