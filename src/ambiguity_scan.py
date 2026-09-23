"""선지 중의성 스캔 (Kaggle 디스커션 #742293 재현): 한 선지가 다른 선지의 부분문자열이거나
공백/대소문자만 다르면 '의심 후보'로 표시. 이미지는 보지 않고 텍스트만 비교.

  baseline\Scripts\python.exe src\ambiguity_scan.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LETTERS, OUT, load_csv

norm = lambda s: re.sub(r"\s+", "", str(s)).lower()


def scan(df, name):
    sub, dup = 0, 0
    flag = []
    for _, r in df.iterrows():
        opts = [norm(r[c]) for c in LETTERS]
        hit = False
        for i in range(4):
            for j in range(4):
                if i == j or not opts[i]:
                    continue
                if opts[i] == opts[j]:
                    if i < j:
                        dup += 1
                elif opts[i] in opts[j]:
                    hit = True
        if hit:
            sub += 1
        flag.append(hit)
    print(f"{name} ({len(df)}행): 부분문자열 포함 {sub}행 ({sub/len(df)*100:.2f}%), 완전동일쌍 {dup}행")
    return pd.Series(flag, index=df.index)


for name, df in [("train", load_csv("train")), ("test", load_csv("test"))]:
    fl = scan(df, name)
    df.assign(ambiguous=fl).to_csv(OUT / "probs" / f"ambiguous_{name}.csv", index=False)

# 홀드아웃에서 의심 후보 문항의 정확도(가진 예측 파일로) 확인
h = pd.read_csv(OUT / "split" / "holdout.csv")
fl = scan(h, "holdout")
h = h.assign(ambiguous=fl)
h.to_csv(OUT / "probs" / "ambiguous_holdout.csv", index=False)

import numpy as np
L = np.array(list("abcd"))
for tag in ["q35_4b_lora_t768", "q3vl_4b_lora_t768"]:
    z = np.load(OUT / "probs" / f"{tag}_holdout.npz", allow_pickle=True)
    assert (z["ids"] == h.id.to_numpy()).all()
    pred = L[z["probs"].argmax(1)]
    ok = pred == h.answer.to_numpy()
    print(f"{tag}: 의심후보 정확도 {ok[fl].mean()*100:.1f}% ({fl.sum()}문항) / 나머지 {ok[~fl].mean()*100:.1f}%")
