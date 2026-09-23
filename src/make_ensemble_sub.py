"""test 확률 npz 여러 개를 균등 평균해 제출 CSV 생성.

  baseline\Scripts\python.exe src\make_ensemble_sub.py 출력이름 태그1 태그2 ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT

name, tags = sys.argv[1], sys.argv[2:]
L = np.array(list("abcd"))
zs = [np.load(OUT / "probs" / f"{t}_test.npz", allow_pickle=True) for t in tags]
for z in zs:
    assert (z["ids"] == zs[0]["ids"]).all()
p = np.mean([z["probs"] for z in zs], 0)
sub = pd.DataFrame({"id": zs[0]["ids"], "answer": L[p.argmax(1)]})
samp = pd.read_csv(OUT.parent / "data" / "sample_submission.csv")
assert len(sub) == len(samp) and set(sub.id) == set(samp.id), "형식 불일치"
sub = sub.set_index("id").loc[samp.id].reset_index()
sub.to_csv(OUT / "submissions" / f"{name}.csv", index=False)
print(name, len(sub), sub.answer.value_counts(normalize=True).mul(100).round(1).to_dict())
