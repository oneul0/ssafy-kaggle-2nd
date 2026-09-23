"""[1단계] train 에서 홀드아웃(모의고사) 1,000장을 분리한다.

- 같은 이미지(파일 해시 동일)는 반드시 같은 쪽에 넣는다 (누수 방지)
- 질문 유형 비율이 train 전체와 비슷하도록 층화 추출
- 시드 고정 -> 항상 같은 결과

산출물 (outputs/split/)
  holdout.csv    : 모의고사 (절대 학습에 쓰지 않는다)
  train_fit.csv  : 나머지 (학습에 사용)
  image_hash.csv : train 이미지 해시 캐시

실행:  baseline\\Scripts\\python.exe src\\make_holdout.py
"""
import hashlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, OUT, load_csv, question_type

SEED = 42
TARGET = 1000


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    split_dir = OUT / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    tr = load_csv("train")
    tr["qtype"] = tr["question"].map(question_type)
    tr["img_hash"] = [md5(DATA / p) for p in tr["path"]]
    tr[["id", "img_hash"]].to_csv(split_dir / "image_hash.csv", index=False)

    # 이미지 단위 그룹 (같은 이미지가 여러 행이면 한 그룹)
    grp = tr.groupby("img_hash").agg(n=("id", "size"), qtype=("qtype", "first")).reset_index()
    rng = np.random.default_rng(SEED)

    share = tr["qtype"].value_counts(normalize=True)
    quota = (share * TARGET).round().astype(int)

    chosen = []
    for qt, q in quota.items():
        g = grp[grp["qtype"] == qt].sample(frac=1.0, random_state=int(rng.integers(1_000_000)))
        got = 0
        for _, row in g.iterrows():
            if got >= q:
                break
            chosen.append(row["img_hash"])
            got += row["n"]

    hold = tr[tr["img_hash"].isin(set(chosen))].copy()
    fit = tr[~tr["img_hash"].isin(set(chosen))].copy()

    # 안전 검사: 겹치는 이미지/ID 가 없어야 한다
    assert set(hold["img_hash"]).isdisjoint(set(fit["img_hash"])), "홀드아웃과 학습 이미지가 겹침!"
    assert set(hold["id"]).isdisjoint(set(fit["id"]))
    assert len(hold) + len(fit) == len(tr)

    cols = ["id", "path", "question", "a", "b", "c", "d", "answer", "qtype", "img_hash"]
    hold[cols].to_csv(split_dir / "holdout.csv", index=False)
    fit[cols].to_csv(split_dir / "train_fit.csv", index=False)

    print(f"train 전체 {len(tr):,} -> 홀드아웃 {len(hold):,} / 학습용 {len(fit):,}")
    cmp = pd.DataFrame({
        "train 전체(%)": (tr["qtype"].value_counts(normalize=True) * 100).round(1),
        "홀드아웃(%)": (hold["qtype"].value_counts(normalize=True) * 100).round(1),
        "홀드아웃(개)": hold["qtype"].value_counts(),
    })
    print(cmp.to_string())
    print("홀드아웃 정답 분포(%):", (hold["answer"].value_counts(normalize=True) * 100).round(1).to_dict())
    print("이미지 겹침 검사: 통과")


if __name__ == "__main__":
    main()
