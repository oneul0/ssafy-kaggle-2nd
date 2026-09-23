"""공통 설정/유틸. 모든 스크립트가 이 파일을 import 한다."""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS = ROOT / "가이드 및 코드" / "downloads" / "models"
LETTERS = "abcd"

# 베이스라인과 동일한 프롬프트 (변경하면 이전 실험과 비교가 안 되므로 함부로 바꾸지 말 것)
SYSTEM = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


def build_prompt(question, a, b, c, d):
    return (
        f"{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


# ---- 질문 유형 (홀드아웃 층화 + 유형별 정확도용). 위에서부터 먼저 걸리는 유형으로 분류 ----
_TYPE_RULES = [
    ("부정형", r"아닌|않은|않는|없는|틀린|잘못|다른 "),
    ("가격/수량", r"얼마|몇|가격|개수|수량"),
    ("위치", r"왼쪽|오른쪽|위쪽|아래|옆|가장|맨 "),
    ("색/모양", r"색|모양|형태"),
]


def question_type(q):
    q = str(q)
    for name, pat in _TYPE_RULES:
        if re.search(pat, q):
            return name
    return "기타"


def wilson_ci(k, n, z=1.96):
    """정확도의 95% 신뢰구간 (Wilson). (low, high) 반환."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_csv(name):
    return pd.read_csv(DATA / f"{name}.csv")
