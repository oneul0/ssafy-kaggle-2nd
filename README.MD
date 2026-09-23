# SSAFY 16기 2회차 AI 챌린지 — 텍스트 이미지 VQA

이미지 속 글자(간판, 가격표 등)를 읽고 질문에 4지선다(a/b/c/d)로 답하는 VQA(Visual Question Answering) 모델을 만드는 대회입니다.
Kaggle: [`ssafy-16-2-ai-9-21-9-28`](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28) (9/21~9/28)

**처음 왔다면 이 순서로 읽으세요**: 이 README(환경 세팅) → `WORKLOG_0922.MD`(지금까지 뭘 했고 뭐가 제일 잘 됐는지) → `TEAM_TASKS.MD`(내가 뭘 하면 되는지) → `STRATEFY.MD`(왜 이렇게 하는지, 전체 근거).

---

## 1. 저장소 가져오기

이 저장소에는 **코드와 문서만** 들어있습니다. 모델 가중치·원본 이미지 데이터·가상환경은 용량이 커서 git에 올리지 않습니다(`.gitignore` 참고). 아래 순서대로 각자 준비해야 합니다.

```powershell
git clone https://github.com/SSAFY-AI-Competition/Kaggle.git
cd Kaggle
```

### 1.1 데이터 준비

Kaggle 대회 페이지의 Data 탭(또는 팀 공유 Google Drive의 `ssafy-16-2-ai.zip`)에서 데이터를 받아 아래 구조로 `data/` 밑에 풀어놓습니다. 상세 스펙은 [`DATA_DESC.MD`](DATA_DESC.MD) 참고.

```
data/
├── train.csv, train/      # 학습용 (정답 포함, 6,714행)
├── test.csv,  test/       # 제출용 (정답 없음, 6,714행)
├── dev.csv,   dev/        # 개발용
└── sample_submission.csv
```

### 1.2 모델 준비

`가이드 및 코드/downloads/models/` 밑에 사용할 모델을 받아둡니다(Hugging Face 사전학습 VQA 모델만 허용 — [`RULE.MD`](RULE.MD) 참고). 이미 받아 확인된 모델:

- `Qwen2.5-VL-3B-Instruct`, `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL-4B-Instruct`, `Qwen3-VL-8B-Instruct`
- `Qwen3.5-4B`

`src/download_models.py`로 이어받기 가능한 다운로드를 할 수 있습니다(대용량이라 처음엔 시간이 걸립니다).

---

## 2. 환경 세팅 (가상환경 2개)

**왜 2개인가**: 가장 최신 모델(Qwen3.5)은 `transformers >= 5.0.0`에서만 동작하는데, 나머지 모델(Qwen2.5-VL, Qwen3-VL)이 검증된 조합은 `< 5.0.0`입니다. 하나로 합치면 버전 충돌이 나서 두 개로 분리했습니다 (`STRATEFY.MD` §4.10).

```powershell
# ① baseline: Qwen2.5-VL, Qwen3-VL 계열용 (transformers 4.57.6)
python -m venv baseline
baseline\Scripts\python.exe -m pip install -r requirements-baseline.txt

# ② baseline5: Qwen3.5 전용 (transformers 5.17.0)
python -m venv baseline5
baseline5\Scripts\python.exe -m pip install -r requirements-baseline5.txt
```

`torch`는 CUDA 버전에 맞춰 별도 인덱스에서 설치해야 할 수 있습니다 ([pytorch.org](https://pytorch.org/get-started/locally/) 참고). GPU는 16GB 기준으로 세팅되어 있습니다(예약 메모리가 16GB를 넘으면 10배 이상 느려지는 함정이 있으니 `STRATEFY.MD` §4.4 확인).

---

## 3. 빠른 실행

모든 명령은 **저장소 루트**에서 실행합니다. 어떤 모델에 어떤 가상환경을 쓰는지는 `src/README.MD` 표를 확인하세요.

```powershell
# 1) 모의고사(홀드아웃 1,000장) 분리 — 이미 outputs/split/ 에 있으면 생략
baseline\Scripts\python.exe src\make_holdout.py

# 2) 기준선 정확도 확인 (학습 없이, 로짓 채점)
baseline\Scripts\python.exe src\score.py --split holdout --max-tokens 768 --tag smoke

# 3) LoRA 학습 (확정 레시피 그대로. WORKLOG_0922.MD §5.3 참고 — 함부로 바꾸지 말 것)
baseline5\Scripts\python.exe src\train_lora.py --model Qwen3.5-4B --tag q35_4b_lora --batch 1 --accum 8 --lr 1e-4 --rank 16 --alpha 32 --epochs 1

# 4) 학습된 adapter로 test 추론 → 제출 파일 생성
baseline5\Scripts\python.exe src\score.py --split test --model Qwen3.5-4B --max-tokens 768 --adapter outputs/adapters/q35_4b_lora/final --tag q35_4b_lora_t768

# 5) 여러 모델 확률을 합쳐 앙상블 제출 파일 생성
baseline\Scripts\python.exe src\make_ensemble_sub.py 내제출이름 태그1 태그2 태그3
```

세부 옵션(`--rot`, `--quant`, `--init-adapter` 등)과 `outputs/` 산출물 설명은 [`src/README.MD`](src/README.MD)에 정리되어 있습니다.

**지금까지 가장 성능이 높았던 조합**(홀드아웃 95.6% / Public 0.95948)과 그 재현 명령은 [`TEAM_TASKS.MD`](TEAM_TASKS.MD) 맨 위에 있습니다. 새로 시작한다면 이걸 기준으로 fork해서 실험하세요.

---

## 4. 문서 지도 — 각 MD 파일이 뭔지

| 파일 | 내용 | 언제 읽나 |
|---|---|---|
| **README.MD** (이 파일) | 저장소 받기, 환경 세팅, 실행 방법 | 맨 처음 |
| [`OVERVIEW.MD`](OVERVIEW.MD) | 대회 소개 (원본 배포 자료) | 대회 자체가 처음일 때 |
| [`RULE.MD`](RULE.MD) | 대회 규칙 요약 (팀 구성, 제출 횟수, 금지 사항 등) | 뭔가 해도 되는지 헷갈릴 때 |
| [`DATA_DESC.MD`](DATA_DESC.MD) | 데이터 구조·컬럼 설명 (원본 배포 자료) | 데이터 형식 확인할 때 |
| [`REFERENCE_SOLUTIONS.MD`](REFERENCE_SOLUTIONS.MD) | 유사 선행 대회(SSAFY 16기 1회차 등) 상위권 풀이 분석 | 전략 배경이 궁금할 때 |
| [`STRATEFY.MD`](STRATEFY.MD) | **우리 팀 전체 전략** — 근거·우선순위·로드맵·검증 설계. 가장 오래되고 큰 문서 | 왜 이렇게 하는지 이해하고 싶을 때 |
| [`MODEL_COMPARISON.MD`](MODEL_COMPARISON.MD) | 학습 전 모델 5종 zero-shot 비교 결과 (학습 대상 2개를 고른 근거) | 모델 선택 근거가 궁금할 때 |
| [`ERROR_ANALYSIS.MD`](ERROR_ANALYSIS.MD) | 오답 19문제 전수 육안 분석 (라벨 문제 vs 진짜 모델 오류 구분) | 정확도 상한이 얼마인지 궁금할 때 |
| [`WORKLOG_0922.MD`](WORKLOG_0922.MD) | **9/22 실험 기록** — 리더보드 제출 내역, TTA·앙상블·rank 비교·2단계 학습 실험, 확정 레시피, Colab 세팅 계획 | 오늘까지 뭘 했는지, 뭐가 됐고 안 됐는지 확인할 때 |
| [`TEAM_TASKS.MD`](TEAM_TASKS.MD) | **팀원 5명 작업 분담** + 최고 성능 조합 재현 명령 + 일정표 | 내가 뭘 해야 하는지 확인할 때 |

## 5. 꼭 지켜야 할 것

- **홀드아웃(`outputs/split/holdout.csv`) 1,000장은 절대 학습에 쓰지 않습니다.** 전원 같은 파일을 써야 결과를 비교·앙상블할 수 있습니다.
- 프롬프트(`src/common.py`의 `SYSTEM`/`build_prompt`)를 바꾸면 이전 실험과 비교가 안 됩니다 — 바꾸려면 팀에 먼저 공유.
- 개선처럼 보여도 홀드아웃 짝비교 **p ≥ 0.1이면 아직 "후보"**입니다. 리더보드 실측 또는 재현 확인 전엔 확정하지 마세요(`WORKLOG_0922.MD` §6 — 실제로 한 번 뒤집힌 사례가 있습니다).
- 제출은 팀 공유 쿼터입니다. 각자 올리지 말고 취합해서 진행하세요.
- 외부 API 호출 방식 추론은 금지, 외부 GPU 대여(Colab 포함)로 공개 가중치 모델을 자체 코드로 돌리는 건 허용입니다(`RULE.MD`, 운영진 확인).
# ssafy-kaggle-2nd
