# src/ — 실험 코드

모든 명령은 프로젝트 루트(`C:\SSAFY\provided`)에서, **baseline 가상환경**의 파이썬으로 실행합니다.

```powershell
baseline\Scripts\python.exe src\make_holdout.py
baseline\Scripts\python.exe src\score.py --split holdout --max-tokens 768 --tag q25vl3b_t768
baseline\Scripts\python.exe src\score.py --split test    --max-tokens 768 --tag q25vl3b_t768
```

**환경 두 개**: Qwen3.5 계열은 `baseline5\Scripts\python.exe`(transformers 5.17.0), 나머지(Qwen2.5-VL, Qwen3-VL)는 `baseline\Scripts\python.exe`(4.57.6)를 씁니다.

| 파일 | 역할 |
|------|------|
| `common.py` | 경로, 프롬프트(베이스라인과 동일), 질문 유형 분류, 신뢰구간 |
| `make_holdout.py` | train 에서 모의고사 1,000장 분리 (이미지 해시 기준, 유형 층화, seed 42) |
| `score.py` | a/b/c/d 점수 방식 추론. 홀드아웃이면 정확도·유형별 정확도, test 면 제출 CSV 생성. `--adapter`로 LoRA 적용, `--quant 4bit`로 압축 로드 |
| `train_lora.py` | **LoRA 학습** (bf16, 정답 토큰만 loss, 보기 섞기, 이어하기, GPU 예약 메모리 상한 설정 포함). `--quant 4bit/8bit`로 QLoRA(base 압축, Colab 대형 모델용), `--init-adapter`로 기존 어댑터에서 이어서 학습(2단계 학습용, 단 WORKLOG_0922.MD §5.2에서 손해로 확인됨) |
| `run_lora.sh` | Qwen3.5-4B → 평가 → Qwen3-VL-4B → 평가를 자동으로 |
| `run_bakeoff.sh` / `compare_models.py` | 5개 모델 비교 실행 / 짝 비교·앙상블·편향·확신도 분석 |
| `run_memtest.sh` / `mem_test.py` | LoRA·QLoRA 학습 메모리·속도 측정 |
| `run_quantcheck.sh` | 압축(4bit)이 정확도에 주는 영향 측정 |
| `download_models.py` | 모델 사전 다운로드 (오프라인 대비, 이어받기 가능) |

## 산출물 (`outputs/`)

| 경로 | 내용 |
|------|------|
| `split/holdout.csv` | 모의고사 1,000장 — **절대 학습에 쓰지 않기** |
| `split/train_fit.csv` | 학습에 쓸 나머지 5,714장 |
| `probs/{tag}_{split}.npz` | 문제별 a~d 확률 (앙상블·TTA·분석용 원재료) |
| `probs/{tag}_holdout_detail.csv` | 홀드아웃 문제별 예측/정답/확률 |
| `probs/{tag}_holdout_errors.csv` | 틀린 문제 목록 (육안 확인용) |
| `submissions/{tag}.csv` | 제출 파일 |
| `experiments.csv` | 실험 기록표 (실행할 때마다 한 줄 추가) |
| `adapters/{tag}/final` | LoRA 학습 결과(어댑터). 중간 저장은 `ckpt/` |
| `logs/train_{tag}.csv` | 학습 로그 (스텝별 loss, 중간 확인 정확도) |

## 주의

- **프롬프트를 바꾸면 이전 실험과 비교가 안 됩니다.** 바꿀 땐 `--tag` 를 새로 붙이고 기록하세요.
- 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행합니다 (100장마다 저장).
- `--model` 에 `downloads/models/` 아래 다른 폴더명을 주면 다른 모델을 쓸 수 있습니다 (예: `--model Qwen3-VL-4B-Instruct`). 처음 쓰는 모델은 `--limit 20` 으로 먼저 확인하세요.
- 사진은 `--max-tokens` 로 크기를 조절합니다 (1토큰 = 28×28px, 원본 약 880). 파일을 직접 리사이즈하지 마세요.
