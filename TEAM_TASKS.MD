# 팀 작업 분담 — 5인 로컬 실험 + Colab 트랙

> 배경: `STRATEFY.MD`(전체 전략), `WORKLOG_0922.MD`(9/22 실험 기록) 참고.
> 오늘(9/22) 기준 **최종 채택 제출은 `ens_lora2_q7b.csv` (Public 0.95948)**. 대회 마감은 9/28 (오늘 기준 6일 남음).

## ★ 지금까지 가장 성능이 높았던 조합 (이걸 기준으로 시작할 것)

**최종 제출**: `ens_lora2_q7b.csv` — 홀드아웃 95.6%, **Public 0.95948** (오늘 만든 것 중 최고, 리더보드로 실측 확인됨)

**구성**: 아래 3개 모델의 a/b/c/d 확률을 **균등 평균**한 것.

| 모델 | 방식 | 단독 홀드아웃 | 단독 Public |
|---|---|---|---|
| Qwen3.5-4B | LoRA 학습(아래 레시피) | 94.7% | 0.95144 |
| Qwen3-VL-4B | LoRA 학습(아래 레시피) | 94.0% | 0.93893 |
| Qwen2.5-VL-7B | 4bit, **학습 없음**(zero-shot) | 93.5% | 0.92344 |

셋 다 단독으로는 앙상블보다 낮지만, 서로 다르게 틀려서 합치면 더 높아진다(앙상블 효과 실측 확인, `WORKLOG_0922.MD` §1).

**LoRA 학습 레시피** (Qwen3.5-4B, Qwen3-VL-4B 둘 다 동일 — `WORKLOG_0922.MD` §5.3에서 확정, **이 값들이 오늘 여러 실험 끝에 걸러낸 결과이니 함부로 바꾸지 말 것**):

| 항목 | 값 | 비고 |
|---|---|---|
| 해상도 | 768토큰(원본 크기) | |
| 채점 방식 | 생성 대신 a/b/c/d 로짓 비교 | 파싱 실패 없음, 재현성 확보 |
| LoRA rank / alpha | **16 / 32** | rank64는 유의하게 더 나쁨(p=0.024)으로 확인, 절대 올리지 말 것 |
| loss | 정답 토큰 1개에만 (`-100` 마스킹) | |
| 보기 순서 | 매 샘플 무작위 셔플 + 정답 위치 동기화 | 위치 버릇 방지 |
| epoch | **1epoch만** | 2단계 저lr 재학습은 홀드아웃에서 개선처럼 보였으나 리더보드에서 −0.45%p 악화로 확인(§6) → **하지 말 것** |
| lr | 1e-4, cosine, warmup 3% | |
| 압축 | 없음(bf16) | 4bit 압축은 정확도 −1.9~−3.2%p 손해 |
| 학습 대상 | 언어모델만(비전 타워 동결) | |

**재현 명령**:
```bash
# 학습 (완성된 adapter는 이미 outputs/adapters/{q35_4b_lora,q3vl_4b_lora}/final 에 있음, 재학습 불필요)
baseline5\Scripts\python.exe src\train_lora.py --model Qwen3.5-4B --tag q35_4b_lora --batch 1 --accum 8 --lr 1e-4 --rank 16 --alpha 32 --epochs 1
baseline\Scripts\python.exe  src\train_lora.py --model Qwen3-VL-4B-Instruct --tag q3vl_4b_lora --batch 2 --accum 4 --lr 1e-4 --rank 16 --alpha 32 --epochs 1

# 채점 (홀드아웃/test)
baseline5\Scripts\python.exe src\score.py --split holdout --model Qwen3.5-4B --max-tokens 768 --adapter outputs/adapters/q35_4b_lora/final --tag q35_4b_lora_t768
baseline\Scripts\python.exe  src\score.py --split holdout --model Qwen2.5-VL-7B-Instruct --max-tokens 768 --quant 4bit --tag q25vl7b_4bit_t768

# 앙상블 제출 파일 생성
baseline\Scripts\python.exe src\make_ensemble_sub.py ens_lora2_q7b q35_4b_lora_t768 q3vl_4b_lora_t768 q25vl7b_4bit_t768
```

**팀원들은 이 adapter/코드를 그대로 fork해서, 아래 배정된 본인 축 하나만 바꿔서 실험한다.** rank·epoch·2단계 학습처럼 이미 걸러낸 방향은 재시도하지 않는다.

---

## 0. 시작 전 공통 규칙 (반드시 지킬 것)

1. **같은 커밋에서 시작**: 이 문서와 함께 받은 코드(`src/`)와 `outputs/split/holdout.csv`, `outputs/split/train_fit.csv`를 그대로 사용한다. 각자 다른 버전으로 실험하면 나중에 결과를 못 합친다.
2. **한 번에 한 축만 변경**: 아래 배정된 축 외에는 기존 확정 레시피를 그대로 둔다 — rank16/alpha32, 정답 토큰만 loss, 보기 셔플, 1epoch, lr 1e-4, 압축 없이 bf16. (2단계 저lr 재학습은 오늘 리더보드에서 손해로 확인됐으니 시도하지 않는다 — `WORKLOG_0922.MD` §5.2, §6 참고)
3. **평가는 항상 같은 홀드아웃(1,000장)으로**: 다른 표본으로 비교하면 의미 없다.
4. **채택 기준(중요, 오늘 직접 겪은 교훈)**: 홀드아웃 짝 비교에서 p≥0.1인 개선은 리더보드에서 뒤집힐 수 있다. 개선처럼 보여도:
   - 팀 채팅/문서에 결과와 p-value를 먼저 공유하고
   - 최소 1회 재현(다른 seed) 되거나 리더보드로 실측 확인되기 전까지는 **"후보"로만 표시**한다.
5. **제출은 개별적으로 하지 않는다**: 오늘 이미 7회 사용(팀 공유 쿼터로 추정). 최종 후보가 나오면 취합해서 한 사람이 제출한다.
6. **산출물 형식**: 학습을 했다면 `outputs/probs/{본인태그}_holdout.npz`, `_test.npz` 형태로 확률을 저장(기존 `score.py`가 자동으로 이렇게 저장함). 텍스트 비교 결과는 `outputs/experiments.csv`에 한 줄 추가.

## 1. 팀원별 작업 (겹치지 않는 5축)

### 팀원 A — 해상도 특화 추론 (재학습 없음, 가장 가벼움)

- 기존 완성 adapter(`outputs/adapters/q35_4b_lora/final`, `q3vl_4b_lora/final`)를 그대로 쓰고 `score.py --max-tokens`만 1024, 1280으로 바꿔 홀드아웃 재채점
- 문항 유형별(`가격/수량` 90.1%로 가장 낮음, §STRATEFY.MD §4.9) 정확도 변화를 특히 확인
- (선택) 확신도 낮은 문항에만 고해상도 재추론하는 선택적 방식도 시도 (§4.2-3, §4.8에 이미 아이디어 있음)
- 명령 예: `baseline\Scripts\python.exe src\score.py --split holdout --model Qwen3.5-4B --max-tokens 1024 --adapter outputs/adapters/q35_4b_lora/final --tag q35_lora_t1024`

### 팀원 B — LoRA 하이퍼파라미터 (lr / 유효배치)

- rank16/alpha32는 고정, lr(5e-5, 2e-4)과 유효배치(8→16, `--batch`/`--accum` 조합)를 **1,500장 부분학습**으로 먼저 비교
- `src/run_recipe_confirm.sh`가 이런 소규모 비교 패턴의 예시(참고만, 그대로 복붙 금지 — 본인 축에 맞게 수정)
- 유의한 개선이 나오면 그때 전체 5,714장으로 재학습해서 확정
- 참고: Kaggle 디스커션 [`Qwen 학습하면서 조절해볼 만한 파라미터들`](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28/discussion/742457) — 증상별(loss 발산/과적합/OOM 등) 조절 대상 정리표. rank 비교 시 alpha/r 비율을 유지하라는 부분은 우리가 이미 한 방식과 일치, 1epoch 최적 lr이 2epoch에서도 최적은 아닐 수 있다는 점 참고

### 팀원 C — LoRA 적용 범위 (비전-언어 커넥터 포함 여부)

- 현재는 언어모델 부분만 학습 중 (`train_lora.py`의 `LoraConfig target_modules`가 비전 타워는 제외)
- 비전-언어 커넥터(merger/projector) 레이어까지 LoRA에 포함했을 때 효과가 있는지 비교 (코드 수정 필요)
- 역시 1,500장 부분학습으로 먼저 방향을 확인한 뒤 전체 학습 여부 결정
- **⚠️ 필수 확인**: 비전 쪽 LoRA를 target_modules에 넣었다고 실제로 학습되는 건 아닐 수 있다. Kaggle 디스커션([`Unsloth + Qwen3-VL: vision LoRA가 학습되지 않는 문제`](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28/discussion/742458))에서 다른 팀이 gradient checkpointing 훅 문제로 **오류 없이 비전 LoRA가 조용히 전혀 학습되지 않는** 사례를 겪었다(체크포인트의 vision `lora_B` 텐서가 전부 0). 우리 코드는 Unsloth가 아니라 원인은 다르지만, **결론("효과 없음"/"효과 있음")을 내리기 전에 학습 몇 스텝 후 아래처럼 비전 쪽 LoRA 파라미터가 실제로 0에서 벗어났는지 직접 확인할 것**:
  ```python
  vis = [p for n, p in model.named_parameters() if "visual" in n and "lora_B" in n]
  print(sum(int(p.abs().max() == 0) for p in vis), "/", len(vis), "이 갱신 안 됨 (0이면 정상)")
  ```

### 팀원 D — 앙상블 다양성용 신규 계열 모델 (학습 없음)

- VARCO-VISION-2.0(경량 1.7B, 한국어 OCR 특화 — 14B는 로컬 16GB엔 무거움), Gemma3-4B 등을 **zero-shot**으로 홀드아웃 채점
- 기존 모델들과 오답 겹침(Jaccard)까지 확인 — `src/compare_models.py`가 이 패턴의 예시
- 목표는 정확도 자체보다 "다른 모델들과 다르게 틀리는가"이다. 정확도가 낮아도 오답 겹침이 낮으면 앙상블 재료로 가치 있음
- 참고: Kaggle 디스커션([`같은 95%라도 틀리는 문제는 다를까?`](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28/discussion/742459))에서 제안한 **"복구율" = (다른 모델이 맞힌 메인모델 오답 수) / (메인모델 전체 오답 수)** 지표가 Jaccard보다 한 단계 더 유용할 수 있음 — 새 모델이 Qwen3.5-4B(메인)의 오답을 얼마나 회복시키는지로 평가해볼 것

### 팀원 E — 라벨 노이즈 / 중의성 문항 정제

- `src/ambiguity_scan.py`(오늘 작성, 선지가 서로 부분문자열인 문항 약 5% 탐지)를 확장
- train에서 의심 문항(약 334행)을 제외하고 학습했을 때, **중의성 없는 문항만 기준으로** 홀드아웃이 개선되는지 확인
- (선택) 확신도(1등-2등 확률차)가 낮은 문항을 자동으로 찾아내는 스크립트 추가 — Kaggle 디스커션(#742293)에서 제안된 아이디어

## 2. Colab 트랙 (1명 전담, 병행)

5명 전부가 아니라 **1명만 전담**해서 로컬 5축과 동시에 진행한다. 근거와 상세 계획은 `WORKLOG_0922.MD` §7 참고. 요약:

- **모델**: Qwen3-VL-30B-A3B(MoE), 4bit, A100 40GB
- **선행 작업**: `train_lora.py`에 QLoRA(4bit) 학습 경로 추가 (현재는 bf16 미압축만 지원)
- **순서**: 코드 추가 → 환경/데이터 이식 → **zero-shot 게이트**(학습 없이 4bit로 홀드아웃 채점, 로컬 최고 대비 우위 확인) → 통과 시에만 학습 진행
- 게이트를 못 넘으면 학습에 시간 쓰지 말고 로컬 5축에 합류

## 3. 일정 (9/22 저녁 ~ 9/28 마감)

| 날짜 | 로컬 5명 | Colab 전담 |
|---|---|---|
| 9/22 저녁~9/23 | 5축 착수 (A·D는 당일 결과 가능) | QLoRA 코드 추가 → 환경/데이터 이식 → zero-shot 게이트 |
| 9/23~9/24 | B·C·E 결과 수렴, 유의한 것만 전체 재학습 | 게이트 통과 시 학습 시작(`--resume` 필수, 세션 제한 대비) |
| 9/24~9/25 | 결과 취합, 새 앙상블 후보 홀드아웃 검증 | 학습 계속/체크포인트 |
| 9/25~9/26 | 재현 확인된 개선만 후보로 좁히기 | 결과를 로컬과 같은 holdout id로 통합 |
| 9/26~9/27 | 최종 앙상블 결정, 리더보드로 실측 확인 (제출 배분 신경) | 통합 완료 |
| 9/27~9/28 | Phase 4 재현성 검증(깨끗한 환경), 최종 제출 선택 | — |

## 4. 참고 파일

| 파일 | 용도 |
|---|---|
| `STRATEFY.MD` | 전체 전략, 근거, 우선순위 |
| `WORKLOG_0922.MD` | 오늘 실험 전체 기록, 확정 레시피(§5.3), Colab 세팅(§7) |
| `src/score.py` | 로짓 채점기 (`--adapter`, `--quant`, `--max-tokens`, `--rot` 지원) |
| `src/train_lora.py` | LoRA 학습 (`--rank`, `--lr`, `--batch`/`--accum`, `--init-adapter` 지원) |
| `src/ensemble.py`, `src/make_ensemble_sub.py` | 홀드아웃 앙상블 비교, test 제출 파일 생성 |
| `src/compare_models.py` | 모델 간 정확도·오답 겹침 비교 |
| `src/ambiguity_scan.py` | 선지 중의성(부분문자열) 스캔 |
| `outputs/experiments.csv` | 모든 실험 결과 누적 기록 (여기에 본인 실험 추가) |
| `outputs/leaderboard_log.csv` | 리더보드 제출 기록 |
