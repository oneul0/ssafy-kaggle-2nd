# 선택적 이미지 크롭 추론 결과 (2026-09-23)

## 방법

- 기존 Qwen3-VL-8B LoRA 모델의 원본 이미지 답안 확률 차이(1위-2위)가 0.5 이하인 문항만 추가 추론한다.
- 원본, 좌측 72%, 우측 72%, 상단 72%, 하단 72% 이미지의 답안 확률을 평균한다.
- 홀드아웃 1,000개 중 43개에 이 규칙을 적용했다. 테스트 6,714개 중 같은 기준으로 고른 265개에 적용했다.
- `submission_crop_m05.csv`는 이 모델의 단독 예측이다. `submission_four_model_crop_m05.csv`는 위 모델 확률과 공유 저장소의 3모델 앙상블 확률을 각각 50%로 평균한다. 두 파일 모두 실제 이미지 모델 추론 확률에서 생성되었다.

## 결과

| 방식 | 홀드아웃 정답 |
|---|---:|
| 기존 Qwen3-VL-8B | 952/1,000 |
| 기존 공유 저장소 3모델 앙상블 | 956/1,000 |
| Qwen3-VL-8B 선택적 크롭 | 956/1,000 |
| 공유 저장소 3모델 + 선택적 크롭 8B | 959/1,000 |

선택적 크롭은 기존 8B의 오답 7개를 맞혔고 정답 3개를 잃었다. 4모델 후보는 기존 공유 저장소 앙상블에 비해 오답 13개를 맞혔고 정답 10개를 잃었다. 사용자 제출로 확인한 공개 점수는 8B 선택적 크롭 **0.95650**, 4모델 + 선택적 크롭 **0.96187**이다. 기존 공유 저장소의 공개 점수는 0.95948, 기존 8B 단독 공개 점수는 0.95442였다.

## 파일과 재현

- 원본·크롭 검증 확률: `output/qwen3vl8b_refrecipe_t768/crop_rescore.csv`, `crop_correct_low_margin_scores.csv`
- 테스트 크롭 확률: `output/qwen3vl8b_refrecipe_t768_submission/crop_test_scores_m05.csv`
- 전체 제출 후보: `output/qwen3vl8b_refrecipe_t768_submission/submission_crop_m05.csv`, `submission_four_model_crop_m05.csv`
- 생성 스크립트: `crop_rescore_qwen3vl.py`, `build_crop_submission.py`
- 요약 JSON: `output/qwen3vl8b_refrecipe_t768_submission/crop_m05_metrics.json`

두 후보 CSV 모두 `id,answer` 열, 샘플과 같은 ID 순서, 중복 없는 6,714행, `a/b/c/d` 답안 범위를 확인했다. 크롭 추론을 재실행할 때는 `crop_rescore_qwen3vl.py --split test --review output/qwen3vl8b_refrecipe_t768_submission/crop_test_review_m05.csv --out output/qwen3vl8b_refrecipe_t768_submission/crop_test_scores_m05.csv`를 사용한다. 스크립트는 완료된 행을 건너뛰므로 중단 후 재개할 수 있다. 그 다음 `build_crop_submission.py`를 실행한다.
