# 0.95948 저장소 기반 앙상블 점검

원본: [SSAFY-AI-Competition/Kaggle](https://github.com/SSAFY-AI-Competition/Kaggle), 2026-09-22에 받은 HEAD `1f7cb71f8afd52bfb21cd30e5f6c18da489eff9a`.

- `WORKLOG_0922.MD`에 기록된 public 0.95948은 Qwen3.5-4B LoRA, Qwen3-VL-4B LoRA, Qwen2.5-VL-7B 4bit의 **확률 균등 평균**이다. 이 셋의 저장된 NPZ를 `analyze_reference_ensemble.py`로 합쳐 6,714행 제출 파일을 재구성했다. 공개 점수는 사용자가 제공한 값이며, 재구성 파일 자체는 제출하지 않았다.
- 원본 저장소의 1,000행 holdout 정확도를 956/1,000으로 재현했다.
- 로컬 Qwen3-VL-8B의 별도 validation과 원본 holdout의 공통 113행을 찾았다. 두 학습 분할 모두에서 해당 ID가 제외됐고 원본 `image_hash.csv`로 동일 이미지가 학습 쪽에 남지 않은 것도 확인했다.
- 이 113행에서 원본 앙상블은 109개, 원본 확률에 로컬 1 epoch 고해상도 확률을 0.25배 더한 후보는 111개를 맞혔다. 2개 회복, 0개 손실이다. 2 epoch 역시 0.25배에서 111개였다. 차이는 단 2문항이므로 개선이 확실하다고 볼 수 없다.
- 로컬 1 epoch 고해상도 public 0.94399가 2 epoch 0.94101보다 높아 1 epoch를 추가 모델로 선택했다. 새 후보는 원본 앙상블의 test 답 52/6,714개를 바꾼다. 기존 public 0.95382 제출은 그 52개 중 34개에서 새 후보와 일치한다. 이 수치는 정답 여부가 아니다.

## 결과 파일

- `output/reference_ensemble_analysis/reference_0_95948_rebuilt.csv`: 원본 3모델 앙상블 재구성. 원본 `make_ensemble_sub.py`의 float32 계산과 6,714개 예측이 모두 일치한다.
- `output/reference_ensemble_analysis/reference_plus_8b_weight025.csv`: 0.25배 로컬 8B를 추가한 신규 후보. 6,714개 ID, 원본 sample 순서, `id,answer` 형식 및 답 범위를 검사했다. **public 점수는 아직 모른다.**
- `analyze_reference_ensemble.py`: 재구성·검증·후보 생성 코드. 외부 추론 API 호출 없음.

다음 실험은 제출 횟수 여유가 있을 때 후보 public 점수를 0.95948과 비교하는 것이다. 성공이 확인되기 전까지 0.95948 제출을 최고로 유지한다. 113행 비교만으로 0.98 달성을 주장하지 않는다.
