# Qwen3-VL-8B 재학습 결과 (2026-09-22)

목적: 제출 파일 재조합이 아닌 로컬 8B 모델의 실제 재학습과 6,714장 새 추론.

## 학습

- 기준 분할: `reference_0_95948/outputs/split/holdout.csv` 1,000개 제외, train 5,714개 학습. dev 합의 라벨 미사용.
- RTX 5060 Ti 16GB, 로컬 Qwen3-VL-8B-Instruct, 4bit NF4 QLoRA.
- 이미지 예산: min 256, max 768, 28×28 pixel/token. LoRA rank/alpha 16/32, 언어모델 선형층만. 1 epoch, lr 1e-4, cosine, warmup 3%, batch 1, 누적 8, 보기 순서 셔플, 저장소와 같은 프롬프트.
- 16개 스모크 테스트 성공. 저장된 LoRA B 텐서 252/252개에서 0이 아닌 값 확인.
- 전체 학습의 optimizer step 100마다 체크포인트 저장, 최근 2개 유지. 최종 어댑터: `output/qwen3vl8b_refrecipe_t768/adapter`.

## 평가·산출물

- 최종 어댑터를 다시 로드해 별도 추론한 홀드아웃 정확도: **952/1,000 (95.2%)**. 공유 저장소 3모델 앙상블은 956/1,000 (95.6%). 단일 모델인 Qwen3.5-4B LoRA 947/1,000과 Qwen3-VL-4B LoRA 940/1,000보다는 높다.
- 공유 저장소 앙상블이 틀리고 새 8B가 맞힌 문항 16개, 반대 20개, 둘 다 틀린 문항 28개. 새 8B가 유용한 보완 신호를 가진다는 뜻이지만 단독으로 최고 앙상블을 넘지는 못했다.
- test 이미지 **6,714개를 새로 추론**해 `output/qwen3vl8b_refrecipe_t768_submission/submission.csv` 생성. `id,answer` 열, 원본 sample의 ID 순서, 답 a/b/c/d, 행 수를 검사했다.
- 파이프라인: `run_refrecipe_8b.ps1`. 학습: `qwen3vl_5060ti.py`. 어댑터 로드 및 추론: `qwen3vl_submit_from_adapter.py`.

이 제출 파일 자체의 public 점수는 아직 측정되지 않았다. 기존 최고 public 0.95948을 대체한다는 근거는 없다. Kaggle 자동 제출이나 외부 추론 API는 사용하지 않았다.
