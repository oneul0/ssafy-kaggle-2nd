# 0.98 목표 실험 기록 (2026-09-22)

현재 팀 최고 public accuracy: **0.95382** (사용자 제공). 해당 제출 파일을 `output/public_0_95382.csv`에 보존했다. SHA-256: `45CB3AD671D7D8A229891178186F9352A3F0A356A8FDAC3109FC3D06B90E42FB`. 목표와 차이는 **2.618%p**다. 공개 리더보드는 test 전체가 아닐 수 있으므로 내부 validation을 우선한다.

## 현재 확인한 자료

- train/test/dev: 6,714 / 6,714 / 2,683개. 모든 이미지 파일이 있다.
- 최고 제출은 사용자 설명에 따르면 **Colab A100의 Qwen3-VL-8B**, train 전체와 합의비율·엔트로피로 정제한 dev, 2 epoch, 정답 토큰 loss, 원본 비율 이미지 해상도, 4지선다 likelihood scoring을 사용했다. 코드와 검증 점수 파일은 아직 작업 폴더에 없다.
- Qwen2.5-VL-3B v4 노트북의 group validation: 671개 중 accuracy 0.9344. 분할과 모델이 달라 최고 제출과 직접 비교할 수 없다.
- 최고 제출은 6,714행의 ID 순서와 a/b/c/d 답 형식이 모두 정상이다. 기존 `output/submission2.csv`와 답 453개, MiniCPM v5 제출과 답 1,461개가 다르다.
- MiniCPM v5 저장 결과의 validation은 31개, accuracy 22/31. 이 수치로 모델 우열을 결정하지 않는다.
- dev 사람 응답에서 3표 이상인 2,026개를 `output/dev_pseudo_3of5.csv`에 분리했다. 기존 최고 모델의 합의비율·엔트로피 정제보다 개선된 방식이 아니므로 현재는 참고 자료로만 둔다.
- train/test 간 공백 정규화 질문 문자열이 같은 test 행 888개. train 내부 split은 질문 단위로 고정한다.
- train/test 이미지 파일의 SHA-256이 같은 24쌍을 확인했다. 23쌍은 질문이 다르므로 train 정답 글자를 test에 복사하면 안 된다. 질문과 선지까지 완전히 같은 문항은 1개이며 기존 기준 제출과 답이 일치한다.

## 실험 순서

1. **로컬 기준선 측정**: 원래 Colab 코드는 없으므로 새 노트북으로 고정 validation을 실행하고 `id/path/question/gold/pred/logp_a..d`를 저장한다. public 0.95382와 직접 비교하지 않는다.
2. **오답 1차 분류**: validation 오답을 작은 글자/OCR, 숫자·가격·전화번호, 위치·관계, 선택지 혼동, 라벨 오류로 나눈다. 빈도가 높은 두 범주만 우선 개선한다.
3. **시각 정보 실험**: 작은 글자와 숫자 오답에 원본 해상도 증가 또는 2~4개 겹치는 crop을 적용해 재추론한다. 전체에 적용하기 전에 고정 validation에서 선택적 재추론의 순이득을 측정한다.
4. **독립 모델 실험**: 로컬 가중치의 새 VLM을 같은 validation에서 평가한다. 단독 accuracy뿐 아니라 현재 모델이 틀린 행을 얼마나 맞히는지 본다. 두 모델의 `logp_a..d` 평균은 validation에서 개선될 때만 사용한다.
5. **dev 정제 실험**: 기존 합의비율·엔트로피 필터의 임계값을 고정 validation에서 비교한다. dev를 validation에 넣지 않는다.
6. **최종화**: 선택한 설정으로 train 전체를 재학습하고, CSV의 ID 순서·행 수·답 형식 검사를 거쳐 제출한다. 코드, 시드, 모델 버전, 전처리, 추론 설정을 남긴다.

`qwen3vl_5060ti_local.ipynb`가 Windows RTX 5060 Ti 16GB 실행용 노트북이다. `qwen3vl_5060ti.py`의 핵심 함수를 포함하므로 노트북 파일 하나로 실행할 수 있다. `make_qwen3vl_notebook.ps1`을 실행하면 스크립트의 최신 함수로 노트북을 다시 생성한다. Qwen3-VL-8B 4bit LoRA, 질문 기준 층화 group split, validation과 가까운 이미지 제외, dev 합의·엔트로피 필터, 정답 글자 하나만의 loss, 문항별 4지선다 점수 저장을 포함한다. 실제 정제 결과는 train 6,042개와 dev 1,263개, validation 671개다. Hugging Face 가중치 4개 shard는 `downloads/models/Qwen3-VL-8B-Instruct`에 받았다. 로컬 TRL 0.24의 VLM collator를 사용해 실제 이미지에서 Qwen3-VL 프로세서와 정답 한 토큰 loss mask를 확인했다. 노트북 JSON과 Python 셀 문법도 검사했다. GPU 학습과 실제 점수는 아직 확인되지 않았다. 기존 0.95382 모델의 정확한 재현으로 간주하지 않는다.

## RTX 5060 Ti 실행 순서

VS Code가 이전 탭의 셀을 유지한다면 `qwen3vl_5060ti_local_v2.ipynb`를 새로 연다. 첫 코드 셀에 `QWEN3VL_5060TI_LOCAL_V2`가 출력되어야 한다. `qwen3vl_a100_experiment.ipynb`는 이전 Colab용 파일이며 로컬 실행 대상이 아니다.

1. 기존 GPU 작업이 끝나 12 GiB 이상 비면 VS Code/Jupyter에서 `qwen3vl_5060ti_local.ipynb`를 열고 `baseline/Scripts/python.exe` 커널을 선택한다. 현재 다른 Python 작업으로 14.6 GiB를 사용 중이다. 해당 작업은 임의로 종료하지 않는다.
2. `SMOKE=16`, `MODE="valid"`로 전체 셀을 실행해 로딩, 4bit 학습, 10문항 추론까지 확인한다. 16GB에서 OOM이면 `MAX_VISUAL_TOKENS=192`로 낮춰 다시 확인한다.
3. 새 커널에서 `SMOKE=0`, `MODE="valid"`로 전체 학습을 수행한다. `output/qwen3vl8b_valid/valid_scores.csv`와 `valid_errors.csv`를 확인하고, 고해상도 선택 재추론은 개선될 때만 적용한다.
4. 채택할 설정이 확정되면 새 커널에서 `SMOKE=0`, `MODE="submit"`을 실행한다. `output/qwen3vl8b_submit/submission.csv`의 6,714행과 ID 순서를 확인한 뒤 제출한다.

동일한 실험은 CLI로도 가능하다: `baseline/Scripts/python.exe qwen3vl_5060ti.py --data dataset --out output/local_valid --mode valid --smoke 16`. CLI와 노트북 모두 외부 추론 API를 쓰지 않는다.

로컬 `transformers 5.17.0`의 `SFTConfig`는 `warmup_ratio` 대신 `warmup_steps`에 0~1 사이 비율을 받는다. 노트북과 CLI 모두 `warmup_steps=0.05`로 맞췄고, 실제 `SFTConfig` 생성 및 100 step 기준 5 step warmup을 확인했다.

## 채택 기준

2026-09-22 로컬 smoke 실행: `SMOKE=16`, 검증 10문항 중 8문항 정답. 2개 오답은 가격 질문(`train_0014.jpg`, `train_0072.jpg`)이다. 가중치 로딩, 4bit LoRA 학습, 어댑터 저장, 추론까지 완료됐다. 표본이 작아 0.98 가능성을 판단할 수 없으며 다음은 `SMOKE=0`, `MODE="valid"`로 671문항 전체 검증이다.

마감 3시간 50분 전 현재 전체 학습이 74/1828 step에 있다. 2 epoch 완료 후 별도 제출용 재학습은 시간상 위험하다. `output/qwen3vl8b_valid/checkpoints/checkpoint-*`의 첫 epoch 체크포인트가 완전히 저장되면 학습을 중단하고 커널을 재시작해 `./make_submission_from_checkpoint.ps1`을 실행할 수 있다. 이 스크립트는 최신 완료 체크포인트 또는 최종 adapter를 사용해 같은 671개 validation과 test 6,714개를 추론하고 `output/qwen3vl8b_candidate/submission.csv`를 만든다. 0.95382 제출 파일은 보존한다. 새 후보의 validation 결과와 제출 형식을 확인한 뒤에만 제출한다.

첫 epoch 체크포인트 `checkpoint-914`를 저장하고 기본 후보를 만들었다. 내부 validation은 626/671(0.93294), public은 사용자 제출 결과 0.91420이다. 검증 점수 최하위 margin 15%를 512 visual-token 예산으로 재추론하면 25개를 새로 맞히고 5개를 잃어 총 646/671(0.96274)이 됐다. 같은 규칙으로 test 1,007개를 재추론한 후보는 `output/qwen3vl8b_highres_selective/submission.csv`이며 6,714행·ID 순서·답 형식을 확인했다. 이후 `checkpoint-914`에서 2 epoch 학습 재개를 시작했다.

고해상도 후보의 public 점수는 사용자 제출 결과 **0.94399**로 확인됐다. 기존 최고 0.95382보다 0.00983 낮아 채택하지 않는다. 2 epoch 학습은 동일한 분할과 로컬 가중치에서 진행 중이며, 완료 뒤 자동 후속 추론이 `finish_two_epoch_pipeline.ps1`에 설정돼 있다.

새 방법은 **동일한 671개 validation의 정답 수가 실제로 증가**하고, OCR·숫자 질문에서 개선이 확인될 때만 채택한다. 0.98은 목표이지 보장된 성능이 아니다. 매 실험마다 이전 최고 submission을 보존한다.

모든 추론은 로컬 모델과 공개 가중치로 수행한다. 외부 추론 API는 사용하지 않는다.
