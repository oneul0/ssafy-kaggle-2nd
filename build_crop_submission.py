"""Build a full submission from validated selective crop inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("output/qwen3vl8b_refrecipe_t768_submission")
TRAIN_ROOT = Path("output/qwen3vl8b_refrecipe_t768")
REFERENCE_PROBS = Path("reference_0_95948/outputs/probs")
REFERENCE_TAGS = ("q35_4b_lora_t768", "q3vl_4b_lora_t768", "q25vl7b_4bit_t768")
THRESHOLD = 0.5


def with_margin(frame):
    result = frame.copy()
    scores = result[[f"logp_{c}" for c in "abcd"]].to_numpy()
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ordered = np.sort(probs, axis=1)
    result["margin"] = ordered[:, -1] - ordered[:, -2]
    return result


def apply_crops(base, crops):
    selected = base[base.margin <= THRESHOLD]
    assert selected.id.is_unique and crops.id.is_unique
    assert set(selected.id) == set(crops.id), "Missing or extra crop predictions"
    result = base.copy()
    crop_preds = crops.set_index("id").pred
    mask = result.margin <= THRESHOLD
    result.loc[mask, "pred"] = result.loc[mask, "id"].map(crop_preds)
    assert result.pred.isin(list("abcd")).all()
    return result, len(selected)


def model_probs(base, crops):
    scores = base[[f"logp_{c}" for c in "abcd"]].to_numpy()
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    mask = base.margin <= THRESHOLD
    crop_probs = crops.set_index("id").loc[base.loc[mask, "id"], [f"p_{c}" for c in "abcd"]]
    probs[mask] = crop_probs.to_numpy()
    return probs


def reference_probs(split, ids):
    arrays = [np.load(REFERENCE_PROBS / f"{tag}_{split}.npz", allow_pickle=True) for tag in REFERENCE_TAGS]
    assert all(np.array_equal(item["ids"], ids) for item in arrays)
    return np.mean([item["probs"] for item in arrays], axis=0)


def main():
    valid = with_margin(pd.read_csv(ROOT / "valid_scores.csv"))
    crop_valid = pd.concat(
        [
            pd.read_csv(TRAIN_ROOT / "crop_rescore.csv"),
            pd.read_csv(TRAIN_ROOT / "crop_correct_low_margin_scores.csv"),
        ],
        ignore_index=True,
    )
    crop_valid = crop_valid[crop_valid.id.isin(valid.loc[valid.margin <= THRESHOLD, "id"])]
    valid_after, valid_selected = apply_crops(valid, crop_valid)
    baseline_correct = int((valid.gold == valid.pred).sum())
    crop_correct = int((valid_after.gold == valid_after.pred).sum())
    valid_four_probs = 0.5 * reference_probs("holdout", valid.id.to_numpy()) + 0.5 * model_probs(valid, crop_valid)
    four_valid_pred = np.array(list("abcd"))[valid_four_probs.argmax(axis=1)]
    four_correct = int((four_valid_pred == valid.gold.to_numpy()).sum())

    test = with_margin(pd.read_csv(ROOT / "test_scores.csv"))
    crop_test = pd.read_csv(ROOT / "crop_test_scores_m05.csv")
    test_after, test_selected = apply_crops(test, crop_test)
    sample = pd.read_csv("dataset/sample_submission.csv")
    assert len(test_after) == len(sample) == 6714
    assert test_after.id.is_unique and set(test_after.id) == set(sample.id)
    submission = test_after.set_index("id").loc[sample.id, ["pred"]].reset_index()
    submission.columns = ["id", "answer"]
    assert submission.id.equals(sample.id) and submission.answer.isin(list("abcd")).all()
    submission.to_csv(ROOT / "submission_crop_m05.csv", index=False)
    test_four_probs = 0.5 * reference_probs("test", test.id.to_numpy()) + 0.5 * model_probs(test, crop_test)
    four_test_pred = np.array(list("abcd"))[test_four_probs.argmax(axis=1)]
    four_submission = pd.DataFrame({"id": test.id, "answer": four_test_pred}).set_index("id").loc[sample.id].reset_index()
    assert four_submission.id.equals(sample.id) and four_submission.answer.isin(list("abcd")).all()
    four_submission.to_csv(ROOT / "submission_four_model_crop_m05.csv", index=False)

    report = {
        "baseline_correct": baseline_correct,
        "crop_correct": crop_correct,
        "four_model_crop_correct": four_correct,
        "validation_count": len(valid),
        "validation_cropped": valid_selected,
        "test_count": len(test),
        "test_cropped": test_selected,
        "test_changed": int((test.pred != test_after.pred).sum()),
        "threshold": THRESHOLD,
        "submission": str(ROOT / "submission_crop_m05.csv"),
        "four_model_submission": str(ROOT / "submission_four_model_crop_m05.csv"),
    }
    (ROOT / "crop_m05_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
