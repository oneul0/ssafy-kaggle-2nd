"""Build a 6,714-row submission from retrained 4B rotation probabilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_crop_submission import model_probs, with_margin


ROOT = Path("reference_0_95948/outputs/probs")
OUT = Path("output/qwen4b_retrain_pipeline")
LETTERS = np.array(list("abcd"))


def load(tag, ids):
    with np.load(ROOT / f"{tag}_test.npz", allow_pickle=True) as item:
        assert np.array_equal(item["ids"], ids), tag
        return item["probs"].copy()


def main():
    gate = json.loads((OUT / "rotation_gate.json").read_text(encoding="utf-8"))
    assert gate["test_gate"], "Fixed holdout did not improve"
    test = with_margin(pd.read_csv("output/qwen3vl8b_refrecipe_t768_submission/test_scores.csv"))
    crop = pd.read_csv("output/qwen3vl8b_refrecipe_t768_submission/crop_test_scores_m05.csv")
    ids = test.id.to_numpy()
    q8 = model_probs(test, crop)
    q25 = load("q25vl7b_4bit_t768", ids)
    t35 = np.mean([load("local_q35_4b_r16_t768_q4" + (f"_rot{k}" if k else ""), ids)
                   for k in range(4)], axis=0)
    t3 = np.mean([load("local_q3vl4b_r16_t768_q4" + (f"_rot{k}" if k else ""), ids)
                  for k in range(4)], axis=0)
    probs = 0.5 * np.mean([t35, t3, q25], axis=0) + 0.5 * q8
    pred = LETTERS[probs.argmax(axis=1)]
    sample = pd.read_csv("dataset/sample_submission.csv")
    assert len(test) == len(sample) == 6714 and test.id.equals(sample.id)
    submission = pd.DataFrame({"id": ids, "answer": pred})
    assert submission.id.is_unique and submission.answer.isin(LETTERS).all()
    path = OUT / "submission_local_q4_rot_t768.csv"
    submission.to_csv(path, index=False)
    np.savez(OUT / "ensemble_test_probs.npz", ids=ids, probs=probs)
    old = pd.read_csv("output/qwen3vl8b_refrecipe_t768_submission/submission_four_model_crop_m05.csv")
    assert old.id.equals(submission.id)
    report = {
        "rows": len(submission),
        "changed_vs_public_0_96187": int((old.answer != submission.answer).sum()),
        "holdout_correct": gate["local_four_rotated_correct"],
        "submission": str(path),
    }
    (OUT / "submission_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
