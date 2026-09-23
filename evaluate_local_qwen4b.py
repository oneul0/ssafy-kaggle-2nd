"""Compare retrained 4B models and rotation TTA on the fixed holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("reference_0_95948/outputs/probs")
OUT = Path("output/qwen4b_retrain_pipeline")
LETTERS = np.array(list("abcd"))


def load(tag, ids):
    with np.load(ROOT / f"{tag}_holdout.npz", allow_pickle=True) as item:
        assert np.array_equal(item["ids"], ids), tag
        return item["probs"].copy()


def cropped_8b(valid):
    scores = valid[[f"logp_{c}" for c in "abcd"]].to_numpy()
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ordered = np.sort(probs, axis=1)
    mask = ordered[:, -1] - ordered[:, -2] <= 0.5
    crops = pd.concat(
        [
            pd.read_csv("output/qwen3vl8b_refrecipe_t768/crop_rescore.csv"),
            pd.read_csv("output/qwen3vl8b_refrecipe_t768/crop_correct_low_margin_scores.csv"),
        ]
    ).set_index("id")
    assert set(valid.loc[mask, "id"]) == set(crops.index) & set(valid.loc[mask, "id"])
    probs[mask] = crops.loc[valid.loc[mask, "id"], [f"p_{c}" for c in "abcd"]].to_numpy()
    return probs


def result(probs, gold):
    return LETTERS[probs.argmax(axis=1)] == gold


def main():
    valid = pd.read_csv("output/qwen3vl8b_refrecipe_t768_submission/valid_scores.csv")
    ids, gold = valid.id.to_numpy(), valid.gold.to_numpy()
    q8 = cropped_8b(valid)
    q25 = load("q25vl7b_4bit_t768", ids)
    old = [load(tag, ids) for tag in ("q35_4b_lora_t768", "q3vl_4b_lora_t768")]
    base = 0.5 * np.mean([*old, q25], axis=0) + 0.5 * q8
    q35 = load("local_q35_4b_r16_t768_q4", ids)
    q3 = load("local_q3vl4b_r16_t768_q4", ids)
    plain = 0.5 * np.mean([q35, q3, q25], axis=0) + 0.5 * q8
    t35 = np.mean([q35, *[load(f"local_q35_4b_r16_t768_q4_rot{k}", ids) for k in (1, 2, 3)]], axis=0)
    t3 = np.mean([q3, *[load(f"local_q3vl4b_r16_t768_q4_rot{k}", ids) for k in (1, 2, 3)]], axis=0)
    rotated = 0.5 * np.mean([t35, t3, q25], axis=0) + 0.5 * q8
    old_ok, plain_ok, rotated_ok = (result(p, gold) for p in (base, plain, rotated))
    report = {
        "reference_four_correct": int(old_ok.sum()),
        "local_four_correct": int(plain_ok.sum()),
        "local_four_rotated_correct": int(rotated_ok.sum()),
        "rotation_gained_vs_reference": int((~old_ok & rotated_ok).sum()),
        "rotation_lost_vs_reference": int((old_ok & ~rotated_ok).sum()),
        "q35_single_correct": int(result(q35, gold).sum()),
        "q3_single_correct": int(result(q3, gold).sum()),
        "test_gate": bool(rotated_ok.sum() > old_ok.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rotation_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
