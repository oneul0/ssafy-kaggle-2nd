"""Rebuild the shared 0.95948 ensemble and test a Qwen3-VL-8B addition.

Only the intersection of the two teams' held-out train IDs is used to
compare models. Neither model has trained on those images.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REF = ROOT / "reference_0_95948" / "outputs"
LOCAL = ROOT / "output"
OUT = LOCAL / "reference_ensemble_analysis"
LETTERS = np.array(list("abcd"))
TAGS = ("q35_4b_lora_t768", "q3vl_4b_lora_t768", "q25vl7b_4bit_t768")


def reference_probs(split):
    # The source repository writes string IDs as NumPy object arrays.
    parts = [np.load(REF / "probs" / f"{tag}_{split}.npz", allow_pickle=True) for tag in TAGS]
    ids = parts[0]["ids"].astype(str)
    assert all(np.array_equal(ids, part["ids"].astype(str)) for part in parts)
    probabilities = np.mean([part["probs"].astype(np.float64) for part in parts], axis=0)
    return ids, probabilities


def local_probs(path):
    rows = pd.read_csv(path)
    scores = rows[[f"logp_{letter}" for letter in LETTERS]].to_numpy(dtype=np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return rows, probabilities


def check_submission(ids, probabilities, path):
    sample = pd.read_csv(ROOT / "dataset" / "sample_submission.csv")
    assert len(ids) == len(sample) == 6714
    assert len(set(ids)) == len(ids)
    table = pd.DataFrame({"id": ids, "answer": LETTERS[probabilities.argmax(axis=1)]})
    assert set(table.id) == set(sample.id)
    table = table.set_index("id").loc[sample.id].reset_index()
    assert set(table.answer) <= set(LETTERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def score_overlap(ref_ids, ref_p, holdout, path, highres_path=None):
    local_valid, valid_p = local_probs(path)
    index = {id_: i for i, id_ in enumerate(local_valid.id)}
    if highres_path is not None:
        high_valid, high_valid_p = local_probs(highres_path)
        for id_, p in zip(high_valid.id, high_valid_p):
            valid_p[index[id_]] = p
    overlap = sorted(set(ref_ids) & set(local_valid.id))
    ref_index = {id_: i for i, id_ in enumerate(ref_ids)}
    gold = holdout.loc[overlap, "answer"].to_numpy()
    assert np.array_equal(gold, local_valid.set_index("id").loc[overlap, "gold"].to_numpy())
    rp = ref_p[[ref_index[id_] for id_ in overlap]]
    lp = valid_p[[index[id_] for id_ in overlap]]
    base_pred = LETTERS[rp.argmax(axis=1)]
    print(f"Independent validation overlap: {len(overlap)}; local correct {sum(LETTERS[lp.argmax(axis=1)] == gold)}")
    for weight in (0, 0.25, 0.5, 1):
        combined = (rp + weight * lp) / (1 + weight)
        pred = LETTERS[combined.argmax(axis=1)]
        wins = int(((pred == gold) & (base_pred != gold)).sum())
        losses = int(((pred != gold) & (base_pred == gold)).sum())
        print(f"Qwen8B weight {weight}: {sum(pred == gold)}/{len(gold)}, "
              f"wins {wins}, losses {losses}")


def main():
    holdout = pd.read_csv(REF / "split" / "holdout.csv").set_index("id")
    ref_ids, ref_p = reference_probs("holdout")
    assert np.array_equal(ref_ids, holdout.index.to_numpy())
    correct = LETTERS[ref_p.argmax(axis=1)] == holdout.answer.to_numpy()
    print(f"Reference holdout: {correct.sum()}/{len(correct)} = {correct.mean():.5f}")

    ref_test_ids, ref_test_p = reference_probs("test")
    reference = check_submission(ref_test_ids, ref_test_p, OUT / "reference_0_95948_rebuilt.csv")
    old = pd.read_csv(LOCAL / "public_0_95382.csv")
    local_submit = pd.read_csv(LOCAL / "qwen3vl8b_2epoch_highres" / "submission.csv")
    old = old.set_index("id").loc[reference.id]
    local_submit = local_submit.set_index("id").loc[reference.id]
    print("Test disagreement with reference:",
          "previous best", int((old.answer.to_numpy() != reference.answer.to_numpy()).sum()),
          "local 8B", int((local_submit.answer.to_numpy() != reference.answer.to_numpy()).sum()))

    # Use the 1-epoch high-resolution run: its measured public score (0.94399)
    # was higher than the 2-epoch run (0.94101). The weight is checked below
    # on the intersection that both runs genuinely held out.
    local_test, local_test_p = local_probs(
        LOCAL / "qwen3vl8b_highres_selective" / "test_scores_highres_selective.csv"
    )
    local_test_index = {id_: i for i, id_ in enumerate(local_test.id)}
    assert len(local_test_index) == 6714
    local_test_p = local_test_p[[local_test_index[id_] for id_ in ref_test_ids]]
    check_submission(ref_test_ids, local_test_p, OUT / "local_8b_check.csv")
    expected_local = pd.read_csv(LOCAL / "qwen3vl8b_highres_selective" / "submission.csv")
    actual_local = pd.read_csv(OUT / "local_8b_check.csv")
    assert expected_local.equals(actual_local), "Local probability rows disagree with submitted answers"
    candidate_p = (ref_test_p + 0.25 * local_test_p) / 1.25
    candidate = check_submission(ref_test_ids, candidate_p, OUT / "reference_plus_8b_weight025.csv")
    changed = candidate.answer.to_numpy() != reference.answer.to_numpy()
    print(f"0.25-weight candidate changes {int(changed.sum())} / 6714 reference answers")

    for label, base, high in (
        ("1 epoch base", "qwen3vl8b_candidate", None),
        ("1 epoch highres", "qwen3vl8b_candidate", "qwen3vl8b_highres_selective"),
        ("2 epoch base", "qwen3vl8b_2epoch_candidate", None),
        ("2 epoch highres", "qwen3vl8b_2epoch_candidate", "qwen3vl8b_2epoch_highres"),
    ):
        print(label)
        score_overlap(ref_ids, ref_p, holdout, LOCAL / base / "valid_scores.csv",
                      None if high is None else LOCAL / high / "valid_scores.csv")


if __name__ == "__main__":
    main()
