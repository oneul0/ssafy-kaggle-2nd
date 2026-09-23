"""Create a test submission from an already trained Qwen3-VL LoRA adapter.

Example:
    baseline/Scripts/python.exe qwen3vl_submit_from_adapter.py \
        --adapter output/qwen3vl8b_valid/adapter \
        --out output/qwen3vl8b_valid_submit

This does not train. It uses local model weights and makes no inference API calls.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

import qwen3vl_5060ti as train_code

CHOICES = train_code.CHOICES
score = train_code.score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--base", type=Path, default=Path("downloads/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--min-visual-tokens", type=int, default=64)
    parser.add_argument("--visual-patch-size", type=int, choices=[28, 32], default=32)
    parser.add_argument("--reference-prompt", action="store_true")
    parser.add_argument("--valid-split", type=Path, default=Path("output/qwen3vl8b_valid/valid_split.csv"))
    parser.add_argument("--validate", action="store_true", help="Score the saved validation split before test")
    parser.add_argument("--valid-only", action="store_true", help="Score validation and exit before test")
    parser.add_argument("--question-regex", help="Score only questions matching this regex; output subset scores, not submission")
    parser.add_argument("--uncertain-base-dir", type=Path, help="Rescore the least confident 15 percent from this run")
    parser.add_argument("--uncertain-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if args.reference_prompt:
        train_code.REFERENCE_PROMPT = True
        train_code.SYSTEM = ("You are a helpful visual question answering assistant. "
                             "Answer using exactly one letter among a, b, c, or d. No explanation.")
    if args.min_visual_tokens > args.max_visual_tokens:
        raise ValueError("min-visual-tokens exceeds max-visual-tokens")

    if args.uncertain_base_dir and args.question_regex:
        raise ValueError("Choose either uncertainty selection or question regex")
    if args.uncertain_base_dir and not 0 < args.uncertain_fraction <= 1:
        raise ValueError("uncertain-fraction must be in (0, 1]")

    if not (args.adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"Trained adapter missing: {args.adapter}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.valid_only:
        args.validate = True
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free: {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    if free < 9 * 2**30:
        raise RuntimeError("Free at least 9 GiB before loading the inference model")

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.image_processor.size["shortest_edge"] = args.min_visual_tokens * args.visual_patch_size**2
    processor.image_processor.size["longest_edge"] = args.max_visual_tokens * args.visual_patch_size**2
    token_ids = [processor.tokenizer.encode(c, add_special_tokens=False) for c in CHOICES]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError(f"Choice tokenization changed: {token_ids}")

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    args.out.mkdir(parents=True, exist_ok=True)

    def select_uncertain(frame, score_file):
        baseline = pd.read_csv(score_file)
        if not baseline.id.equals(frame.id):
            raise RuntimeError(f"Score IDs do not match source: {score_file}")
        columns = [f"logp_{letter}" for letter in CHOICES]
        baseline["margin"] = baseline[columns].apply(lambda row: sorted(row)[-1] - sorted(row)[-2], axis=1)
        selected_ids = set(baseline.nsmallest(max(1, round(len(frame) * args.uncertain_fraction)), "margin").id)
        return frame.loc[frame.id.isin(selected_ids)].reset_index(drop=True), baseline

    if args.validate:
        split = pd.read_csv(args.valid_split)
        train = pd.read_csv(args.data / "train.csv").set_index("id", drop=False)
        valid = train.loc[split.id].reset_index(drop=True)
        if not valid.id.equals(split.id) or not valid.answer.equals(split.answer):
            raise RuntimeError("Saved validation split does not match train.csv")
        if args.question_regex:
            valid = valid.loc[valid.question.str.contains(args.question_regex, regex=True)].reset_index(drop=True)
        if args.uncertain_base_dir:
            valid, baseline_valid = select_uncertain(valid, args.uncertain_base_dir / "valid_scores.csv")
        score(valid, model, processor, [ids[0] for ids in token_ids], args.data, args.out / "valid_scores.csv")
        if args.uncertain_base_dir:
            highres = pd.read_csv(args.out / "valid_scores.csv").set_index("id")
            combined = baseline_valid.set_index("id")
            old_correct = (combined.gold == combined.pred).sum()
            combined.update(highres)
            new_correct = (combined.gold == combined.pred).sum()
            print(f"Selective validation: {old_correct} -> {new_correct} / {len(combined)}")
            if new_correct <= old_correct:
                print("No validation gain; keeping the original submission candidate.")
                return
    if args.valid_only:
        return
    test = pd.read_csv(args.data / "test.csv").fillna("")
    sample = pd.read_csv(args.data / "sample_submission.csv")
    if len(test) != 6714 or not sample.id.equals(test.id):
        raise RuntimeError("Unexpected test count or submission ID order")
    if args.question_regex:
        test = test.loc[test.question.str.contains(args.question_regex, regex=True)].reset_index(drop=True)
    if args.uncertain_base_dir:
        test, baseline_test = select_uncertain(test, args.uncertain_base_dir / "test_scores.csv")

    scores_path = args.out / "test_scores.csv"
    score(test, model, processor, [ids[0] for ids in token_ids], args.data, scores_path)
    if args.uncertain_base_dir:
        highres = pd.read_csv(scores_path).set_index("id")
        combined = baseline_test.set_index("id")
        combined.update(highres)
        scored = combined.reset_index()
        scored.drop(columns=["margin"]).to_csv(args.out / "test_scores_highres_selective.csv", index=False)
        sample["answer"] = scored.pred
        sample.to_csv(args.out / "submission.csv", index=False)
        print(f"Selective high-resolution submission: {args.out / 'submission.csv'}")
        return
    if args.question_regex:
        print(f"Subset scores saved: {scores_path} ({len(test)} rows)")
        return
    scored = pd.read_csv(scores_path)
    if not scored.id.equals(test.id) or not scored.pred.isin(list(CHOICES)).all():
        raise RuntimeError("Incomplete or invalid test predictions")
    sample["answer"] = scored.pred
    submission_path = args.out / "submission.csv"
    sample.to_csv(submission_path, index=False)
    print(f"Submission: {submission_path} ({len(sample)} predictions)")


if __name__ == "__main__":
    main()
