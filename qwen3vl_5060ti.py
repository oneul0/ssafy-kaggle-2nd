"""Reproducible VQA experiment for a Windows RTX 5060 Ti 16GB.

Uses the existing baseline virtual environment and downloads only model weights.
Run: baseline/Scripts/python.exe qwen3vl_5060ti.py --data dataset --out output/local_valid --mode valid
Then, only after selecting a configuration on validation:
     baseline/Scripts/python.exe qwen3vl_5060ti.py --data dataset --out output/local_final --mode submit
No inference API is used. The Hugging Face Hub only supplies model weights.
"""

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Image as DatasetImage, Sequence
from peft import LoraConfig, prepare_model_for_kbit_training
from PIL import Image, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


CHOICES = "abcd"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SYSTEM = "이미지와 질문을 보고 정답을 고르세요. 이미지 속 작은 글자와 숫자를 주의 깊게 읽으세요. a, b, c, d 중 한 글자만 답하세요."
REFERENCE_PROMPT = False


def normalized_question(value):
    return " ".join(str(value).casefold().split())


def image_hash(path):
    """Small difference hash; used only to keep near-identical images out of validation training."""
    with Image.open(path) as image:
        pixels = list(ImageOps.grayscale(image).resize((9, 8)).getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def avoid_validation_leakage(train, valid, data_root):
    valid_questions = set(valid.question.map(normalized_question))
    valid_hashes = [image_hash(data_root / path) for path in valid.path]
    keep = []
    for row in train.itertuples():
        if normalized_question(row.question) in valid_questions:
            keep.append(False)
            continue
        signature = image_hash(data_root / row.path)
        keep.append(all((signature ^ other).bit_count() > 3 for other in valid_hashes))
    return train.loc[keep].reset_index(drop=True)


def dev_consensus(dev):
    selected = []
    for row in dev.itertuples():
        votes = [str(getattr(row, f"answer{i}")).lower() for i in range(1, 6)]
        votes = [vote for vote in votes if vote in ("a", "b", "c", "d")]
        if len(votes) < 3:
            continue
        counts = Counter(votes)
        answer, count = counts.most_common(1)[0]
        entropy = -sum((n / len(votes)) * math.log(n / len(votes)) for n in counts.values())
        entropy /= math.log(4)
        if count >= 3 and count / len(votes) >= 0.6 and entropy <= 0.5:
            selected.append({key: getattr(row, key) for key in ("id", "path", "question", *CHOICES)} | {"answer": answer})
    return pd.DataFrame(selected)


def make_split(train):
    groups = {}
    for index, question in enumerate(train.question.map(normalized_question)):
        groups.setdefault(question, []).append(index)
    overall = train.answer.value_counts(normalize=True).reindex(list(CHOICES), fill_value=0)
    target_size = round(len(train) * 0.1)
    best = None
    for trial in range(100):
        items = list(groups.values())
        random.Random(42 + trial).shuffle(items)
        held_out = []
        for indices in items:
            if len(held_out) >= target_size:
                break
            held_out.extend(indices)
        proportions = train.iloc[held_out].answer.value_counts(normalize=True).reindex(list(CHOICES), fill_value=0)
        distance = (proportions - overall).abs().sum() + abs(len(held_out) - target_size) / len(train)
        if best is None or distance < best[0]:
            best = (distance, set(held_out))
    fit = train.iloc[[index for index in range(len(train)) if index not in best[1]]]
    valid = train.iloc[sorted(best[1])]
    return fit.reset_index(drop=True), valid.reset_index(drop=True)


def prompt(row):
    if REFERENCE_PROMPT:
        return (
            f"{row['question']}\n(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
        )
    return "질문: {}\n(a) {}\n(b) {}\n(c) {}\n(d) {}\n정답:".format(
        row["question"], row["a"], row["b"], row["c"], row["d"]
    )


def messages(row, image=None, answer=None):
    image_part = {"type": "image"}
    if image is not None:
        image_part["image"] = image
    result = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [image_part, {"type": "text", "text": prompt(row)}]},
    ]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def training_dataset(frame, data_root, shuffle_options=False, seed=42):
    records = []
    for index, row in enumerate(frame.to_dict("records")):
        if shuffle_options:
            old_answer = row["answer"]
            order = list(CHOICES)
            random.Random(seed + index).shuffle(order)
            original_options = {letter: row[letter] for letter in CHOICES}
            row = row.copy()
            for new_letter, old_letter in zip(CHOICES, order):
                row[new_letter] = original_options[old_letter]
            row["answer"] = CHOICES[order.index(old_answer)]
        records.append({
            "images": [str((data_root / row["path"]).resolve())],
            "messages": messages(row, answer=row["answer"]),
        })
    return Dataset.from_list(records).cast_column("images", Sequence(DatasetImage()))


def make_answer_only_collator(base_collator, token_ids):
    allowed = torch.tensor(token_ids)

    def collate(features):
        batch = base_collator(features)
        labels = batch["labels"]
        original = labels.clone()
        labels.fill_(-100)
        for row, feature in enumerate(features):
            end = int(batch["attention_mask"][row].sum())
            positions = torch.where(torch.isin(original[row, :end], allowed))[0]
            if len(positions) == 0 or end - int(positions[-1]) > 4:
                raise RuntimeError("Could not locate the assistant answer token near the sequence end.")
            answer = feature["messages"][-1]["content"][0]["text"]
            expected_id = token_ids[CHOICES.index(answer)]
            if int(original[row, positions[-1]]) != expected_id:
                raise RuntimeError("Last choice token does not match the training answer.")
            labels[row, positions[-1]] = expected_id
        if (labels != -100).sum().item() != len(features):
            raise RuntimeError("Expected exactly one supervised answer token per example.")
        return batch

    return collate


def load_model(max_tokens, min_tokens=64, visual_patch_size=32):
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=Path(MODEL_ID).exists())
    processor.image_processor.size["shortest_edge"] = min_tokens * visual_patch_size**2
    processor.image_processor.size["longest_edge"] = max_tokens * visual_patch_size**2
    processor.tokenizer.padding_side = "right"
    ids = [processor.tokenizer.encode(c, add_special_tokens=False) for c in CHOICES]
    if any(len(item) != 1 for item in ids):
        raise ValueError(f"Choice letters must be single tokens: {ids}")
    token_ids = [item[0] for item in ids]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, module in model.named_modules()
               if "language_model" in name and name.rsplit(".", 1)[-1] in target_names
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language-model LoRA modules found; inspect model.named_modules().")
    print(f"LoRA target modules: {len(targets)}; examples: {targets[:3]}")
    return model, processor, token_ids, targets


@torch.inference_mode()
def score(frame, model, processor, token_ids, data_root, output):
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    answer_ids = torch.tensor(token_ids, device=device)
    columns = ["id", "path", "question", "gold", "pred", "logp_a", "logp_b", "logp_c", "logp_d"]
    correct = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(frame.to_dict("records"), 1):
            with Image.open(data_root / row["path"]) as image:
                chat = messages(row, image=image.convert("RGB"))
                inputs = processor.apply_chat_template(
                    chat, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
            inputs = {key: value.to(device) if torch.is_tensor(value) else value
                      for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits[0, -1].index_select(0, answer_ids)
            logp = torch.log_softmax(logits.float(), dim=0).cpu().tolist()
            predicted = CHOICES[max(range(4), key=logp.__getitem__)]
            gold = row.get("answer", "")
            correct += predicted == gold
            writer.writerow(dict(zip(columns, [row["id"], row["path"], row["question"],
                                               gold, predicted, *logp])))
            if index % 100 == 0:
                print(f"Scored {index}/{len(frame)}", flush=True)
    if "answer" in frame:
        print(f"Validation: {correct}/{len(frame)} = {correct / len(frame):.5f}")


def main():
    global MODEL_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train.csv/test.csv and image folders")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["valid", "submit"], default="valid")
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--min-visual-tokens", type=int, default=64)
    parser.add_argument("--visual-patch-size", type=int, choices=[28, 32], default=32)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--reference-prompt", action="store_true")
    parser.add_argument("--no-dev", action="store_true")
    parser.add_argument("--valid-ids", type=Path, help="CSV with holdout IDs to use instead of a new split")
    parser.add_argument("--skip-duplicate-filter", action="store_true", help="Use with an image-grouped holdout CSV")
    parser.add_argument("--save-steps", type=int, default=0, help="Checkpoint every N optimizer steps; 0 means each epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0, help="Train on this many rows and score 10 rows; never submit")
    parser.add_argument("--resume-from", type=Path, help="Resume optimizer and scheduler from a saved checkpoint")
    parser.add_argument("--model", help="Local model directory; defaults to the downloaded workspace weights")
    args = parser.parse_args()
    global SYSTEM, REFERENCE_PROMPT
    REFERENCE_PROMPT = args.reference_prompt
    if REFERENCE_PROMPT:
        SYSTEM = ("You are a helpful visual question answering assistant. "
                  "Answer using exactly one letter among a, b, c, or d. No explanation.")
    if args.rank <= 0 or args.alpha <= 0 or args.min_visual_tokens > args.max_visual_tokens:
        parser.error("Invalid LoRA rank/alpha or visual-token range")
    local_model = Path(__file__).resolve().parent / "downloads/models/Qwen3-VL-8B-Instruct"
    MODEL_ID = args.model or (str(local_model) if (local_model / "config.json").exists() else MODEL_ID)
    print(f"Model weights: {MODEL_ID}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable NVIDIA GPU is required.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB free")
    if free_bytes < 12 * 2**30:
        raise RuntimeError("Free at least 12 GiB of GPU memory before loading Qwen3-VL-8B.")
    random.seed(42)
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data / "train.csv").fillna("")
    test = pd.read_csv(args.data / "test.csv").fillna("")
    if args.mode == "valid":
        if args.valid_ids:
            ids = pd.read_csv(args.valid_ids).id.astype(str)
            if len(ids) != len(set(ids)) or not set(ids).issubset(set(train.id)):
                raise ValueError("Holdout IDs must be unique and present in train.csv")
            valid = train.set_index("id").loc[ids].reset_index()
            fit = train.loc[~train.id.isin(ids)].reset_index(drop=True)
        else:
            fit, valid = make_split(train)
        valid[["id", "path", "question", "answer"]].to_csv(args.out / "valid_split.csv", index=False)
        if not args.skip_duplicate_filter:
            fit = avoid_validation_leakage(fit, valid, args.data)
    else:
        fit, valid = train, None
    dev = pd.DataFrame()
    if not args.no_dev:
        dev = dev_consensus(pd.read_csv(args.data / "dev.csv").fillna(""))
        if valid is not None:
            dev = avoid_validation_leakage(dev, valid, args.data)
        fit = pd.concat([fit, dev], ignore_index=True)
    print(f"Training rows: {len(fit)} (dev consensus: {len(dev)})")
    if args.smoke:
        fit = fit.sample(n=min(args.smoke, len(fit)), random_state=42).reset_index(drop=True)
        if valid is not None:
            valid = valid.head(10)

    model, processor, token_ids, targets = load_model(
        args.max_visual_tokens, args.min_visual_tokens, args.visual_patch_size
    )
    config = SFTConfig(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_fraction,
        lr_scheduler_type=args.scheduler,
        bf16=True,
        max_length=None,
        assistant_only_loss=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=max(1, args.save_steps),
        save_total_limit=args.save_total_limit,
        logging_steps=20,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=training_dataset(fit, args.data, args.shuffle_options),
        processing_class=processor,
        peft_config=LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                               target_modules=targets, task_type="CAUSAL_LM"),
    )
    trainer.data_collator = make_answer_only_collator(trainer.data_collator, token_ids)
    probe = trainer.data_collator([trainer.train_dataset[0]])
    assert (probe["labels"] != -100).sum().item() == 1
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    trainer.model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "adapter")
    if valid is not None:
        score(valid, trainer.model, processor, token_ids, args.data, args.out / "valid_scores.csv")
    elif not args.smoke:
        score(test, trainer.model, processor, token_ids, args.data, args.out / "test_scores.csv")
        scored = pd.read_csv(args.out / "test_scores.csv")
        sample = pd.read_csv(args.data / "sample_submission.csv")
        if len(sample) != len(test) or not sample.id.equals(test.id):
            raise RuntimeError("sample_submission IDs do not match test.csv")
        sample["answer"] = scored.pred
        sample.to_csv(args.out / "submission.csv", index=False)
        print(f"Saved {args.out / 'submission.csv'}")


if __name__ == "__main__":
    main()
