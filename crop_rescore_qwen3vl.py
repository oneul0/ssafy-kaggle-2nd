"""Score selected train or test images using five image views."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

import qwen3vl_5060ti as code


def boxes(w, h):
    return {
        "full": (0, 0, w, h),
        "left": (0, 0, int(w * 0.72), h),
        "right": (int(w * 0.28), 0, w, h),
        "top": (0, 0, w, int(h * 0.72)),
        "bottom": (0, int(h * 0.28), w, h),
    }


@torch.inference_mode()
def run_one(row, model, processor, token_ids, data_root, max_tokens):
    path = data_root / row.path
    with Image.open(path) as src:
        src = src.convert("RGB")
        outputs = {}
        for name, box in boxes(*src.size).items():
            image = src.crop(box)
            chat = code.messages(row.to_dict(), image=image)
            inputs = processor.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits[0, -1].index_select(0, torch.tensor(token_ids, device=device))
            outputs[name] = torch.log_softmax(logits.float(), dim=0).cpu().numpy()
    # Mean probability is less sensitive to one bad crop than max pooling.
    import numpy as np
    values = np.stack(list(outputs.values()))
    probs = np.exp(values)
    probs /= probs.sum(axis=1, keepdims=True)
    mean_prob = probs.mean(axis=0)
    pred = code.CHOICES[int(mean_prob.argmax())]
    best_name = list(outputs)[int(np.stack(list(outputs.values())).max(axis=1).argmax())]
    return pred, mean_prob, best_name, outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--adapter", type=Path, default=Path("output/qwen3vl8b_refrecipe_t768/adapter"))
    ap.add_argument("--base", type=Path, default=Path("downloads/models/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--review", type=Path, default=Path("output/qwen3vl8b_refrecipe_t768/holdout_errors_for_review.csv"))
    ap.add_argument("--out", type=Path, default=Path("output/qwen3vl8b_refrecipe_t768/crop_rescore.csv"))
    ap.add_argument("--split", choices=("train", "test"), default="train")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    # The adapter was trained with the repository prompt.
    code.REFERENCE_PROMPT = True
    code.SYSTEM = "You are a helpful visual question answering assistant. Answer using exactly one letter among a, b, c, or d. No explanation."
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.image_processor.size["shortest_edge"] = 256 * 28 * 28
    processor.image_processor.size["longest_edge"] = 768 * 28 * 28
    token_ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in code.CHOICES]
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    source = pd.read_csv(args.data / f"{args.split}.csv").set_index("id", drop=False)
    review = pd.read_csv(args.review)
    rows = source.loc[review.id].reset_index(drop=True)
    records = pd.read_csv(args.out).to_dict("records") if args.out.exists() else []
    completed = {r["id"] for r in records}
    for i, row in rows.iterrows():
        if row.id in completed:
            continue
        pred, probs, best, raw = run_one(row, model, processor, token_ids, args.data, 768)
        gold = row.get("answer", "")
        records.append({"id": row.id, "gold": gold, "pred": pred, "best_crop": best,
                        **{f"p_{c}": float(probs[j]) for j, c in enumerate(code.CHOICES)}})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.out, index=False)
        print(f"{i+1}/{len(rows)} {row.id}: {pred} gold={gold} crop={best}", flush=True)
    out = pd.DataFrame(records)
    if args.split == "train":
        print(f"crop accuracy: {(out.gold == out.pred).sum()}/{len(out)}")


if __name__ == "__main__":
    main()
