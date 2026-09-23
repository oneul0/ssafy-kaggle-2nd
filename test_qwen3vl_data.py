"""CPU-only checks for the local VQA data split and dev filtering."""

from pathlib import Path

import pandas as pd

from qwen3vl_5060ti import (
    avoid_validation_leakage,
    dev_consensus,
    make_answer_only_collator,
    make_split,
    normalized_question,
    training_dataset,
)


root = Path(__file__).parent / "dataset"
train = pd.read_csv(root / "train.csv").fillna("")
test = pd.read_csv(root / "test.csv").fillna("")
fit, valid = make_split(train)
assert len(fit) + len(valid) == len(train)
assert not set(fit.question.map(normalized_question)) & set(valid.question.map(normalized_question))
assert valid.answer.isin(list("abcd")).all()

dev = dev_consensus(pd.read_csv(root / "dev.csv").fillna(""))
assert len(dev) == 1269, len(dev)
assert dev.set_index("id").loc["dev_0004.jpg", "answer"] == "c"

duplicate = train.loc[train.id == "train_1852.jpg"]
same_image = test.loc[test.id == "test_0480.jpg"]
assert avoid_validation_leakage(duplicate, same_image, root).empty
fit_clean = avoid_validation_leakage(fit, valid, root)
dev_clean = avoid_validation_leakage(dev, valid, root)
assert len(fit_clean) <= len(fit) and len(dev_clean) <= len(dev)

# The installed TRL version expects an `images` list even for one image.
from transformers import AutoProcessor
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

dataset = training_dataset(train.head(1), root)
feature = dataset[0]
assert len(feature["images"]) == 1
for model_dir in ("Qwen2.5-VL-3B-Instruct", "Qwen3-VL-8B-Instruct"):
    processor = AutoProcessor.from_pretrained(root.parent / "downloads/models" / model_dir, local_files_only=True)
    choice_ids = [processor.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in "abcd"]
    collator = DataCollatorForVisionLanguageModeling(processor)
    batch = make_answer_only_collator(collator, choice_ids)([feature])
    assert (batch["labels"] != -100).sum().item() == 1
print(f"OK: {len(fit_clean)} train, {len(valid)} validation, {len(dev_clean)} dev candidates")
