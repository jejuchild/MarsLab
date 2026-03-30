"""
Task 1: Fine-tune ViLT on DoMars16k for Mars landform classification VQA.
Input: image + "What type of Mars landform is this?" → 15-class classification
"""
import os
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    ViltProcessor,
    ViltForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from PIL import Image

from mars_captions import LABEL_NAMES, LABEL_FULL_NAMES

# ── Config ──────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "dandelin/vilt-b32-mlm"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "vilt_mars")
NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# ── Label mappings ──────────────────────────────────────────────────────
id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)

# ── Questions (randomly picked per sample for diversity) ────────────────
QUESTIONS = [
    "What type of Mars landform is this?",
    "What geological feature is visible?",
    "Classify this Mars surface terrain.",
    "What surface pattern does this image show?",
    "Identify the landform in this Mars image.",
]


def main():
    print(f"Loading DoMars16k dataset...")
    ds = load_dataset("gremlin97/domars16k")
    train_ds = ds["train"]
    val_ds = ds["val"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"Loading ViLT processor from {MODEL_CHECKPOINT}...")
    processor = ViltProcessor.from_pretrained(MODEL_CHECKPOINT)

    def preprocess(examples):
        images = []
        texts = []
        targets = []

        for img, label_idx in zip(examples["image"], examples["label"]):
            # Convert grayscale to RGB (ViLT expects RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

            # Random question for diversity
            question = random.choice(QUESTIONS)
            texts.append(question)

            # Soft label: one-hot with the correct class
            target = torch.zeros(NUM_LABELS)
            target[label_idx] = 1.0
            targets.append(target)

        encoding = processor(
            images, texts, padding="max_length", truncation=True, return_tensors="pt"
        )
        for k, v in encoding.items():
            encoding[k] = v.squeeze() if v.dim() > 2 else v

        encoding["labels"] = targets
        return encoding

    print("Preprocessing train set...")
    train_processed = train_ds.map(
        preprocess, batched=True, batch_size=32,
        remove_columns=train_ds.column_names,
    )
    print("Preprocessing val set...")
    val_processed = val_ds.map(
        preprocess, batched=True, batch_size=32,
        remove_columns=val_ds.column_names,
    )

    print(f"Loading ViLT model with {NUM_LABELS} labels...")
    model = ViltForQuestionAnswering.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=20,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=train_processed,
        eval_dataset=val_processed,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    # Quick eval
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
