"""
Task 2: Fine-tune BLIP-2 on DoMars16k with generated captions.
Uses LoRA (PEFT) to fine-tune only the Q-Former + projection layers.
Input: image + "Describe the geological features in this Mars image."
Output: natural language caption about the landform.
"""
import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image

from mars_captions import LABEL_NAMES, CAPTION_TEMPLATES, get_caption

# ── Config ──────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "Salesforce/blip2-opt-2.7b"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "blip2_mars_lora")
NUM_EPOCHS = 10
BATCH_SIZE = 2  # BLIP-2 is large, keep small
LEARNING_RATE = 2e-4  # LoRA-friendly LR
MAX_LENGTH = 80

# Prompts for training (model sees these as input)
INPUT_PROMPTS = [
    "Describe the geological features visible in this Mars satellite image.",
    "What surface patterns and landforms are present in this Mars image?",
    "Analyze the terrain visible in this Mars surface tile.",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading DoMars16k...")
    ds = load_dataset("gremlin97/domars16k")
    train_ds = ds["train"]
    val_ds = ds["val"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"Loading BLIP-2 processor...")
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)

    print(f"Loading BLIP-2 model...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_CHECKPOINT, torch_dtype=dtype,
    )

    # ── Apply LoRA ──────────────────────────────────────────────────────
    # Target the language model's attention layers for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # OPT attention layers
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess(examples):
        images = []
        input_texts = []
        target_texts = []

        for img, label_idx in zip(examples["image"], examples["label"]):
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

            # Random prompt
            prompt = random.choice(INPUT_PROMPTS)
            input_texts.append(prompt)

            # Random caption variant for the label
            label_name = LABEL_NAMES[label_idx]
            variant = random.randint(0, len(CAPTION_TEMPLATES[label_name]) - 1)
            caption = get_caption(label_name, variant)
            target_texts.append(caption)

        # Process images + input prompts
        encoding = processor(
            images=images,
            text=input_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Process target captions as labels
        labels = processor.tokenizer(
            target_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).input_ids

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

    print("Preprocessing train set...")
    train_processed = train_ds.map(
        preprocess, batched=True, batch_size=16,
        remove_columns=train_ds.column_names,
    )
    print("Preprocessing val set...")
    val_processed = val_ds.map(
        preprocess, batched=True, batch_size=16,
        remove_columns=val_ds.column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device == "cuda"),
        gradient_accumulation_steps=4,  # effective batch = 8
        warmup_ratio=0.05,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save LoRA adapter
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"LoRA adapter saved to {final_dir}")

    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
