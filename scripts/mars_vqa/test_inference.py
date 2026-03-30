"""
Test inference: Compare ViLT (classification) vs BLIP-2 LoRA (caption generation)
on HiRISE tiles from the MarsLab pipeline.
"""
import sys
import os
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from analysis.hirise_landforms.preprocessing import tile_image
from mars_captions import LABEL_NAMES, LABEL_FULL_NAMES


def test_vilt(tiles: list[Image.Image], model_path: str):
    """Run ViLT classification VQA on tiles."""
    from transformers import ViltProcessor, ViltForQuestionAnswering

    processor = ViltProcessor.from_pretrained(model_path)
    model = ViltForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    question = "What type of Mars landform is this?"
    print("\n=== ViLT Classification VQA ===")

    for i, tile in enumerate(tiles):
        rgb = tile.convert("RGB")
        inputs = processor(rgb, question, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)

        results = []
        for idx, prob in zip(top3.indices, top3.values):
            label = model.config.id2label[idx.item()]
            full = LABEL_FULL_NAMES.get(label, label)
            results.append(f"{full} ({prob:.1%})")

        print(f"  Tile {i:2d}: {' | '.join(results)}")


def test_blip2(tiles: list[Image.Image], model_path: str):
    """Run BLIP-2 LoRA captioning on tiles."""
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    from peft import PeftModel

    base_model_id = "Salesforce/blip2-opt-2.7b"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_path)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    prompt = "Describe the geological features visible in this Mars satellite image."
    print("\n=== BLIP-2 LoRA Caption Generation ===")

    for i, tile in enumerate(tiles):
        rgb = tile.convert("RGB")
        inputs = processor(rgb, text=prompt, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=80)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"  Tile {i:2d}: {text}")


def main():
    script_dir = os.path.dirname(__file__)
    vilt_path = os.path.join(script_dir, "output", "vilt_mars", "final")
    blip2_path = os.path.join(script_dir, "output", "blip2_mars_lora", "final")

    # Load HiRISE image and tile it
    hirise_path = os.path.join(script_dir, "..", "..", "backend", "hirise_quickview", "ESP_011323_2265.jpg")
    if not os.path.exists(hirise_path):
        print(f"HiRISE image not found: {hirise_path}")
        return

    image = Image.open(hirise_path).convert("RGB")
    all_tiles = tile_image(image, tile_size=224, min_content=0.3)
    print(f"Total tiles: {len(all_tiles)}")

    # Sample 6 tiles from different regions
    n = len(all_tiles)
    sample_idx = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    tiles = [all_tiles[i][2] for i in sample_idx]
    coords = [(all_tiles[i][0], all_tiles[i][1]) for i in sample_idx]
    print(f"Testing on {len(tiles)} tiles: {coords}")

    if os.path.exists(vilt_path):
        test_vilt(tiles, vilt_path)
    else:
        print(f"\nViLT model not found at {vilt_path} — train first with train_vilt.py")

    if os.path.exists(blip2_path):
        test_blip2(tiles, blip2_path)
    else:
        print(f"\nBLIP-2 LoRA not found at {blip2_path} — train first with train_blip2_caption.py")


if __name__ == "__main__":
    main()
