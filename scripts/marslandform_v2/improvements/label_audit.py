#!/usr/bin/env python3
"""VLM-assisted label audit: find and fix noisy labels using model cross-validation + GroqVLM reasoning.

Pipeline:
1. Cross-validate all 639 images (5-fold) to get unbiased predictions
2. Flag suspicious labels: model disagrees OR low confidence
3. Use GroqVLM to reason about flagged images using MOLA terrain features
4. Output cleaned label set
"""
import json
import os
import sys
import time
import random
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    AttentionMILClassifier,
    FocalLoss,
    MILDataset,
    StratifiedBatchSampler,
    build_scheduler,
    compute_metrics,
    load_embeddings,
    load_labels,
    load_mola_features,
    make_class_weights,
    mil_collate_fn,
    run_epoch_eval,
    set_seed,
)

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"

# MOLA feature names for interpretable audit
MOLA_FEATURE_NAMES = [
    "elevation_mean", "elevation_std", "elevation_min", "elevation_max", "elevation_range",
    "slope_mean", "slope_std", "slope_max",
    "roughness_mean", "roughness_std", "roughness_max",
    "tpi_mean", "tpi_std",
    "aspect_sin_mean", "aspect_cos_mean",
    "latitude", "longitude",
    "elevation_skew", "elevation_kurtosis",
    "slope_skew", "slope_kurtosis",
    "local_relief", "curvature_mean",
]


def train_fold_model(
    train_ids: List[str],
    val_ids: List[str],
    emb_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    cfg: Any,
    device: torch.device,
    fold: int,
    epochs: int = 30,
    patience: int = 10,
) -> AttentionMILClassifier:
    """Train a single fold model quickly for cross-validation."""
    mil_cfg = deepcopy(cfg)
    mil_cfg.epochs = epochs
    mil_cfg.patience = patience
    
    train_ds = MILDataset(train_ids, emb_dict, mola_dict, labels_dict,
                          min_tiles_per_image=mil_cfg.min_tiles_per_image,
                          max_tiles_per_image=mil_cfg.max_tiles_per_image)
    val_ds = MILDataset(val_ids, emb_dict, mola_dict, labels_dict,
                        min_tiles_per_image=mil_cfg.min_tiles_per_image,
                        max_tiles_per_image=mil_cfg.max_tiles_per_image)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=mil_collate_fn)
    
    model = AttentionMILClassifier(mil_cfg).to(device)
    class_weights = make_class_weights((labels_dict[i] for i in train_ids), mil_cfg.num_classes, device)
    criterion = FocalLoss(weight=class_weights, gamma=1.5, label_smoothing=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mil_cfg.lr, weight_decay=mil_cfg.weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)
    
    best_metric = -1.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            tiles = batch["tile_embeddings"].to(device)
            mask = batch["tile_mask"].to(device)
            mola = batch["mola_features"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tiles, mask, mola)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        val_metrics = run_epoch_eval(model, val_loader, device, criterion, False)
        metric = val_metrics["landform_macro_f1"]
        
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"  Fold {fold}: best LF F1={best_metric:.4f} @ {epoch - patience_counter} epochs")
    return model


@torch.no_grad()
def predict_with_model(
    model: AttentionMILClassifier,
    image_ids: List[str],
    emb_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    cfg: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Get predictions for given image IDs."""
    model.eval()
    ds = MILDataset(image_ids, emb_dict, mola_dict, labels_dict,
                    min_tiles_per_image=cfg.min_tiles_per_image,
                    max_tiles_per_image=cfg.max_tiles_per_image)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    
    results = []
    for batch in loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        labels = batch["labels"]
        img_ids = batch["image_ids"]
        
        logits, att = model(tiles, mask, mola)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        for i, img_id in enumerate(img_ids):
            pred_cls = int(np.argmax(probs[i]))
            conf = float(probs[i][pred_cls])
            results.append({
                "image_id": img_id,
                "true_label": int(labels[i]),
                "pred_label": pred_cls,
                "confidence": conf,
                "probabilities": probs[i].tolist(),
            })
    
    return results


def cross_validate_all(
    emb_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    cfg: Any,
    device: torch.device,
    n_folds: int = 5,
) -> List[Dict[str, Any]]:
    """5-fold CV to get unbiased predictions for ALL 639 images."""
    print(f"\n{'='*40}")
    print(f"Cross-validating {len(labels_dict)} images ({n_folds}-fold)")
    print(f"{'='*40}")
    
    all_ids = sorted(labels_dict.keys())
    all_labels = np.array([labels_dict[i] for i in all_ids])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_predictions = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(all_ids, all_labels)):
        fold_test_ids = [all_ids[i] for i in test_idx]
        fold_train_val_ids = [all_ids[i] for i in train_val_idx]
        
        # Split train_val into train and val (80/20)
        fold_labels = [labels_dict[i] for i in fold_train_val_ids]
        from sklearn.model_selection import train_test_split
        fold_train_ids, fold_val_ids = train_test_split(
            fold_train_val_ids, test_size=0.2, random_state=42, stratify=fold_labels
        )
        
        print(f"\nFold {fold+1}/{n_folds}: train={len(fold_train_ids)}, val={len(fold_val_ids)}, test={len(fold_test_ids)}")
        
        model = train_fold_model(
            fold_train_ids, fold_val_ids, emb_dict, mola_dict, labels_dict,
            cfg, device, fold + 1, epochs=30, patience=10
        )
        
        fold_preds = predict_with_model(model, fold_test_ids, emb_dict, mola_dict, labels_dict, cfg, device)
        all_predictions.extend(fold_preds)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_predictions


def groq_audit_image(
    image_id: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    probabilities: List[float],
    mola_features: Dict[str, float],
    api_key: str,
) -> Dict[str, Any]:
    """Use GroqVLM to reason about whether a label is correct."""
    import requests
    
    # Format MOLA features
    mola_str = "\n".join([f"  {k}: {v:.3f}" for k, v in mola_features.items()])
    prob_str = ", ".join([f"{cls}={p:.3f}" for cls, p in zip(CLASS_ORDER, probabilities)])
    
    prompt = f"""You are a Mars geomorphology expert. Analyze this HiRISE image classification:

Image: {image_id}
Current label: {true_label}
Model prediction: {pred_label} (confidence: {confidence:.3f})
Full probabilities: {prob_str}

MOLA terrain features:
{mola_str}

Key definitions:
- LDA (Lobate Debris Apron): Gently sloping, lobate features at base of scarps/massifs. Elevation typically -2000 to 0m, slope 2-8°.
- LVF (Lineated Valley Fill): Linear textures filling valleys between massifs. Often at higher elevations, moderate slopes.
- CCF (Concentric Crater Fill): Concentric ridges/rings inside craters. Distinctive circular patterns.
- GLF (Glacier-Like Form): Tongue-shaped features extending from alcoves. Steep slopes, high relief.
- BACKGROUND: No clear glacial/periglacial landform.

Based on the terrain features and model predictions, answer:
1. Is the current label "{true_label}" likely CORRECT or SUSPICIOUS?
2. If suspicious, what should it be? Pick from: LDA, LVF, CCF, GLF, BACKGROUND, MULTI (if multiple landforms present)
3. Confidence in your assessment (0-1)
4. Brief reasoning (1-2 sentences)

Reply in JSON format:
{{"verdict": "correct" or "suspicious", "suggested_label": "...", "confidence": 0.X, "reasoning": "..."}}"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a Mars geomorphology expert. Reply only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 256,
            },
            timeout=15,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        
        # Parse JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"verdict": "error", "reasoning": f"Could not parse: {content[:200]}"}
    except Exception as e:
        return {"verdict": "error", "reasoning": str(e)}


def run_label_audit():
    print("=" * 60)
    print("VLM-ASSISTED LABEL AUDIT")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    
    # Load data
    print("\nLoading data...")
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_raw = np.load(DATA_ROOT / "mola_features_by_image.npy", allow_pickle=True).item()
    mola_dict = {str(k): np.asarray(v, dtype=np.float32) for k, v in mola_raw.items()}
    labels_raw = json.loads((DATA_ROOT / "labels_simple.json").read_text())
    labels_dict = {k: CLASS_ORDER.index(v) if isinstance(v, str) else v for k, v in labels_raw.items()}
    
    # Filter to images with embeddings
    valid_ids = sorted(set(emb_dict.keys()) & set(mola_dict.keys()) & set(labels_dict.keys()))
    labels_dict = {k: labels_dict[k] for k in valid_ids}
    print(f"Valid images: {len(valid_ids)}")
    print(f"Label distribution: {Counter(labels_dict.values())}")
    
    # Output dir
    out_dir = DATA_ROOT / "label_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Cross-validate to get unbiased predictions
    cv_cache = out_dir / "cv_predictions.json"
    if cv_cache.exists():
        print(f"\nLoading cached CV predictions from {cv_cache}")
        all_preds = json.loads(cv_cache.read_text())
    else:
        all_preds = cross_validate_all(emb_dict, mola_dict, labels_dict, mil_cfg, device)
        cv_cache.write_text(json.dumps(all_preds, indent=2))
        print(f"\nSaved CV predictions to {cv_cache}")
    
    # Step 2: Analyze predictions
    print(f"\n{'='*40}")
    print(f"Analyzing {len(all_preds)} predictions")
    print(f"{'='*40}")
    
    # Overall CV metrics
    y_true = [p["true_label"] for p in all_preds]
    y_pred = [p["pred_label"] for p in all_preds]
    cv_metrics = compute_metrics(y_true, y_pred, num_classes=5)
    print(f"\nCV Macro F1: {cv_metrics['macro_f1_all']:.4f}")
    print(f"CV Landform F1: {cv_metrics['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"  {cls}: P={cv_metrics['precision'][i]:.3f} R={cv_metrics['recall'][i]:.3f} F1={cv_metrics['f1'][i]:.3f} (n={cv_metrics['support'][i]})")
    
    # Flag suspicious labels
    suspicious = []
    confident_correct = []
    multi_label_candidates = []
    
    for p in all_preds:
        true_cls = CLASS_ORDER[p["true_label"]]
        pred_cls = CLASS_ORDER[p["pred_label"]]
        conf = p["confidence"]
        probs = p["probabilities"]
        
        # Suspicious: model disagrees AND is confident about it
        is_disagreement = p["true_label"] != p["pred_label"]
        is_confident_wrong = is_disagreement and conf > 0.4
        is_low_conf = conf < 0.35
        
        # Multi-label: two classes both have > 0.25 probability
        high_probs = [(CLASS_ORDER[i], probs[i]) for i in range(4) if probs[i] > 0.25]  # exclude BG
        is_multi = len(high_probs) >= 2
        
        if is_confident_wrong or is_low_conf:
            suspicious.append({
                **p,
                "true_name": true_cls,
                "pred_name": pred_cls,
                "reason": "confident_wrong" if is_confident_wrong else "low_confidence",
            })
        else:
            confident_correct.append(p)
        
        if is_multi:
            multi_label_candidates.append({
                "image_id": p["image_id"],
                "true_label": true_cls,
                "high_prob_classes": high_probs,
            })
    
    print(f"\n--- Flagged ---")
    print(f"  Suspicious labels: {len(suspicious)} ({len(suspicious)/len(all_preds)*100:.1f}%)")
    print(f"  Confident correct: {len(confident_correct)} ({len(confident_correct)/len(all_preds)*100:.1f}%)")
    print(f"  Multi-label candidates: {len(multi_label_candidates)} ({len(multi_label_candidates)/len(all_preds)*100:.1f}%)")
    
    # Group suspicious by reason
    by_reason = Counter(s["reason"] for s in suspicious)
    print(f"  By reason: {dict(by_reason)}")
    
    # Group by confusion pair
    confusion_pairs = Counter(f'{s["true_name"]}->{s["pred_name"]}' for s in suspicious)
    print(f"\n  Top confusion pairs:")
    for pair, count in confusion_pairs.most_common(10):
        print(f"    {pair}: {count}")
    
    # Step 3: Use GroqVLM for suspicious images
    groq_key = os.getenv("GROQ_API_KEY")
    vlm_results = {}
    
    if groq_key:
        print(f"\n{'='*40}")
        print(f"Querying GroqVLM for {min(len(suspicious), 200)} suspicious images...")
        print(f"{'='*40}")
        
        # Limit to most suspicious (confident wrong first, then low confidence)
        suspicious_sorted = sorted(suspicious, key=lambda s: (
            0 if s["reason"] == "confident_wrong" else 1,
            -s["confidence"]
        ))[:200]  # Cap at 200 API calls
        
        for idx, s in enumerate(suspicious_sorted):
            img_id = s["image_id"]
            
            # Get MOLA features
            mola_feats = {}
            if img_id in mola_dict:
                mola_arr = mola_dict[img_id]
                for i, name in enumerate(MOLA_FEATURE_NAMES[:min(len(mola_arr), len(MOLA_FEATURE_NAMES))]):
                    mola_feats[name] = float(mola_arr[i])
            
            result = groq_audit_image(
                image_id=img_id,
                true_label=s["true_name"],
                pred_label=s["pred_name"],
                confidence=s["confidence"],
                probabilities=s["probabilities"],
                mola_features=mola_feats,
                api_key=groq_key,
            )
            vlm_results[img_id] = {
                "true_label": s["true_name"],
                "model_pred": s["pred_name"],
                "model_conf": s["confidence"],
                "vlm_verdict": result,
            }
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx+1}/{len(suspicious_sorted)} images")
                time.sleep(1)  # Rate limit
            
            # Tiny delay between calls
            time.sleep(0.2)
        
        print(f"\nVLM audit complete: {len(vlm_results)} images reviewed")
        
        # Analyze VLM verdicts
        verdicts = Counter(v["vlm_verdict"].get("verdict", "error") for v in vlm_results.values())
        print(f"  Verdicts: {dict(verdicts)}")
        
        # Build correction suggestions
        corrections = {}
        for img_id, v in vlm_results.items():
            vlm = v["vlm_verdict"]
            if vlm.get("verdict") == "suspicious" and vlm.get("suggested_label"):
                suggested = vlm["suggested_label"].upper()
                if suggested in CLASS_ORDER or suggested == "MULTI":
                    corrections[img_id] = {
                        "old_label": v["true_label"],
                        "suggested_label": suggested,
                        "vlm_confidence": vlm.get("confidence", 0),
                        "reasoning": vlm.get("reasoning", ""),
                    }
        
        print(f"\n  Corrections suggested: {len(corrections)}")
        if corrections:
            correction_types = Counter(c["old_label"] + "->" + c["suggested_label"] for c in corrections.values())
            for ct, count in correction_types.most_common(10):
                print(f"    {ct}: {count}")
    else:
        print("\n  GROQ_API_KEY not set — skipping VLM audit (using model-only heuristics)")
        
        # Heuristic corrections without VLM
        corrections = {}
        for s in suspicious:
            if s["reason"] == "confident_wrong" and s["confidence"] > 0.55:
                corrections[s["image_id"]] = {
                    "old_label": s["true_name"],
                    "suggested_label": s["pred_name"],
                    "vlm_confidence": s["confidence"],
                    "reasoning": f"Model strongly disagrees (conf={s['confidence']:.2f})",
                }
    
    # Step 4: Build cleaned label set
    print(f"\n{'='*40}")
    print("Building cleaned label set")
    print(f"{'='*40}")
    
    cleaned_labels = dict(labels_raw)  # Start with original string labels
    changes_applied = 0
    multi_labels = {}
    
    for img_id, corr in corrections.items():
        suggested = corr["suggested_label"]
        vlm_conf = corr.get("vlm_confidence", 0)
        
        if suggested == "MULTI":
            multi_labels[img_id] = corr
            continue
        
        if suggested in CLASS_ORDER and vlm_conf >= 0.6:
            old = cleaned_labels.get(img_id, "?")
            cleaned_labels[img_id] = suggested
            changes_applied += 1
    
    print(f"  Changes applied: {changes_applied}")
    print(f"  Multi-label flagged: {len(multi_labels)}")
    
    # Save cleaned labels
    (out_dir / "labels_cleaned.json").write_text(json.dumps(cleaned_labels, indent=2))
    
    # Save audit report
    report = {
        "total_images": len(all_preds),
        "cv_landform_f1": cv_metrics["landform_macro_f1"],
        "cv_macro_f1": cv_metrics["macro_f1_all"],
        "suspicious_count": len(suspicious),
        "confident_correct_count": len(confident_correct),
        "multi_label_candidates": len(multi_label_candidates),
        "vlm_reviewed": len(vlm_results),
        "corrections_suggested": len(corrections),
        "corrections_applied": changes_applied,
        "multi_label_flagged": len(multi_labels),
        "corrections": corrections,
        "multi_labels": multi_labels,
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))
    
    # Save multi-label candidates for Phase 3
    (out_dir / "multi_label_candidates.json").write_text(json.dumps(multi_label_candidates, indent=2))
    
    # Save confident correct labels (clean subset for semi-supervised)
    clean_ids = [p["image_id"] for p in confident_correct]
    (out_dir / "clean_subset_ids.json").write_text(json.dumps(clean_ids, indent=2))
    
    print(f"\nSaved to {out_dir}/:")
    print(f"  labels_cleaned.json ({len(cleaned_labels)} images)")
    print(f"  audit_report.json")
    print(f"  multi_label_candidates.json ({len(multi_label_candidates)} candidates)")
    print(f"  clean_subset_ids.json ({len(clean_ids)} clean images)")
    
    # Distribution comparison
    print(f"\n--- Label Distribution Change ---")
    old_dist = Counter(labels_raw.values())
    new_dist = Counter(cleaned_labels.values())
    for cls in CLASS_ORDER:
        old_n = old_dist.get(cls, 0)
        new_n = new_dist.get(cls, 0)
        delta = new_n - old_n
        print(f"  {cls}: {old_n} -> {new_n} ({delta:+d})")
    
    return report


if __name__ == "__main__":
    run_label_audit()
