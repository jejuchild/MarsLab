# MarsRefSR Experiment Plan

## Overview
- **Goal**: Validate MarsRefSR approach and establish benchmark baselines
- **Total GPU Budget**: ~42 hours
- **Dataset**: MarsOrtho (47 sols, ~1,100+ LR-HR pairs from SPICE coregistration)
- **Split**: 40 sols train / 7 sols test (cross-sol generalization)

## Experiment Blocks

### Block 1: SISR Baselines (E1)
**Priority**: MUST | **GPU**: 4h | **Order**: Run first

| Model | Source | x4 Config |
|---|---|---|
| EDSR | `eugenesiow/edsr-base` or official | baseline_edsr |
| SwinIR | JingyunLiang/SwinIR | swinir_classical_sr_x4 |
| HAT | XPixelGroup/HAT | HAT-L_SRx4 |
| Real-ESRGAN | xinntao/Real-ESRGAN | RealESRGAN_x4plus |

**Metrics**: PSNR, SSIM, LPIPS, FID (patch-level)
**Notes**: Pretrained on DIV2K, no fine-tuning first → then fine-tuned on MarsOrtho train set

### Block 2: RefSR Baselines (E2)
**Priority**: MUST | **GPU**: 8h | **Order**: Run in parallel with E1

| Model | Reference Input | Notes |
|---|---|---|
| RRDB + Ref Attention | Mastcam-Z ortho | Simple concatenation baseline |
| MASA-SR | Mastcam-Z ortho | Match & select attention |
| CRefDiff | Mastcam-Z ortho | Diffusion-based, current SOTA |

**Notes**: Need to adapt reference input pipeline for each model

### Block 3: MarsRefSR — Proposed Model (E3)
**Priority**: MUST | **GPU**: 12h | **Order**: Start after E1 confirms data pipeline works

**Architecture**:
```
Input: HiRISE patch (64x64) + Mastcam-Z reference (256x256) + DTM patch (64x64)
                ↓                      ↓                        ↓
          RRDB encoder          Ref feature extractor     DTM encoder (3ch: elev, slope, aspect)
                ↓                      ↓                        ↓
          Cross-attention ←── Reference features ──→ DTM-guided spatial attention
                ↓
          RRDB decoder → SR output (256x256)
```

**Training**:
1. Stage 1: Pretrain encoder/decoder on SEN2NAIP (transfer from Earth RS)
2. Stage 2: Fine-tune on MarsOrtho with DTM branch
3. Loss: L1 + Perceptual (VGG) + GAN (optional)

### Block 4: Ablation Studies (E4)
**Priority**: MUST | **GPU**: 8h

| Ablation | What it tests |
|---|---|
| MarsRefSR w/o DTM | Is DTM guidance helpful? |
| MarsRefSR w/o Ref | Is reference image helpful? (degrades to SISR) |
| SPICE registration vs SIFT feature matching | Is SPICE better? |
| L1 only vs L1+Perceptual vs L1+Perceptual+GAN | Loss function impact |

### Block 5: Generalization (E5)
**Priority**: MUST | **GPU**: 4h

- Train: 40 sols (diverse terrain: bedrock, regolith, sand)
- Test: 7 sols (held out, different terrain types)
- Report per-terrain-type metrics
- Cross-dataset: test on simulated Curiosity Mastcam pairs (if available)

### Block 6: Multi-Frame Comparison (E6)
**Priority**: SHOULD | **GPU**: 4h

- Compare: Single-frame RefSR vs Multi-frame aggregation (MFSR + RefSR)
- Use overlapping frames within same sol for MFSR
- Question: Does multi-frame sub-pixel information help beyond reference image?

### Block 7: Scientific Evaluation (E7)
**Priority**: MUST | **GPU**: 2h

**Geological Preservation Metrics**:
- Edge preservation ratio (Canny edge comparison)
- Texture fidelity (Gram matrix distance in VGG feature space)
- Spectral consistency (if multispectral bands available)
- Geologist visual evaluation (qualitative, 10 random samples)

**Failure Case Analysis**:
- Regions with high temporal change
- Regions with poor SPICE registration
- Regions with insufficient Mastcam-Z coverage

## Run Order (Optimized for Dependencies)

```
Week 1: E1 (SISR baselines) + E2 (RefSR baselines) — parallel
Week 2: E3 (MarsRefSR training) — depends on data pipeline from E1
Week 3: E4 (Ablations) + E5 (Generalization) — parallel, uses E3 checkpoint
Week 4: E6 (Multi-frame) + E7 (Scientific eval) — parallel
```

## Data Preparation Checklist

- [ ] Extract LR-HR patch pairs from ortho_hirise.py output
- [ ] Normalize to consistent format (PNG, 256x256 patches)
- [ ] Generate DTM condition maps (elevation, slope, aspect)
- [ ] Create train/val/test splits by sol
- [ ] Compute dataset statistics (mean, std per channel)
- [ ] Prepare HuggingFace dataset card for public release
