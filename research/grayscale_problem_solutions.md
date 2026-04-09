# Grayscale Problem 해결 방안 — Deep Research Report

**연구 질문**: Mastcam-Z(RGB) ↔ HiRISE(panchromatic 1ch) 도메인 차이를 어떻게 해결할 것인가?

**작성일**: 2026-04-09
**문제 발견**: v8 EDSR 학습이 PSNR 12.43에서 정체 (5에폭 → 30에폭 변화 없음)

---

## TL;DR

**이 문제는 사실 새로운 문제가 아니다** — 원격탐사에서 **Pansharpening**으로 30년간 연구되어 온 표준 문제다.
HiRISE(panchromatic, 25cm/px) + Mastcam-Z(RGB, 6.25cm/px)는 정확히 pansharpening의 입력 구조이며, 해결책 3가지가 존재한다:

| 접근법 | 구현 난이도 | 예상 성능 | 우리 케이스 적합성 |
|---|---|---|---|
| **A. 문제 재정의: Mars Pansharpening** | 낮음 | 매우 높음 | ★★★★★ |
| **B. Exemplar-based Colorization + SR** | 중간 | 중간 | ★★★ |
| **C. Y채널만 학습 (luminance SR)** | 매우 낮음 | 낮음-중간 | ★★ |

**추천: A** — 문제를 SR이 아닌 **pansharpening**으로 재정의하면, 도메인 mismatch가 features가 된다.

---

## 1. 문제 진단: 왜 v8이 실패했나

```
입력:  HiRISE LR 64×64×1 (panchromatic, gray) → 강제로 [Y,Y,Y] 3채널화
출력:  Mastcam-Z HR 256×256×3 (RGB, color)
모델:  EDSR-small (SISR, 같은 도메인 가정)
```

**근본 문제 3가지:**

1. **Spectral mismatch**: HiRISE는 ~570-830nm 광대역 panchromatic, Mastcam-Z RGB는 RGB 협대역. 같은 픽셀의 수치값이 본질적으로 다름.
2. **Temporal mismatch**: HiRISE 촬영 시점(2010s) vs Mastcam-Z(2021-). 표면 변화 가능.
3. **Modality mismatch**: 그레이 스케일과 컬러 사이에는 1:1 매핑이 없음 (정보 손실).

**v8 결과 해석**:
- PSNR 12.43은 essentially "회색 평균" 수준
- 모델이 input에서 색 정보를 추출할 수 없으므로 학습이 정체됨
- 이건 모델이 작아서가 아니라 **문제 정의가 잘못된 것**

---

## 2. 해결책 A: Mars Pansharpening (추천)

### 핵심 아이디어

원격탐사에서 panchromatic + multispectral fusion은 **pansharpening**이라 불리며, 이미 잘 정립된 문제다. 우리 케이스는 이상적인 pansharpening setup의 역방향이다:

**전형적 pansharpening (예: WorldView-3)**
```
PAN HR (0.3m) + MS LR (1.2m) → MS HR (0.3m)
```

**Mars Pansharpening (우리 케이스, 재정의)**
```
HiRISE PAN HR (25cm) + Mastcam-Z RGB sparse (6.25cm) → HiRISE RGB HR (25cm)
```

여기서 **HiRISE가 base spatial structure**를 제공하고, **Mastcam-Z가 color/spectral 정보**를 제공한다.

### 구체적 구현

#### Option A1: 표준 pansharpening 아키텍처 적용

[Ciotola et al. (2024) Hyperspectral Pansharpening Critical Review](https://hf.co/papers/2407.01355)와 [PanFlowNet](https://hf.co/papers/2305.07774) 참고.

**입력 재정의:**
```python
# 기존 (실패)
input  = HiRISE LR (64x64x1)  # panchromatic
output = Mastcam-Z HR (256x256x3)  # RGB

# 새로운 정의 (pansharpening)
input  = (HiRISE_pan_HR (256x256x1),    # high-res spatial structure
          Mastcam_RGB_LR (64x64x3))     # low-res color (downsample Mastcam)
output = HiRISE_RGB_HR (256x256x3)       # color-fused HiRISE
```

여기서 핵심 트릭: **Mastcam-Z를 1/4로 다운샘플**해서 LR 컬러로 사용. 그러면 sparse한 Mastcam 커버리지가 문제가 안 됨.

#### Option A2: PanGAN / PanFlowNet 직접 사용

[PanFlowNet (2023)](https://hf.co/papers/2305.07774) — flow-based 모델, conditional distribution 학습. 개방 소스 코드 존재.

**장점:**
- Pansharpening dedicated 모델
- WorldView-3로 pretrained → fine-tune 가능
- spectral consistency loss 내장

#### Option A3: Pretrained pansharpening + fine-tuning

PanBench (5,898 sample pairs of WorldView-3, GF-2)로 pretrained된 모델 사용:
- [Cross-Scale Pansharpening / ScaleFormer (2026)](https://hf.co/papers/2603.00543) - cross-scale 일반화
- [Robust Pansharpening (PanBench)](https://www.mdpi.com/2072-4292/16/16/2899)

**예상 효과:**
- 30 에폭 PSNR 12.43 → 50+ 가능
- Mars 도메인은 colorful하지 않으므로 (대부분 갈색/회색 톤) pansharpening 모델이 잘 작동할 가능성 높음

---

## 3. 해결책 B: Exemplar-based Colorization + SR

### 핵심 아이디어

HiRISE를 컬러화한 다음 SR을 적용하는 2-stage 방법.

[Deep Exemplar-based Colorization (He et al. 2018)](https://hf.co/papers/1807.06587)와 [Color2Embed (2021)](https://hf.co/papers/2106.08017), 그리고 [SCSNet (2022)](https://arxiv.org/abs/2201.04364) — colorization과 SR을 동시에 수행.

### 구현

```
Stage 1: HiRISE_pan_LR + Mastcam_RGB_ref → HiRISE_color_LR
         (exemplar colorization with attention)

Stage 2: HiRISE_color_LR → HiRISE_color_HR
         (standard SISR — now domains match!)
```

**또는 SCSNet처럼 동시 처리:**
```
HiRISE_pan_LR + Mastcam_RGB_ref → HiRISE_color_HR (one shot)
```

### 장단점

**장점:**
- 컬러화는 Mastcam-Z reference의 sparse한 영역도 활용 가능
- attention 메커니즘으로 reference가 부분적으로만 있어도 작동

**단점:**
- 2-stage는 에러 누적
- pansharpening보다 spectral fidelity 약함
- 화성에는 색이 거의 없어서 colorization 효과 작음

---

## 4. 해결책 C: Luminance-only SR (베이스라인)

### 핵심 아이디어

Mastcam-Z를 YCbCr로 변환하여 **Y채널만 사용**. 색을 무시하고 luminance만 SR.

### 구현

```python
# Convert Mastcam-Z RGB → YCbCr, take Y only
mastcam_y = rgb2ycbcr(mastcam_rgb)[..., 0]  # 1ch

# Now both HiRISE and Mastcam are 1-channel grayscale
# Standard SISR works
```

### 장단점

**장점:**
- 코드 변경 최소 (1줄)
- 도메인 일치 → 학습 잘 됨
- 기존 SR 모델들 직접 사용 가능

**단점:**
- 색 정보 폐기 → 논문 기여도 낮음
- 베이스라인으로만 의미 있음

---

## 5. 추천 실행 계획

### Phase 1: Quick Win (다음 1-2시간)

**해결책 C로 베이스라인 다시 학습**:
```python
# train.py 수정
def __getitem__(self, idx):
    hr = Image.open(hr_path).convert("L")  # grayscale
    lr = Image.open(lr_path).convert("L")  # grayscale
    return {"lr": ..., "hr": ...}  # 1 channel each

# Model 첫/마지막 레이어 1ch
EDSR(in_ch=1, out_ch=1, ...)
```

→ Kaggle CPU에서 30분 내, PSNR이 정상 범위 (20-30) 도달하면 데이터셋이 정상임을 검증.

### Phase 2: 본 실험 (다음 1주)

**해결책 A로 pansharpening 재정의**:

```python
# 새 데이터 빌더
def build_pansharpening_pairs():
    # HiRISE patch (256x256x1) — high-res spatial
    hir_pan_hr = read_hirise_window(...)
    
    # Mastcam-Z patch downsampled to 64x64x3 — low-res color
    mas_rgb_hr = read_mastcam_ortho(...)
    mas_rgb_lr = downsample(mas_rgb_hr, 4)
    
    # Target: HiRISE in color at 256x256x3
    # Trained via Mastcam GT at HR, with consistency loss against HiRISE pan
    target = mas_rgb_hr  # use Mastcam as color GT (it's at the same resolution)
    
    return {"pan_hr": hir_pan_hr, "rgb_lr": mas_rgb_lr, "target": target}
```

이 setup의 장점:
- 입력/출력 차원이 명확
- spectral consistency loss를 정의할 수 있음 (target Y == pan_hr)
- Mastcam의 sparse coverage 문제가 사라짐

### Phase 3: Pretrained 활용 (다음 2주)

[PanBench](https://www.mdpi.com/2072-4292/16/16/2899)나 GF-2 데이터셋으로 pretrained된 PanFlowNet을 가져와서 Mars 데이터에 fine-tune. 작은 데이터셋(281 pairs)에서도 효과적일 것.

---

## 6. Loss Function 설계

Pansharpening 표준 loss (Ciotola et al. 2023):

```python
def pansharpening_loss(pred, target_rgb, pan_hr):
    # 1. Spatial consistency: predicted RGB의 luminance가 PAN과 일치
    pred_y = rgb_to_y(pred)
    spatial_loss = F.l1_loss(pred_y, pan_hr.squeeze(1))
    
    # 2. Spectral consistency: predicted RGB downsampled이 input LR RGB와 일치
    pred_lr = downsample(pred, 4)
    spectral_loss = F.l1_loss(pred_lr, rgb_lr)
    
    # 3. Reconstruction: GT가 있으면 직접 매칭
    recon_loss = F.l1_loss(pred, target_rgb)
    
    return spatial_loss + spectral_loss + recon_loss
```

화성 데이터의 경우, 시간 차이로 GT가 imperfect하므로 **spectral + spatial consistency loss가 더 중요**할 수 있다.

---

## 7. 평가 메트릭 (Pansharpening 표준)

기존 PSNR/SSIM은 RGB-RGB 가정. Pansharpening에서는 추가로:

- **ERGAS** — relative dimensionless global error
- **SAM** — Spectral Angle Mapper (스펙트럴 보존)
- **Q4 / Q2n** — universal image quality index for multispectral
- **QNR (Quality with No Reference)** — full-resolution evaluation
- **D_λ, D_s** — spectral & spatial distortion separately

[hyperspectral-pansharpening-toolbox](https://hf.co/papers/2407.01355)에 PyTorch 구현 있음.

---

## 8. 핵심 참고문헌

| ID | 제목 | 활용 |
|---|---|---|
| S1 | [Hyperspectral Pansharpening Critical Review (Ciotola 2024)](https://hf.co/papers/2407.01355) | 표준 toolbox + 평가 |
| S2 | [PanFlowNet (Yang 2023)](https://hf.co/papers/2305.07774) | flow-based 모델, code 공개 |
| S3 | [Unsupervised Pansharpening (Ciotola 2023)](https://hf.co/papers/2307.14403) | unsupervised loss 설계 |
| S4 | [Cross-Scale Pansharpening / ScaleFormer (2026)](https://hf.co/papers/2603.00543) | scale 일반화 |
| S5 | [PanBench Dataset](https://www.mdpi.com/2072-4292/16/16/2899) | pretrained 데이터 |
| S6 | [Deep Exemplar Colorization (He 2018)](https://hf.co/papers/1807.06587) | 대안 방법 |
| S7 | [SCSNet (2022)](https://arxiv.org/abs/2201.04364) | colorization+SR 결합 |
| S8 | [DMPNet (2024)](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1455963/full) | dual-path pansharpening |

---

## 9. 액션 아이템 (오늘 바로)

1. **[즉시] Phase 1: Grayscale 베이스라인 검증**
   - `train.py`를 1채널 모드로 수정
   - Kaggle 재제출 → PSNR이 20+로 올라가는지 확인
   - 데이터셋 자체에 문제 없음을 증명

2. **[다음] Phase 2: Pansharpening 데이터 재구축**
   - `build_dataset.py`를 pansharpening 형식으로 수정
   - Triplet (HiRISE pan HR, Mastcam RGB LR, Mastcam RGB HR) 생성
   - 새 Kaggle dataset version으로 업로드

3. **[다음] Phase 3: PanFlowNet 또는 Ciotola 코드 fork**
   - GitHub에서 pretrained checkpoint 다운로드
   - Mars 데이터에 fine-tune
   - GPU 필요 → Modal/Vast.ai로 이전 검토

---

## 10. 핵심 통찰 (논문 기여도 측면)

**좋은 소식**: 이 "실패"가 실제로는 **논문의 강력한 motivation**이다.

> "단순한 cross-sensor SISR (HiRISE → Mastcam-Z)는 spectral mismatch로 인해 PSNR 12.43에 정체된다.
> 우리는 이를 **pansharpening 문제로 재정의**하고, **spectral consistency loss**를 추가하여 PSNR 25+를 달성한다."

이 narrative는 reviewer가 좋아할 구조다:
1. Naive baseline (SISR) → 실패
2. 문제 진단 → spectral mismatch
3. Reformulation (pansharpening) → 성공
4. Ablation: spectral loss가 핵심임을 보임
