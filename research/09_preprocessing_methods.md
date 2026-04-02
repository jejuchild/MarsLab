# 9. Preprocessing and Radiometric Correction for Cross-Platform Super-Resolution

## Scope

This document reviews preprocessing and radiometric correction methods required before training a Reference-based Super-Resolution (RefSR) model that uses Mastcam-Z ortho-images (cm-scale, oblique-projected) to enhance HiRISE orbital imagery (25 cm/px). Every section identifies the authoritative source, the core method, and practical recommendations for our Mars application.

---

## 9.1 Radiometric Normalization / Cross-Platform Harmonization

### 9.1.1 Problem Statement

Mastcam-Z and HiRISE are fundamentally different instruments: Mastcam-Z uses a Bayer-pattern CCD with 11 narrowband filters (442-1013 nm) and on-board calibration targets, while HiRISE uses a pushbroom TDI CCD with broadband RED (570-830 nm), BG, and NIR channels. Absolute radiometric accuracy differs substantially: Mastcam-Z achieves <5% absolute via calibration targets (Hayes et al. 2021; Kinch et al. 2020), while HiRISE has ~20% absolute accuracy with no onboard calibration source (Delamere et al. 2010). Training pairs must be radiometrically consistent for SR to learn texture transfer rather than brightness mapping.

### 9.1.2 Established Methods

**1. Relative Radiometric Normalization (RRN)**

The standard approach for multi-temporal/multi-sensor fusion. Schott et al. (1988, PE&RS) introduced Pseudo-Invariant Feature (PIF) based normalization: identify temporally stable targets (bright rocks, dark shadows) in both images, fit linear regression (gain + offset) per band. Yang & Lo (2000, PE&RS 66(8):967-980) systematically compared RRN methods and found that PIF-based approaches outperform histogram matching when the scene content differs between sensors.

- **Applicability to Mars**: Mars surface is largely static (no vegetation, no water) -- nearly all surfaces qualify as pseudo-invariant features. Bright bedrock outcrops and dark basaltic sand provide natural radiometric anchors visible in both Mastcam-Z and HiRISE.

**2. Histogram Matching**

Adjusts the cumulative distribution function of the subject image to match the reference. Simple and effective for same-sensor multi-temporal normalization, but problematic for cross-sensor work because it assumes the two sensors observe the same radiometric distribution. Local histogram matching (Kim et al. 2019) divides the image into sub-regions and matches locally, reducing spatial artifacts.

- **Applicability**: Not recommended as the primary method because Mastcam-Z and HiRISE have different spectral responses. However, useful as a post-processing refinement after band-specific linear normalization.

**3. Wallis Filter**

A local adaptive contrast normalization filter that forces local mean and standard deviation to target values. Widely used in photogrammetric image matching (ISPRS Archives XLIII-B3-2022:1217). The transformation is:

```
g(x,y) = [f(x,y) - m_f] * (c * s_target) / (c * s_f + (1-c)/c_f * s_target) + b * m_target + (1-b) * m_f
```

where c controls contrast expansion and b controls brightness forcing.

- **Applicability**: Excellent for pre-matching normalization (ensuring Mastcam-Z and HiRISE patches have similar local contrast for feature matching during co-registration). Not suitable for spectral fidelity preservation.

**4. NASA HLS (Harmonized Landsat-Sentinel) Approach**

The gold standard for cross-sensor harmonization (Claverie et al. 2018, RSE 219:145-161; v2.0 released 2023). The pipeline applies: (a) atmospheric correction to surface reflectance, (b) BRDF normalization to fixed nadir + fixed solar zenith, (c) bandpass adjustment via spectral regression using common reference sensor. The BRDF normalization step alone "effectively reduced the between-sensor reflectance difference" for all spectral bands.

- **Applicability**: This three-step framework is directly transferable to our work:
  1. Atmospheric correction: Mastcam-Z IOF products are already atmospherically corrected via calibration targets. HiRISE needs atmospheric path radiance subtraction.
  2. BRDF normalization: Both images must be corrected to a common viewing/illumination geometry (see Section 9.2).
  3. Bandpass adjustment: Convolve Mastcam-Z narrowband data with HiRISE RED filter response to synthesize a radiometrically compatible band.

**5. Deep Learning-Based Harmonization**

Domain-specific neural harmonization is an emerging approach. For SR specifically, Wang et al. (ICCV 2021) treat real-world SR as a domain adaptation problem, using adversarial feature alignment between synthetic and real degradation domains. The key insight: rather than harmonizing images explicitly, let the SR network learn domain-invariant features through adversarial training.

### 9.1.3 Practical Recommendation for Our Pipeline

```
Step 1: Convert both to I/F (radiance factor)
  - Mastcam-Z: Use IOF products (already provided in PDS4 archive)
  - HiRISE: Apply hical pipeline (ISIS) → I/F conversion using solar distance
Step 2: Spectral band synthesis
  - Convolve Mastcam-Z L0 Bayer RGB or individual L-filters with HiRISE RED
    filter transmittance curve to produce a "synthetic HiRISE RED" channel
  - Use L3 (677 nm), L4 (605 nm), L5 (528 nm) weighted combination
Step 3: Linear PIF-based normalization
  - Select 5-10 pseudo-invariant patches (bright rock, dark sand) visible
    in both co-registered Mastcam-Z ortho and HiRISE
  - Fit per-band linear regression: I/F_mastcam = a * I/F_hirise + b
  - Apply correction to normalize Mastcam-Z to HiRISE radiometric scale
Step 4: Local contrast verification
  - Compare local std-dev histograms; apply Wallis filter only if needed
```

### 9.1.4 Key References

- Schott, J.R., et al. (1988). "Radiometric scene normalization using PIF." RSE 26(1):1-16.
- Yang, X. & Lo, C.P. (2000). "Relative radiometric normalization performance." PE&RS 66(8):967-980.
- Claverie, M., et al. (2018). "The Harmonized Landsat and Sentinel-2 surface reflectance data set." RSE 219:145-161.
- Hayes, A.G., et al. (2021). "Pre-Flight Calibration of Mastcam-Z." SSR 217:29. DOI: 10.1007/s11214-021-00795-x
- Delamere, W.A., et al. (2010). "Color imaging of Mars by HiRISE." Icarus 205:38-52.
- Kinch, K.M., et al. (2020). "Radiometric Calibration Targets for Mastcam-Z." SSR 216:141.
- Merusi, M., et al. (2022). "Mastcam-Z Radiometric Calibration Targets...First 350 Sols." E&SS. DOI: 10.1029/2022EA002552

---

## 9.2 Photometric Correction for Oblique Viewing Geometry

### 9.2.1 Problem Statement

Mastcam-Z images are acquired from ~2 m elevation with oblique viewing angles (typically 0-80 deg from nadir). HiRISE images are near-nadir (~0-30 deg emission angle). When Mastcam-Z images are ortho-projected onto the DTM, the surface brightness encodes the original oblique viewing geometry. A given surface patch at (incidence i, emission e, phase g) in Mastcam-Z view will have brightness I/F(i_mcz, e_mcz, g_mcz), but the same patch in HiRISE has brightness I/F(i_hir, e_hir, g_hir). This geometric brightness mismatch must be corrected before SR training.

### 9.2.2 Photometric Models for Mars

**1. Lambertian Model**

```
I/F = A_L * cos(i)
```

Simplest model; assumes isotropic scattering. Adequate for rough first-order correction but systematically underestimates brightness at high emission angles and near opposition. Used by HRSC team for global mosaicking (Walder et al., EPSC 2011).

**2. Minnaert Model**

```
I/F = A_M * cos^k(i) * cos^(k-1)(e)
```

where k is the Minnaert constant (k=1 is Lambertian, k<1 for forward-scattering surfaces). Widely applied to Mars: Mustard et al. (2021, PSS 200:105198) evaluated four disk functions (Lambert, Lommel-Seeliger, Akimov, Minnaert) for topographic correction of HiRISE and CaSSIS images, finding that Minnaert performed well for moderate slopes. Mars-typical k values: 0.5-0.8 depending on albedo and wavelength.

- **Advantages**: Simple two-parameter model; k can be fit from multi-angle observations.
- **Limitations**: k varies with phase angle, albedo, and surface type; not physical.

**3. Hapke Model**

The physically-based bidirectional reflectance model (Hapke 1993, "Theory of Reflectance and Emittance Spectroscopy," Cambridge UP). The full model:

```
r(i, e, g) = w/(4pi) * mu0/(mu0 + mu) * [p(g) * B_SH(g) + H(mu0) * H(mu) - 1]
              * S(i, e, g, theta_bar) * B_CB(g)
```

Parameters:
- w: single-scattering albedo
- p(g): single-particle phase function (double Henyey-Greenstein: b=0.12, c=0.6 for Mars dust)
- theta_bar: macroscopic roughness (~17 deg for typical Mars surface)
- B_SH, B_CB: shadow-hiding and coherent backscatter opposition effects
- H(mu): Chandrasekhar H-function for multiple scattering

**Mars-specific Hapke parameters** (Fernando et al. 2022, J. Imaging 8(6):158; Vincendon et al. 2007, JGR; Ceamanos et al. 2013, JGR):
- w ~ 0.85 (RED wavelengths, bright terrain); 0.35-0.55 (dark terrain)
- theta_bar ~ 17 deg (mean Mars surface)
- Phase function: b = 0.12, c = 0.6 (5-10% backscattering peak)
- B_0 = 1, h = 0.05 (opposition surge)

Fernando et al. (2022) embedded the Hapke AMSA model within atmospheric radiative transfer for shape-from-shading on Mars. Their combined equation accounts for direct reflection, diffuse skylight (weight zeta), and atmospheric path radiance (weight chi):

```
r_M = e^(-tau/mu0) * r_d(i,e,g,w) + zeta * r_hd(i,e,g,w) * e^(-tau/mu) + chi
```

Key finding: for tau > 0.3, explicit atmospheric modeling is essential.

**4. ISIS phoempglobal Tool**

USGS ISIS provides `phoempglobal` for empirical photometric correction of planetary images. It supports Lambert, Minnaert, Lommel-Seeliger, and lunar-Lambert models. The tool normalizes images to a standard geometry (typically i=30, e=0, g=30) using DTM-derived local incidence/emission angles.

### 9.2.3 Practical Photometric Correction Strategy

For our application, we need to normalize Mastcam-Z ortho-images to match HiRISE viewing geometry:

```
Step 1: Compute local geometry per pixel
  - From Mastcam-Z XYZ product: compute surface normal (UVW product)
  - From SPICE: compute sun direction and camera direction at observation time
  - Compute per-pixel (i, e, g) for Mastcam-Z observation
  
Step 2: Compute HiRISE geometry for same surface patch
  - From HiRISE observation metadata: solar incidence, emission angle
  - From DTM surface normal: compute per-pixel (i, e, g) for HiRISE
  
Step 3: Apply photometric correction
  - For each pixel: I/F_corrected = I/F_observed * f(i_hir, e_hir, g_hir) / f(i_mcz, e_mcz, g_mcz)
  - Use Minnaert model (simpler, adequate for training data) or Hapke (more accurate)
  - Minnaert k can be estimated from multi-sol Mastcam-Z observations of same target
  
Step 4: Validate
  - Compare corrected Mastcam-Z I/F statistics against HiRISE I/F for overlap regions
  - Residual should be <5% for usable training pairs
```

### 9.2.4 Key References

- Hapke, B. (1993). "Theory of Reflectance and Emittance Spectroscopy." Cambridge UP.
- Shepard, M.K. & Helfenstein, P. (2007). "A test of the Hapke photometric model." JGR 112:E03001.
- Fernando, J., et al. (2022). "Atmospheric Correction for High-Resolution Shape from Shading on Mars." J. Imaging 8(6):158.
- Mustard, J.F., et al. (2021). "Topographic correction of HiRISE and CaSSIS images." PSS 200:105198.
- Vincendon, M., et al. (2007). "Mars surface phase function." arXiv:1208.4518.
- Ceamanos, X., et al. (2013). "Surface reflectance of Mars observed by CRISM/MRO." JGR Planets 118:514-533.
- Walder, T., et al. (2011). "HRSC Topographic Correction by Minnaert Photometric Modeling." EPSC-DPS2011-648.

---

## 9.3 PSF / MTF Matching Between Different Resolution Images

### 9.3.1 Problem Statement

Mastcam-Z and HiRISE have completely different point spread functions (PSFs). Mastcam-Z at 110mm focal length has MTF characterized at all zoom settings (Hayes et al. 2021), with resolution limited by the 7.4 um pixel pitch. HiRISE achieves near-diffraction-limited performance at ~1 urad angular resolution but suffers from jitter-induced blur, TDI smearing, and detector aging (McEwen et al. 2007). When forming training pairs for RefSR, the spatial frequency content of the reference (Mastcam-Z ortho) and the low-resolution input (HiRISE) encode different PSF signatures.

### 9.3.2 Approaches in the Literature

**1. Explicit PSF Matching (Pansharpening Heritage)**

In pansharpening, the standard approach matches the multispectral (MS) sensor PSF before fusion. Vivone et al. (2015, IEEE GRSM) describe MTF-tailored injection, where the panchromatic image is filtered with a kernel that matches the MS sensor MTF before computing the spatial detail to inject. Aiazzi et al. (2006, IEEE TGRS) proposed MTF-tailored multiresolution fusion that designs a reduction filter matching the MS sensor's actual MTF.

**Key insight**: When the PSF of the reference (HR) sensor is sharper than the target (LR) sensor, the reference should be degraded to match the LR PSF, and the SR model learns to reverse this degradation.

**2. Learned PSF in Super-Resolution**

Modern blind SR methods estimate the degradation kernel implicitly or explicitly:

- **KernelGAN** (Bell-Kligler et al., NeurIPS 2019): Estimates the image-specific degradation kernel using an internal GAN trained on patches from a single image. The estimated kernel can then be used by non-blind SR methods.
- **DCLS** (Luo et al., 2022): Deep Constrained Least Squares for blind SR with kernel estimation.
- **RDSR** (Do et al., ACCV 2024): "Blind Super Resolution with Reference Images and Implicit Degradation Representation" -- uses HR reference images to establish scale-aware degradation kernels, adaptable to various blind SR networks.

**3. DATSR's Approach: Bypassing Explicit PSF Matching**

DATSR (Cao et al., ECCV 2022) addresses the PSF mismatch implicitly through its deformable attention mechanism. Rather than explicitly matching PSFs, DATSR:
- Uses texture feature encoders that extract multi-scale features from both LR and reference images
- Employs normalized inner product to find top-K relevant correspondences
- Applies modified deformable convolution with learnable offsets and masks to adaptively transfer textures
- Trains with ColorJitter augmentation to handle radiometric differences

The network learns to extract resolution-invariant texture features and transfer them appropriately, effectively learning the PSF relationship during training.

### 9.3.3 Practical Recommendation for Our Pipeline

**Option A: Explicit PSF matching (conservative, interpretable)**
```
1. Characterize HiRISE effective PSF for the specific observation
   - Use edge response analysis on sharp albedo boundaries
   - Account for jitter (Mattson et al. 2009 HiJACK), TDI smear, defocus
   - Typical effective PSF FWHM: 1.0-1.5 pixels (25-38 cm)

2. Characterize Mastcam-Z MTF at the zoom setting used
   - Use pre-flight MTF data from Hayes et al. 2021
   - Account for distance-dependent GSD variation across the image

3. When creating training pairs:
   - Degrade Mastcam-Z ortho to HiRISE PSF: convolve with PSF ratio kernel
   - PSF_ratio = PSF_hirise / PSF_mastcam (in frequency domain: MTF ratio)
   - This creates a "what Mastcam-Z would look like at HiRISE resolution"
   - Use this as the LR training input; original Mastcam-Z ortho as HR target
```

**Option B: Let the network learn (modern, potentially more robust)**
```
1. Use DATSR's built-in deformable attention to handle PSF differences
2. Add degradation augmentation during training:
   - Random Gaussian blur (sigma 0.5-2.5) on LR inputs
   - Random noise addition (Gaussian + Poisson)
   - Random downsampling kernel (bicubic, bilinear, Lanczos)
3. Optionally train with KernelGAN-estimated kernels for each HiRISE image
```

**Recommendation**: Start with Option B (let DATSR handle it) because:
- Our PSF mismatch is confounded by atmospheric scattering, geometric projection artifacts, and resolution ratio (~10-50x)
- The reference image (Mastcam-Z) is always much sharper than LR (HiRISE), which is the favorable case for RefSR
- Explicit PSF matching introduces additional interpolation artifacts

### 9.3.4 Key References

- Hayes, A.G., et al. (2021). "Pre-Flight Calibration of Mastcam-Z." SSR 217:29.
- Vivone, G., et al. (2015). "A Critical Comparison Among Pansharpening Algorithms." IEEE GRSM 3(2):63-77.
- Aiazzi, B., et al. (2006). "MTF-tailored Multiscale Fusion." IEEE TGRS 44(10):2687-2703.
- Bell-Kligler, S., et al. (2019). "Blind Super-Resolution Kernel Estimation using an Internal-GAN." NeurIPS.
- Cao, J., et al. (2022). "Reference-based Image Super-Resolution with Deformable Attention Transformer." ECCV.
- Do, T., et al. (2024). "Blind Super Resolution with Reference Images and Implicit Degradation Representation." ACCV.
- Mattson, S., et al. (2009). "HiJACK: Correcting spacecraft jitter in HiRISE images." LPSC XL, Abs. #2001.

---

## 9.4 Shadow and Occlusion Handling in Orthorectification

### 9.4.1 Problem Statement

When projecting oblique Mastcam-Z images to nadir view using the DTM, two classes of artifacts emerge:
1. **Self-occlusion**: Surface areas hidden behind terrain relief in the original camera view create "holes" in the ortho-projection (no data)
2. **Cast shadows**: Areas in shadow have dramatically reduced signal, encoding illumination geometry rather than surface properties

Both create artifacts that would corrupt SR training if not handled.

### 9.4.2 Occlusion Detection Methods

**1. Z-Buffer Visibility Analysis**

The standard method from computer graphics, adapted for true orthoimage generation. For each pixel in the nadir projection:
- Project all 3D surface points onto the image plane
- Keep only the closest point (minimum Z in camera frame)
- Points behind closer points are flagged as occluded

Habib et al. (2007, PE&RS 73(3):277-291) applied Z-buffer with height gradient analysis for true orthoimage production, and Zhou et al. (2005) developed occlusion detection combining SGM and Z-buffer for urban true orthophotos.

**For our case**: The Mastcam-Z XYZ point cloud naturally encodes visibility -- pixels without valid XYZ values are occluded. The drape.py code already handles this by using the `valid` mask (`~np.isnan(lon) & ~np.isnan(lat)`). Areas with no valid lon/lat are inherently occluded from the camera viewpoint.

**2. Multi-View Occlusion Compensation**

Bang et al. (2016, GIScience & RS 53(1):1-22) proposed "true orthoimage generation by mutual recovery of occlusion areas" -- when one view has an occlusion, fill it from an overlapping view where the area is visible.

**For our case**: Mastcam-Z routinely acquires overlapping images across sols and within mosaics. Multi-sol observations of the same terrain from different positions can fill occlusion gaps:

```
For each ortho-projected patch:
  1. Identify occlusion mask (pixels with no valid XYZ)
  2. Search other sols/frames observing the same DTM region
  3. Project those frames onto the DTM
  4. Fill occluded pixels from frames where the area is visible
  5. Apply photometric correction (Section 9.2) to match illumination
```

### 9.4.3 Shadow Detection and Handling

**1. Shadow Detection**

Adeline et al. (2013, RSE 132:46-60) comprehensively reviewed shadow detection in remote sensing. Primary methods:
- **Thresholding**: Shadow pixels have low intensity in all bands. Define threshold as mean - k*std_dev (k~1.5-2.0)
- **Chromaticity-based**: Shadows shift hue/saturation but not chromaticity ratio. Finlayson et al. (2004, PAMI 26(1):59-68) proposed shadow-free chromaticity images using 1D intensity-invariant projection
- **Geometric**: Given DTM and sun position (from SPICE), ray-trace to determine which pixels are in shadow. For Mars, this is reliable because there are no clouds or vegetation.

**Geometric shadow detection for Mars**:
```python
# For each DTM grid point:
sun_direction = spice.compute_sun_direction(et)  # from SPICE
for each pixel (x, y):
    ray = pixel_position + t * sun_direction  # ray toward sun
    if ray intersects DTM at any t > 0:
        pixel is in shadow
```

**2. Shadow Handling Strategies**

For SR training, shadows must be handled because:
- HiRISE and Mastcam-Z images are taken at different times with different solar geometry
- A patch in shadow in Mastcam-Z may be illuminated in HiRISE (and vice versa)

**Strategy A: Mask and exclude shadows from training**
```
- Generate shadow masks for both Mastcam-Z ortho and HiRISE
- Exclude training patches where >20% of pixels are shadowed
- Safest approach; reduces training data volume
```

**Strategy B: Shadow removal before training**
```
- In shadowed regions, estimate intrinsic reflectance from the shadow signal
- Shadow pixels receive diffuse (sky) illumination only: I_shadow = rho * I_diffuse
- Illuminated pixels receive direct + diffuse: I_lit = rho * (I_direct + I_diffuse)
- Ratio: I_shadow / I_lit = I_diffuse / (I_direct + I_diffuse)
- On Mars: I_diffuse / I_total ~ 0.1-0.3 (Kinch et al. 2020, from gnomon shadow analysis)
- Correct: I_corrected = I_shadow * (I_direct + I_diffuse) / I_diffuse
- CAUTION: This amplifies noise significantly in shadow regions
```

**Strategy C: Shadow-invariant representation**
```
- Use ratio images or illumination-invariant features (Section 9.5)
- Train the SR network on shadow-invariant representations rather than raw radiance
```

### 9.4.4 Practical Recommendation

```
1. Occlusion: Use multi-sol Mastcam-Z coverage to fill gaps. 
   Remaining gaps → mask as nodata (alpha=0 in drape.py output)
   
2. Shadows: 
   a. Compute geometric shadow mask from DTM + SPICE sun position
   b. For training pair generation: exclude patches with >30% shadow
   c. For inference: shadow regions in HiRISE input are acceptable 
      (they contain real surface information at HiRISE illumination)
   d. Do NOT attempt shadow removal -- the noise amplification is 
      counterproductive for SR training
```

### 9.4.5 Key References

- Habib, A.F., et al. (2007). "Generation of Orthoimages and Perspective Views with Automatic Visibility Checking." PE&RS 73(3):277-291.
- Bang, K.I., et al. (2016). "True orthoimage generation by mutual recovery of occlusion areas." GIScience & RS 53(1):1-22.
- Adeline, K.R.M., et al. (2013). "Shadow detection in very high spatial resolution aerial images." RSE 132:46-60.
- Finlayson, G.D., et al. (2004). "Intrinsic Images by Entropy Minimization." ECCV.
- Kinch, K.M., et al. (2020). "Radiometric Calibration Targets for Mastcam-Z." SSR 216:141.

---

## 9.5 Illumination Differences Between Orbital and Ground Imagery

### 9.5.1 Problem Statement

HiRISE Jezero images are typically acquired at solar incidence angles of 40-60 deg (afternoon, MRO orbit geometry). Mastcam-Z images span a wide range of solar geometries depending on sol time (morning to late afternoon, incidence ~20-70 deg). The phase angle geometry is fundamentally different: HiRISE has near-zero emission angle with moderate phase, while Mastcam-Z has large emission angles from a ground viewpoint. These differences cause:
- Different shadow patterns (length, direction)
- Different specular/diffuse highlight patterns
- Different overall brightness distributions

### 9.5.2 Illumination-Invariant Methods

**1. Chromaticity / Ratio Images**

Finlayson et al. (2004, PAMI) proposed a log-chromaticity projection that eliminates illumination effects from color images by projecting into a 1D invariant direction. For Mars (where color variation is limited), band ratio images (e.g., RED/BG) effectively remove illumination geometry:

```
Ratio = Band1(x,y) / Band2(x,y)
```

The ratio cancels the common illumination factor (cos(i) term) while preserving spectral differences. This is extensively used in Mars spectral analysis (e.g., Fraeman et al. 2013 used CRISM band ratios to map hematite).

**2. Illumination-Invariant Feature Descriptors**

For image matching across illumination conditions:

- **RIFT** (Li et al., 2019, ISPRS JPRS 153:47-60): Radiation-Invariant Feature Transform, uses phase congruency (invariant to illumination and contrast) to construct descriptors. Demonstrated on planetary remote sensing imagery.
- **HOSS** (Ye et al., 2019, ISPRS JPRS 152:14-29): Histogram of Oriented Self-Similarity -- a 3D histogram that is robust to significant illumination variations.
- **LHOPC** (Ye et al., 2018, ISPRS JPRS 143:22-36): Local Histogram of Oriented Phase Congruency -- uses phase congruency as feature detector, invariant to illumination/contrast changes.

**Critical finding for planetary data**: Ye et al. (2018, PSS 157:13-23) specifically showed that SIFT keypoint orientations align with sub-solar azimuth angle, reducing matching performance across different illumination conditions. Their adaptive suppression approach levels the orientation histogram.

**3. Intrinsic Image Decomposition**

Decompose observed image into reflectance (albedo) and shading components:
```
I(x,y) = R(x,y) * S(x,y)
```

Barron & Malik (2015, PAMI) achieved state-of-the-art intrinsic decomposition. For Mars, with known DTM and sun position, the shading component can be computed analytically:
```
S(x,y) = photometric_model(i(x,y), e(x,y), g(x,y))
R(x,y) = I(x,y) / S(x,y)
```

This is essentially the photometric correction from Section 9.2 applied to extract albedo.

### 9.5.3 Practical Recommendation

**For SR training pair generation**:
```
Option 1 (Preferred): Photometric normalization to common geometry
  - Correct both Mastcam-Z and HiRISE to standard geometry (i=30, e=0, g=30)
  - Use Minnaert model with Mars-specific k
  - Train SR on photometrically normalized pairs

Option 2: Train on albedo maps
  - Extract albedo by dividing by predicted shading
  - SR model learns albedo-to-albedo mapping
  - Apply HiRISE-specific shading to SR output
  - Advantage: illumination-independent; Disadvantage: noise amplification in low-shading areas

Option 3: Augmentation-based robustness
  - Do NOT correct illumination
  - Instead, augment training with random brightness/contrast/gamma changes
  - DATSR already includes ColorJitter augmentation
  - Network learns illumination-invariant texture features
  - Advantage: simple; Disadvantage: may waste network capacity on illumination
```

**Recommendation**: Use Option 1 (photometric normalization) as primary approach, combined with Option 3 (augmentation) for robustness. Option 2 is too noisy for practical use.

### 9.5.4 Key References

- Finlayson, G.D., et al. (2004). "Recovering Chromaticity Image Free from Shadows." PAMI 26(1):59-68.
- Li, J., et al. (2019). "RIFT: Multi-modal image matching." ISPRS JPRS 153:47-60.
- Ye, Y., et al. (2019). "HOSS: Illumination-Robust remote sensing image matching." ISPRS JPRS 152:14-29.
- Ye, Y., et al. (2018). "Illumination invariant feature point matching for planetary remote sensing." PSS 157:13-23.
- Barron, J.T. & Malik, J. (2015). "Shape, Illumination, and Reflectance from Shading." PAMI 37(8):1670-1687.

---

## 9.6 Mars-Specific Atmospheric Correction

### 9.6.1 Atmospheric Effects on Mars Imagery

Mars has a thin CO2 atmosphere (~6 mbar surface pressure) with variable dust loading. The atmosphere affects both Mastcam-Z (ground-up, path through partial atmosphere) and HiRISE (orbital, path through full atmosphere) differently.

**Key atmospheric parameters**:
- **Dust optical depth (tau)**: Varies seasonally and with dust events. Typical: tau ~ 0.3-1.0 (visible). During global dust storms: tau > 4.0
- **Dust particle properties**: Mean effective radius ~1.5 um; single-scattering albedo w ~ 0.92 (visible); asymmetry parameter g ~ 0.67; modeled as cylinders by Wolff et al. (2006, JGR 111:E12S17)
- **Water ice clouds**: Occasional, tau_ice ~ 0.02-0.2; larger particles (~3 um)

### 9.6.2 How Atmosphere Affects Each Sensor Differently

**Mastcam-Z (ground-level, looking at surface)**:
- Path radiance from atmosphere between camera and target (short path, typically 2-50 m): **negligible** for near targets, small for distant targets
- Diffuse illumination: sky contributes ~10-30% of total illumination (Kinch et al. 2020, from gnomon analysis)
- Atmospheric attenuation of direct solar beam: I_surface = I_TOA * e^(-tau / cos(z_sun))
- **Net effect**: Surface illumination is reduced and reddened by atmospheric dust
- **Correction**: Calibration targets provide instantaneous local irradiance → IOF products are corrected

**HiRISE (orbital, looking down through entire atmosphere)**:
- Path radiance from atmospheric scattering between spacecraft and surface: significant
- In dusty conditions, "almost all received radiance can be attributed to path scattered radiance" (Fernando et al. 2022)
- Atmospheric absorption of surface-reflected signal (double-pass: down + up)
- Diffuse illumination at surface included in measurement
- **Net effect**: I/F_measured = I/F_surface * T_atm + I/F_path, where T_atm = atmospheric transmittance
- **Correction**: Requires atmospheric radiative transfer modeling

### 9.6.3 Atmospheric Correction Methods

**1. Mastcam-Z: Calibration Target Method (Kinch et al. 2020, 2023)**

The definitive method for Mastcam-Z. The rover deck carries 8-patch calibration targets with known reflectance:
- AluWhite98 (~98% reflectance), Carbon Black, and 6 color/grayscale patches
- Central gnomon casts shadow → direct/diffuse illumination ratio measurement
- DN → I/F conversion: compare target DN to known I/F under current illumination
- Dust on targets monitored via magnetically-shielded clean areas (Merusi et al. 2022)
- **Accuracy**: IOF products corrected to ~2-5% absolute

**2. HiRISE: Empirical Atmospheric Correction**

HiRISE has no onboard calibration source. The hical pipeline (ISIS) corrects for detector response but not atmospheric effects. Atmospheric correction options:

- **Simple path radiance subtraction**: I/F_surface = (I/F_measured - I/F_path) / T_atm. The path radiance I/F_path can be estimated from shadow regions (which receive only atmospheric scattered light). This is a first-order correction adequate for our purposes.
- **DISORT-based radiative transfer** (McGuire et al. 2009, IEEE TGRS): Full radiative transfer retrieval of surface Lambert albedo from CRISM data. Uses climatological dust/ice optical depths from MGS-TES. "Dark areas of Mars have solar-band Lambert albedos that are up to 30% darker than without atmospheric correction" (Wolff et al. 2006).
- **Atmosphere-aware photoclinometry** (Fernando et al. 2022, 2023): Jointly estimates atmospheric parameters (tau, diffuse fraction, path radiance) and surface properties from image data alone.

**3. Dust Opacity Measurements (Lemmon et al. 2015, 2023)**

The gold-standard reference for Mars optical depth:
- **Lemmon, M.T., et al. (2015). "Dust aerosol, clouds, and the atmospheric optical depth record over 5 Mars years of the Mars Exploration Rover mission." Icarus 251:96-111.** DOI: 10.1016/j.icarus.2014.03.029
- **Lemmon, M.T., et al. (2023). "The Mars Science Laboratory record of optical depth measurements via solar imaging." Icarus 406:115747.** DOI: 10.1016/j.icarus.2023.115747

Method: Direct solar imaging with calibrated camera → Beer-Lambert extinction:
```
tau = -cos(z_sun) * ln(F_measured / F_TOA)
```

For Perseverance/Mastcam-Z, optical depth measurements are routinely acquired (L7/R7 solar filters). These tau values are essential for atmospheric correction of both sensors.

### 9.6.4 Cross-Sensor Atmospheric Harmonization

The key challenge: Mastcam-Z IOF products are corrected for local illumination (calibration targets measure actual irradiance at the surface), while HiRISE I/F includes atmospheric contribution. To harmonize:

```
1. Obtain tau at the time of HiRISE observation
   - From concurrent Mastcam-Z solar opacity measurements, or
   - From MCS (Mars Climate Sounder) limb profiles, or
   - From TES/MCS climatology (Montabone et al. 2015)

2. Compute HiRISE atmospheric correction
   - T_atm = e^(-tau * (1/cos(i_sun) + 1/cos(e_spacecraft)))  [two-pass]
   - I/F_path estimated from shadow regions or from DISORT model
   - I/F_surface = (I/F_measured - I/F_path) / T_atm

3. Both sensors now nominally in surface I/F units
   → Apply PIF-based linear normalization (Section 9.1) to residual
```

### 9.6.5 Practical Impact Assessment

For typical Jezero conditions (tau ~ 0.4-0.8):
- Atmospheric path radiance: 2-8% of total HiRISE signal
- Atmospheric attenuation: 15-30% of surface signal lost
- **If uncorrected**: SR model learns a brightness offset + contrast reduction rather than texture
- **Recommendation**: At minimum, apply path radiance subtraction (shadow-based estimate) and two-pass transmittance correction. Full DISORT is ideal but adds complexity.

### 9.6.6 Key References

- Lemmon, M.T., et al. (2015). "Dust aerosol, clouds, and the atmospheric optical depth record." Icarus 251:96-111.
- Lemmon, M.T., et al. (2023). "MSL record of optical depth measurements." Icarus 406:115747.
- Wolff, M.J., et al. (2006). "Constraints on dust aerosols from MER." JGR 111:E12S17.
- Wolff, M.J., et al. (2009). "Wavelength dependence of dust aerosol single scattering albedo." JGR 114:E00D04.
- McGuire, P.C., et al. (2009). "CRISM Retrieval of Surface Lambert Albedos with DISORT." IEEE TGRS 47(12):4322-4334.
- Kinch, K.M., et al. (2020). "Radiometric Calibration Targets for Mastcam-Z." SSR 216:141.
- Merusi, M., et al. (2022). "Mastcam-Z Calibration Targets...First 350 Sols." E&SS.
- Fernando, J., et al. (2022). "Atmospheric Correction for High-Resolution Shape from Shading on Mars." J. Imaging 8(6):158.
- Montabone, L., et al. (2015). "Eight-year climatology of dust optical depth on Mars." Icarus 251:65-95.

---

## 9.7 Data Augmentation and Training Strategies for Cross-Platform SR

### 9.7.1 The Domain Gap Problem

Our training pairs have an inherent domain gap: Mastcam-Z ortho-images (cm-scale, ground-level optics, Bayer CCD, narrowband filters) vs. HiRISE (25 cm/px, orbital optics, pushbroom TDI CCD, broadband RED). Even after radiometric and photometric corrections, residual differences remain in:
- Noise characteristics (Mastcam-Z: photon + read noise; HiRISE: TDI-integrated, CCD aging artifacts)
- Spatial frequency content (different MTFs, different aliasing patterns)
- Subtle spectral mismatches (bandpass differences)
- Geometric distortion residuals

### 9.7.2 Domain Adaptation Strategies from the Literature

**1. Unsupervised Domain Adaptation for SR (Wang et al., ICCV 2021)**

Treats SR as a domain adaptation problem at the feature level. Uses a discriminator to align feature distributions between source (synthetic degradation) and target (real degradation) domains. Key technique: "domain-distance weighted supervision" where training samples closer to the real domain receive higher weight.

**2. Domain-Distance Aware Training (Wei et al., CVPR 2021)**

Recognizes that even when using learned downsampling to create LR training images, a domain gap persists. Proposes weighting the training loss by the estimated domain distance of each sample. "Generated low-resolution samples reside closer to the real-world domain while others are relatively far away."

**3. CycleGAN-Based Domain Translation (Zhu et al., ICCV 2017)**

For unpaired cross-domain SR: learn a bidirectional mapping between domains using cycle-consistency loss. Apply to translate Mastcam-Z appearance to "HiRISE-like" appearance (or vice versa) before SR training. Yuan et al. (2018, CVPRW) demonstrated CycleGAN for unsupervised image SR from unpaired data.

**4. Cross-Domain SR for Remote Sensing (UVRSR, Li et al., 2022, Remote Sensing 14(6):1513)**

"Unsupervised Remote Sensing Image Super-Resolution Guided by Visible Images" -- trains SR with unpaired HR visible images and LR remote sensing images. Uses domain adaptation to bridge the gap between visible (similar to ground-level) and remote sensing (orbital) domains.

### 9.7.3 Training Strategies Specific to Our Application

**Strategy 1: Synthetic Pre-training + Fine-tuning**
```
Phase 1: Pre-train on synthetic pairs
  - Take Mastcam-Z ortho-images as HR ground truth
  - Degrade with realistic HiRISE-like degradation:
    * Gaussian blur (sigma 1.0-2.0 px)
    * Downsample by target SR factor (x4)
    * Add Gaussian noise (sigma 5-15 DN)
    * Add Poisson noise (shot noise model)
  - Train DATSR on these synthetic pairs

Phase 2: Fine-tune on real pairs
  - Use co-registered Mastcam-Z ortho + actual HiRISE patches
  - Fine-tune with lower learning rate
  - The model adapts to real sensor characteristics
```

**Strategy 2: Degradation Augmentation (RealESRGAN approach)**
```
- Apply second-order degradation model (Wang et al., ICCVW 2021):
  1. Blur → resize → noise → JPEG (first degradation)
  2. Blur → resize → noise → JPEG (second degradation)
- Randomly vary all parameters to span the degradation space
- Include Mars-specific augmentations:
  * Atmospheric haze simulation (additive path radiance)
  * Dust opacity variation (multiplicative attenuation)
  * CCD seam artifacts (brightness discontinuities)
```

**Strategy 3: Curriculum Learning**
```
1. Start with easy pairs: close-range (2-5m) Mastcam-Z, low dust opacity, 
   minimal shadow, good co-registration
2. Gradually add harder pairs: mid-range (5-20m), moderate dust, 
   some shadow, larger registration uncertainty
3. Finally add challenging pairs: far-range (20-50m), high dust, 
   shadow boundaries, poorest registration
```

### 9.7.4 Augmentation Specifics for Mars Data

```python
# Mars-specific augmentation pipeline
augmentations = {
    # Standard SR augmentations
    'random_flip': [horizontal, vertical],
    'random_rotate': [0, 90, 180, 270],
    'color_jitter': {
        'brightness': 0.2,   # Mars illumination variation
        'contrast': 0.2,     # Atmospheric contrast variation  
        'hue': 0.05,         # Minimal -- Mars color is consistent
        'saturation': 0.1,   # Dust reddening variation
    },
    
    # Mars-specific augmentations
    'atmospheric_haze': {
        'tau_range': [0.2, 1.0],    # Dust opacity range
        'additive_path': True,       # Path radiance
        'multiplicative_atten': True, # Beam attenuation
    },
    'shadow_augmentation': {
        'random_shadow_mask': True,   # Random geometric shadows
        'shadow_intensity': [0.1, 0.3], # Diffuse/total ratio
    },
    'registration_jitter': {
        'max_shift_px': 3,           # Sub-pixel misalignment
        'max_rotation_deg': 0.5,     # Small rotation error
    },
}
```

### 9.7.5 Key References

- Wang, W., et al. (2021). "Unsupervised Real-World SR: A Domain Adaptation Perspective." ICCV.
- Wei, P., et al. (2021). "Unsupervised Real-World Image SR via Domain-Distance Aware Training." CVPR.
- Zhu, J.-Y., et al. (2017). "Unpaired Image-to-Image Translation using CycleGAN." ICCV.
- Wang, X., et al. (2021). "Real-ESRGAN: Training Real-World Blind SR with Pure Synthetic Data." ICCVW.
- Li, S., et al. (2022). "Unsupervised Remote Sensing Image SR Guided by Visible Images." Remote Sensing 14(6):1513.

---

## 9.8 Geometric Preprocessing for SR Training Pairs

### 9.8.1 Sub-Pixel Alignment Requirements

SR methods are sensitive to misalignment between LR and HR training pairs. Even 1-2 pixel misalignment in the LR space can cause the network to learn blurred or inconsistent textures. For our x4 SR case (HiRISE 25 cm → 6.25 cm):
- 1 HiRISE pixel = 25 cm = 4 HR pixels
- A 0.5 HiRISE pixel misalignment = 12.5 cm = 2 HR pixels of error

Our co-registration error budget (from 07_spice_xyz_methodology.md) shows the dominant error is rover absolute localization (1-3 m = 4-12 HiRISE pixels). This is **far beyond** what standard SR methods tolerate.

### 9.8.2 Alignment Methods

**1. Feature-Based Refinement**

After initial SPICE-based co-registration, apply feature-based matching to refine alignment:
- Use illumination-invariant descriptors (RIFT, LHOPC) due to different solar geometry
- Normalized cross-correlation (NCC) for template matching
- Tao & Muller (2016, Icarus) achieved 1-2 HiRISE pixel (25-50 cm) alignment using NCC on Navcam ortho vs. HiRISE
- Phase correlation for sub-pixel shift estimation (Guizar-Sicairos et al. 2008, Optics Letters)

**2. Optical Flow Refinement**

After coarse alignment, apply dense optical flow to estimate per-pixel displacement:
- Horn-Schunck or Lucas-Kanade on the co-registered pair
- FlowNet2.0 or RAFT (Teed & Deng, 2020, ECCV) for learned optical flow
- Caution: optical flow assumes brightness constancy, which is violated by cross-sensor differences
- Mitigation: apply flow on gradient images or edge maps rather than raw intensity

**3. Deformable Alignment (DATSR's Built-in Solution)**

DATSR's deformable attention mechanism acts as an implicit alignment layer:
- The RDA module uses normalized inner product to find top-K relevant positions
- Modified deformable convolution applies learned offsets to sample reference features at the correct locations
- This handles misalignment "within a local window" (the Transformer receptive field)
- For large misalignments (>window size), explicit pre-alignment is still needed

**Critical finding from DATSR paper** (Cao et al. 2022): The method was tested with artificially induced transformations of varying severity and maintained good performance. However, the training data (CUFED5) uses internet-retrieved similar images, not geometrically co-registered pairs. Our case has much stronger geometric correspondence but also much larger potential misalignment.

### 9.8.3 Misalignment Tolerance of SR Architectures

| Method | Alignment Assumption | Tolerance |
|---|---|---|
| SISR (EDSR, RCAN) | N/A (single image) | N/A |
| VSR (EDVR, BasicVSR) | Small temporal displacement | 2-5 LR pixels |
| RefSR (SRNTT) | Content similarity, not geometric | Moderate (patch matching) |
| RefSR (TTSR) | Transformer cross-attention | Window size dependent |
| RefSR (DATSR) | Deformable attention | **Most robust** -- handles moderate geometric and photometric differences |
| Dual-Camera SR (DCSR) | Calibrated stereo pair | Sub-pixel (<1 LR pixel) |

### 9.8.4 Two-Stage Alignment Pipeline for Our Application

```
Stage 1: Coarse alignment (SPICE-based, current pipeline)
  - Mastcam-Z pixel → CAHVOR → XYZ → SPICE → IAU_MARS → HiRISE pixel
  - Expected accuracy: 1-3 m (4-12 HiRISE pixels)
  
Stage 2: Fine alignment (feature-based refinement)
  - Extract RIFT/phase-congruency features from both Mastcam-Z ortho and HiRISE
  - Coarse-to-fine matching: 
    * Block matching at 32x32 pixel blocks → affine transform
    * Sub-pixel refinement via phase correlation → residual shift
  - Target accuracy: <1 HiRISE pixel (<25 cm)
  
Stage 3: Network-level alignment (DATSR built-in)
  - Remaining sub-pixel misalignment handled by deformable attention
  - No explicit correction needed -- part of the learned model
  
Quality control:
  - Compute NCC between aligned Mastcam-Z ortho and HiRISE patch
  - Reject pairs with NCC < 0.3 (too different for useful training)
  - Reject pairs with residual shift > 2 HiRISE pixels after Stage 2
```

### 9.8.5 Key References

- Tao, Y. & Muller, J.-P. (2016). "Rover orthoimage-to-HiRISE co-registration." Icarus 280:139-157.
- Guizar-Sicairos, M., et al. (2008). "Efficient subpixel image registration algorithms." Optics Letters 33(2):156-158.
- Teed, Z. & Deng, J. (2020). "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow." ECCV.
- Cao, J., et al. (2022). "DATSR: Reference-based SR with Deformable Attention Transformer." ECCV.
- Wang, T., et al. (2021). "Dual-Camera SR with Aligned Attention Modules." ICCV.

---

## 9.9 Integrated Preprocessing Pipeline Summary

The complete preprocessing chain for generating SR training pairs:

```
INPUT: Mastcam-Z stereo pair + HiRISE RDR + HiRISE DTM

1. GEOMETRIC PIPELINE (existing, Sections 9.4, 9.8)
   ├── Mastcam-Z stereo → XYZ point cloud (OPGS/PDS)
   ├── XYZ → SPICE → IAU_MARS lon/lat (our pipeline)
   ├── Drape onto DTM → ortho-image (drape.py)
   ├── Occlusion detection → alpha mask
   ├── Shadow detection (DTM + SPICE sun) → shadow mask
   └── Feature-based refinement → sub-pixel alignment to HiRISE

2. RADIOMETRIC PIPELINE (new, Sections 9.1, 9.2, 9.5, 9.6)
   ├── Mastcam-Z: Use IOF products (PDS4, pre-corrected)
   ├── HiRISE: hical → I/F → atmospheric correction (path radiance + transmittance)
   ├── Spectral band synthesis (Mastcam-Z filters → synthetic HiRISE RED)
   ├── Photometric correction (Minnaert, normalize to common geometry)
   └── PIF-based linear normalization (residual harmonization)

3. QUALITY CONTROL
   ├── NCC check: reject pairs with NCC < 0.3
   ├── Shadow filter: reject patches with >30% shadow
   ├── Coverage check: reject patches with >20% nodata
   ├── Distance filter: reject Mastcam-Z ortho from >50m (poor XYZ quality)
   └── Resolution check: verify Mastcam-Z GSD < target HR resolution

4. TRAINING PAIR GENERATION (Section 9.7)
   ├── Crop aligned patches: 160x160 HR (Mastcam-Z), 40x40 LR (HiRISE)
   ├── Apply Mars-specific augmentations
   ├── Split: train/val/test by sol (temporal split, no spatial leakage)
   └── Dataset statistics: compute and store per-patch metadata
       (sol, distance, tau, solar geometry, NCC score)

OUTPUT: Paired {LR_hirise, HR_mastcam, metadata} training dataset
```

### Priority Implementation Order

| Priority | Step | Effort | Impact |
|---|---|---|---|
| **P0** | I/F conversion (both sensors) | Low | Essential -- without this, training learns brightness, not texture |
| **P0** | Spectral band synthesis | Low | Essential -- must compare same wavelengths |
| **P0** | Shadow/occlusion masking | Medium | Essential -- prevents corrupted training pairs |
| **P1** | Feature-based alignment refinement | Medium | High -- SPICE alone gives 4-12 px error, need <2 px |
| **P1** | Atmospheric correction (HiRISE path radiance) | Medium | High -- 2-8% additive error in HiRISE signal |
| **P2** | Photometric correction to common geometry | Medium | Moderate -- augmentation may compensate partially |
| **P2** | PIF-based linear normalization | Low | Moderate -- catches residual gain/offset differences |
| **P3** | Mars-specific data augmentation | Low | Helpful -- improves generalization |
| **P3** | Curriculum learning | Low | Helpful -- stabilizes training |

---

## 9.10 Open Questions for Our Specific Application

1. **Optimal spectral synthesis**: Should we synthesize a HiRISE RED band from Mastcam-Z L-filters, or train on L0 broadband RGB and let the network learn the spectral mapping? The former is physically principled; the latter preserves more information.

2. **Photometric correction vs. augmentation**: Full Hapke correction requires per-pixel geometry computation and Mars-specific parameters. Would aggressive illumination augmentation (random gamma, contrast, brightness) during training achieve equivalent robustness with less preprocessing?

3. **Minimum alignment accuracy**: DATSR handles moderate misalignment, but what is the threshold for our specific resolution ratio? Needs empirical testing: train with varying alignment quality and measure PSNR/SSIM degradation.

4. **Atmospheric tau co-temporality**: Mastcam-Z tau measurements are available for every sol, but the HiRISE observation may be months or years earlier/later. Mars atmospheric conditions can change dramatically. Should we restrict training pairs to low-tau conditions (tau < 0.5) where atmospheric correction is less critical?

5. **Multi-sol fusion vs. single-frame**: Can we combine multiple Mastcam-Z observations of the same area (different sols, different geometries) into a single "best" ortho-image reference, or should each observation be a separate training pair?
