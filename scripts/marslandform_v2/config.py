"""
MarsLandformNet V2/V3 — Central configuration.
V2: Image-level MIL (deprecated).
V3: Tile-level classification with Levy 2014 polygon labels.
All hyperparams, paths, and constants in one place.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(os.getenv("MARSLAB_ROOT", "/disk1/cspark/MarsLab"))
DATA_DIR = ROOT / "Data" / "HiRISE"
MOLA_DEM = ROOT / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
METADATA_JSON = DATA_DIR / "midlat_metadata.json"
BROWSE_DIR = DATA_DIR / "midlat_browse"
V1_OUTPUT = DATA_DIR / "pipeline_output"

# V2 outputs (deprecated — kept for backward compat)
V2_OUTPUT = DATA_DIR / "v2_output"
EMBEDDINGS_DIR = V2_OUTPUT / "embeddings"
MODELS_DIR = V2_OUTPUT / "models"
PREDICTIONS_DIR = V2_OUTPUT / "predictions"
EVAL_DIR = V2_OUTPUT / "eval"
GEOJSON_DIR = V2_OUTPUT / "geojson"

# V3 outputs (tile-level)
V3_OUTPUT = DATA_DIR / "v3_output"
V3_MODELS_DIR = V3_OUTPUT / "models"
V3_EVAL_DIR = V3_OUTPUT / "eval"

# External datasets
EXTERNAL_DATA = ROOT / "Data" / "external_datasets"
LEVY_SHAPEFILE = EXTERNAL_DATA / "levy_2014_glacial" / "extracted" / "AreaIceUnits.shp"
# RAG
RAG_CORPUS_DIR = ROOT / "scripts" / "marslandform_v2" / "rag" / "corpus"
RAG_DB_DIR = V2_OUTPUT / "rag_db"

# Knowledge base (existing)
KNOWLEDGE_DIR = ROOT / "knowledge"

# ─── V2 Class Definitions (deprecated) ────────────────────────────────────────
CLASS_NAMES = ["LDA", "LVF", "CCF", "GLF"]
CLASS_ORDER = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
NUM_CLASSES = 5  # 4 landform + BACKGROUND
NUM_LANDFORM_CLASSES = 4

# ─── V3 Class Definitions (tile-level, Levy 2014 only) ────────────────────────
V3_CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
V3_NUM_CLASSES = 4  # 3 landform + OTHER
V3_NUM_LANDFORM_CLASSES = 3
V3_CLASS_DESCRIPTIONS = {
    "LDA": "Lobate Debris Apron — lobate aprons at bases of scarps, radially spreading",
    "LVF": "Lineated Valley Fill — valley-confined ice with longitudinal flow lineations",
    "CCF": "Concentric Crater Fill — concentric ridges filling craters, brain terrain",
    "OTHER": "Non-target terrain or insufficient glacial/periglacial evidence",
}

CLASS_DESCRIPTIONS = {
    "LDA": "Lobate Debris Apron — lobate-shaped aprons at bases of scarps, radially spreading, convex profiles",
    "LVF": "Lineated Valley Fill — valley-confined ice with longitudinal flow lineations, tributary patterns",
    "CCF": "Concentric Crater Fill — concentric ridges filling impact craters, brain terrain textures",
    "GLF": "Glacier-Like Form — small valley glaciers <10km, steep headwalls, nested moraines",
}

# ─── DINOv2 + LoRA Config ────────────────────────────────────────────────────
@dataclass
class DINOv2Config:
    model_name: str = "facebook/dinov2-base"  # ViT-B/14, 768-dim
    embed_dim: int = 768
    patch_size: int = 14
    image_size: int = 224  # input tile size

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value",  # HuggingFace DINOv2 attention projections
    ])
    unfreeze_last_n_blocks: int = 2  # unfreeze last 2 transformer blocks

    # SSL training
    ssl_lr: float = 5e-5
    ssl_layerwise_decay: float = 0.9
    ssl_epochs: int = 50
    ssl_warmup_epochs: int = 5
    ssl_batch_size: int = 64
    ssl_temperature: float = 0.04  # DINO loss temperature
    ssl_crop_scale: tuple[float, float] = (0.4, 1.0)  # Mars: larger crops

    # Mars-specific augmentation (NO color jitter — Mars is grayscale/near-IR)
    aug_rotation_degrees: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_gaussian_noise_std: float = 0.02
    aug_use_color_jitter: bool = False  # CRITICAL: disabled for Mars

# ─── MOLA Feature Config ─────────────────────────────────────────────────────
@dataclass
class MOLAConfig:
    dem_path: str = str(MOLA_DEM)
    # V3: extract per-tile (5.6km tiles → ~28 DEM pixels per side → adequate)
    extraction_level: str = "tile"  # "image" (V2) or "tile" (V3)
    scales_km: List[float] = field(default_factory=lambda: [1.0, 5.0, 20.0])
    features_per_scale: List[str] = field(default_factory=lambda: [
        "slope_mean", "slope_std", "curvature_mean",
        "TPI", "TRI", "roughness", "lobateness",
    ])
    global_features: List[str] = field(default_factory=lambda: [
        "elevation_mean", "abs_latitude",
    ])
    num_features: int = 25  # 7 × 3 scales + 2 global + 2 relative (V3)

@dataclass
class LabelConfig:
    sglf_max_distance_km: float = 5.0
    spatial_split_radius_km: float = 20.0
    title_regex_mode: str = "weak"
    reclassify_periglacial: bool = True
    min_confidence_for_train: str = "weak"

# ─── MIL Classifier Config ───────────────────────────────────────────────────
@dataclass
class MILConfig:
    # Attention-based MIL for tile→image aggregation
    embed_dim: int = 768  # DINOv2 ViT-B/14 output
    mola_dim: int = 23
    hidden_dim: int = 256
    attention_dim: int = 128
    num_attention_heads: int = 4
    num_classes: int = 5  # 4 landform + BACKGROUND
    dropout: float = 0.3

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15  # early stopping
    batch_size: int = 16  # 16 images (bags) per batch
    min_tiles_per_image: int = 5
    max_tiles_per_image: int = 200  # subsample if more

    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

# ─── V3 Tile Classifier Config ──────────────────────────────────────────────
@dataclass
class TileClassifierConfig:
    """Pure tile-level classifier: DINOv2 embedding + MOLA → per-tile class."""
    embed_dim: int = 768  # DINOv2 ViT-B/14 CLS output
    mola_dim: int = 25  # 23 base + 2 relative (elev_rel, slope_rel)
    hidden_dim: int = 256
    num_classes: int = 4  # LDA, LVF, CCF, OTHER
    dropout: float = 0.3

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15  # early stopping on landform macro-F1
    batch_size: int = 256  # tiles per batch (not bags)

    # Focal loss
    focal_gamma: float = 1.5
    label_smoothing: float = 0.1

    # Soft label handling
    confident_threshold: float = 0.6  # polygon coverage → hard label
    mixed_threshold: float = 0.2  # → soft label (KL-div loss)
    mixed_loss_weight: float = 0.5  # weight for soft-label tiles
    other_subsample_ratio: float = 0.3  # subsample OTHER to prevent dominance

    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
# ─── RAG Config ───────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/allenai-specter"  # SPECTER for scientific text
    collection_name: str = "mars_landform_papers"
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64
    top_k: int = 5  # retrieval top-k
    similarity_threshold: float = 0.7
    db_path: str = str(RAG_DB_DIR)

# ─── Agent Config ─────────────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    # ReACT loop
    max_steps: int = 5
    confidence_threshold: float = 0.7  # below this → invoke agent
    vlm_model: str = "claude-sonnet-4-20250514"  # or local: "Qwen/Qwen2.5-VL-7B"
    vlm_temperature: float = 0.3
    vlm_max_tokens: int = 2048

    # Runtime modes
    mode: str = "fast"  # "fast" (classifier only) or "agent" (full ReACT)

# ─── Pipeline Config (top-level) ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    dinov2: DINOv2Config = field(default_factory=DINOv2Config)
    mola: MOLAConfig = field(default_factory=MOLAConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    mil: MILConfig = field(default_factory=MILConfig)  # V2 (deprecated)
    tile_classifier: TileClassifierConfig = field(default_factory=TileClassifierConfig)  # V3
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Global settings
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"  # auto-detect: falls back to CPU
    mixed_precision: bool = True  # fp16 on GPU
    pipeline_version: str = "v3"  # "v2" or "v3"

    def __post_init__(self):
        """Auto-detect device and create output dirs."""
        import torch
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.mixed_precision = False
            self.dinov2.ssl_batch_size = 8  # reduce for CPU
            self.mil.batch_size = 4
            self.tile_classifier.batch_size = 64

        # Create output directories
        for d in [V2_OUTPUT, V3_OUTPUT, V3_MODELS_DIR, V3_EVAL_DIR,
                  EMBEDDINGS_DIR, MODELS_DIR,
                  PREDICTIONS_DIR, EVAL_DIR, GEOJSON_DIR,
                  RAG_DB_DIR, EXTERNAL_DATA]:
            d.mkdir(parents=True, exist_ok=True)

# Convenience: default config singleton
def get_config() -> PipelineConfig:
    return PipelineConfig()
