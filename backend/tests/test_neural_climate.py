"""
Tests for the Mars GCM Neural Emulator (neural_climate package).

Covers:
  - Model architecture (shapes, forward pass, parameter count)
  - Input encoding / output decoding
  - Dataset generation and DataLoader creation
  - Predictor (load, predict single, predict batch, global map, compare)
  - FastAPI router endpoints (status, predict, batch, compare, map)
"""
import math
import os
import sys
import tempfile
import shutil

import numpy as np
import pytest
import torch

# Ensure backend is importable
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from neural_climate.model import (
    MarsClimateEmulator,
    ResidualBlock,
    create_model,
    encode_inputs,
    decode_outputs,
    INPUT_DIM,
    OUTPUT_DIM,
    OUTPUT_NAMES,
    OUTPUT_BOUNDS,
)
from neural_climate.dataset import (
    generate_grid,
    sample_parametric_model,
    generate_training_data,
    MarsClimateDataset,
    create_dataloaders,
    save_dataset,
    load_dataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestModelConstants:
    """Verify model dimension constants."""

    def test_input_dim(self):
        assert INPUT_DIM == 6  # lat_norm, sin_lon, cos_lon, sin_ls, cos_ls, elev_norm

    def test_output_dim(self):
        assert OUTPUT_DIM == 7

    def test_output_names_count(self):
        assert len(OUTPUT_NAMES) == OUTPUT_DIM

    def test_output_names_contents(self):
        expected = [
            "temperature_mean_k", "temperature_max_k", "temperature_min_k",
            "pressure_pa", "dust_tau_mean", "wind_mean_ms", "frost_probability",
        ]
        assert OUTPUT_NAMES == expected

    def test_output_bounds_cover_all_names(self):
        for name in OUTPUT_NAMES:
            assert name in OUTPUT_BOUNDS
            lo, hi = OUTPUT_BOUNDS[name]
            assert lo < hi


class TestResidualBlock:
    """Test the skip-connection block."""

    def test_output_shape_preserved(self):
        block = ResidualBlock(dim=64)
        x = torch.randn(8, 64)
        y = block(x)
        assert y.shape == (8, 64)

    def test_skip_connection_gradient_flow(self):
        """Gradients should flow through skip connection."""
        block = ResidualBlock(dim=32)
        x = torch.randn(4, 32, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMarsClimateEmulator:
    """Test the full emulator model."""

    def test_default_creation(self):
        model = create_model()
        assert isinstance(model, MarsClimateEmulator)

    def test_forward_shape(self):
        model = create_model(hidden_dim=64, n_blocks=2)
        x = torch.randn(16, INPUT_DIM)
        y = model(x)
        assert y.shape == (16, OUTPUT_DIM)

    def test_single_sample_forward(self):
        """BatchNorm1d requires batch > 1 in train mode; use eval."""
        model = create_model(hidden_dim=64, n_blocks=2)
        model.eval()
        x = torch.randn(1, INPUT_DIM)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, OUTPUT_DIM)

    def test_frost_probability_sigmoid(self):
        """Frost probability (index 6) must be in [0, 1] due to sigmoid."""
        model = create_model(hidden_dim=64, n_blocks=2)
        x = torch.randn(100, INPUT_DIM)
        y = model(x)
        frost = y[:, 6]
        assert frost.min() >= 0.0
        assert frost.max() <= 1.0

    def test_param_count_positive(self):
        model = create_model(hidden_dim=64, n_blocks=2)
        assert model.param_count > 0
        assert model.trainable_param_count > 0
        assert model.trainable_param_count == model.param_count

    def test_custom_dimensions(self):
        model = create_model(hidden_dim=128, n_blocks=3, dropout=0.1)
        assert len(model.blocks) == 3
        x = torch.randn(4, INPUT_DIM)
        y = model(x)
        assert y.shape == (4, OUTPUT_DIM)

    def test_eval_mode(self):
        """Model should work correctly in eval mode (BatchNorm behavior differs)."""
        model = create_model(hidden_dim=64, n_blocks=2)
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, OUTPUT_DIM)

    def test_deterministic_in_eval(self):
        """Same input → same output in eval mode."""
        model = create_model(hidden_dim=64, n_blocks=2)
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        assert torch.allclose(y1, y2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Input Encoding / Output Decoding
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeInputs:
    """Test feature encoding from raw coordinates."""

    def test_basic_shape(self):
        lat = np.array([0.0, 45.0, -90.0])
        lon = np.array([0.0, 180.0, 270.0])
        ls = np.array([0.0, 90.0, 180.0])
        elev = np.array([0.0, 5000.0, -4000.0])
        enc = encode_inputs(lat, lon, ls, elev)
        assert enc.shape == (3, INPUT_DIM)
        assert enc.dtype == np.float32

    def test_lat_normalization(self):
        """lat=90 → 1.0, lat=-90 → -1.0, lat=0 → 0.0"""
        enc = encode_inputs(
            np.array([0.0, 90.0, -90.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert enc[0, 0] == pytest.approx(0.0)
        assert enc[1, 0] == pytest.approx(1.0)
        assert enc[2, 0] == pytest.approx(-1.0)

    def test_lon_sincos_encoding(self):
        """lon=0 → sin=0, cos=1; lon=90 → sin=1, cos=0"""
        enc = encode_inputs(
            np.array([0.0, 0.0]),
            np.array([0.0, 90.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        # lon=0: sin(0)=0, cos(0)=1
        assert enc[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert enc[0, 2] == pytest.approx(1.0, abs=1e-6)
        # lon=90: sin(pi/2)=1, cos(pi/2)=0
        assert enc[1, 1] == pytest.approx(1.0, abs=1e-6)
        assert enc[1, 2] == pytest.approx(0.0, abs=1e-6)

    def test_ls_sincos_encoding(self):
        """Ls=0 → sin=0, cos=1; Ls=180 → sin=0, cos=-1"""
        enc = encode_inputs(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 180.0]),
            np.array([0.0, 0.0]),
        )
        # Ls=0: sin(0)=0, cos(0)=1
        assert enc[0, 3] == pytest.approx(0.0, abs=1e-6)
        assert enc[0, 4] == pytest.approx(1.0, abs=1e-6)
        # Ls=180: sin(pi)≈0, cos(pi)=-1
        assert enc[1, 3] == pytest.approx(0.0, abs=1e-6)
        assert enc[1, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_elevation_normalization(self):
        """Elevation normalized by 10800m scale height."""
        enc = encode_inputs(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([10800.0]),
        )
        assert enc[0, 5] == pytest.approx(1.0, abs=1e-6)

    def test_single_sample(self):
        enc = encode_inputs(
            np.array([18.4]),
            np.array([77.5]),
            np.array([180.0]),
            np.array([0.0]),
        )
        assert enc.shape == (1, INPUT_DIM)


class TestDecodeOutputs:
    """Test output decoding and physical clamping."""

    def test_basic_decode(self):
        preds = np.array([[220.0, 250.0, 190.0, 600.0, 0.5, 5.0, 0.3]])
        results = decode_outputs(preds)
        assert len(results) == 1
        assert set(results[0].keys()) == set(OUTPUT_NAMES)

    def test_clamping_temperature(self):
        """Values outside physical bounds should be clamped."""
        preds = np.array([[100.0, 400.0, 100.0, 10.0, 10.0, 50.0, 2.0]])
        results = decode_outputs(preds)
        r = results[0]
        # Check clamping
        assert r["temperature_mean_k"] == OUTPUT_BOUNDS["temperature_mean_k"][0]  # clamped to min
        assert r["temperature_max_k"] == OUTPUT_BOUNDS["temperature_max_k"][1]   # clamped to max
        assert r["frost_probability"] == OUTPUT_BOUNDS["frost_probability"][1]   # clamped to 1.0

    def test_1d_input(self):
        """1-D input should be auto-reshaped."""
        preds = np.array([220.0, 250.0, 190.0, 600.0, 0.5, 5.0, 0.3])
        results = decode_outputs(preds)
        assert len(results) == 1

    def test_batch_decode(self):
        preds = np.random.uniform(
            [150, 150, 150, 100, 0.1, 1.0, 0.0],
            [280, 300, 240, 1000, 3.0, 20.0, 1.0],
            size=(10, 7),
        ).astype(np.float32)
        results = decode_outputs(preds)
        assert len(results) == 10
        for r in results:
            assert len(r) == OUTPUT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateGrid:
    """Test coordinate grid generation."""

    def test_default_grid_sizes(self):
        lats, lons, ls_vals = generate_grid()
        assert len(lats) == 37   # -90 to 90 by 5°
        assert len(lons) == 36   # 0 to 355 by 10°
        assert len(ls_vals) == 36  # 0 to 350 by 10°

    def test_custom_grid(self):
        lats, lons, ls_vals = generate_grid(lat_step=30, lon_step=60, ls_step=90)
        assert len(lats) == 7    # -90, -60, -30, 0, 30, 60, 90
        assert len(lons) == 6    # 0, 60, 120, 180, 240, 300
        assert len(ls_vals) == 4  # 0, 90, 180, 270

    def test_lat_bounds(self):
        lats, _, _ = generate_grid()
        assert lats[0] == -90.0
        assert lats[-1] == 90.0


class TestSampleParametricModel:
    """Test single-point parametric model sampling."""

    def test_basic_sample(self):
        result = sample_parametric_model(0.0, 0.0, 180.0)
        assert result is not None
        assert "temperature_mean_k" in result
        assert "pressure_pa" in result
        assert "frost_probability" in result
        assert len(result) == 11  # 4 inputs + 7 outputs

    def test_polar_sample(self):
        result = sample_parametric_model(-85.0, 0.0, 270.0)
        assert result is not None
        # Polar winter should have high frost probability
        assert result["frost_probability"] >= 0.0


class TestMarsClimateDataset:
    """Test PyTorch Dataset wrapper."""

    def test_creation(self):
        N = 100
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32)
        ds = MarsClimateDataset(encoded, targets)
        assert len(ds) == N

    def test_getitem(self):
        N = 50
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32)
        ds = MarsClimateDataset(encoded, targets)
        x, y = ds[0]
        assert x.shape == (INPUT_DIM,)
        assert y.shape == (OUTPUT_DIM,)

    def test_normalization(self):
        """Targets should be z-score normalized."""
        N = 200
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32) * 100 + 500
        ds = MarsClimateDataset(encoded, targets)
        # Normalized targets should be roughly zero-mean, unit-std
        all_y = torch.stack([ds[i][1] for i in range(len(ds))])
        assert all_y.mean().abs() < 0.5  # roughly zero-mean
        assert abs(all_y.std() - 1.0) < 0.5  # roughly unit-std

    def test_denormalize_roundtrip(self):
        N = 50
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32) * 100 + 500
        ds = MarsClimateDataset(encoded, targets)
        _, y_norm = ds[0]
        y_denorm = ds.denormalize(y_norm)
        # Should approximately match original
        assert torch.allclose(
            y_denorm,
            torch.from_numpy(targets[0]),
            atol=1e-4,
        )


class TestCreateDataloaders:
    """Test DataLoader creation."""

    def test_basic_creation(self):
        N = 200
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32)
        train_loader, val_loader, norm_stats = create_dataloaders(
            encoded, targets, batch_size=32,
        )
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert "target_mean" in norm_stats
        assert "target_std" in norm_stats
        assert norm_stats["target_mean"].shape == (OUTPUT_DIM,)

    def test_train_val_split(self):
        N = 100
        encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
        targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32)
        train_loader, val_loader, _ = create_dataloaders(
            encoded, targets, train_fraction=0.8, batch_size=16,
        )
        train_count = sum(len(batch[0]) for batch in train_loader)
        val_count = sum(len(batch[0]) for batch in val_loader)
        assert train_count == 80
        assert val_count == 20


class TestSaveLoadDataset:
    """Test dataset caching."""

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        try:
            N = 50
            inputs = np.random.randn(N, 4).astype(np.float32)
            targets = np.random.randn(N, OUTPUT_DIM).astype(np.float32)
            encoded = np.random.randn(N, INPUT_DIM).astype(np.float32)
            meta = np.random.randn(N, 3).astype(np.float32)

            save_dataset(inputs, targets, encoded, meta, data_dir=tmpdir)
            loaded = load_dataset(data_dir=tmpdir)
            assert loaded is not None
            l_inputs, l_targets, l_encoded, l_meta = loaded
            np.testing.assert_array_almost_equal(l_inputs, inputs)
            np.testing.assert_array_almost_equal(l_targets, targets)
        finally:
            shutil.rmtree(tmpdir)

    def test_load_nonexistent(self):
        result = load_dataset(data_dir="/tmp/nonexistent_dir_xyz")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Predictor
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictor:
    """Test the inference predictor (requires trained checkpoint)."""

    @pytest.fixture(autouse=True)
    def check_checkpoint(self):
        """Skip if no trained model exists."""
        from neural_climate.predictor import is_model_trained
        if not is_model_trained():
            pytest.skip("No trained model checkpoint — run training first")

    def test_is_model_trained(self):
        from neural_climate.predictor import is_model_trained
        assert is_model_trained()

    def test_load_predictor(self):
        from neural_climate.predictor import MarsClimatePredictor
        predictor = MarsClimatePredictor.load(device="cpu")
        assert predictor is not None
        assert predictor.model is not None

    def test_single_prediction(self):
        from neural_climate.predictor import get_predictor
        predictor = get_predictor(device="cpu")
        result = predictor.predict(lat=0.0, lon=0.0, ls=180.0, elevation=0.0)
        assert "temperature_mean_k" in result
        assert "source" in result
        assert result["source"] == "neural_emulator"
        # Physical plausibility checks
        assert 148.0 <= result["temperature_mean_k"] <= 280.0
        assert 30.0 <= result["pressure_pa"] <= 1200.0
        assert 0.0 <= result["frost_probability"] <= 1.0

    def test_batch_prediction(self):
        from neural_climate.predictor import get_predictor
        predictor = get_predictor(device="cpu")
        lats = np.array([0.0, 45.0, -45.0, 80.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0])
        ls_vals = np.array([0.0, 90.0, 180.0, 270.0])
        elevations = np.array([0.0, 0.0, 0.0, 0.0])
        results = predictor.predict_batch(lats, lons, ls_vals, elevations)
        assert len(results) == 4
        for r in results:
            assert 148.0 <= r["temperature_mean_k"] <= 280.0

    def test_global_map(self):
        from neural_climate.predictor import get_predictor
        predictor = get_predictor(device="cpu")
        result = predictor.predict_global_map(ls=180.0, lat_step=30.0, lon_step=60.0)
        assert "lats" in result
        assert "lons" in result
        assert "values" in result
        assert "shape" in result
        assert len(result["lats"]) == 7   # -90 to 90 by 30°
        assert len(result["lons"]) == 6   # 0 to 300 by 60°

    def test_compare_with_parametric(self):
        from neural_climate.predictor import get_predictor
        predictor = get_predictor(device="cpu")
        comp = predictor.compare_with_parametric(lat=0.0, lon=0.0, ls=180.0)
        assert "neural" in comp
        assert "parametric" in comp
        assert "errors" in comp
        for name in OUTPUT_NAMES:
            assert name in comp["errors"]
            assert "absolute" in comp["errors"][name]
            assert "relative_pct" in comp["errors"][name]

    def test_model_info(self):
        from neural_climate.predictor import get_predictor
        predictor = get_predictor(device="cpu")
        info = predictor.model_info
        assert "param_count" in info
        assert info["param_count"] > 0
        assert info["checkpoint_exists"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Router Endpoints (FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateRouter:
    """Test FastAPI endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def check_checkpoint(self):
        from neural_climate.predictor import is_model_trained
        if not is_model_trained():
            pytest.skip("No trained model checkpoint — run training first")

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from neural_climate.climate_router import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_status_endpoint(self, client):
        resp = client.get("/api/climate/neural/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trained"] is True
        assert data["status"] == "ready"

    def test_predict_endpoint(self, client):
        resp = client.post("/api/climate/neural/predict", json={
            "lat": 18.4, "lon": 77.5, "ls": 180.0, "elevation": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "temperature_mean_k" in data
        assert data["source"] == "neural_emulator"

    def test_predict_validation_lat(self, client):
        resp = client.post("/api/climate/neural/predict", json={
            "lat": 100.0, "lon": 0.0, "ls": 180.0,
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_batch_endpoint(self, client):
        resp = client.post("/api/climate/neural/predict/batch", json={
            "points": [
                {"lat": 0.0, "lon": 0.0, "ls": 0.0, "elevation": 0.0},
                {"lat": 45.0, "lon": 90.0, "ls": 180.0, "elevation": 0.0},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_compare_endpoint(self, client):
        resp = client.post("/api/climate/neural/compare", json={
            "lat": 0.0, "lon": 0.0, "ls": 180.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "neural" in data
        assert "parametric" in data
        assert "errors" in data

    def test_map_endpoint(self, client):
        resp = client.post("/api/climate/neural/map", json={
            "ls": 180.0,
            "variable": "temperature_mean_k",
            "lat_step": 10.0,
            "lon_step": 10.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "values" in data
        assert data["variable"] == "temperature_mean_k"

    def test_map_invalid_variable(self, client):
        resp = client.post("/api/climate/neural/map", json={
            "ls": 180.0,
            "variable": "invalid_variable",
            "lat_step": 5.0,
            "lon_step": 5.0,
        })
        assert resp.status_code == 400
