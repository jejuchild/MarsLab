import os
import sys
import importlib

import numpy as np
import pytest
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

pi = importlib.import_module("pinns_interior")

MarsInteriorPINN = pi.MarsInteriorPINN
MARS_RADIUS_KM = pi.MARS_RADIUS_KM
compute_travel_time = pi.compute_travel_time
generate_synthetic_data = pi.generate_synthetic_data
get_reference_model = pi.get_reference_model


class TestReferenceModel:
    def test_shapes(self):
        radius, vp, vs, density = get_reference_model(n_layers=120)
        assert radius.shape == (120,)
        assert vp.shape == (120,)
        assert vs.shape == (120,)
        assert density.shape == (120,)

    def test_physical_bounds(self):
        radius, vp, vs, density = get_reference_model(n_layers=150)
        depth = MARS_RADIUS_KM - radius
        assert np.all(vp > 0.0)
        assert np.all(density > 0.0)
        assert np.all(vs[depth > 1500.0] == 0.0)
        assert np.all(vs[depth <= 1500.0] >= 0.0)

    def test_layer_transition_core_drop(self):
        radius, vp, _, _ = get_reference_model(n_layers=600)
        depth = MARS_RADIUS_KM - radius
        mantle_idx = np.argmin(np.abs(depth - 1490.0))
        core_idx = np.argmin(np.abs(depth - 1510.0))
        assert vp[core_idx] < vp[mantle_idx]


class TestForwardModel:
    def test_travel_time_positive(self):
        ref = get_reference_model(n_layers=180)
        t = compute_travel_time(ref, 60.0)
        assert t > 0.0

    def test_travel_time_monotonic_with_distance(self):
        ref = get_reference_model(n_layers=180)
        t1 = compute_travel_time(ref, 30.0)
        t2 = compute_travel_time(ref, 60.0)
        t3 = compute_travel_time(ref, 120.0)
        assert t1 < t2 < t3

    def test_synthetic_data_shapes(self):
        ref = get_reference_model(n_layers=180)
        d, t = generate_synthetic_data(ref, n_events=25)
        assert d.shape == (25,)
        assert t.shape == (25,)


class TestPINNModel:
    def test_forward_shape(self):
        model = MarsInteriorPINN(hidden_dim=64)
        r = torch.rand(16, 1)
        v = model(r)
        assert v.shape == (16, 1)

    def test_output_positive(self):
        model = MarsInteriorPINN(hidden_dim=64)
        r = torch.linspace(0.0, 1.0, 128).unsqueeze(-1)
        v = model(r)
        assert float(v.min().detach()) > 0.0


class TestPredictor:
    @pytest.fixture(autouse=True)
    def check_checkpoint(self):
        if not pi.is_model_trained():
            pytest.skip("No trained PINNs checkpoint")

    def test_predict_velocity(self):
        predictor = pi.MarsInteriorPredictor.load(device="cpu")
        vp = predictor.predict_velocity(np.array([0.0, 100.0, 500.0], dtype=np.float32))
        assert vp.shape == (3,)
        assert np.all(vp > 0.0)


class TestRouter:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        pinns_router = importlib.import_module("pinns_interior.pinns_router").router

        app = FastAPI()
        app.include_router(pinns_router)
        return TestClient(app)

    def test_reference_endpoint(self, client):
        resp = client.get("/api/pinns/reference-model")
        assert resp.status_code == 200
        payload = resp.json()
        assert "vp_km_s" in payload
        assert "radius_km" in payload

    def test_status_endpoint(self, client):
        resp = client.get("/api/pinns/status")
        assert resp.status_code == 200
        payload = resp.json()
        assert "trained" in payload
        assert "status" in payload

    def test_train_predict_profile_compare_endpoints(self, client):
        train_resp = client.post("/api/pinns/train", json={"epochs": 120, "lr": 0.001, "seed": 3})
        assert train_resp.status_code == 200
        assert train_resp.json()["status"] == "trained"

        predict_resp = client.post("/api/pinns/predict", json={"depths_km": [0.0, 200.0, 1000.0]})
        assert predict_resp.status_code == 200
        pred = predict_resp.json()
        assert len(pred["vp_km_s"]) == 3

        profile_resp = client.post("/api/pinns/profile", json={"n_points": 64})
        assert profile_resp.status_code == 200
        profile = profile_resp.json()
        assert len(profile["depth_km"]) == 64

        compare_resp = client.post("/api/pinns/compare", json={"n_points": 64})
        assert compare_resp.status_code == 200
        compare = compare_resp.json()
        assert "mae_km_s" in compare
