"""
Unit tests for Phase 3: Hyperbola-based εr estimation + cross-validation.

Tests cover:
  - _cross_validate_epsilon_methods: pairwise agreement, consensus, conflicts
  - hyperbola_epsilon_estimation output structure (mocked)
  - Evidence pack hyperbola and cross-validation fields
  - Method hierarchy with hyperbola
  - Agreement thresholds
"""
import math
import sys
import os
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from api.agent_tasks import _cross_validate_epsilon_methods


# ─────────────────────────────────────────────────────────────────────────────
# 1. _cross_validate_epsilon_methods
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossValidateEpsilonMethods:
    """Test the Phase 3 multi-method εr cross-validation function."""

    def test_two_consistent_methods(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.1},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.3},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["n_methods"] == 2
        assert result["overall_agreement"] == "consistent"
        assert len(result["pairwise_comparisons"]) == 1
        assert result["pairwise_comparisons"][0]["delta_epsilon"] == 0.2
        assert result["pairwise_comparisons"][0]["agreement"] == "consistent"

    def test_two_marginal_methods(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 4.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["overall_agreement"] == "marginal"
        assert result["pairwise_comparisons"][0]["delta_epsilon"] == 1.0

    def test_two_inconsistent_methods(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 6.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["overall_agreement"] == "inconsistent"
        assert result["n_inconsistent"] == 1
        assert len(result["conflicts"]) == 1

    def test_three_methods_all_consistent(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.2},
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 3.1},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["n_methods"] == 3
        # 3 choose 2 = 3 pairwise comparisons
        assert len(result["pairwise_comparisons"]) == 3
        assert result["overall_agreement"] == "consistent"
        assert result["n_consistent"] == 3

    def test_three_methods_one_inconsistent(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.1},
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 7.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["overall_agreement"] == "inconsistent"
        assert result["n_inconsistent"] >= 1
        assert len(result["conflicts"]) >= 1

    def test_consensus_epsilon_r_weighted(self):
        """Consensus should weight physics_inversion (1.0) > terraced_crater (0.8)."""
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 4.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        consensus = result["consensus_epsilon_r"]
        # Weighted: (3.0*1.0 + 4.0*0.8) / (1.0+0.8) = (3.0+3.2)/1.8 = 3.44
        assert abs(consensus - 3.44) < 0.01

    def test_consensus_with_hyperbola(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 3.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        # (3.0*1.0 + 3.0*0.6) / (1.0+0.6) = 4.8/1.6 = 3.0
        assert result["consensus_epsilon_r"] == 3.0

    def test_method_weights_present(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.0},
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 3.0},
        ]
        result = _cross_validate_epsilon_methods(methods)
        weights = result["consensus_method_weights"]
        assert weights["physics_inversion"] == 1.0
        assert weights["terraced_crater"] == 0.8
        assert weights["hyperbola_curvature"] == 0.6

    def test_agreement_threshold_exact_boundary(self):
        """Delta exactly 0.5 should be marginal (not consistent)."""
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.5},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["pairwise_comparisons"][0]["agreement"] == "marginal"

    def test_agreement_threshold_exact_1_5(self):
        """Delta exactly 1.5 should be inconsistent."""
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 4.5},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["pairwise_comparisons"][0]["agreement"] == "inconsistent"

    def test_conflicts_empty_when_consistent(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.1},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert result["conflicts"] == []

    def test_methods_list_in_output(self):
        methods = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "terraced_crater", "status": "success", "epsilon_r": 3.1},
        ]
        result = _cross_validate_epsilon_methods(methods)
        assert len(result["methods"]) == 2
        assert result["methods"][0]["method"] == "physics_inversion"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hyperbola estimation output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestHyperbolaEpsilonEstimation:
    """Test the hyperbola_epsilon_estimation function structure."""

    def test_no_subsurface_result(self):
        from api.agent_tasks import hyperbola_epsilon_estimation, TaskResult
        result = hyperbola_epsilon_estimation(None, [])
        assert result.task_type == "hyperbola_epsilon"
        assert result.success is True
        assert result.data["estimates_count"] == 0

    def test_no_sharad_products(self):
        from api.agent_tasks import hyperbola_epsilon_estimation, TaskResult
        sub = TaskResult(task_type="subsurface", success=True, data={"tracks": []})
        result = hyperbola_epsilon_estimation(sub, [])
        assert result.data["estimates_count"] == 0
        assert result.data["reason"] == "no_sharad_highres"

    def test_no_sharad_products_with_other_instruments(self):
        from api.agent_tasks import hyperbola_epsilon_estimation, TaskResult
        sub = TaskResult(task_type="subsurface", success=True, data={"tracks": []})
        prods = [{"instrument": "CRISM", "product_id": "crism_001"}]
        result = hyperbola_epsilon_estimation(sub, prods)
        assert result.data["estimates_count"] == 0

    def test_failed_subsurface_result(self):
        from api.agent_tasks import hyperbola_epsilon_estimation, TaskResult
        sub = TaskResult(task_type="subsurface", success=False, data={})
        prods = [{"instrument": "SHARAD_HIGHRES", "product_id": "test_001"}]
        result = hyperbola_epsilon_estimation(sub, prods)
        assert result.data["estimates_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evidence Pack Phase 3 fields
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidencePackPhase3:
    """Test evidence_pack assembly with Phase 3 hyperbola and cross-validation."""

    def _make_session(self, hyperbola_data=None, cross_val_data=None):
        from types import SimpleNamespace
        synth = {
            "subsurface_coverage": {},
            "dielectric_analysis": {},
            "terrace_dielectric": {},
            "sharad_physics_inversion": {},
            "hyperbola_epsilon": hyperbola_data or {},
            "epsilon_cross_validation": cross_val_data or {},
            "dielectric_method_hierarchy": [],
            "mineral_signatures": {},
            "crism_spectral_analysis": {},
            "terrain_favorability": {},
            "climate_compatibility": {},
            "scoring": {},
            "isru_assessment": {},
        }
        return SimpleNamespace(
            synthesis=synth,
            bbox=None,
            region_name="Test Region",
            session_id="test-session-003",
            objective="Test Phase 3",
            narrative="",
        )

    def test_hyperbola_fields_present(self):
        from api.evidence_pack import assemble_evidence_pack
        session = self._make_session(hyperbola_data={
            "estimates_count": 3,
            "median_epsilon_r": 3.2,
            "epsilon_r_ci95": [2.8, 3.6],
            "n_good": 2,
            "ice_consistent_count": 1,
            "fits": [],
        })
        pack = assemble_evidence_pack(session)
        diel = pack["dielectric"]
        assert diel["hyperbola_estimates_count"] == 3
        assert diel["hyperbola_median_epsilon_r"] == 3.2
        assert diel["hyperbola_epsilon_r_ci95"] == [2.8, 3.6]
        assert diel["hyperbola_n_good"] == 2
        assert diel["hyperbola_ice_consistent"] == 1

    def test_cross_validation_present(self):
        from api.evidence_pack import assemble_evidence_pack
        cv = {
            "n_methods": 2,
            "consensus_epsilon_r": 3.15,
            "pairwise_comparisons": [{
                "method_a": "physics_inversion",
                "method_b": "terraced_crater",
                "epsilon_r_a": 3.1,
                "epsilon_r_b": 3.2,
                "delta_epsilon": 0.1,
                "agreement": "consistent",
            }],
            "overall_agreement": "consistent",
        }
        session = self._make_session(cross_val_data=cv)
        pack = assemble_evidence_pack(session)
        diel = pack["dielectric"]
        assert diel["cross_validation"]["n_methods"] == 2
        assert diel["cross_validation"]["consensus_epsilon_r"] == 3.15

    def test_no_hyperbola_defaults(self):
        from api.evidence_pack import assemble_evidence_pack
        session = self._make_session()
        pack = assemble_evidence_pack(session)
        diel = pack["dielectric"]
        assert diel["hyperbola_estimates_count"] == 0
        assert diel["hyperbola_median_epsilon_r"] is None

    def test_consensus_preferred_for_eps_val(self):
        """When cross_validation consensus is available, it should be used."""
        from api.evidence_pack import assemble_evidence_pack
        cv = {
            "consensus_epsilon_r": 3.25,
            "n_methods": 2,
        }
        session = self._make_session(cross_val_data=cv)
        pack = assemble_evidence_pack(session)
        # Consensus εr should be picked (no physics_inv.best_epsilon_r)
        assert pack["dielectric"]["epsilon_r"] == 3.25


# ─────────────────────────────────────────────────────────────────────────────
# 4. Method hierarchy integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMethodHierarchyPhase3:
    """Test that hyperbola is properly added to method hierarchy."""

    def test_hyperbola_in_successful_methods(self):
        """Verify the method hierarchy logic recognizes hyperbola_curvature."""
        # Simulate what synthesize_results does
        method_hierarchy = [
            {"method": "physics_inversion", "status": "success", "epsilon_r": 3.0},
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 3.1},
        ]
        successful_physics = [
            m for m in method_hierarchy
            if m["status"] == "success" and m["method"] in ("physics_inversion", "terraced_crater", "hyperbola_curvature")
        ]
        assert len(successful_physics) == 2

    def test_hyperbola_as_epsilon_source(self):
        """Hyperbola should be accepted as a valid εr source."""
        method_hierarchy = [
            {"method": "hyperbola_curvature", "status": "success", "epsilon_r": 3.1},
        ]
        successful_physics = [
            m for m in method_hierarchy
            if m["status"] == "success" and m["method"] in ("physics_inversion", "terraced_crater", "hyperbola_curvature")
        ]
        assert len(successful_physics) == 1
        assert successful_physics[0]["method"] == "hyperbola_curvature"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Agreement thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestAgreementThresholds:
    """Test the cross-validation agreement classification thresholds."""

    def _check_agreement(self, eps1, eps2):
        methods = [
            {"method": "a", "status": "success", "epsilon_r": eps1},
            {"method": "b", "status": "success", "epsilon_r": eps2},
        ]
        return _cross_validate_epsilon_methods(methods)["pairwise_comparisons"][0]["agreement"]

    def test_consistent_below_0_5(self):
        assert self._check_agreement(3.0, 3.4) == "consistent"

    def test_marginal_at_0_5(self):
        assert self._check_agreement(3.0, 3.5) == "marginal"

    def test_marginal_at_1_0(self):
        assert self._check_agreement(3.0, 4.0) == "marginal"

    def test_inconsistent_at_1_5(self):
        assert self._check_agreement(3.0, 4.5) == "inconsistent"

    def test_inconsistent_large_delta(self):
        assert self._check_agreement(3.0, 8.0) == "inconsistent"

    def test_consistent_identical(self):
        assert self._check_agreement(3.1, 3.1) == "consistent"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bug fix verification: epsilon_r_curvature field name
# ─────────────────────────────────────────────────────────────────────────────

class TestBugFixEpsilonFieldName:
    """Verify the bug fix: epsilon_r_hyperbola → epsilon_r_curvature."""

    def test_field_name_in_model(self):
        from analysis.sharad_inversion.models import HyperbolaValidation
        hv = HyperbolaValidation(
            performed=True,
            epsilon_r_curvature=3.15,
        )
        assert hv.epsilon_r_curvature == 3.15
        # Ensure the old field name doesn't exist
        assert not hasattr(hv, "epsilon_r_hyperbola")
