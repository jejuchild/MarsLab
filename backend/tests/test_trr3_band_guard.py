"""TRR3 wavelengths: 438-band IR cubes use the fixed table, S-detector (VNIR)
cubes use the CRISM WA CDR named in their label, others fail with 422."""

import numpy as np
import pytest

import api.crism_trr3.loader as L


def test_438_band_ir_cube_is_supported():
    L.require_supported_bands(438)


@pytest.mark.parametrize("bands", [90, 107, 3])
def test_ir_only_operations_reject_other_cubes_with_reason(bands):
    with pytest.raises(L.UnsupportedCubeError, match=f"{bands} bands"):
        L.require_supported_bands(bands)


@pytest.mark.parametrize("name,bands,samples", [
    ("CDR450924300802_WA0300000S_2.IMG", 107, 64),
    ("CDR421012262433_WA0200030S_2.IMG", 90, 128),
])
def test_bundled_vnir_wavelength_tables(name, bands, samples):
    if not (L.WA_DIR / name.lower()).exists():
        pytest.skip("WA CDR not present")
    t = L._wa_table(name, bands)
    assert t.shape == (bands, samples)
    wl = np.nanmedian(t, axis=1)
    assert np.isfinite(wl).all()
    assert 0.35 < wl.min() < 0.40 and 1.0 < wl.max() < 1.07  # S detector, µm
    assert np.all(np.diff(wl) > 0)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "CDR1/../x.IMG", "foo.img"])
def test_wavelength_file_name_is_validated(bad, monkeypatch, tmp_path):
    monkeypatch.setattr(L, "WA_DIR", tmp_path)
    with pytest.raises(L.UnsupportedCubeError):
        L._wa_file(bad)


def test_cube_without_wavelength_table_is_unsupported(monkeypatch):
    monkeypatch.setattr(L, "get_cube_memmap", lambda _o: (None, 10, 10, 3))
    monkeypatch.setattr(L, "resolve_trr_files", lambda _o: {"trr_lbl": "x"})
    monkeypatch.setattr(L, "parse_pds3_label", lambda _p: {})
    with pytest.raises(L.UnsupportedCubeError, match="3 bands"):
        L.get_cube_wavelengths("hrl0000a061")


def test_column_specific_wavelengths(monkeypatch):
    # 12 bands x 2 columns; column 1 is shifted by 0.01 µm (spectral smile).
    base = np.linspace(0.40, 0.95, 12)
    table = np.stack([base, base + 0.01], axis=1)
    table[3, 1] = np.nan  # a fill value in column 1
    monkeypatch.setattr(L, "get_cube_memmap", lambda _o: (None, 10, 2, 12))
    monkeypatch.setattr(L, "resolve_trr_files", lambda _o: {"trr_lbl": "x"})
    monkeypatch.setattr(L, "parse_pds3_label", lambda _p: {"MRO:WAVELENGTH_FILE_NAME": "CDR1_WA.IMG"})
    monkeypatch.setattr(L, "_wa_table", lambda _n, _b: table)
    np.testing.assert_allclose(L.get_cube_wavelengths("x", 0), base)
    # Column with fill values falls back to the median over columns.
    np.testing.assert_allclose(L.get_cube_wavelengths("x", 1)[3], base[3])
    np.testing.assert_allclose(L.get_cube_wavelengths("x")[0], base[0] + 0.005)


def test_truncated_wavelength_table_is_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "WA_DIR", tmp_path)
    (tmp_path / "cdr1_wa_short.img").write_bytes(np.zeros(10, "<f4").tobytes())
    with pytest.raises(L.UnsupportedCubeError, match="does not match"):
        L._wa_table.__wrapped__("CDR1_WA_SHORT.IMG", 107)


def test_l_detector_table_runs_long_to_short():
    """438-band TRR3 cubes are stored long→short wavelength; the CO2 2.0 µm
    band of real spectra lands at band ~284, not where an ascending table puts it."""
    name = "CDR410803692813_WA0000000L_3.IMG"
    if not (L.WA_DIR / name.lower()).exists():
        pytest.skip("WA CDR not present")
    with np.errstate(all="ignore"):
        wl = np.nanmedian(L._wa_table(name, 438), axis=1)
    assert 3.9 < wl[1] < 3.95 and 1.0 < wl[437] < 1.01
    assert np.all(np.diff(wl[1:]) < 0)
    assert abs(wl[284] - 2.01) < 0.01
