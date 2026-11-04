"""TRR3 cubes without a wavelength table must fail cleanly (422), not 500."""

import pytest

from api.crism_trr3.loader import UnsupportedCubeError, require_supported_bands


def test_438_band_ir_cube_is_supported():
    require_supported_bands(438)


@pytest.mark.parametrize("bands", [90, 107, 3])
def test_vnir_and_survey_cubes_are_rejected_with_reason(bands):
    with pytest.raises(UnsupportedCubeError, match=f"{bands} bands"):
        require_supported_bands(bands)
