"""
Geochemical group taxonomy and paleo-environment sequence matching.

Maps CNN mineral class IDs to broader geochemical groups, and matches
observed mineral sequences against canonical paleo-environment patterns.
"""

from typing import Dict, List, Optional

# ── Geochemical Groups ────────────────────────────────────────────
# Maps group name → list of CNN mineral class IDs (from mineral_cnn/constants.py CLASS_NAME)
GEOCHEM_GROUPS: Dict[str, List[int]] = {
    "Fe/Mg phyllosilicates": [6, 7, 10, 31, 38],        # Fe smectite, Mg smectite, Serpentine, Chlorite, Chlorite-smectite
    "Al phyllosilicates":    [14, 18, 15, 23],            # Al smectite 1/2, Kaolinite, Illite
    "Sulfates":              [3, 9, 11, 16, 19, 26, 29],  # Gypsum, Jarosite, Alunite, Bassanite, Polyhydrated, Monohydrated, Ferricopiapite
    "Silica/Zeolite":        [25, 27],                     # Analcime, Hydrated silica
    "Ices":                  [1, 2],                        # CO2 ice, H2O ice
    "Fe oxides/hydroxides":  [4, 12],                      # Ferric hydroxysulfate, Akaganeite
    "Other hydrated":        [8, 17],                      # Prehnite, Epidote
}

# Reverse lookup: class ID → group name
_CLASS_TO_GROUP: Dict[int, str] = {}
for group_name, class_ids in GEOCHEM_GROUPS.items():
    for cid in class_ids:
        _CLASS_TO_GROUP[cid] = group_name


def group_for_class(class_id: int) -> Optional[str]:
    """Return geochemical group for a CNN mineral class ID, or None if unclassified."""
    if class_id == 100:  # Water-unrelated
        return None
    return _CLASS_TO_GROUP.get(class_id)


# ── Canonical Paleo-Environment Sequences ─────────────────────────
# Each environment maps to a list of possible group subsequences.
# If any subsequence appears in the observed transect, it's a match.
PALEO_SEQUENCES: Dict[str, List[List[str]]] = {
    "Evaporite lake":        [["Fe/Mg phyllosilicates", "Sulfates"]],
    "Acid leaching":         [["Fe/Mg phyllosilicates", "Al phyllosilicates", "Silica/Zeolite"]],
    "Deep alteration":       [["Fe/Mg phyllosilicates", "Other hydrated"]],
    "Groundwater upwelling": [["Sulfates", "Silica/Zeolite"]],
    "Ice-mineral contact":   [["Ices", "Fe/Mg phyllosilicates"], ["Ices", "Sulfates"]],
}


def match_sequence(observed_groups: List[str]) -> List[str]:
    """Match observed group sequence against canonical paleo-environment patterns.

    Args:
        observed_groups: Ordered list of geochemical group names along a transect
                         (with consecutive duplicates already removed).

    Returns:
        List of matched environment names (may be empty).
    """
    if len(observed_groups) < 2:
        return []

    matched: List[str] = []
    for env_name, patterns in PALEO_SEQUENCES.items():
        for pattern in patterns:
            if _is_subsequence(pattern, observed_groups):
                matched.append(env_name)
                break  # Only match each environment once
    return matched


def _is_subsequence(pattern: List[str], sequence: List[str]) -> bool:
    """Check if pattern appears as a contiguous subsequence in sequence."""
    plen = len(pattern)
    for i in range(len(sequence) - plen + 1):
        if sequence[i:i + plen] == pattern:
            return True
    return False
