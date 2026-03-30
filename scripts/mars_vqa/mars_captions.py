"""
Mars landform class → natural language caption mappings for DoMars16k.
Used to generate training data for generative VQA models.
"""

LABEL_NAMES = [
    "aec", "ael", "cli", "cra", "fse", "fsf", "fsg", "fss",
    "mix", "rid", "rou", "sfe", "sfx", "smo", "tex",
]

LABEL_FULL_NAMES = {
    "aec": "Aeolian Erosion / Channel",
    "ael": "Aeolian Bedforms (dunes/ripples)",
    "cli": "Cliff / Scarp",
    "cra": "Impact Crater",
    "fse": "Fractured Surface (erosional)",
    "fsf": "Fractured Surface (filled)",
    "fsg": "Fractured Surface (graben)",
    "fss": "Fractured Surface (stressed)",
    "mix": "Mixed Terrain",
    "rid": "Ridge",
    "rou": "Rough Terrain",
    "sfe": "Smooth Feature (ejecta)",
    "sfx": "Smooth Feature (extended)",
    "smo": "Smooth Terrain",
    "tex": "Textured Surface",
}

# Multiple caption templates per class for training diversity
CAPTION_TEMPLATES = {
    "aec": [
        "This Mars surface tile shows aeolian erosion features with wind-carved channels and streamlined landforms.",
        "Wind-driven erosion has sculpted channels and grooves across this Martian terrain.",
        "The surface displays characteristic aeolian erosion patterns including deflation channels.",
    ],
    "ael": [
        "This tile contains aeolian bedforms such as sand dunes and wind-blown ripple patterns on the Martian surface.",
        "Prominent dune fields and ripple structures formed by Martian winds are visible.",
        "The terrain shows well-developed aeolian bedforms including transverse dunes and ripples.",
    ],
    "cli": [
        "A cliff or scarp feature is visible, showing a steep elevation change on the Mars surface.",
        "The image shows a prominent escarpment or cliff face with exposed layered stratigraphy.",
        "A sharp topographic break marks a cliff or scarp cutting across the terrain.",
    ],
    "cra": [
        "An impact crater is visible with a raised rim and bowl-shaped depression on the Martian surface.",
        "The tile shows a circular impact structure with characteristic crater morphology.",
        "A well-preserved impact crater dominates this area, showing rim, floor, and possible ejecta.",
    ],
    "fse": [
        "The surface displays fractures and cracks caused by erosional processes on Mars.",
        "Erosion-driven fracturing has created a network of cracks across this Martian terrain.",
        "Fractured surface patterns from erosional degradation are visible in this tile.",
    ],
    "fsf": [
        "Fractures filled with secondary material are visible on this Martian surface.",
        "The terrain shows a fractured surface where cracks have been infilled with sediment or lava.",
        "Filled fracture networks indicate past fluid or sediment transport across this surface.",
    ],
    "fsg": [
        "Graben structures — downdropped blocks bounded by faults — are visible on this Mars surface.",
        "The tile shows extensional tectonic features forming graben and fracture systems.",
        "Linear graben formed by crustal extension cut across this Martian terrain.",
    ],
    "fss": [
        "Stress-induced fractures and polygonal cracking patterns are visible on this Mars surface.",
        "The terrain displays fractures caused by thermal or tectonic stress.",
        "Stressed surface patterns with intersecting fracture networks characterize this tile.",
    ],
    "mix": [
        "This tile shows a mix of multiple landform types on the Martian surface.",
        "Several distinct geomorphologic features coexist in this area of mixed terrain.",
        "The surface displays a heterogeneous combination of different Martian landforms.",
    ],
    "rid": [
        "A ridge or linear elevated feature runs across this Martian terrain.",
        "The tile shows a prominent ridge structure, possibly wrinkle ridge or pressure ridge.",
        "An elongated topographic high forms a ridge feature across the surface.",
    ],
    "rou": [
        "The Martian surface here is rough and irregular with varied small-scale topography.",
        "Rough terrain with a chaotic, uneven surface texture dominates this tile.",
        "The surface shows rough, hummocky terrain with irregular elevation variations.",
    ],
    "sfe": [
        "A smooth feature associated with crater ejecta is visible on this Mars surface.",
        "The terrain shows a smooth ejecta blanket surrounding or adjacent to an impact site.",
        "Smooth material interpreted as impact ejecta covers this portion of the surface.",
    ],
    "sfx": [
        "An extended smooth feature — possibly a lava flow or sedimentary plain — covers this area.",
        "The tile shows a broad, smooth surface extending across the terrain.",
        "A smooth, featureless plain extends across this portion of the Martian surface.",
    ],
    "smo": [
        "The Martian surface here is smooth and relatively featureless.",
        "A smooth, flat terrain with minimal topographic variation is visible.",
        "The tile shows smooth terrain typical of sedimentary infill or dust mantling.",
    ],
    "tex": [
        "The surface displays distinctive textural patterns on the Martian terrain.",
        "Repetitive surface textures — possibly polygonal ground or patterned terrain — are visible.",
        "The tile shows textured terrain with regular or semi-regular surface patterns.",
    ],
}


def get_caption(label_name: str, variant: int = 0) -> str:
    """Get a caption for a given label name."""
    templates = CAPTION_TEMPLATES[label_name]
    return templates[variant % len(templates)]


def get_all_captions(label_name: str) -> list[str]:
    """Get all caption variants for a given label name."""
    return CAPTION_TEMPLATES[label_name]


# QA format for VQA training
QA_TEMPLATES = {
    "describe": "Question: Describe the geological features visible in this Mars satellite image. Answer: {caption}",
    "classify": "Question: What type of Mars landform is this? Answer: {full_name}",
    "identify": "Question: What surface patterns can you identify? Answer: {caption}",
}
