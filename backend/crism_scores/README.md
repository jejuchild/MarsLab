# CRISM Score Layers

This directory contains score-based analysis layers for CRISM data.

## Directory Structure

```
crism_scores/
├── ice_score/           # Ice detection score maps
│   └── {product_id}.png # Score overlay images (0-255 grayscale or colored)
├── hydrated_score/      # Hydrated mineral score maps
│   └── {product_id}.png # Score overlay images
└── index.geojson        # Index of available score products
```

## Expected GeoJSON Schema (index.geojson)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "product_id": "frt00003156_07_brcarj_mtr3",
        "base_key": "frt00003156",
        "ice_score": "/crism/scores/ice/{product_id}.png",
        "hydrated_score": "/crism/scores/hydrated/{product_id}.png",
        "ice_score_value": 0.85,
        "hydrated_score_value": 0.42
      },
      "geometry": {
        "type": "Point",
        "coordinates": [134.316775, 47.730082]
      }
    }
  ]
}
```

## Score Image Format

- PNG format with transparency
- Grayscale or colormap (e.g., viridis, plasma)
- Same georeferencing as corresponding browse products
- Bounds from associated LBL file

## API Endpoints (to be implemented)

- `GET /crism/scores/ice/{product_id}.png` - Ice score overlay
- `GET /crism/scores/hydrated/{product_id}.png` - Hydrated mineral score overlay
- `GET /crism_scores_index.geojson` - Score products index
