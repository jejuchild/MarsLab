# MarsLab Architecture

This document describes the high-level architecture of MarsLab, including system components, data flows, and key subsystems.

---

## System Overview

MarsLab is a client-server web application for exploring Mars orbital science data:

```mermaid
graph TB
    subgraph "Client Browser"
        UI[React UI]
        Cesium[Cesium 3D Engine]
        FM[FootprintManager]
    end

    subgraph "Backend Server"
        FastAPI[FastAPI Application]
        Registry[Instrument Registry]
        Cache[LRU Cache Layer]
        DM[Download Manager]
    end

    subgraph "Data Storage"
        GeoJSON[(GeoJSON Indexes)]
        GeoTIFF[(GeoTIFF Images)]
        ENVI[(ENVI Cubes)]
        Browse[(Browse PNGs)]
        Score[(Score Maps)]
    end

    subgraph "External APIs"
        ODE[ODE REST API]
        PDS[PDS Archive]
    end

    UI --> Cesium
    UI --> FM
    FM --> FastAPI
    Cesium --> FM

    FastAPI --> Registry
    FastAPI --> Cache
    FastAPI --> DM

    Cache --> GeoJSON
    Cache --> GeoTIFF
    FastAPI --> ENVI
    FastAPI --> Browse
    FastAPI --> Score

    DM --> ODE
    ODE --> PDS
```

---

## Component Architecture

### Frontend Components

```mermaid
graph TB
    subgraph "React Application"
        App[App.tsx]
        MainPage[MainPage.tsx<br/>State Management Hub]

        subgraph "UI Components"
            TopBar[TopBar<br/>Search & Navigation]
            LayerPanel[LayerPanel<br/>Map Controls]
            MapView[MapView<br/>Cesium Viewer]
            Inspector[Inspector<br/>Product Details]
        end

        subgraph "Utilities"
            FootprintMgr[FootprintManager]
            InstrumentReg[instrumentRegistry]
        end
    end

    App --> MainPage
    MainPage --> TopBar
    MainPage --> LayerPanel
    MainPage --> MapView
    MainPage --> Inspector

    MapView --> FootprintMgr
    MapView --> InstrumentReg
    LayerPanel --> InstrumentReg
```

**Key Component Responsibilities:**

| Component | File | Responsibility |
|-----------|------|----------------|
| `MainPage` | `pages/MainPage.tsx` | Central state management, coordinates all child components |
| `MapView` | `components/MapView.tsx` | Cesium viewer, footprint rendering, overlay management |
| `LayerPanel` | `components/LayerPanel.tsx` | Map controls, footprint toggles, active products list |
| `Inspector` | `components/Inspector.tsx` | Product details, spectral analysis, overlay controls |
| `TopBar` | `components/TopBar.tsx` | Search bar with typeahead |
| `FootprintManager` | `utils/FootprintManager.ts` | Explicit footprint loading, Cesium entity management |

### Backend Components

```mermaid
graph TB
    subgraph "FastAPI Application"
        App[app.py<br/>Main Entry]

        subgraph "Routers"
            SearchRouter[search_router.py]
            FootprintRouter[footprints_router.py]
            CRISMRouter[crism/router.py]
            HiRISERouter[hirise_pixel.py]
        end

        subgraph "Services"
            Registry[registry.py<br/>Instrument Registry]
            DownloadMgr[download_manager.py]
            ODEClient[ode_client.py]
        end

        subgraph "CRISM Processing"
            Loader[loader.py]
            Processing[processing.py]
            Spectrum[spectrum.py]
        end
    end

    App --> SearchRouter
    App --> FootprintRouter
    App --> CRISMRouter
    App --> HiRISERouter

    SearchRouter --> ODEClient
    SearchRouter --> DownloadMgr
    FootprintRouter --> Registry
    CRISMRouter --> Loader
    CRISMRouter --> Processing
    CRISMRouter --> Spectrum
```

---

## Data Flow Diagrams

### 1. Product Discovery Flow

How users find and select products:

```mermaid
sequenceDiagram
    participant User
    participant TopBar
    participant MainPage
    participant FootprintMgr
    participant Backend
    participant Index

    User->>LayerPanel: Click "Load Footprints"
    LayerPanel->>MainPage: setLoadFootprintsTrigger()
    MainPage->>MapView: loadFootprintsTrigger prop
    MapView->>FootprintMgr: loadFootprints(instrument)
    FootprintMgr->>FootprintMgr: Compute viewport bbox
    FootprintMgr->>Backend: GET /api/footprints?bbox=...
    Backend->>Index: Load index.geojson (cached)
    Backend->>Backend: Filter by bbox
    Backend-->>FootprintMgr: GeoJSON FeatureCollection
    FootprintMgr->>MapView: Render Cesium entities
    MapView->>MainPage: onVisibleProductsChange()
    MainPage->>LayerPanel: Update visibleProducts
```

### 2. Footprint Loading Flow

Detailed footprint loading with LOD:

```mermaid
flowchart TB
    Start([User clicks Load])
    ComputeBbox[Compute viewport bbox]
    CheckLOD{Camera height?}
    LODNone[LOD: none<br/>Return empty]
    LODPoint[LOD: point<br/>Centroids only]
    LODPoly[LOD: poly<br/>Full polygons]
    FetchAPI[Fetch /api/footprints]
    FilterBbox[Filter features by bbox]
    Simplify{Simplify?}
    ApplySimplify[Douglas-Peucker<br/>simplification]
    ReturnGeoJSON[Return GeoJSON]
    RenderEntities[Render Cesium entities]
    UpdateUI[Update visible products]

    Start --> ComputeBbox
    ComputeBbox --> CheckLOD
    CheckLOD -->|> 15,000 km| LODNone
    CheckLOD -->|5,000-15,000 km| LODPoint
    CheckLOD -->|< 5,000 km| LODPoly

    LODNone --> ReturnGeoJSON
    LODPoint --> FetchAPI
    LODPoly --> FetchAPI

    FetchAPI --> FilterBbox
    FilterBbox --> Simplify
    Simplify -->|Yes| ApplySimplify
    Simplify -->|No| ReturnGeoJSON
    ApplySimplify --> ReturnGeoJSON
    ReturnGeoJSON --> RenderEntities
    RenderEntities --> UpdateUI
```

### 3. Overlay Rendering Flow

How overlays (quickview, high-res, browse) are displayed:

```mermaid
sequenceDiagram
    participant User
    participant LayerPanel
    participant MainPage
    participant MapView
    participant Backend

    User->>LayerPanel: Toggle overlay type
    LayerPanel->>MainPage: handleSetOverlay(productId, type)
    MainPage->>MainPage: Update activeOverlays Map
    MainPage->>MapView: Updated overlay props

    alt Quickview Overlay
        MapView->>Backend: GET /crism/quickview/{id}.png
        Backend-->>MapView: PNG with transparency
    else High-Res Overlay
        MapView->>Backend: GET /hirise/overlay/{id}.png
        Backend->>Backend: Check disk cache
        Backend->>Backend: Process GeoTIFF if not cached
        Backend-->>MapView: PNG with transparency
    else Browse Product (CRISM)
        MapView->>Backend: GET /crism/browse/{filename}.png
        Backend-->>MapView: Browse PNG (HYD/ICE/IC2)
    end

    MapView->>MapView: Create ImageMaterialProperty entity
    MapView->>MapView: Apply opacity from activeOverlays
```

### 4. Inspector Interaction Flow

Product inspection and analysis:

```mermaid
sequenceDiagram
    participant User
    participant MapView
    participant MainPage
    participant Inspector
    participant Backend

    User->>MapView: Click footprint
    MapView->>MainPage: setSelected(context)
    MainPage->>Inspector: Render with selected prop

    alt HiRISE Product
        Inspector->>Backend: GET /hirise/window_xy?...
        Backend-->>Inspector: Pixel statistics
        Inspector->>Inspector: Display histogram
    else CRISM Product
        Inspector->>Backend: POST /crism/{id}/spectrum
        Backend->>Backend: Load ENVI cube
        Backend-->>Inspector: Spectral data
        Inspector->>Inspector: Plot spectrum chart

        User->>Inspector: Adjust RGB wavelengths
        Inspector->>MainPage: onRGBChange()
        MainPage->>MapView: rgbWavelengths prop
        MapView->>Backend: POST /crism/{id}/rgb
        Backend->>Backend: Generate RGB image
        Backend-->>MapView: PNG blob
        MapView->>MapView: Update overlay entity
    end
```

---

## Map Rendering Subsystem

### Cesium Configuration

The Cesium viewer is configured for Mars:

```typescript
// Mars ellipsoid (IAU-defined)
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  3396190.0,  // Equatorial radius (m)
  3396190.0,
  3376200.0   // Polar radius (m)
);

// Viewer setup
const viewer = new Cesium.Viewer(container, {
  baseLayerPicker: false,
  animation: false,
  timeline: false,
  sceneModePicker: false,
  // ... other disabled UI elements
});
```

**Code Reference:** `frontend/src/components/MapView.tsx:50-100`

### Base Layers

Two base layer options from NASA Trek:

| Layer | Source | Description |
|-------|--------|-------------|
| MOLA | NASA Trek | Mars Global Surveyor elevation data |
| HRSC | NASA Trek | Viking color mosaic |

### Entity Types

| Entity Type | Geometry | Usage |
|-------------|----------|-------|
| Rectangle | Polygon bounds | Footprint outlines, overlays |
| Point | Centroid | Point-mode footprints |
| Polyline | LineString | SHARAD radar tracks |
| Label | Text | Product ID labels |

---

## Caching Architecture

### Backend Caching Layers

```mermaid
graph LR
    subgraph "Request"
        API[API Request]
    end

    subgraph "Memory Cache"
        LRU1[GeoJSON Index<br/>lru_cache maxsize=8]
        LRU2[Rasterio Datasets<br/>lru_cache maxsize=32]
        LRU3[Tile Cache<br/>lru_cache maxsize=8192]
        Global[Score Stats<br/>Global dict]
    end

    subgraph "Disk Cache"
        Overlay[.overlay_cache/<br/>Processed PNGs]
    end

    subgraph "Storage"
        Files[(Data Files)]
    end

    API --> LRU1
    API --> LRU2
    API --> LRU3
    API --> Global
    LRU1 --> Files
    LRU2 --> Files
    LRU3 --> Overlay
    Overlay --> Files
```

**Cache Details:**

| Cache | Location | Max Size | TTL | Purpose |
|-------|----------|----------|-----|---------|
| `load_geojson_index` | `footprints_router.py:53` | 8 | Forever | GeoJSON indexes |
| `open_ds` | `app.py:72` | 32 | Forever | Rasterio dataset handles |
| `load_world_tile` | `app.py:136` | 8192 | Forever | Tile PNG bytes |
| `.overlay_cache/` | Disk | Unlimited | Forever | Processed HiRISE overlays |
| `_score_stats_cache` | `app.py:448` | 1 | Forever | Score statistics JSON |

### Frontend Caching

- **LBL bounds cache:** Parsed label file bounds (in-memory Map)
- **Blob URL management:** CRISM RGB images as blob URLs
- **FootprintManager:** Maintains feature maps per instrument

---

## State Management Architecture

### Frontend State Flow

```mermaid
stateDiagram-v2
    [*] --> MainPage

    state MainPage {
        selected: InspectorContext
        baseLayer: BaseLayerType
        mapMode: MapMode
        viewBounds: BoundingBox
        showCRISM: boolean
        showHiRISE: boolean
        showSHARAD: boolean
        visibleProducts: VisibleProduct[]
        activeOverlays: Map<productId, ProductOverlay>
        rgbWavelengths: RGBWavelengths
        iceScoreFilter: IceScoreFilter
        filteredProductIds: Set<string>
    }

    MainPage --> LayerPanel: Derived props
    MainPage --> MapView: Derived props
    MainPage --> Inspector: selected, activeOverlay

    LayerPanel --> MainPage: Callbacks
    MapView --> MainPage: Callbacks
    Inspector --> MainPage: Callbacks
```

**State Variables (MainPage.tsx:93-152):**

| State | Type | Purpose |
|-------|------|---------|
| `selected` | `InspectorContext \| null` | Currently selected product |
| `baseLayer` | `"MOLA" \| "HRSC"` | Active base map |
| `mapMode` | `"2D" \| "3D"` | View mode |
| `viewBounds` | `BoundingBox \| null` | Optional view restriction |
| `showCRISM/HiRISE/SHARAD` | `boolean` | Footprint visibility |
| `visibleProducts` | `VisibleProduct[]` | Products in current view |
| `activeOverlays` | `Map<string, ProductOverlay>` | Active overlay per product |
| `rgbWavelengths` | `RGBWavelengths` | CRISM RGB composition |
| `iceScoreFilter` | `IceScoreFilter` | Ice score filter config |

---

## Security Considerations

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Code Reference:** `backend/app.py:30-36`

**Note:** This permissive CORS is suitable for development. For production, restrict `allow_origins` to specific domains.

### Input Validation

- **Pydantic models** validate request bodies
- **Query parameter validation** via FastAPI Query()
- **Coordinate bounds checking** in footprints API
- **File path sanitization** to prevent path traversal

---

## Error Handling Patterns

### Backend Error Handling

```python
# HTTP exceptions for client errors
raise HTTPException(status_code=404, detail="Product not found")

# Graceful degradation for optional features
try:
    from shapely.geometry import shape
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
```

### Frontend Error Handling

- **AbortController** for request cancellation
- **Try-catch** around async operations
- **Null checks** before entity operations
- **Fallback rendering** when data unavailable

---

## Key Architectural Decisions

### 1. Explicit Footprint Loading

**Decision:** Footprints load only on explicit user action, not automatically on camera movement.

**Rationale:**
- Prevents request spam during rapid panning
- Gives user control over data loading
- Reduces server load

**Code Reference:** `frontend/src/utils/FootprintManager.ts:1-9`

### 2. Single Overlay Per Product

**Decision:** Each product can have only one active overlay type at a time.

**Rationale:**
- Simplifies z-ordering management
- Clearer user experience
- Reduces Cesium entity count

**Code Reference:** `frontend/src/pages/MainPage.tsx:192-210`

### 3. Server-Side LOD Enforcement

**Decision:** Server enforces LOD based on camera height, overriding client requests.

**Rationale:**
- Prevents abuse/mistakes
- Consistent performance
- Reduces bandwidth for distant views

**Code Reference:** `backend/api/footprints_router.py:246-258`

### 4. Disk Cache for Processed Overlays

**Decision:** HiRISE overlays are cached to disk after processing.

**Rationale:**
- Expensive processing (GeoTIFF to PNG with transparency)
- Frequent re-requests for same products
- Persistent across server restarts

**Code Reference:** `backend/app.py:184-245`
