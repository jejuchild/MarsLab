# Frontend Documentation

This document provides a deep dive into the MarsLab React/TypeScript frontend, including component architecture, state management, and Cesium integration.

---

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **React** | 19.2.0 | UI framework |
| **React Router** | 7.12.0 | Client-side routing |
| **TypeScript** | 5.9.3 | Type-safe JavaScript |
| **Vite** | 7.2.4 | Build tool & dev server |
| **Cesium** | 1.137.0 | 3D geospatial rendering |
| **Recharts** | 3.6.0 | Data visualization |
| **Tailwind CSS** | 3.4.17 | Utility-first styling |
| **PostCSS** | 8.4.49 | CSS transformations |

**Code Reference:** `frontend/package.json`

---

## Build Toolchain

### Vite Configuration

**File:** `frontend/vite.config.ts`

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: ["**/public/tiles/**", "**/node_modules/**"],
    },
    fs: {
      allow: [
        path.resolve(__dirname),
        path.resolve(__dirname, "../Data"),
        path.resolve(__dirname, "node_modules"),
      ],
    },
    proxy: {
      "/api": "http://localhost:8000",
      "/hirise": "http://localhost:8000",
      "/crism": "http://localhost:8000",
      "/sharad": "http://localhost:8000",
      // ... other proxies
    },
  },
});
```

### TypeScript Configuration

**File:** `frontend/tsconfig.json`

Key settings:
- `strict: true` - Full type checking
- `target: "ES2022"` - Modern JavaScript
- `moduleResolution: "bundler"` - Vite compatibility
- `jsx: "react-jsx"` - React 17+ JSX transform

### Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  }
}
```

---

## UI Layout

### Visual Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                         TopBar (Search & Navigation)            │
├──────────────┬─────────────────────────────┬────────────────────┤
│              │                             │                    │
│  LayerPanel  │        MapView              │     Inspector      │
│  (Left)      │        (Center)             │     (Right)        │
│              │                             │                    │
│  - Map Mode  │        Cesium 3D Globe      │  - Metadata        │
│  - Base Layer│        or 2D Map            │  - Spectrum        │
│  - Footprints│                             │  - Overlays        │
│  - Filters   │                             │  - Pixel Info      │
│  - Products  │                             │                    │
│              │                             │                    │
└──────────────┴─────────────────────────────┴────────────────────┘
```

### Component Hierarchy

```
App (routing)
└── MainPage (state hub)
    └── AppShell (layout)
        ├── TopBar (header slot)
        │   └── Search dropdown
        ├── LayerPanel (left slot)
        │   ├── Map Mode selector
        │   ├── Base Layer selector
        │   ├── View Bounds inputs
        │   ├── Footprint toggles & load buttons
        │   ├── Ice Score Filter
        │   └── Active Products list
        ├── MapView (main slot)
        │   ├── Cesium Viewer
        │   └── FootprintManager
        └── Inspector (right slot)
            ├── Metadata tab
            ├── Spectrum tab (CRISM)
            ├── Pixel tab (HiRISE)
            └── Overlay controls
```

---

## Component Documentation

### MainPage

**File:** `frontend/src/pages/MainPage.tsx`

The central state management hub. All application state lives here and is passed down via props.

**State Variables:**

| State | Type | Purpose |
|-------|------|---------|
| `selected` | `InspectorContext \| null` | Selected product for inspector |
| `baseLayer` | `"MOLA" \| "HRSC"` | Active base map |
| `mapMode` | `"2D" \| "3D"` | Globe vs flat view |
| `viewBounds` | `BoundingBox \| null` | View restriction |
| `showCRISM/HiRISE/SHARAD` | `boolean` | Footprint visibility |
| `loadFootprintsTrigger` | `{instrument, timestamp}` | Explicit load trigger |
| `footprintsLoading` | `{crism, hirise, sharad: boolean}` | Loading states |
| `footprintCounts` | `{count, truncated, total}` | Load results |
| `visibleProducts` | `VisibleProduct[]` | Products in view |
| `activeOverlays` | `Map<productId, ProductOverlay>` | Active overlays |
| `flyToProductId` | `string \| null` | Product to fly to |
| `bringToFrontId` | `string \| null` | Overlay z-ordering |
| `rgbWavelengths` | `RGBWavelengths` | CRISM RGB composition |
| `sharadPopup` | `SHARADPopup \| null` | SHARAD quickview modal |
| `productsWithHighRes` | `Set<string>` | Products with hi-res data |
| `iceScoreFilter` | `IceScoreFilter` | Filter configuration |
| `filteredProductIds` | `Set<string> \| null` | Filtered product IDs |

**Key Callbacks:**

```typescript
const handleSetOverlay = useCallback((productId: string, type: OverlayType | null, opacity?: number) => {
  // Enforces single-overlay-per-product rule
  setActiveOverlays((prev) => {
    const newMap = new Map(prev);
    if (type === null) {
      newMap.delete(productId);
    } else {
      newMap.set(productId, { type, opacity: opacity ?? existingOpacity });
    }
    return newMap;
  });
}, []);
```

---

### MapView

**File:** `frontend/src/components/MapView.tsx` (~1700 lines)

The main map rendering component using Cesium.

**Key Responsibilities:**
- Initialize and configure Cesium Viewer
- Manage FootprintManager instance
- Handle click/hover interactions
- Render overlay entities
- Process fly-to requests
- Manage 2D/3D mode switching

**Props:**

| Prop | Type | Purpose |
|------|------|---------|
| `mapMode` | `MapMode` | 2D or 3D view |
| `baseLayer` | `BaseLayerType` | Base map selection |
| `viewBounds` | `BoundingBox` | Optional view restriction |
| `showCRISM/HiRISE/SHARAD` | `boolean` | Footprint visibility |
| `quickviewOverlays` | `string[]` | Products with quickview |
| `highResOverlays` | `string[]` | Products with high-res |
| `browseOverlays` | `Map<string, Set<BrowseType>>` | Browse product overlays |
| `scoreOverlays` | `Map<string, Set<ScoreType>>` | Score map overlays |
| `overlayOpacities` | `Map<string, number>` | Per-product opacity |
| `flyToProductId` | `string \| null` | Product to fly to |
| `rgbWavelengths` | `RGBWavelengths` | CRISM RGB composition |
| `crismFilteredIds` | `Set<string> \| null` | Ice score filter |

**Cesium Setup:**

```typescript
// Mars ellipsoid
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  3396190.0,  // equatorial radius
  3396190.0,
  3376200.0   // polar radius
);

// Viewer initialization
const viewer = new Cesium.Viewer(containerRef.current, {
  baseLayerPicker: false,
  animation: false,
  timeline: false,
  sceneModePicker: false,
  homeButton: false,
  geocoder: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
  creditContainer: document.createElement("div"),
});
```

---

### Inspector

**File:** `frontend/src/components/Inspector.tsx`

Right panel showing selected product details.

**Tabs by Instrument:**

| Instrument | Tabs |
|------------|------|
| HiRISE | Metadata, Pixel Inspector |
| CRISM | Metadata, Spectrum, Bands (RGB) |
| SHARAD | Metadata |

**Props:**

| Prop | Type | Purpose |
|------|------|---------|
| `selected` | `InspectorContext` | Selected product context |
| `onClose` | `() => void` | Close handler |
| `activeOverlay` | `ProductOverlay \| null` | Current overlay state |
| `onSetOverlay` | `(type) => void` | Set overlay type |
| `onSetOpacity` | `(opacity) => void` | Set opacity |
| `rgbWavelengths` | `RGBWavelengths` | CRISM RGB values |
| `onRGBChange` | `(rgb) => void` | RGB change handler |
| `hasHighResData` | `boolean` | Hi-res availability |

**Spectrum Fetching:**

```typescript
useEffect(() => {
  const fetchSpectrum = async () => {
    const res = await fetch(`/crism/${productId}/spectrum`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ line: pixelLine, sample: pixelSample }),
    });
    const data = await res.json();
    setSpectrum(data);
  };
  if (instrument === "CRISM" && pixelLine && pixelSample) {
    fetchSpectrum();
  }
}, [productId, pixelLine, pixelSample]);
```

---

### LayerPanel

**File:** `frontend/src/components/LayerPanel.tsx`

Left sidebar with map controls and product list.

**Sections:**

1. **Map Mode** - 2D/3D toggle
2. **Base Layer** - MOLA/HRSC selection
3. **View Bounds** - Optional coordinate restriction
4. **Footprint Layers** - Toggle visibility + load buttons
5. **Ice Score Filter** - CRISM filtering by ice score
6. **Active Products** - List with overlay controls

**Explicit Load Pattern:**

```typescript
const handleLoadCRISM = () => {
  onLoadFootprints("CRISM");
};

// Button with loading state
<button
  onClick={handleLoadCRISM}
  disabled={footprintsLoading.crism}
  className="..."
>
  {footprintsLoading.crism ? "Loading..." : "Load CRISM"}
</button>
```

---

### TopBar

**File:** `frontend/src/components/TopBar.tsx`

Header with search functionality.

**Features:**
- Typeahead search across visible products
- Keyboard navigation (arrow keys, Enter)
- Search by product ID or title
- Click to fly-to and select

---

### FootprintManager

**File:** `frontend/src/utils/FootprintManager.ts`

Manages footprint loading and Cesium entity rendering.

**Architecture:**
- **Explicit loading only** - No automatic camera-driven updates
- **Snapshot-based** - Loads replace previous footprints
- **Per-instrument tracking** - Separate state for each instrument

**Key Methods:**

```typescript
class FootprintManager {
  // Load footprints for current viewport
  async loadFootprints(instrument: InstrumentType): Promise<LoadResult | null>

  // Clear all footprints for instrument
  clearFootprints(instrument: InstrumentType): void

  // Toggle visibility without reloading
  setVisible(instrument: InstrumentType, visible: boolean): void

  // Get loaded features
  getFeatures(instrument: InstrumentType): FootprintFeature[]

  // Cleanup
  dispose(): void
}
```

**Entity Rendering:**

```typescript
private renderFootprints(instrument: InstrumentType, features: FootprintFeature[]): void {
  viewer.entities.suspendEvents();

  for (const feature of features) {
    if (geom.type === "Polygon") {
      // Create rectangle entity
      viewer.entities.add({
        id: entityId,
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
          material: color.withAlpha(0.4),
          outline: true,
          outlineColor: color,
        },
        properties: { product_id, instrument, kind: "FOOTPRINT_RECT" },
      });

      // Add label
      viewer.entities.add({
        id: labelId,
        position: centroidPosition,
        label: { text: productId, ... },
      });
    } else if (geom.type === "LineString") {
      // SHARAD tracks
      viewer.entities.add({
        id: entityId,
        polyline: { positions, width: 2, material: color },
      });
    }
  }

  viewer.entities.resumeEvents();
}
```

---

## State Management

### Pattern: Props Drilling with Callbacks

MarsLab uses a simple state management pattern:

1. **All state in MainPage** - Single source of truth
2. **Props down** - State passed to children
3. **Callbacks up** - Children call handlers to update state

```
MainPage (state)
    ↓ props
    ├── LayerPanel
    │       ↑ callbacks (onToggle, onLoad, etc.)
    ├── MapView
    │       ↑ callbacks (onSelect, onVisibleProductsChange)
    └── Inspector
            ↑ callbacks (onSetOverlay, onRGBChange)
```

### TypeScript Types

**Core Types (`MainPage.tsx`):**

```typescript
export type VisibleProduct = {
  productId: string;
  instrument: "HIRISE" | "CRISM" | "SHARAD";
  title?: string;
};

export type OverlayType =
  | "quickview"
  | "highres"
  | "browse_HYD"
  | "browse_ICE"
  | "browse_IC2"
  | "score_ice"
  | "score_hyd";

export type ProductOverlay = {
  type: OverlayType;
  opacity: number;  // 0-100
};

export type ActiveOverlays = Map<string, ProductOverlay>;

export type BaseLayerType = "MOLA" | "HRSC";
export type MapMode = "2D" | "3D";

export type BoundingBox = {
  minLat: number;
  maxLat: number;
  westLon: number;
  eastLon: number;
} | null;
```

**Inspector Context:**

```typescript
export interface InspectorContext {
  instrument: "HIRISE" | "CRISM" | "SHARAD";
  productId: string;
  lat: number;
  lon: number;
  pixelLine?: number;
  pixelSample?: number;
  title?: string;
}
```

---

## Map Engine Integration

### Cesium Configuration

**Mars-Specific Setup:**

```typescript
// Custom Mars ellipsoid
const MARS_ELLIPSOID = new Cesium.Ellipsoid(
  3396190.0,  // equatorial radius (m)
  3396190.0,
  3376200.0   // polar radius (m)
);

// Geographic projection for Mars
const projection = new Cesium.GeographicProjection(MARS_ELLIPSOID);

// Scene configuration
viewer.scene.globe.enableLighting = false;
viewer.scene.backgroundColor = Cesium.Color.BLACK;
```

### Base Layers

```typescript
const BASE_LAYERS = {
  MOLA: new Cesium.UrlTemplateImageryProvider({
    url: "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.png",
    tilingScheme: new Cesium.GeographicTilingScheme({ ellipsoid: MARS_ELLIPSOID }),
  }),
  HRSC: new Cesium.UrlTemplateImageryProvider({
    url: "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
    tilingScheme: new Cesium.GeographicTilingScheme({ ellipsoid: MARS_ELLIPSOID }),
  }),
};
```

### 2D/3D Mode Switching

```typescript
const switchTo2D = () => {
  viewer.scene.morphTo2D(1.5);  // 1.5 second transition
};

const switchTo3D = () => {
  viewer.scene.morphTo3D(1.5);
};
```

### Entity Picking

```typescript
// Click handler
const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction((click) => {
  const picked = viewer.scene.pick(click.position);
  if (Cesium.defined(picked) && picked.id) {
    const props = picked.id.properties;
    if (props && props.product_id) {
      const productId = props.product_id.getValue();
      const instrument = props.instrument.getValue();
      onSelect({ instrument, productId, lat, lon });
    }
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
```

---

## Layer & Overlay Management

### Overlay Lifecycle

1. **Activation** - User clicks overlay button in LayerPanel or Inspector
2. **State Update** - `handleSetOverlay()` updates `activeOverlays` Map
3. **Prop Derivation** - `derivedOverlays` useMemo computes legacy format
4. **Entity Creation** - MapView creates ImageMaterialProperty entity
5. **Deactivation** - User clicks again or different overlay type

### Overlay Entity Structure

```typescript
viewer.entities.add({
  id: `overlay_${productId}`,
  rectangle: {
    coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
    material: new Cesium.ImageMaterialProperty({
      image: imageUrl,
      transparent: true,
      alpha: opacity,
    }),
  },
  properties: { product_id: productId, kind: "OVERLAY" },
});
```

### Z-Ordering

Overlays are z-ordered by creation time. To bring an overlay to front:

```typescript
const bringToFront = (productId: string) => {
  const entity = viewer.entities.getById(`overlay_${productId}`);
  if (entity) {
    viewer.entities.remove(entity);
    viewer.entities.add(entity);  // Re-add at end
  }
};
```

---

## Event Flow Examples

### Toggle Overlay

```
User clicks "Quickview" button
    ↓
LayerPanel.handleSetOverlay("quickview")
    ↓
MainPage.handleSetOverlay(productId, "quickview")
    ↓
setActiveOverlays(Map with new entry)
    ↓
derivedOverlays computed (useMemo)
    ↓
MapView receives quickviewOverlays prop
    ↓
MapView effect creates Cesium entity
    ↓
Cesium renders overlay on globe
```

### Fly-To Product

```
User clicks product ID in LayerPanel
    ↓
LayerPanel.handleFlyToProduct(productId)
    ↓
MainPage.handleFlyToProduct(productId)
    ↓
setFlyToProductId(productId)
    ↓
MapView receives flyToProductId prop
    ↓
MapView effect calls viewer.camera.flyTo(...)
    ↓
Camera animates to product bounds
    ↓
MapView calls onFlyToComplete()
    ↓
MainPage.setFlyToProductId(null)
```

### Per-Product Opacity

```
User drags opacity slider
    ↓
LayerPanel.onSetOpacity(productId, 75)
    ↓
MainPage.handleSetOpacity(productId, 75)
    ↓
setActiveOverlays(Map with updated opacity)
    ↓
derivedOverlays.opacities updated
    ↓
MapView receives overlayOpacities prop
    ↓
MapView effect updates entity.rectangle.material.alpha
```

---

## Instrument Registry

**File:** `frontend/src/config/instrumentRegistry.ts`

Centralized instrument configuration:

```typescript
export const INSTRUMENTS = {
  crism: {
    id: "crism",
    name: "CRISM",
    displayName: "Compact Reconnaissance Imaging Spectrometer for Mars",
    geometryType: "Polygon",
    color: { r: 0, g: 1, b: 1 },  // Cyan
    productIdPattern: /^(frt|hrl|hrs|frs|arl|atl)/i,
    supportsSpectrum: true,
    supportsRgb: true,
  },
  hirise: {
    id: "hirise",
    name: "HiRISE",
    displayName: "High Resolution Imaging Science Experiment",
    geometryType: "Polygon",
    color: { r: 1, g: 1, b: 0 },  // Yellow
    productIdPattern: /^(ESP|PSP|TRA)_/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
  sharad: {
    id: "sharad",
    name: "SHARAD",
    displayName: "SHAllow RADar",
    geometryType: "LineString",
    color: { r: 1, g: 0.65, b: 0 },  // Orange
    productIdPattern: /^S_/i,
    supportsSpectrum: false,
    supportsRgb: false,
  },
};

// Helper functions
export function getInstrument(id: string): InstrumentConfig | undefined
export function detectInstrument(productId: string): InstrumentConfig | undefined
export function getInstrumentCesiumColor(id: InstrumentId): { r, g, b }
```

---

## Styling

### Tailwind Configuration

**File:** `frontend/tailwind.config.js`

```javascript
module.exports = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: "#135bec",
        "bg-dark": "#0a0e17",
        "surface-dark": "#161e2d",
      },
      fontFamily: {
        sans: ["Space Grotesk", "sans-serif"],
      },
    },
  },
};
```

### Design System

| Element | Color | Usage |
|---------|-------|-------|
| Primary | `#135bec` | Buttons, links, accents |
| Background | `#0a0e17` | Main background |
| Surface | `#161e2d` | Cards, panels |
| Border | `#232f48` | Panel borders |
| Text Primary | `#ffffff` | Main text |
| Text Secondary | `#92a4c9` | Labels, hints |

### Icons

Using Material Symbols Outlined:

```html
<span class="material-symbols-outlined">close</span>
<span class="material-symbols-outlined">visibility</span>
<span class="material-symbols-outlined">flight</span>
```
