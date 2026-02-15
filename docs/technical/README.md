# MarsLab Technical Documentation

**MarsLab** is a web-based application for exploring and analyzing Mars orbital science data from multiple instruments: CRISM (hyperspectral imaging), HiRISE (high-resolution imaging), and SHARAD (subsurface radar).

This documentation provides a comprehensive engineering handbook for understanding, developing, and maintaining the MarsLab system.

---

## Navigation

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | High-level system architecture, data flows, and component interactions |
| [Backend](./backend.md) | Python/FastAPI backend deep-dive: services, caching, configuration |
| [Frontend](./frontend.md) | React/TypeScript frontend: components, state management, Cesium integration |
| [Data Model](./data_model.md) | Product registry, index schemas, storage layout conventions |
| [API Reference](./api_reference.md) | Complete REST API documentation with examples |
| [Performance](./performance.md) | Bottlenecks, LOD strategy, caching, profiling techniques |
| [Dev Workflow](./dev_workflow.md) | Local development setup, scripts, linting, building |
| [Deployment](./deployment.md) | Production deployment, environment variables, infrastructure |
| [Troubleshooting](./troubleshooting.md) | Common issues, debugging techniques, known problems |

---

## Repository Structure

```
MarsLab/
├── backend/                          # Python FastAPI backend
│   ├── api/                          # API route modules
│   │   ├── crism/                    # CRISM-specific processing
│   │   │   ├── loader.py             # ENVI hyperspectral cube loading
│   │   │   ├── processing.py         # RGB generation, normalization
│   │   │   ├── resolver.py           # File path resolution
│   │   │   ├── rgb.py                # RGB image generation
│   │   │   ├── router.py             # CRISM API endpoints
│   │   │   └── spectrum.py           # Spectral extraction
│   │   ├── search_router.py          # Search & download endpoints
│   │   ├── footprints_router.py      # Viewport-based footprint API
│   │   ├── download_manager.py       # Async download task management
│   │   ├── ode_client.py             # ODE REST API client
│   │   ├── registry.py               # Instrument registry singleton
│   │   └── hirise_pixel.py           # HiRISE pixel inspection
│   ├── scripts/                      # Build and processing scripts
│   │   ├── build_crism_quickviews.py # Generate CRISM thumbnails
│   │   ├── build_hirise_quickviews.py# Generate HiRISE thumbnails
│   │   ├── generate_score_maps.py    # Compute ice/hydration scores
│   │   └── crop_score_maps.py        # Crop score maps to bounds
│   ├── data/                         # Configuration files
│   │   └── registry.json             # Instrument definitions
│   ├── crism_data/                   # CRISM observation data
│   │   └── index.geojson             # CRISM footprint index
│   ├── hirise_data/                  # HiRISE product data
│   │   └── index.geojson             # HiRISE footprint index
│   ├── sharad_data/                  # SHARAD radargram data
│   │   └── index.geojson             # SHARAD track index
│   ├── crism_quickview/              # CRISM preview thumbnails
│   ├── crism_browse/                 # CRISM browse products (HYD, ICE, etc.)
│   ├── crism_score/                  # Computed score maps
│   ├── hirise_quickview/             # HiRISE preview thumbnails
│   ├── sharad_quickview/             # SHARAD preview images
│   ├── sharad_highres/               # SHARAD high-resolution data
│   ├── .overlay_cache/               # Cached overlay images
│   ├── app.py                        # Main FastAPI application
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # React + TypeScript frontend
│   ├── src/
│   │   ├── api/                      # API client functions
│   │   │   ├── hirise.ts             # HiRISE API helpers
│   │   │   └── search.ts             # Search & download API
│   │   ├── components/               # React components
│   │   │   ├── MapView.tsx           # Cesium 3D globe/2D map
│   │   │   ├── Inspector.tsx         # Product detail panel
│   │   │   ├── LayerPanel.tsx        # Map controls sidebar
│   │   │   ├── TopBar.tsx            # Search & navigation
│   │   │   └── layout/               # Layout components
│   │   │       └── AppShell.tsx      # Main layout wrapper
│   │   ├── pages/                    # Page components
│   │   │   ├── MainPage.tsx          # Main app page (state hub)
│   │   │   └── DataDownloadPage.tsx  # Download management page
│   │   ├── utils/                    # Utility modules
│   │   │   ├── FootprintManager.ts   # Footprint loading & rendering
│   │   │   ├── search.ts             # Search utilities
│   │   │   └── perfMonitor.ts        # Performance monitoring
│   │   ├── config/                   # Configuration
│   │   │   └── instrumentRegistry.ts # Instrument definitions
│   │   ├── App.tsx                   # Root component with routing
│   │   ├── main.tsx                  # Application entry point
│   │   └── index.css                 # Global styles
│   ├── public/                       # Static assets
│   ├── dist/                         # Production build output
│   ├── vite.config.ts                # Vite build configuration
│   ├── tsconfig.json                 # TypeScript configuration
│   ├── tailwind.config.js            # Tailwind CSS configuration
│   ├── package.json                  # Node.js dependencies
│   └── README.md                     # Frontend-specific docs
│
├── docs/                             # Documentation
│   └── technical/                    # Technical documentation (this folder)
│
├── PERF_NOTES.md                     # Performance optimization notes
└── .claude/                          # Claude AI configuration
```

---

## Directory Purpose Quick Reference

| Directory | Purpose |
|-----------|---------|
| `backend/api/` | FastAPI route handlers and business logic |
| `backend/api/crism/` | CRISM-specific hyperspectral processing |
| `backend/scripts/` | Data processing and build scripts |
| `backend/*_data/` | Instrument data storage (GeoTIFF, ENVI, GeoJSON) |
| `backend/*_quickview/` | Preview thumbnail images |
| `frontend/src/components/` | React UI components |
| `frontend/src/pages/` | Top-level page components |
| `frontend/src/utils/` | Shared utility modules |
| `frontend/src/config/` | Frontend configuration |
| `frontend/src/api/` | API client functions |

---

## Key Technologies

### Backend
- **FastAPI** - Modern async Python web framework
- **Rasterio** - GeoTIFF reading and processing
- **NumPy/OpenCV** - Image processing
- **aiohttp/aiofiles** - Async HTTP and file I/O
- **Pydantic** - Data validation

### Frontend
- **React 19** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Cesium** - 3D geospatial rendering
- **Tailwind CSS** - Utility-first styling

---

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

---

## Related Documents

- [PERF_NOTES.md](../../PERF_NOTES.md) - Detailed performance optimization notes
- [frontend/README.md](../../frontend/README.md) - Frontend-specific setup guide
