# Development Workflow

This document covers local development setup, common commands, linting, building, and testing for MarsLab.

---

## Prerequisites

### System Requirements

- **Python** 3.10+
- **Node.js** 18+ (with npm)
- **GDAL** (optional, for HiRISE JP2 conversion)

### Installing GDAL (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install gdal-bin python3-gdal
```

### Installing GDAL (macOS)

```bash
brew install gdal
```

---

## Initial Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd MarsLab
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install Shapely for geometry simplification
pip install shapely
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Data Setup

Ensure the following directories exist with data:

```
backend/
├── crism_data/index.geojson
├── hirise_data/index.geojson
├── sharad_data/index.geojson
├── crism_quickview/
├── hirise_quickview/
└── sharad_quickview/
```

If indexes are missing, the API will return 404 errors.

---

## Running Development Servers

### Start Backend

```bash
cd backend
source venv/bin/activate  # if using venv

# Development mode with auto-reload
uvicorn app:app --reload --port 8000

# Or with more workers
uvicorn app:app --reload --port 8000 --workers 2
```

Backend will be available at: `http://localhost:8000`

### Start Frontend

```bash
cd frontend

# Development mode with HMR
npm run dev
```

Frontend will be available at: `http://localhost:5173`

The Vite dev server proxies API requests to the backend at port 8000.

### Running Both (Recommended)

Use two terminal windows:

**Terminal 1 (Backend):**
```bash
cd backend && uvicorn app:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend && npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## Common Commands

### Backend Commands

| Command | Description |
|---------|-------------|
| `uvicorn app:app --reload` | Start dev server |
| `uvicorn app:app --host 0.0.0.0` | Listen on all interfaces |
| `pip install -r requirements.txt` | Install dependencies |
| `pip freeze > requirements.txt` | Update dependencies |

### Frontend Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server (port 5173) |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

### Data Processing Scripts

| Script | Description |
|--------|-------------|
| `python scripts/build_crism_quickviews.py` | Generate CRISM thumbnails |
| `python scripts/build_hirise_quickviews.py` | Generate HiRISE thumbnails |
| `python scripts/generate_score_maps.py` | Compute ice/hydration scores |
| `python scripts/crop_score_maps.py` | Crop score maps to bounds |

---

## Linting & Formatting

### Frontend Linting

```bash
cd frontend

# Run ESLint
npm run lint

# Fix auto-fixable issues
npx eslint . --fix
```

**ESLint Configuration:** `frontend/eslint.config.js`

Enforces:
- React Hooks rules
- React Refresh compatibility
- TypeScript strict checks

### Backend Linting (Optional)

Install and run:

```bash
pip install ruff

# Check
ruff check .

# Fix
ruff check . --fix
```

### Type Checking

**Frontend:**
```bash
cd frontend
npx tsc --noEmit
```

**Backend (optional):**
```bash
pip install mypy
mypy app.py
```

---

## Building for Production

### Frontend Production Build

```bash
cd frontend

# Build
npm run build

# Output in frontend/dist/
```

Build outputs:
- `dist/index.html`
- `dist/assets/*.js`
- `dist/assets/*.css`

### Serving Production Build

**Option 1: Vite Preview**
```bash
npm run preview
```

**Option 2: Static Server**
```bash
npx serve dist
```

**Option 3: From Backend**

Copy `dist/` contents to backend static directory and configure FastAPI to serve them.

---

## Environment Variables

### Backend

Currently, all configuration is hardcoded. For production, consider:

```python
# app.py
import os

HIRISE_DATA_DIR = os.getenv("HIRISE_DATA_DIR", os.path.join(BASE_DIR, "hirise_data"))
CRISM_DATA_DIR = os.getenv("CRISM_DATA_DIR", os.path.join(BASE_DIR, "crism_data"))
```

### Frontend

Vite environment variables (create `.env` file):

```env
VITE_API_URL=http://localhost:8000
```

Access in code:
```typescript
const apiUrl = import.meta.env.VITE_API_URL || '';
```

---

## Project Structure for Development

### Adding a New API Endpoint

1. Create router in `backend/api/`:

```python
# backend/api/my_router.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/my-endpoint")
async def my_endpoint():
    return {"status": "ok"}
```

2. Include in `app.py`:

```python
from api.my_router import router as my_router
app.include_router(my_router)
```

### Adding a New Component

1. Create component file:

```typescript
// frontend/src/components/MyComponent.tsx
export default function MyComponent() {
  return <div>My Component</div>;
}
```

2. Import and use in parent:

```typescript
import MyComponent from "./MyComponent";

// In JSX:
<MyComponent />
```

### Adding a New Page

1. Create page component:

```typescript
// frontend/src/pages/MyPage.tsx
export default function MyPage() {
  return <div>My Page</div>;
}
```

2. Add route in `App.tsx`:

```typescript
import MyPage from "./pages/MyPage";

// In Routes:
<Route path="/my-page" element={<MyPage />} />
```

---

## Debugging

### Backend Debugging

**Add breakpoints with debugpy:**

```bash
pip install debugpy

# Run with debugger
python -m debugpy --listen 5678 -m uvicorn app:app --reload
```

Then attach VS Code debugger.

**Print debugging:**

```python
import json
print(json.dumps(data, indent=2))
```

### Frontend Debugging

**Browser DevTools:**
- React DevTools extension
- Console logging
- Network tab for API calls
- Performance tab for profiling

**VS Code debugging:**

`.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Launch Chrome",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/frontend/src"
    }
  ]
}
```

### Cesium Debugging

```typescript
// Enable Cesium inspector
viewer.extend(Cesium.viewerCesiumInspectorMixin);

// Log entity count
console.log("Entities:", viewer.entities.values.length);

// Log camera position
console.log("Camera:", viewer.camera.positionCartographic);
```

---

## Testing

### Manual Testing Checklist

- [ ] Load page without errors
- [ ] Pan/zoom map smoothly
- [ ] Load CRISM footprints
- [ ] Load HiRISE footprints
- [ ] Load SHARAD tracks
- [ ] Click footprint to select
- [ ] Toggle overlay types
- [ ] Adjust overlay opacity
- [ ] Search for product
- [ ] Fly to product
- [ ] Switch 2D/3D mode
- [ ] Change base layer
- [ ] Apply ice score filter

### API Testing

```bash
# Test footprints endpoint
curl "http://localhost:8000/api/footprints?instrument=CRISM&bbox=-10,-5,10,5" | jq .

# Test search endpoint
curl "http://localhost:8000/api/search?q=frt00009" | jq .

# Test existence check
curl "http://localhost:8000/api/exists/crism/frt00009312_07_if165l_trr3" | jq .
```

---

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation

### Commit Messages

Follow conventional commits:

```
feat: add ice score filtering API
fix: handle antimeridian crossing in footprints
refactor: extract FootprintManager class
docs: add API reference documentation
```

### Before Committing

1. Run linting:
   ```bash
   cd frontend && npm run lint
   ```

2. Type check:
   ```bash
   cd frontend && npx tsc --noEmit
   ```

3. Test API endpoints manually

4. Review changes:
   ```bash
   git diff --staged
   ```

---

## Troubleshooting Development Issues

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000
# or
netstat -tulpn | grep 8000

# Kill process
kill -9 <PID>
```

### Module Not Found (Python)

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Module Not Found (Node)

```bash
# Clear node_modules and reinstall
rm -rf node_modules
npm install
```

### Vite Proxy Not Working

Check `vite.config.ts` proxy configuration. Ensure backend is running on port 8000.

### Cesium Assets Not Loading

Ensure `vite-plugin-cesium` is configured in `vite.config.ts`.

### TypeScript Errors

```bash
# Check for type errors
npx tsc --noEmit

# Fix common issues
npm run lint -- --fix
```
