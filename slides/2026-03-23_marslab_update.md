# MarsLab Platform Update — 2026.03.23

---

## Agenda

1. **Bug Fixes** — Display System & Download Pipeline
2. **New Feature** — Save to Local (Direct-to-PC Download)

---

## 1. Bug Fixes Overview

| Issue | Root Cause | Impact |
|-------|-----------|--------|
| File download 500 error | slowapi parameter naming conflict | All downloads blocked |
| AI Search hangs forever | Ollama unresponsive (local LLM timeout) | Smart search unusable |
| AI Search parse failure | LLM returning schema description literally | Search returns empty |
| HiRISE images not displayed | LBL path mismatch + missing quickview | HiRISE overlay broken |

---

## 1-1. Download Pipeline — 500 Internal Server Error

**Symptom**: `POST /api/download` returns HTTP 500 for all products

**Root Cause**: `slowapi` rate limiter requires the HTTP request parameter to be named exactly `request`. The endpoint named it `http_request`, causing an internal exception.

```python
# Before (broken)
async def start_download(http_request: Request, request: DownloadRequest):

# After (fixed)
async def start_download(request: Request, body: DownloadRequest):
```

**Result**: All instrument downloads (CRISM, HiRISE, SHARAD) restored

---

## 1-2. AI Search — Ollama → Groq Migration

**Symptom**: Smart AI search hangs indefinitely (>60s timeout)

**Root Cause**: Local Ollama (llama3.1:8b) unresponsive — GPU resource contention

**Fix**: Replaced Ollama with **Groq API** (`llama-3.1-8b-instant`)

| Metric | Ollama (Before) | Groq (After) |
|--------|----------------|--------------|
| Parse latency | >60s (timeout) | ~0.4s |
| Selection latency | N/A | ~0.5s |
| Availability | Unreliable | 99.9% |

---

## 1-2. AI Search — Prompt Engineering Fix

**Symptom**: LLM returns `"type": "named_region | bbox | point | global"` — the schema description verbatim

**Root Cause**: Pipe-separated union types in the prompt confused the model

**Fix (two-layer)**:
1. **Prompt**: Replaced union syntax with concrete example values
2. **Parser**: Added auto-recovery — infers `region_type` from available fields when invalid

```
Before: "type": "named_region | bbox | point | global"
After:  "type": "named_region"  (with explicit enum docs)
```

---

## 1-2. AI Search — Performance Optimization

**Additional optimization**: Local-first search strategy

- Search local GeoJSON index first (instant, 0.004s)
- Skip ODE API call if local results exist
- Parallel instrument search with `asyncio.gather`

| Step | Before | After |
|------|--------|-------|
| Parse (Groq) | — | 0.4s |
| Search (ODE) | 4.0s | 0.1s (local-first) |
| Select (Groq) | — | 0.5s |
| **Total** | **>60s** | **~1.0s** |

---

## 1-3. HiRISE Display — LBL Path Mismatch

**Symptom**: HiRISE image overlays fail to render on the map — repeated 404 on LBL files

**Root Cause**: Frontend requests flat path (`/hirise_lbl/ESP_xxx_RED.lbl`), but downloaded files are in subdirectories (`/hirise_lbl/ESP_xxx_RED/ESP_xxx_RED.LBL`)

**Fix**: Multi-pattern LBL loader with fallback

```
Try order:
  1. /{id}_RED/{id}_RED.LBL   (new downloads, subdirectory)
  2. /{id}_RED.lbl             (legacy flat files)
  3. /{id}_RED/{id}_RED.lbl    (case variant)
```

---

## 1-3. HiRISE Display — On-the-fly Quickview Generation

**Symptom**: Products with JP2 data downloaded but no quickview thumbnail → overlay fails

**Root Cause**: Quickview generation was not included in download pipeline

**Fix**: Added on-the-fly quickview generation from JP2

1. Request: `GET /hirise/quickview/ESP_xxx.png`
2. No cached quickview → open JP2 with **rasterio**
3. Downsample to ~800px wide, percentile stretch to 8-bit
4. Save as JPG (cached for subsequent requests)
5. Apply black→transparent conversion, return PNG

**Result**: All downloaded HiRISE products now display correctly

---

## 2. New Feature — Save to Local

### Problem Statement

Previously, users could only download data **to the MarsLab server**. To get files on their local PC, they had to:

1. Download from PDS → MarsLab server
2. Then download from MarsLab server → local PC (ZIP)

**This required the data to exist on MarsLab first.**

---

## 2. Save to Local — Architecture

```
                    ┌─────────────┐
                    │   PDS/ODE   │
                    │   Server    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
     ┌─ Download ──┐  ┌─ Save to ──┐  ┌── Save to ──┐
     │  to MarsLab │  │ Local (PDS)│  │ Local (ZIP) │
     │  (existing) │  │   (NEW)    │  │ (existing)  │
     └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
            │               │                │
            ▼               ▼                ▼
     ┌────────────┐  ┌────────────┐   ┌────────────┐
     │  MarsLab   │  │  User's    │   │  User's    │
     │  Server    │  │  Browser   │   │  Browser   │
     └────────────┘  └────────────┘   └────────────┘
                     Direct from PDS   From MarsLab
```

---

## 2. Save to Local — How It Works

### Direct PDS Download (New)
- Click **"Save to Local"** on any search result
- Backend resolves PDS URL via `/api/product-urls/` endpoint
- Browser downloads directly from NASA PDS — **no MarsLab storage needed**
- Supports: CRISM (.img), HiRISE (.jp2), labels (.lbl)

### ZIP Download (Enhanced)
- For products already on MarsLab, **"ZIP"** button creates streaming archive
- Available in all search modes and download history

---

## 2. Save to Local — UI Integration

**Available in all search modes:**

| Search Mode | Save to Local | Status |
|-------------|:---:|:---:|
| Product ID search | ✓ | New |
| Spatial (bbox) search | ✓ | New |
| Coordinate (point) search | ✓ | New |
| Product proximity search | ✓ | New |
| AI Smart search | ✓ | New |
| Download history | ✓ | Existing |
| Inspector panel | ✓ | Existing |

**Works for both Remote and MarsLab products**

---

## 2. Terminology Update

Unified labeling across the platform to avoid confusion:

| Before | After | Meaning |
|--------|-------|---------|
| "Local" badge | **"MarsLab"** badge | Product exists on MarsLab server |
| "Complete" badge | **"MarsLab"** badge | All files downloaded to server |
| "Downloaded" badge | **"MarsLab"** badge | Consistent naming |
| "Save to Local" button | **"Save to Local"** button | Download to user's PC (unchanged) |

Clear distinction: **MarsLab** = server storage, **Local** = user's computer

---

## Summary

### Bug Fixes
- ✓ Download 500 error — parameter naming fix
- ✓ AI Search — migrated to Groq API (60s → 1s)
- ✓ AI Search — prompt + parser hardening
- ✓ HiRISE display — LBL path resolution + on-the-fly quickview

### New Feature
- ✓ Save to Local — direct PDS download to user's PC
- ✓ Available in all 5 search modes
- ✓ No MarsLab storage required for PDS downloads
- ✓ Consistent MarsLab / Local terminology

---

## Thank You
