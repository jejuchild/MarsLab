# MarsLab Update — 2026.03.23

## 1. Display System Fixes

- **Download pipeline**: Fixed HTTP 500 error blocking all file downloads (CRISM, HiRISE, SHARAD)
- **AI Search**: Migrated from Ollama (local, 60s+ timeout) to Groq API — response time reduced to ~1s
- **HiRISE overlay**: Fixed LBL path resolution for subdirectory structure; added on-the-fly quickview generation from JP2 when cached thumbnail is missing

## 2. Save to Local

- **Direct-to-PC download**: Users can now download PDS files (IMG, JP2, LBL) directly to their local computer without storing on MarsLab server first
- **Available across all search modes**: ID, Spatial, Coordinate, Proximity, and AI search
- **Terminology**: "MarsLab" badge = on server, "Save to Local" = download to user's PC
