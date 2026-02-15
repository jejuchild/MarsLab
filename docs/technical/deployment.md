# Deployment Documentation

This document covers production deployment, environment configuration, infrastructure recommendations, and operational considerations for MarsLab.

---

## Deployment Topology

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer / CDN                      │
│                    (Cloudflare, AWS CloudFront)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Reverse Proxy                            │
│                   (Nginx / Caddy)                            │
│                                                              │
│  - SSL termination                                           │
│  - Static file serving (frontend)                            │
│  - API proxying to backend                                   │
│  - Gzip compression                                          │
│  - Rate limiting                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│   Frontend Static    │   │   Backend API        │
│   (built files)      │   │   (FastAPI + Uvicorn)│
│                      │   │                      │
│   /index.html        │   │   /api/*             │
│   /assets/*          │   │   /crism/*           │
│                      │   │   /hirise/*          │
└──────────────────────┘   │   /sharad/*          │
                           └──────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Data Storage       │
                           │                      │
                           │   - GeoJSON indexes  │
                           │   - GeoTIFF images   │
                           │   - ENVI cubes       │
                           │   - Overlay cache    │
                           └──────────────────────┘
```

### Simple Deployment (Single Server)

For smaller deployments:

```
┌─────────────────────────────────────────┐
│              Single Server              │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         Nginx                    │   │
│  │  - Serve frontend at /          │   │
│  │  - Proxy /api/* to :8000        │   │
│  └─────────────────────────────────┘   │
│                 │                       │
│                 ▼                       │
│  ┌─────────────────────────────────┐   │
│  │    Uvicorn (systemd service)    │   │
│  │    Port 8000                    │   │
│  └─────────────────────────────────┘   │
│                 │                       │
│                 ▼                       │
│  ┌─────────────────────────────────┐   │
│  │    /var/www/marslab/data/       │   │
│  │    (GeoJSON, GeoTIFF, etc.)     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Deployment Steps

### 1. Build Frontend

```bash
cd frontend
npm ci
npm run build

# Output in dist/
ls dist/
# index.html  assets/
```

### 2. Prepare Backend

```bash
cd backend

# Install production dependencies
pip install -r requirements.txt

# Optional: Install Shapely for geometry simplification
pip install shapely
```

### 3. Configure Nginx

**/etc/nginx/sites-available/marslab:**

```nginx
server {
    listen 80;
    server_name marslab.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name marslab.example.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/marslab.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/marslab.example.com/privkey.pem;

    # Frontend static files
    root /var/www/marslab/frontend/dist;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript image/svg+xml;
    gzip_min_length 1000;

    # Frontend routes (SPA)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for large file operations
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # CRISM/HiRISE/SHARAD API proxy
    location ~ ^/(crism|hirise|sharad|hirise_lbl|crism_lbl|world_meta|world_tiles)/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Cache control for static assets
        add_header Cache-Control "public, max-age=3600";
    }

    # Index files proxy
    location ~ ^/(hirise_index|crism_index|sharad_index)\.geojson$ {
        proxy_pass http://127.0.0.1:8000;
        add_header Cache-Control "public, max-age=3600";
    }

    # Rate limiting for API
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location /api/footprints {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/marslab /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Create Systemd Service

**/etc/systemd/system/marslab.service:**

```ini
[Unit]
Description=MarsLab Backend API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/marslab/backend
Environment="PATH=/var/www/marslab/backend/venv/bin"
ExecStart=/var/www/marslab/backend/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

# Resource limits
MemoryLimit=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable marslab
sudo systemctl start marslab
sudo systemctl status marslab
```

### 5. SSL Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d marslab.example.com
```

---

## Environment Variables

### Backend Configuration

Create `/var/www/marslab/backend/.env`:

```env
# Data directories (optional - defaults to relative paths)
HIRISE_DATA_DIR=/var/www/marslab/data/hirise_data
CRISM_DATA_DIR=/var/www/marslab/data/crism_data
SHARAD_DATA_DIR=/var/www/marslab/data/sharad_data

# Performance tuning
MAX_WORKERS=4
OVERLAY_CACHE_SIZE_GB=10

# CORS (restrict in production)
CORS_ORIGINS=https://marslab.example.com

# Logging
LOG_LEVEL=INFO
```

Update `app.py` to read environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

HIRISE_DATA_DIR = os.getenv("HIRISE_DATA_DIR", os.path.join(BASE_DIR, "hirise_data"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
```

### Frontend Configuration

For frontend environment variables, create `.env.production`:

```env
VITE_API_URL=https://marslab.example.com
```

Build with production config:
```bash
npm run build
```

---

## Infrastructure Recommendations

### Minimum Server Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 50 GB SSD | 200+ GB SSD |
| Network | 100 Mbps | 1 Gbps |

### Storage Sizing

| Data Type | Typical Size | Notes |
|-----------|--------------|-------|
| CRISM indexes | 2 MB | Per index file |
| HiRISE indexes | 500 KB | Per index file |
| CRISM quickviews | 5-20 KB each | ~5700 files = ~50 MB |
| HiRISE quickviews | 50-200 KB each | ~2000 files = ~200 MB |
| CRISM data (per obs) | 100-500 MB | Full hyperspectral cube |
| HiRISE data (per product) | 200 MB - 2 GB | GeoTIFF images |
| Overlay cache | Varies | ~1 MB per cached overlay |

**Example storage calculation:**
- 1000 CRISM observations × 300 MB = 300 GB
- 500 HiRISE products × 500 MB = 250 GB
- Overlay cache × 100 MB = 100 MB
- Total: ~550 GB + headroom = 750 GB recommended

### Scaling Options

**Vertical Scaling:**
- Increase CPU cores for more Uvicorn workers
- Increase RAM for larger LRU caches
- Use NVMe SSD for faster file I/O

**Horizontal Scaling:**
- Load balancer + multiple backend instances
- Shared storage (NFS/S3) for data files
- Sticky sessions for overlay cache consistency

---

## Docker Deployment (Optional)

### Dockerfile (Backend)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - overlay_cache:/app/.overlay_cache
    environment:
      - HIRISE_DATA_DIR=/app/data/hirise_data
      - CRISM_DATA_DIR=/app/data/crism_data
    restart: unless-stopped

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./frontend/dist:/usr/share/nginx/html:ro
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  overlay_cache:
```

Build and run:
```bash
cd frontend && npm run build && cd ..
docker-compose up -d
```

---

## Monitoring & Logging

### Application Logs

```bash
# View backend logs
sudo journalctl -u marslab -f

# View Nginx access logs
sudo tail -f /var/log/nginx/access.log

# View Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Health Check Endpoint

Add to `app.py`:

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "indexes_loaded": {
            "crism": os.path.exists(os.path.join(CRISM_DATA_DIR, "index.geojson")),
            "hirise": os.path.exists(os.path.join(HIRISE_DATA_DIR, "index.geojson")),
            "sharad": os.path.exists(os.path.join(SHARAD_DATA_DIR, "index.geojson")),
        }
    }
```

### Monitoring Tools

**Prometheus + Grafana:**
```python
pip install prometheus-fastapi-instrumentator

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

**Simple Uptime Monitoring:**
```bash
# Add to crontab
*/5 * * * * curl -sf http://localhost:8000/health || echo "MarsLab down" | mail -s "Alert" admin@example.com
```

---

## Backup & Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/backups/marslab
DATE=$(date +%Y%m%d)

# Backup indexes
tar -czvf $BACKUP_DIR/indexes_$DATE.tar.gz \
    /var/www/marslab/data/crism_data/index.geojson \
    /var/www/marslab/data/hirise_data/index.geojson \
    /var/www/marslab/data/sharad_data/index.geojson

# Backup configuration
tar -czvf $BACKUP_DIR/config_$DATE.tar.gz \
    /etc/nginx/sites-available/marslab \
    /etc/systemd/system/marslab.service \
    /var/www/marslab/backend/.env

# Rotate old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Procedure

1. Restore configuration files
2. Restore index files
3. Verify data directories exist
4. Restart services:
   ```bash
   sudo systemctl restart marslab
   sudo systemctl restart nginx
   ```
5. Verify health endpoint

---

## Security Recommendations

### Production Checklist

- [ ] Restrict CORS origins to production domain
- [ ] Enable HTTPS with valid certificate
- [ ] Configure firewall (allow 80, 443 only)
- [ ] Use non-root user for application
- [ ] Set appropriate file permissions
- [ ] Enable rate limiting
- [ ] Disable debug mode
- [ ] Review and restrict API endpoints
- [ ] Monitor for suspicious activity

### Firewall (UFW)

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

### File Permissions

```bash
# Set ownership
sudo chown -R www-data:www-data /var/www/marslab

# Restrict write access
sudo chmod -R 755 /var/www/marslab
sudo chmod -R 750 /var/www/marslab/backend/.env
```

---

## Configuration Defaults

| Setting | Default | Production Recommendation |
|---------|---------|---------------------------|
| Uvicorn workers | 1 | CPU cores × 2 + 1 |
| CORS origins | `*` | Production domain only |
| Log level | DEBUG | INFO or WARNING |
| Overlay max size | 2048 | 2048-4096 |
| Footprint limit | 2000 | 2000-5000 |
| Cache sizes | Various | Increase with available RAM |
| Request timeout | 120s | 120-300s |

---

## Troubleshooting Deployment

### Backend Not Starting

```bash
# Check systemd status
sudo systemctl status marslab

# Check logs
sudo journalctl -u marslab -n 50

# Test manually
cd /var/www/marslab/backend
source venv/bin/activate
uvicorn app:app --port 8000
```

### 502 Bad Gateway

- Backend not running: `sudo systemctl start marslab`
- Wrong proxy port: Check Nginx config
- Firewall blocking: Check local firewall rules

### Slow Response Times

- Check overlay cache: Clear `.overlay_cache/` if corrupted
- Increase workers: Adjust `--workers` in service file
- Check disk I/O: Use `iotop` to monitor

### SSL Certificate Issues

```bash
# Renew certificate
sudo certbot renew

# Check certificate
sudo certbot certificates
```
