# Wuthering Waves OCR Backend

FastAPI OCR service for WuWaBuilds import scans.
Hosted at `https://ocr.wuwabuilds.moe`.

**Stack**: Python 3.13 · FastAPI · Tesseract + RapidOCR · OpenCV · SIFT/FLANN · Railway

---

## Overview

Processes cropped screenshot regions from the WuWaBuilds frontend and returns structured game data (character, weapon, echo, forte, sequence, watermark). Uses a hybrid OCR pipeline combining Tesseract and RapidOCR with computer vision techniques (SIFT feature matching, perceptual hashing, HSV color analysis) for accurate echo identification.

---

## Runtime Model

- Region-based OCR processing via `card.py`
- Multiprocessing with 8-worker pool for CPU-intensive image analysis
- Data loaded from `backend/Data/*.json` and template PNGs at startup
- Pre-computes SIFT features, perceptual hashes, and HSV histograms for all echo/element templates
- Echo, character, and weapon OCR results include IDs for robust frontend matching

---

## Start

```bash
pip install -r requirements.txt
python server.py
```

Default port is `5000` (`PORT` env var supported).

---

## API

### `POST /api/ocr`

Process one cropped region image.

Request body:

```json
{
  "image": "base64_encoded_image"
}
```

Recommended header:

```http
X-OCR-Region: character | weapon | watermark | forte | sequences | echo1 | echo2 | echo3 | echo4 | echo5
```

Compatibility fallbacks (optional body fields):

- `region`: direct region key
- `type`: legacy `import-<region>` format

### Example Responses

Character region:

```json
{
  "success": true,
  "analysis": {
    "name": "Aemeath",
    "id": "1210",
    "level": 90
  }
}
```

Weapon region:

```json
{
  "success": true,
  "analysis": {
    "name": "Everbright Polestar",
    "id": "21020076",
    "level": 90
  }
}
```

Echo region (`echo1`–`echo5`):

```json
{
  "success": true,
  "analysis": {
    "name": {
      "name": "Sigillum",
      "id": "60001915",
      "confidence": 0.87
    },
    "main": { "name": "Crit DMG", "value": "44%" },
    "substats": [{ "name": "Crit Rate", "value": "8.7%" }],
    "element": "Trailblazing"
  }
}
```

### Other Endpoints

- `GET /health` → health check
- `GET /` → API status metadata with endpoint documentation

---

## Limits and Errors

- Rate limit: `60` requests/minute per IP
- Timeout: `60s` per OCR request
- Auto pool restart after 3+ consecutive 500 errors
- Common statuses:
  - `400` invalid image/region/request
  - `408` processing timeout
  - `429` rate limit exceeded
  - `500` internal server error

---

## Echo Matching Pipeline

Echo identification uses a multi-stage pipeline balancing speed and accuracy:

1. **pHash ordering** — Perceptual hash distance ranks all echo templates; top 12 candidates selected
2. **SIFT feature matching** — FLANN-based keypoint matching scores each candidate; best match wins if confidence is clear
3. **Disambiguation** (if ambiguous) — Expands candidates and applies:
   - Element icon SIFT matching
   - HSV histogram color correlation (normal vs nightmare variants)
   - Cost OCR detection (1, 3, or 4)
   - Color vibrancy scoring for nightmare detection

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` / `uvicorn` | ASGI web framework + server |
| `opencv-python` | Image processing and SIFT feature detection |
| `pytesseract` | Tesseract OCR engine |
| `rapidocr-onnxruntime` | Fast ONNX-based OCR engine |
| `rapidfuzz` | Fuzzy string matching for name resolution |
| `imagehash` / `Pillow` | Perceptual hashing |
| `numpy` | Numerical arrays |
| `slowapi` | IP-based rate limiting |
| `pydantic` | Request/response validation |
| `python-dotenv` | Environment variable loading |

---

## Deployment

### Dockerfile

Multi-stage build on Python 3.13-slim. Runtime includes `tesseract-ocr`, OpenGL, and headless OpenCV support. Exposes port 5000.

### railway.toml

```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "python server.py"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
healthcheckPath = "/"
healthcheckTimeout = 30
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `5000` | Server listen port |

---

## Data Directory

```
Data/
├── Characters.json    # 54 characters (name, id, element, weaponType)
├── Weapons.json       # ~160 weapons by type (name, id)
├── Echoes.json        # 200+ echoes (name, id, cost, elements) incl. nightmare variants
├── Mainstat.json      # Valid main stats by echo cost (1, 3, 4)
├── Substats.json      # Valid substat values per stat type (13 types × 8 tiers)
├── Echoes/            # 180+ echo icon PNGs (188×188) for SIFT template matching
└── Elements/          # 28 element icon PNGs for disambiguation
```

---

## Data Sync

The backend does not fetch runtime game data from production frontend URLs.
Keep `backend/Data` synchronized from `wuwabuilds/scripts`:

```bash
python sync_all.py                                # Full pipeline
python download_echo_icons.py --clean --force    # Refresh echo template PNGs
```

