# Wuthering Waves Scanner Backend

FastAPI-based backend service for analyzing screenshots from Wuthering Waves. Features OCR (Optical Character Recognition) and element detection capabilities to extract information from both Echo screenshots and character cards. 
Currently hosted at https://ocr.wuwabuilds.moe/

## Features

- Screenshot analysis using multiple OCR engines (RapidOCR and Tesseract)
- Element detection via HSV color analysis
- Name validation and matching with rapidfuzz
- Stat normalization against known valid values
- Character and weapon validation
- Rate limiting (60 requests per minute per IP)
- CORS support for web integration
- Parallel processing with timeout protection

## Prerequisites

- Run Dockerfile & requirements.txt

## Usage

1. Start the server:
```bash
python server.py
```

Server will start on port 5000 by default (configurable via PORT environment variable).

## API Endpoints

### `POST /api/ocr`
Process split card regions from the frontend import flow.

Request:
```json
{
    "image": "base64_encoded_image_string"
}
```

Headers:
```http
X-OCR-Region: character | weapon | watermark | forte | sequences | echo1 | echo2 | echo3 | echo4 | echo5
```

#### Echo Split Response (for `echo1`..`echo5`)
```json
{
    "success": true,
    "analysis": {
        "name": {
            "name": "Echo Name",
            "id": "60000425",
            "confidence": 0.85
        },
        "main": {
            "name": "Stat Name",
            "value": "Value"
        },
        "substats": [
            { "name": "Substat Name", "value": "Value" }
        ],
        "element": "Element Type"
    }
}
```

### `GET /health`
Health check endpoint.

### `GET /`
API documentation and status.

## Rate Limiting

- 60 requests per minute per IP address
- Requests exceeding the limit will receive a 429 status code

## Error Handling

- 400: Invalid image or processing error
- 408: Processing timeout (60 seconds)
- 429: Rate limit exceeded
- 500: Server error

## Known Limitations

- Only handles English for now
- Expects split regions from the import workflow (not the removed legacy full-screen mode)
