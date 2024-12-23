# Wuthering Waves Echo Scanner Backend

FastAPI-based backend service for analyzing Echo screenshots from Wuthering Waves. OCR (Optical Character Recognition) and element detection capabilities to extract Echo information from screenshots. 
Currently hosted at https://ocr.wuwabuilds.moe/

## Features

- Screenshot analysis using OCR (Tesseract)
- Element detection via HSV color analysis
- Name validation and matching with rapidfuzz
- Stat normalization against known valid values
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
Process an Echo screenshot.

Request body:
```json
{
    "image": "base64_encoded_image_string"
}
```

Response:
```json
{
    "success": true,
    "analysis": {
        "type": "Echo",
        "name": "Echo Name",
        "element": "Element Type",
        "echoLevel": "25",
        "main": {
            "name": "Stat Name",
            "value": "Value"
        },
        "subs": [
            {
                "name": "Substat Name",
                "value": "Value"
            }
            // ... up to 5 substats
        ]
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
- Image must be a screenshot of an Echo
- Assumes the crop is of the region {"top": 0.11, "left": 0.72, "width": 0.25, "height": 0.35} from a full screenshot