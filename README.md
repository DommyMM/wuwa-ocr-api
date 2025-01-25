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
Process screenshots for Echo analysis or Character import.

Request:
```json
{
    "image": "base64_encoded_image_string",
    "type": "echo | card"
}
```

#### Echo Response (type: "echo"):
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

#### Character Card Response (type: "import"):
```json
{
    "character": {
        "name": "Character Name",
        "level": 90
    },
    "watermark": {
        "username": "Player Name",
        "uid": 500000000
    },
    "weapon": {
        "name": "Weapon Name",
        "level": 90
    },
    "fortes": [
        "10", "10", "10", "10", "10"
    ],
    "echoes": [
        {
            "name": {
                "name": "Echo Name",
                "confidence": 0.85
            },
            "main": {
                "name": "Main Stat",
                "value": "Value"
            },
            "substats": [
                {
                    "name": "Substat Name",
                    "value": "Value"
                }
                // ... up to 5 substats
            ],
            "element": "Element Type"
        }
        // ... 5 echoes total
    ]
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
- For echo type: Image must be a screenshot of an individual Echo
- For import type: Image must be a full screenshot from the character builder