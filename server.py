from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from typing import Optional
from echo import process_echo
import time
from collections import defaultdict
import os
import asyncio
from contextlib import asynccontextmanager

MAX_WORKERS = 6
PROCESS_TIMEOUT = 60
REQUESTS_PER_MINUTE = 60
PORT = int(os.getenv("PORT", "5000"))

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > minute_ago]
        if len(self.requests[ip]) < REQUESTS_PER_MINUTE:
            self.requests[ip].append(now)
            return True
        return False

class ImageRequest(BaseModel):
    image: str

class OCRResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    analysis: Optional[dict] = None

class APIStatus(BaseModel):
    status: str = "running"
    endpoints: dict = {
        "ocr": {
            "path": "/api/ocr",
            "method": "POST",
            "request": {
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "base64",
                        }
                    }
                },
                "required": ["images"]
            }
        }
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
rate_limiter = RateLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/ocr":
        client_ip = request.client.host
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                }
            )
    response = await call_next(request)
    return response

async def process_echo_image(image_bytes: bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(executor, process_echo, image)
        result = await asyncio.wait_for(future, timeout=PROCESS_TIMEOUT)
        return result
            
    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing error: {str(e)}"
        )
@app.get("/", response_model=APIStatus)
async def homepage():
    return APIStatus()

@app.post("/api/ocr", response_model=OCRResponse)
async def process_image_request(request: Request, image_data: ImageRequest):
    request_start = time.perf_counter()
    print("\n=== New Echo Image Request ===")
    print(f"Origin: {request.headers.get('origin', 'unknown')}")
    print("=================================")
        
    try:
        image_str = image_data.image
        if ',' in image_str:
            image_str = image_str.split(',')[1]
        image_bytes = base64.b64decode(image_str)
        
        result = await process_echo_image(image_bytes)
        print(f"Total request time: {time.perf_counter() - request_start:.2f}s")
        return result
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)