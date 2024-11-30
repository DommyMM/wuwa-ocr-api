from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import cv2
import numpy as np
import base64
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
from typing import Optional
from ocr import process_image
from contextlib import asynccontextmanager
import time
from collections import defaultdict
import os
import sys
from io import StringIO

MAX_WORKERS = max(2, multiprocessing.cpu_count() - 1)
PROCESS_TIMEOUT = 30
REQUESTS_PER_MINUTE = 20
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
    version: str = "1.0.0"
    endpoints: dict = {
        "ocr": "/api/ocr"
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

async def process_image_bytes(image_bytes: bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        stdout = StringIO()
        sys.stdout = stdout
        
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, process_image, image)
            result = await asyncio.wait_for(future, timeout=PROCESS_TIMEOUT)
            
            debug_output = stdout.getvalue()
            
            if result["success"]:
                result["analysis"]["debug_output"] = debug_output
            else:
                result["debug_output"] = debug_output
            
            return result
        finally:
            sys.stdout = sys.__stdout__
            
    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds allowed)"
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
    try:
        image_str = image_data.image
        if ',' in image_str:
            image_str = image_str.split(',')[1]
            
        try:
            image_bytes = base64.b64decode(image_str)
            result = await process_image_bytes(image_bytes)
            return result
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Image processing error: {str(e)}"
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)