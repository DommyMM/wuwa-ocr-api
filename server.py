from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from concurrent.futures import TimeoutError, ProcessPoolExecutor
from typing import Optional
from card import process_card
from char import process_char
import time
from collections import defaultdict
import os
import asyncio
from contextlib import asynccontextmanager
import sys

MAX_WORKERS = 8
PROCESS_TIMEOUT = 60
REQUESTS_PER_MINUTE = 60
PORT = int(os.getenv("PORT", "5000"))
consecutive_500s = 0
MAX_CONSECUTIVE_500S = 3

# Ensure output is flushed for Railway
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def force_restart(reason: str):
    print(f"FORCING RESTART: {reason}", flush=True)
    time.sleep(1)  # Give time for log to be written
    os._exit(1)  # Hard exit that Railway will detect
    
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
    type: str

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
                "image": "string (base64 encoded image)",
                "type": "string ('char-type' or 'import-type')"
            }
        }
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Server starting on port {PORT}", flush=True)
    yield
    print("Server shutting down", flush=True)
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
def worker_init():
    """Ensure worker output is flushed"""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

executor = ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init
)
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

async def process_card_image(image_bytes: bytes, type: str):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(executor, process_card, image, type)
        result = await asyncio.wait_for(future, timeout=PROCESS_TIMEOUT)
        return result
            
    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds)")
    except Exception as e:
        error_msg = str(e)
        
        if "terminated abruptly" in error_msg.lower():
            force_restart(f"ProcessPool worker terminated abruptly: {error_msg}")
            
        raise HTTPException(status_code=400, detail=f"Image processing error: {error_msg}")

async def process_char_image(image_bytes: bytes, type: str):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(executor, process_char, image, type)
        result = await asyncio.wait_for(future, timeout=PROCESS_TIMEOUT)
        return result
        
    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Processing timeout exceeded ({PROCESS_TIMEOUT} seconds)")
    except Exception as e:
        error_msg = str(e)
        
        if "terminated abruptly" in error_msg.lower():
            force_restart(f"ProcessPool worker terminated abruptly: {error_msg}")
            
        raise HTTPException(status_code=400, detail=f"Image processing error: {error_msg}")

@app.post("/api/ocr", response_model=OCRResponse)
async def process_image_request(request: Request, image_data: ImageRequest):
    global consecutive_500s

    request_start = time.perf_counter()
    print(f"{image_data.type}: Processing request", flush=True)
        
    try:
        image_str = image_data.image
        if ',' in image_str:
            image_str = image_str.split(',')[1]
        image_bytes = base64.b64decode(image_str)
        
        if image_data.type.startswith("char-"):
            type_name = image_data.type.replace("char-", "")
            result = await process_char_image(image_bytes, type_name)
        elif image_data.type.startswith("import-"):
            type_name = image_data.type.replace("import-", "")
            result = await process_card_image(image_bytes, type_name)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Unsupported type: {image_data.type}"
                }
            )
            
        print(f"{image_data.type}: Completed in {time.perf_counter() - request_start:.2f}s", flush=True)
        
        consecutive_500s = 0
        return result
        
    except Exception as e:
        print(f"{image_data.type}: Failed - {str(e)}", flush=True)
        
        consecutive_500s += 1
        if consecutive_500s > 1:
            print(f"Consecutive errors: {consecutive_500s}/{MAX_CONSECUTIVE_500S}", flush=True)
        
        if consecutive_500s >= MAX_CONSECUTIVE_500S:
            force_restart(f"Too many consecutive 500 errors ({consecutive_500s})")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/", response_model=APIStatus)
async def homepage():
    return APIStatus()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print(f"Uvicorn starting on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)