import cv2
from pathlib import Path
import json
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import time
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CARD_REGIONS = {
    "name": {"left": 0.035, "top": 0.015, "width": 0.27, "height": 0.067},
    "uid": {"left": 0.005, "top": 0.078, "width": 0.089, "height": 0.056},
    "weapon": {"left": 0.833, "top": 0.410, "width": 0.095, "height": 0.100},
    "na": {"left": 0.548, "top": 0.166, "width": 0.056, "height": 0.035},
    "skill": {"left": 0.433, "top": 0.310, "width": 0.055, "height": 0.038},
    "circuit": {"left": 0.613, "top": 0.542, "width": 0.052, "height": 0.033},
    "liberation": {"left": 0.657, "top": 0.310, "width": 0.052, "height": 0.040},
    "intro": {"left": 0.474, "top": 0.543, "width": 0.055, "height": 0.033}
}

def ensure_debug_dir():
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    return debug_dir

def ensure_results_dir():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir

def save_results(results, filename="results.json"):
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return filepath

def process_card(image):
    """Process full character card image"""
    if image is None:
        return {"success": False, "error": "Failed to process image"}
    endpoint = os.getenv("AZURE_ENDPOINT")
    key = os.getenv("AZURE_KEY")
        
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    is_success, buffer = cv2.imencode(".jpg", image)
    image_bytes = io.BytesIO(buffer).read()
    
    image_data = io.BytesIO(image_bytes)
    
    read_response = client.read_in_stream(image_data, raw=True)
    operation_id = read_response.headers["Operation-Location"].split("/")[-1]
    
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
        
    return {
        "success": True,
        "analysis": {
            "type": "Character",
            "azure_raw": result.as_dict()
        }
    }

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    test_image_path = root_dir / "wuwa.png"
    
    if not test_image_path.exists():
        print(f"Test image not found at {test_image_path}")
        exit(1)
        
    image = cv2.imread(str(test_image_path))
    if image is None:
        print("Failed to load test image")
        exit(1)
        
    result = process_card(image)
    
    # Save results
    results_file = save_results(result)
    print("\n=== Card Processing Results ===")
    print(f"Results saved to: {results_file}")
    print("=============================")