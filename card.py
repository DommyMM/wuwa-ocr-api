import cv2
import numpy as np
from pathlib import Path
import json
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from echo import preprocess_echo_image

CARD_REGIONS = {
    "name": {"left": 0.035, "top": 0.015, "width": 0.27, "height": 0.067},
    "uid": {"left": 0.005, "top": 0.078, "width": 0.089, "height": 0.056},
    "weapon": {"left": 0.833, "top": 0.410, "width": 0.095, "height": 0.100},
    "na": {"left": 0.548, "top": 0.166, "width": 0.056, "height": 0.035},
    "skill": {"left": 0.433, "top": 0.310, "width": 0.055, "height": 0.038},
    "circuit": {"left": 0.613, "top": 0.542, "width": 0.052, "height": 0.033},
    "liberation": {"left": 0.657, "top": 0.310, "width": 0.052, "height": 0.040},
    "intro": {"left": 0.474, "top": 0.543, "width": 0.055, "height": 0.033},
    "echo1": {"left": 0.015, "top": 0.602, "width": 0.187, "height": 0.380}
}

ECHO_POSITIONS = {
    "echo1": {"base_x": 28},
    "echo2": {"base_x": 400},
    "echo3": {"base_x": 775},
    "echo4": {"base_x": 1148},
    "echo5": {"base_x": 1521}
}

ECHO_SUBREGIONS = {
    "element": {"left": 0.1375, "top": 0.613, "right": 0.167, "bottom": 0.664},
    "main": {"left": 0.1125, "top": 0.665, "right": 0.203, "bottom": 0.733},
    "sub1": {"left": 0.0297, "top": 0.808, "right": 0.201, "bottom": 0.856},
    "sub2": {"left": 0.0307, "top": 0.846, "right": 0.202, "bottom": 0.885},
    "sub3": {"left": 0.0323, "top": 0.878, "right": 0.201, "bottom": 0.911},
    "sub4": {"left": 0.0313, "top": 0.907, "right": 0.202, "bottom": 0.943},
    "sub5": {"left": 0.0328, "top": 0.94, "right": 0.203, "bottom": 0.98}
}

TESSERACT_CONFIG = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.0123456789: '

def ensure_debug_dir():
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    return debug_dir

def crop_regions(image: np.ndarray) -> dict:
    """Crop image into processing regions including echo panels"""
    h, w = image.shape[:2]
    regions = {}
    
    # Crop main regions
    for name, coords in CARD_REGIONS.items():
        x = int(w * coords["left"])
        y = int(h * coords["top"])
        width = int(w * coords["width"])
        height = int(h * coords["height"])
        regions[name] = image[y:y+height, x:x+width]
    
    # Crop echo panels and their subregions
    for echo_num in range(1, 6):
        echo_name = f"echo{echo_num}"
        base_x = ECHO_POSITIONS[echo_name]["base_x"]
        
        # Crop subregions for this echo
        for subname, coords in ECHO_SUBREGIONS.items():
            left = int(w * coords["left"]) + base_x - 28  # Subtract echo1 base_x
            top = int(h * coords["top"])
            right = int(w * coords["right"]) + base_x - 28
            bottom = int(h * coords["bottom"])
            
            region_name = f"{echo_name}_{subname}"
            regions[region_name] = image[top:bottom, left:right]
    
    return regions

def process_ocr(image: np.ndarray) -> str:
    return pytesseract.image_to_string(image, lang='eng', config=TESSERACT_CONFIG).strip()

def process_name(image: np.ndarray) -> str:
    """Process name region (The Shorekeeper Lv.90)"""
    return process_ocr(image)

def process_uid(image: np.ndarray) -> str:
    """Process UID region (Player ID:Dommy, UID:500006092)"""
    return process_ocr(image)

def process_weapon(image: np.ndarray) -> str:
    """Process weapon region (Stellar Symphony LV.90)"""
    processed = preprocess_echo_image(image)
    return process_ocr(processed)

def process_forte_node(image: np.ndarray) -> str:
    """Process forte node text (LV.10/10)"""
    return process_ocr(image)

def process_card(image):
    if image is None:
        return {"success": False, "error": "Failed to process image"}

    regions = crop_regions(image)
    debug_dir = ensure_debug_dir()
    
    for name, region in regions.items():
        cv2.imwrite(str(debug_dir / f"{name}.png"), region)

    with ThreadPoolExecutor() as executor:
        futures = {
            'name': executor.submit(process_name, regions['name']),
            'uid': executor.submit(process_uid, regions['uid']),
            'weapon': executor.submit(process_weapon, regions['weapon']),
            'na': executor.submit(process_forte_node, regions['na']),
            'skill': executor.submit(process_forte_node, regions['skill']),
            'circuit': executor.submit(process_forte_node, regions['circuit']),
            'liberation': executor.submit(process_forte_node, regions['liberation']),
            'intro': executor.submit(process_forte_node, regions['intro'])
        }
        
        results = {k: v.result() for k, v in futures.items()}

    return {
        "success": True,
        "analysis": {
            "type": "Character",
            "raw": {
                "name": results['name'],
                "uid": results['uid'],
                "weapon": results['weapon'],
                "forte": {
                    "na": results['na'],
                    "skill": results['skill'],
                    "circuit": results['circuit'],
                    "liberation": results['liberation'],
                    "intro": results['intro']
                }
            }
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
    print("\n=== Card Processing Results ===")
    print(json.dumps(result, indent=2))
    print("=== Debug images saved to /debug directory ===")
    print("=============================")