import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple
from pathlib import Path
import json
from rapidfuzz import process
from rapidfuzz.utils import default_process

BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / 'Public' / 'Data'

ORIGINAL_IMAGE = None

try:
    with open(DATA_DIR / 'Echoes.json', 'r', encoding='utf-8') as f:
        echoes_data = json.load(f)
        ECHO_NAMES = [echo['name'] for echo in echoes_data]
        ECHO_ELEMENTS = {echo['name']: echo['elements'] for echo in echoes_data}
    with open(DATA_DIR / 'Mainstat.json', 'r', encoding='utf-8') as f:
        main_data = json.load(f)
        MAIN_STAT_NAMES = set()
        for cost_data in main_data.values():
            if "mainStats" in cost_data:
                for stat_name in cost_data["mainStats"].keys():
                    if stat_name in ["HP%", "ATK%", "DEF%"]:
                        MAIN_STAT_NAMES.add(stat_name.replace("%", ""))
                    else:
                        MAIN_STAT_NAMES.add(stat_name)
    with open(DATA_DIR / 'Substats.json', 'r', encoding='utf-8') as f:
        sub_data = json.load(f)
        SUB_STATS = sub_data["subStats"]
        SUB_STAT_NAMES = set(SUB_STATS.keys())
except FileNotFoundError:
    print("Warning: Reference data files not found")
    MAIN_STAT_NAMES = set()
    SUB_STATS = {}
    ECHO_NAMES = []
    ECHO_ELEMENTS = {}
except json.JSONDecodeError as e:
    print(f"Warning: Invalid JSON format in data files: {e}")
    MAIN_STAT_NAMES = set()
    SUB_STATS = {}
    ECHO_NAMES = []
    ECHO_ELEMENTS = {}

ECHO_REGIONS = {
    "name": {"top": 0.052, "left": 0.055, "width": 0.8, "height": 0.11},
    "level": {"top": 0.23, "left": 0.08, "width": 0.1, "height": 0.08},
    "main": {"top": 0.31, "left": 0.145, "width": 0.78, "height": 0.085},
    "sub1": {"top": 0.53, "left": 0.115, "width": 0.81, "height": 0.08},
    "sub2": {"top": 0.6, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub3": {"top": 0.685, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub4": {"top": 0.773, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub5": {"top": 0.86, "left": 0.115, "width": 0.81, "height": 0.09}
}

def preprocess_echo_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(bilateral)
    _, thresh1 = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY_INV)
    _, thresh = cv2.threshold(thresh1, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return clean

def read_region(image: np.ndarray, region_key: str) -> str:
    region = ECHO_REGIONS[region_key]
    h, w = image.shape[:2]
    x = int(w * region["left"])
    y = int(h * region["top"])
    width = int(w * region["width"])
    height = int(h * region["height"])
    cropped = image[y:y+height, x:x+width]
    processed = preprocess_echo_image(cropped)
    
    text = pytesseract.image_to_string(processed, lang='eng', config='--psm 7').strip()
    return text

def merge_nearby_text(entries: List[Tuple[int, int, str, int, int]]) -> List[str]:
    entries.sort()
    merged_lines = []
    current_line = []
    last_y = None
    
    for y, x, text, w, h in entries:
        if last_y is None or abs(y - last_y) > 10:
            if current_line:
                merged_lines.append(sorted(current_line, key=lambda x: x[1]))
            current_line = [(y, x, text, w, h)]
            last_y = y
        else:
            current_line.append((y, x, text, w, h))
    
    if current_line:
        merged_lines.append(sorted(current_line, key=lambda x: x[1]))
        
    return [" ".join(item[2] for item in line) for line in merged_lines]

def get_name(text_lines: List[str]) -> str:
    raw_name = text_lines[0] if text_lines else "Unknown"
    matched_name = raw_name
    if ECHO_NAMES:
        match = process.extractOne(raw_name, ECHO_NAMES, processor=default_process)
        if match and match[1] > 70:
            matched_name = match[0]
            print(f"Matched '{raw_name}' to '{matched_name}' with score {match[1]}")
    return matched_name

def get_element_circle(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    center_x = int(0.89270833 * w)
    center_y = int(0.12830687 * h)
    radius = int(0.040625 * w)
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    if image.shape[2] == 3:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        result = image.copy()
    
    alpha_mask = mask.copy()
    result[:, :, 3] = alpha_mask
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(w, center_x + radius)
    y2 = min(h, center_y + radius)
    cropped = result[y1:y2, x1:x2]
    
    return cropped

def get_element(name: str, image: np.ndarray) -> str:
    ELEMENT_COLORS = {
    'Healing': {'lower': np.array([30, 60, 120]), 'upper': np.array([50, 210, 240])},
    'Electro': {'lower': np.array([100, 70, 140]), 'upper': np.array([179, 170, 255])},
    'Fusion': {'lower': np.array([0, 150, 150]), 'upper': np.array([20, 180, 255])},
    'Havoc': {'lower': np.array([140, 50, 70]), 'upper': np.array([179, 90, 255])},
    'Spectro': {'lower': np.array([20, 100, 200]), 'upper': np.array([40, 160, 255])},
    'Glacio': {'lower': np.array([90, 150, 210]), 'upper': np.array([110, 210, 255])},
    'Aero': {'lower': np.array([60, 150, 210]), 'upper': np.array([80, 180, 255])},
    'Attack': {'lower': np.array([0, 190, 120]), 'upper': np.array([5, 220, 220])},
    'ER': {'lower': np.array([0, 0, 190]), 'upper': np.array([140, 30, 255])}
    }
    
    possible_elements = ECHO_ELEMENTS.get(name, ["Unknown"])
    print(f"\nPossible elements for {name}: {possible_elements}")
    
    element_region = get_element_circle(image)
    
    hsv = cv2.cvtColor(element_region, cv2.COLOR_BGR2HSV)
    
    matches = []
    for element in possible_elements:
        if element in ELEMENT_COLORS:
            color_range = ELEMENT_COLORS[element]
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            match_ratio = np.count_nonzero(mask) / mask.size
            matches.append((element, match_ratio))
            print(f"Match ratio for {element}: {match_ratio:.3f}")
    
    best_match = max(matches, key=lambda x: x[1]) if matches else (possible_elements[0], 0)
    print(f"Selected element: {best_match[0]} with ratio {best_match[1]:.3f}")
    print()
    
    return best_match[0]

def get_level(text_lines: List[str]) -> str:
    if len(text_lines) < 2:
        return "0"
        
    level_line = text_lines[1]
    words = level_line.split()
    
    for word in words:
        try:
            level = int(word.strip('+ '))
            return str(level)
        except ValueError:
            continue
            
    return "0"

def get_main(text_lines: List[str]) -> Dict:
    if len(text_lines) < 3:
        return {"name": "Unknown", "value": "0"}
    
    main_line = text_lines[2]
    
    parts = main_line.split()
    if len(parts) < 2:
        return {"name": "Unknown", "value": "0"}
    
    value = parts[-1]
    raw_name = " ".join(parts[:-1])
    
    if MAIN_STAT_NAMES:
        match = process.extractOne(raw_name, list(MAIN_STAT_NAMES), processor=default_process)
        print(f"Matched '{raw_name}' to '{match[0]}' with score {match[1]}")
        if match and match[1] > 70:
            result = {"name": match[0], "value": value}
            return result
    return {"name": "Unknown", "value": value}

def get_subs(text_lines: List[str]) -> List[Dict]:
    sub_lines = text_lines[-5:] if len(text_lines) >= 5 else text_lines[3:]
    sub_stats = []
    
    for i, line in enumerate(sub_lines, 1):
        last_space = line.rindex(' ')
        raw_name = line[:last_space].strip()
        raw_value = line[last_space:].strip()
        
        print(f"\nSub {i}: '{line}'")
        
        if SUB_STAT_NAMES:
            match = process.extractOne(raw_name, list(SUB_STAT_NAMES), processor=default_process)
            print(f"Best match: '{match[0]}' with score {match[1]}")
            
            if match and match[1] > 80:
                name = match[0]
                had_percent = "%" in raw_value
                
                if match[0].upper() in ["ATK", "HP", "DEF"]:
                    if had_percent:
                        name = f"{match[0].upper()}%"
                    else:
                        name = match[0].upper().replace("%", "")
                
                try:
                    value = float(raw_value.replace('%', ''))
                    valid_values = [float(v) for v in SUB_STATS[name]]
                    closest = min(valid_values, key=lambda x: abs(x - value))
                    
                    normalized_value = f"{closest}%" if had_percent else str(closest)
                    print(f"Value normalized: {raw_value} -> {normalized_value}")
                    sub_stats.append({"name": name.replace("Resonance ", ""), "value": normalized_value})
                    continue
                except (ValueError, KeyError):
                    print(f"Could not normalize value: {raw_value}")
                    sub_stats.append({"name": name.replace("Resonance ", ""), "value": raw_value})
                    continue
                
        sub_stats.append({"name": "Unknown", "value": raw_value})
    
    return sub_stats

def clean_lines(text_lines: List[str]) -> List[str]:
    cleaned = [
        line.replace('é', 'e') 
        for line in text_lines 
        if (len(line) > 1 or line in ["+", "·"]) and "COST" not in line
    ]
    return cleaned

def process_echo(image: np.ndarray):
    name = read_region(image, "name")
    level = read_region(image, "level")
    main_stat = read_region(image, "main")
    sub_stats = [read_region(image, f"sub{i}") for i in range(1, 6)]
    
    text_lines = [name, level, main_stat] + sub_stats
    text_lines = clean_lines(text_lines)
    print("OCR Output:")
    for line in text_lines:
        print(f"{line}")
    print()
    
    name = get_name(text_lines)
    element = get_element(name, image)
    
    response = {
        "success": True,
        "analysis": {
            "type": "Echo",
            "name": name,
            "element": element,
            "echoLevel": get_level(text_lines),
            "main": get_main(text_lines),
            "subs": get_subs(text_lines)
        }
    }
    
    print("\n=== Final Response ===")
    print(json.dumps(response, indent=2))
    print("===================")
    
    return response