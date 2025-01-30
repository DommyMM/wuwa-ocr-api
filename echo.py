import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple
import json
from rapidfuzz import process
from rapidfuzz.utils import default_process
from data import ECHO_NAMES, MAIN_STAT_NAMES, SUB_STATS, SUB_STAT_NAMES, ECHO_ELEMENTS, ECHO_REGIONS, ELEMENT_FEATURES
from cv2 import SIFT_create, FlannBasedMatcher

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
    possible_elements = ECHO_ELEMENTS.get(name, ["Unknown"])
    print(f"\nPossible elements for {name}: {possible_elements}")
    
    element_region = get_element_circle(image)
    
    sift = SIFT_create()
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    kp1, des1 = sift.detectAndCompute(element_region, None)
    if des1 is None:
        return "Unknown"
    
    matches = []
    for element in possible_elements:
        if element in ELEMENT_FEATURES:
            kp2, des2 = ELEMENT_FEATURES[element]
            matches_list = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
            confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
            matches.append((element, confidence))
            print(f"SIFT match for {element}: {confidence:.3f}")
    
    best_match = max(matches, key=lambda x: x[1]) if matches else (possible_elements[0], 0)
    print(f"Selected element: {best_match[0]} with confidence {best_match[1]:.3f}\n")
    
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
    if not text_lines:
        return []
        
    sub_lines = text_lines[-5:] if len(text_lines) >= 5 else text_lines[3:]
    sub_stats = []
    
    for i, line in enumerate(sub_lines, 1):
        if not line or not line.strip() or ' ' not in line:
            continue
            
        try:
            last_space = line.rindex(' ')
            raw_name = line[:last_space].strip()
            raw_value = line[last_space:].strip()
            
            if not raw_name or not raw_value:
                continue
                
            print(f"\nSub {i}: '{line}'")
            
            if SUB_STAT_NAMES:
                match = process.extractOne(raw_name, list(SUB_STAT_NAMES), processor=default_process)
                print(f"Best match: '{match[0]}' with score {match[1]}")
                
                if match and match[1] > 70:
                    name = match[0]
                    had_percent = "%" in raw_value
                    
                    if match[0].upper().replace("%", "") in ["ATK", "HP", "DEF"]:
                        if had_percent:
                            name = f"{match[0].upper().replace('%','')}%"
                        else:
                            name = match[0].upper().replace("%", "")
                    
                    try:
                        clean_value = raw_value.replace('%', '')
                        valid_values = [str(v) for v in SUB_STATS[name]]
                        match = process.extractOne(clean_value, valid_values)
                        print(f"Value matched: {clean_value} -> {match[0]} (score: {match[1]})")
                        
                        normalized_value = f"{match[0]}%" if had_percent else match[0]
                        clean_name = name.replace("Resonance ", "").replace(" DMG Bonus", "")
                        sub_stats.append({"name": clean_name, "value": normalized_value})
                        continue
                    except (ValueError, KeyError) as e:
                        print(f"Value normalization error: {e}")
                        sub_stats.append({"name": "Unknown", "value": raw_value})
                        continue
            
            sub_stats.append({"name": "Unknown", "value": raw_value})
            
        except (ValueError, AttributeError):
            continue
    
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