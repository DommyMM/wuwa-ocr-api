import cv2
from pathlib import Path
import pytesseract
from rapidocr_onnxruntime import RapidOCR
import re
from echo import ELEMENT_COLORS, MAIN_STAT_NAMES, SUB_STATS
import numpy as np
from rapidfuzz import process
import json

DATA_DIR = Path(__file__).parent / 'Data'
try:
    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        characters_data = json.load(f)
        CHARACTER_NAMES = [char['name'] for char in characters_data]
except (FileNotFoundError, json.JSONDecodeError):
    print("Warning: Characters.json not found or invalid")
    CHARACTER_NAMES = []

ECHO_GRID = {
    "echo1": {"x1": 0.000, "x2": 0.195, "y1": 0, "y2": 1.0},
    "echo2": {"x1": 0.201, "x2": 0.396, "y1": 0, "y2": 1.0},
    "echo3": {"x1": 0.402, "x2": 0.597, "y1": 0, "y2": 1.0},
    "echo4": {"x1": 0.603, "x2": 0.798, "y1": 0, "y2": 1.0},
    "echo5": {"x1": 0.804, "x2": 0.999, "y1": 0, "y2": 1.0}
}

def process_ocr(name: str, image: np.ndarray) -> str:
    """Process image with appropriate OCR engine"""
    if name in ["character"] or name.startswith("echo"):
        ocr = RapidOCR(lang='en')
        result, _ = ocr(image)
        if result:
            return "\n".join(text for _, text, _ in result)
        return ""
    return pytesseract.image_to_string(image)

def preprocess_region(image):
    """Lighter preprocessing to preserve text clarity"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(bilateral, (0,0), 3)
    sharp = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
    return thresh

def clean_value(value: str) -> str:
    return value.strip().replace(" ", "")

def clean_stat_name(name: str, value: str) -> str:
    name = name.strip().replace(" DMG Bonus", "")
    if name.upper() in ["ATK", "HP", "DEF"] and "%" in value:
        return f"{name.upper()}%"
    return name.upper() if name.upper() in ["ATK", "HP", "DEF"] else name

def validate_stat(name: str, valid_names: set) -> str:
    if not valid_names:
        return name
    match = process.extractOne(name, list(valid_names))
    return match[0] if match and match[1] > 70 else name

def validate_value(value: str, stat_name: str) -> str:
    if not SUB_STATS or stat_name not in SUB_STATS:
        return value
        
    had_percent = "%" in value
    clean_value = value.replace('%', '')
    
    try:
        valid_values = [str(v) for v in SUB_STATS[stat_name]]
        match = process.extractOne(clean_value, valid_values)
        if match and match[1] > 70:
            return f"{match[0]}%" if had_percent else match[0]
    except (ValueError, KeyError):
        pass
    return value

def validate_character_name(raw_name: str) -> str:
    if not CHARACTER_NAMES:
        return raw_name
    match = process.extractOne(raw_name, CHARACTER_NAMES)
    return match[0] if match and match[1] > 70 else raw_name

def parse_region_text(name, text):
    if name == "character":
        parts = [p for p in text.split() if p.strip()]
        level = 0
        for part in parts:
            if "LV." in part:
                match = re.search(r'LV\.(\d+)', part)
                if match:
                    level = int(match.group(1))
                    parts.remove(part)
                    break
        raw_name = " ".join(parts)
        char_name = validate_character_name(raw_name)
        return {"name": char_name, "level": level}
    
    elif name == "watermark":
        lines = text.split('\n')
        uid = lines[1].split("UID:")[-1].strip() if len(lines) > 1 else "0"
        return {
            "username": lines[0].split("ID:")[-1].strip() if lines else "",
            "uid": int(uid) if uid.isdigit() else 0
        }
    
    elif name == "forte":
        levels = []
        clean_text = text.replace('+', ' ').strip()
        for line in clean_text.split('\n'):
            matches = re.finditer(r'LV\.(\d+)(?:/10)?', line)
            for match in matches:
                levels.append(int(match.group(1)))
        while len(levels) < 5:
            levels.append(0)
        return {"levels": levels[:5]}
    
    elif name == "weapon":
        lines = text.split('\n')
        name = lines[0].strip() if lines else "Unknown"
        level = 0
        for line in lines[1:]:
            if "LV." in line:
                match = re.search(r'LV\.(\d+)', line)
                if match:
                    level = int(match.group(1))
                    break
        return {
            "name": name,
            "level": level
        }
    
    elif name.startswith("echo"):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        pairs = []
        current_key = None
        previous_line = None
        
        for line in lines:
            if line in ['Bonus', 'DMG Bonus']:
                continue
                
            if any(c.isdigit() for c in line) or "%" in line:
                if current_key:
                    if len(current_key) == 1 and previous_line:
                        current_key = previous_line
                    pairs.append((current_key, line))
                    current_key = None
            else:
                previous_line = current_key if current_key else previous_line
                current_key = line
        
        if len(pairs) < 3:
            return []
        
        # Process main stat
        main_name, main_value = pairs[0]
        main_value = clean_value(main_value)
        main_name = clean_stat_name(main_name, main_value)
        main_name = validate_stat(main_name, MAIN_STAT_NAMES)
        
        # Process substats (skip base stat at pairs[1])
        substats = []
        for stat_name, stat_value in pairs[2:]:
            value = clean_value(stat_value)
            name = clean_stat_name(stat_name, value)
            name = validate_stat(name, SUB_STATS.keys())
            name = name.replace(" DMG Bonus", "")
            value = validate_value(value, name)
            substats.append({"name": name, "value": value})
        
        return {
            "main": {"name": main_name, "value": main_value},
            "substats": substats
        }
    
    return text

def get_element_region(image):
    """Extract element region from individual echo image"""
    h, w = image.shape[:2]
    x1 = int(w * 0.664)
    x2 = int(w * 0.812)
    y1 = int(h * 0.024)
    y2 = int(h * 0.160)
    
    return image[y1:y2, x1:x2]

def determine_element(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    matches = []
    for element, ranges in ELEMENT_COLORS.items():
        mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
        ratio = np.count_nonzero(mask) / mask.size
        matches.append((element, ratio))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:2] if len(matches) >= 2 else matches + [("Unknown", 0)]
    
    return {
        "primary": top_matches[0][0],
        "secondary": top_matches[1][0],
        "primary_ratio": float(top_matches[0][1]),
        "secondary_ratio": float(top_matches[1][1])
    }

def split_echo_image(image):
    """Split full echo section into individual echo regions"""
    h, w = image.shape[:2]
    echo_regions = []
    
    # Create debug directory if it doesn't exist
    debug_dir = Path(__file__).parent / 'debug'
    debug_dir.mkdir(exist_ok=True)
    
    # Split and save individual regions
    for i in range(1, 6):
        region = ECHO_GRID[f"echo{i}"]
        x1 = int(w * region["x1"])
        x2 = int(w * region["x2"])
        echo_img = image[:, x1:x2]
        echo_regions.append(echo_img)
        
        # Save debug images
        cv2.imwrite(str(debug_dir / f'echo{i}.png'), echo_img)
    
    return echo_regions

def process_card(image, region: str):
    if image is None:
        return {"success": False, "error": "Failed to process image"}
    
    try:
        if region == "echoes":
            echo_regions = split_echo_image(image)
            all_echoes = []
            
            for idx, echo_img in enumerate(echo_regions):
                processed = preprocess_region(echo_img)
                text = process_ocr(f"echo{idx+1}", processed)
                cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                
                echo_data = parse_region_text(f"echo{idx+1}", cleaned_text)
                element_region = get_element_region(echo_img)
                element_data = determine_element(element_region)
                
                all_echoes.append({
                    "main": echo_data.get("main", {}),
                    "substats": echo_data.get("substats", []),
                    "element": element_data
                })
            
            return {
                "success": True,
                "analysis": {"echoes": all_echoes}
            }
        else:
            processed = preprocess_region(image)
            text = process_ocr(region, processed)
            cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            result = parse_region_text(region, cleaned_text)
            
            return {
                "success": True,
                "analysis": result
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }