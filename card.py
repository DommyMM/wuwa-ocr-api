import cv2
from pathlib import Path
import pytesseract
from process import REGIONS
from rapidocr_onnxruntime import RapidOCR
import re
from echo import ELEMENT_COLORS, MAIN_STAT_NAMES, SUB_STATS
import numpy as np
from rapidfuzz import process
import json
from pathlib import Path

# Add to top-level constants
DATA_DIR = Path(__file__).parent / 'Data'
try:
    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        characters_data = json.load(f)
        CHARACTER_NAMES = [char['name'] for char in characters_data]
except (FileNotFoundError, json.JSONDecodeError):
    print("Warning: Characters.json not found or invalid")
    CHARACTER_NAMES = []

OCR_INSTANCES = {}

ELEMENT_REGIONS = {
    "element1": {"x1": 0.138, "y1": 0.611, "x2": 0.166, "y2": 0.663},
    "element2": {"x1": 0.331, "y1": 0.610, "x2": 0.361, "y2": 0.665},
    "element3": {"x1": 0.525, "y1": 0.610, "x2": 0.556, "y2": 0.665},
    "element4": {"x1": 0.722, "y1": 0.611, "x2": 0.751, "y2": 0.665},
    "element5": {"x1": 0.917, "y1": 0.611, "x2": 0.947, "y2": 0.665}
}

def init_ocr_instances():
    """Initialize OCR pool"""
    global OCR_INSTANCES
    for region in ["character"] + [f"echo{i}" for i in range(1,6)]:
        OCR_INSTANCES[region] = RapidOCR(lang='en')

def preprocess_region(image):
    """Lighter preprocessing to preserve text clarity"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(bilateral, (0,0), 3)
    sharp = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
    
    return thresh

def ocr_with_rapid(image, region_name):
    """Use dedicated RapidOCR instance"""
    rapid = OCR_INSTANCES[region_name]
    result, _ = rapid(image)
    if result:
        text_only = []
        for box, text, score in result:
            text_only.append(text)
        return "\n".join(text_only)
    return ""

def ocr_region(name, image):
    """Choose OCR engine based on region type"""
    if name in ["character"] or name.startswith("echo"):
        return ocr_with_rapid(image, name)
    else:
        return pytesseract.image_to_string(image)

def clean_value(value: str) -> str:
    """Standardize value format"""
    return value.strip().replace(" ", "")

def clean_stat_name(name: str, value: str) -> str:
    """Format stat name, adding % if needed"""
    name = name.strip().replace(" DMG Bonus", "")
    if name.upper() in ["ATK", "HP", "DEF"] and "%" in value:
        return f"{name.upper()}%"
    return name.upper() if name.upper() in ["ATK", "HP", "DEF"] else name

def validate_stat(name: str, valid_names: set) -> str:
    """Fuzzy match stat name"""
    if not valid_names:
        return name
    match = process.extractOne(name, list(valid_names))
    return match[0] if match and match[1] > 70 else name

def validate_value(value: str, stat_name: str) -> str:
    """Match value against known valid values"""
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
    """Fuzzy match character name against known list"""
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
        return levels[:5]
    
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
        
        for line in lines:
            if line in ['Bonus', 'DMG Bonus']:
                continue
                
            if any(c.isdigit() for c in line) or "%" in line:
                if current_key:
                    pairs.append((current_key, line))
                    current_key = None
            else:
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
            name = name.replace("Resonance ", "").replace(" DMG Bonus", "")
            value = validate_value(value, name)
            substats.append({"name": name, "value": value})
        
        return {
            "main": {"name": main_name, "value": main_value},
            "substats": substats
        }
    
    return text

def get_element_region(image, region_key):
    """Extract element region from image"""
    region = ELEMENT_REGIONS[region_key]
    h, w = image.shape[:2]
    
    x1 = int(w * region["x1"])
    y1 = int(h * region["y1"])
    x2 = int(w * region["x2"])
    y2 = int(h * region["y2"])
    
    return image[y1:y2, x1:x2]

def determine_element(image):
    """Get top 2 matching elements from color matching"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    matches = []
    for element, ranges in ELEMENT_COLORS.items():
        mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
        ratio = np.count_nonzero(mask) / mask.size
        matches.append((element, ratio))
    
    # Sort by ratio and get top 2
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:2] if len(matches) >= 2 else matches + [("Unknown", 0)]
    
    return {
        "primary": top_matches[0][0],
        "secondary": top_matches[1][0],
        "primary_ratio": float(top_matches[0][1]),
        "secondary_ratio": float(top_matches[1][1])
    }

def process_card(image):
    if image is None:
        return {"success": False, "error": "Failed to process image"}
    
    # Save debug image
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    # Process each region
    results = {}
    for name, region in REGIONS.items():
        # Extract region
        roi = image[
            int(region["y1"]):int(region["y2"]), 
            int(region["x1"]):int(region["x2"])
        ]
        
        # Preprocess region
        processed = preprocess_region(roi)
        
        # Save region debug image
        cv2.imwrite(str(debug_dir / f"{name}_region.png"), processed)
        
        # OCR region with appropriate engine
        text = ocr_region(name, processed)
        cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        # Parse region text based on type
        results[name] = parse_region_text(name, cleaned_text)
    
    # Save element regions for debugging
    for i in range(1, 6):
        element_key = f"element{i}"
        element_roi = get_element_region(image, element_key)
        cv2.imwrite(str(debug_dir / f"{element_key}_region.png"), element_roi)
    
    # Create echo array with elements
    echo_results = []
    for i in range(1, 6):
        echo_key = f"echo{i}"
        element_roi = get_element_region(image, f"element{i}")
        element_data = determine_element(element_roi)
        
        echo_data = results.get(echo_key, {})
        echo_results.append({
            "main": echo_data.get("main", {}),
            "substats": echo_data.get("substats", []),
            "element": element_data
        })
    return {
        "character": results["character"],
        "watermark": results["watermark"],
        "weapon": results["weapon"],
        "fortes": results["forte"],
        "echoes": echo_results
    }

def format_results(results):
    """Format results with proper indentation"""
    print("\n=== OCR Results ===")
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        if key == "echoes":
            print("    [")
            for echo in value:
                print("        {")
                print(f"            'main': {{")
                print(f"                'name': '{echo['main'].get('name', '')}',")
                print(f"                'value': '{echo['main'].get('value', '')}'")
                print("            },")
                print("            'substats': [")
                for substat in echo['substats']:
                    print("                {")
                    print(f"                    'name': '{substat['name']}',")
                    print(f"                    'value': '{substat['value']}'")
                    print("                },")
                print("            ],")
                print(f"            'element': {echo['element']}")
                print("        },")
            print("    ]")
        else:
            print(f"    {value}")

if __name__ == "__main__":
    init_ocr_instances()
    image_path = Path(__file__).parent / "wuwa1.png"
    
    if not image_path.exists():
        print(f"Test image not found at {image_path}")
        exit(1)
    
    results = process_card(cv2.imread(str(image_path)))
    print("\nRaw Results:")
    print(results)
    print("\nFormatted Results:")
    format_results(results)