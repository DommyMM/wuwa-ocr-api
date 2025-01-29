import cv2
import pytesseract
import re
from data import CHARACTER_NAMES, WEAPON_NAMES, MAIN_STAT_NAMES, SUB_STATS, ELEMENT_COLORS, ECHO_ELEMENTS, ECHO_COSTS, TEMPLATE_FEATURES, Rapid
import numpy as np
from rapidfuzz import process
from typing import Tuple
from cv2 import SIFT_create, FlannBasedMatcher


WEAPON_REGIONS = {
    "name": {"x1": 152, "y1": 25, "x2": 437, "y2": 79},
    "level": {"x1": 191, "y1": 79, "x2": 269, "y2": 133}
}

SEQUENCE_REGIONS = {
    "S1": {"center": (55, 58), "width": 30, "height": 26},
    "S2": {"center": (130, 58), "width": 30, "height": 26},
    "S3": {"center": (210, 58), "width": 30, "height": 26},
    "S4": {"center": (290, 58), "width": 30, "height": 26},
    "S5": {"center": (369, 58), "width": 30, "height": 26},
    "S6": {"center": (449, 58), "width": 30, "height": 26}
}

ECHO_REGIONS = {
    "main": {"x1": 195, "y1": 66, "x2": 366, "y2": 148},
    "subs_names": {"x1": 40, "y1": 228, "x2": 290, "y2": 395},
    "subs_values": {"x1": 292, "y1": 228, "x2": 359, "y2": 395}
}

def process_ocr(name: str, image: np.ndarray) -> str:
    """Process image with appropriate OCR engine"""
    if name in ["character", "weapon"]:
        result, _ = Rapid(image)
        if result:
            return "\n".join(text for _, text, _ in result)
        return ""
    image = preprocess_region(image)
    return pytesseract.image_to_string(image)

def preprocess_region(image):
    """Lighter preprocessing to preserve text clarity"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(bilateral, (0,0), 3)
    sharp = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
    return thresh

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
            float_value = float(clean_value)
            matched_value = float(match[0])
            if abs(float_value - matched_value) > 2.0:
                closest = min(SUB_STATS[stat_name], key=lambda x: abs(float_value - x))
                if abs(float_value - closest) <= 1.0:
                    return f"{closest}%" if had_percent else str(closest)
            else:
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
    match name:
        case "character":
            parts = [p for p in text.split() if p.strip()]
            level = 1
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
            
        case "watermark":
            lines = text.split('\n')
            uid = lines[1].split("UID:")[-1].strip() if len(lines) > 1 else "0"
            return {
                "username": lines[0].split("ID:")[-1].strip() if lines else "",
                "uid": int(uid) if uid.isdigit() else 0
            }
            
        case "forte":
            levels = []
            clean_text = text.replace('+', ' ').strip()
            for line in clean_text.split('\n'):
                matches = re.finditer(r'LV\.(\d+)(?:/10)?', line)
                for match in matches:
                    levels.append(int(match.group(1)))
            while len(levels) < 5:
                levels.append(0)
            return {"levels": levels[:5]}
            
        case "weapon":
            def validate_weapon_name(raw_name: str) -> str:
                if not WEAPON_NAMES:
                    return raw_name
                match = process.extractOne(raw_name, WEAPON_NAMES)
                return match[0] if match and match[1] > 70 else raw_name
            lines = text.split('\n')
            raw_name = lines[0].strip() if lines else "Unknown"
            weapon_name = validate_weapon_name(raw_name)
            level = 1
            for line in lines[1:]:
                if "LV." in line:
                    match = re.search(r'LV\.(\d+)', line)
                    if match:
                        level = int(match.group(1))
                        break
            return {
                "name": weapon_name,
                "level": level
            }
        case _ if name.startswith("echo"):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines:
                return []
            
            main_parts = lines[0].rsplit(' ', 1)
            if len(main_parts) != 2:
                return []
            main_name, main_value = main_parts
            main_name = clean_stat_name(main_name, main_value)
            main_name = validate_stat(main_name, MAIN_STAT_NAMES)
            main_value = validate_value(main_value, main_name)
            
            substats = []
            for line in lines[1:]:
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    continue
                    
                stat_name, stat_value = parts
                name = clean_stat_name(stat_name, stat_value)
                name = validate_stat(name, SUB_STATS.keys())
                value = validate_value(stat_value, name)
                substats.append({"name": name.replace("DMG Bonus", ""), "value": value})
            
            return {
                "main": {"name": main_name, "value": main_value},
                "substats": substats
            }
            
        case _:
            return text

def get_element_region(image):
    """Extract element region from individual echo image"""
    h, w = image.shape[:2]
    x1 = int(w * 0.664)
    x2 = int(w * 0.812)
    y1 = int(h * 0.024)
    y2 = int(h * 0.160)
    
    return image[y1:y2, x1:x2]

def determine_element(image, echo_name: str):
    """Match element colors only against possible elements for echo"""
    possible_elements = ECHO_ELEMENTS.get(echo_name, ["Unknown"])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    matches = []
    for element in possible_elements:
        if element in ELEMENT_COLORS:
            ranges = ELEMENT_COLORS[element]
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            ratio = np.count_nonzero(mask) / mask.size
            matches.append((element, ratio))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0] if matches else "Unknown"

def get_echo_cost(image: np.ndarray) -> int:
    """Get echo cost from image region"""
    cost_img = image[9:61, 302:345]
    
    result, _ = Rapid(cost_img)
    if result:
        cost = int(result[0][1])
        if cost in [1, 3, 4]:
            return cost
    return 0

def match_icon(image: np.ndarray) -> Tuple[str, float]:
    """SIFT-based icon matching - returns best match with confidence check"""
    icon_img = image[0:182, 0:188]
    sift = SIFT_create()
    kp1, des1 = sift.detectAndCompute(icon_img, None)
    matches = []
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    for name, (kp2, des2) in TEMPLATE_FEATURES.items():
        matches_list = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
        confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        matches.append((name, confidence))
    
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    best_match, best_conf = sorted_matches[0]
    secondary_matches = [m for m in sorted_matches[1:5] if m[1] > 0.1]
    
    if secondary_matches and (best_conf - secondary_matches[0][1]) < 0.25:
        actual_cost = get_echo_cost(image)
        if actual_cost in [1, 3, 4]:
            best_cost = ECHO_COSTS.get(best_match, 0)
            if best_cost != actual_cost:
                for name, conf in secondary_matches:
                    if ECHO_COSTS.get(name, 0) == actual_cost:
                        return (name, conf)
    return sorted_matches[0]

def parse_sequence_region(image) -> int:
    """Count active sequence nodes using HSV gray detection"""
    GRAY_HSV = {
        'lower': np.array([0, 0, 160]),
        'upper': np.array([40, 180, 255])
    }
    GRAY_THRESHOLD = 0.75
    active_count = 0
    
    for seq_num, region in SEQUENCE_REGIONS.items():
        center_x, center_y = region["center"]
        half_w = region["width"] // 2
        half_h = region["height"] // 2
        
        x1 = max(0, center_x - half_w)
        x2 = min(image.shape[1], center_x + half_w)
        y1 = max(0, center_y - half_h)
        y2 = min(image.shape[0], center_y + half_h)
        
        sequence_img = image[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(sequence_img, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv, GRAY_HSV['lower'], GRAY_HSV['upper'])
        gray_ratio = np.count_nonzero(gray_mask) / gray_mask.size
        
        if gray_ratio > GRAY_THRESHOLD:
            active_count += 1
    
    return active_count

def merge_stat_lines(names: list, values: list) -> str:
    """Merge stat names with their values"""
    return "\n".join(f"{name} {value}" for name, value in zip(names, values))

def process_card(image, region: str):
    if image is None:
        return {"success": False, "error": "No image data provided"}
    
    try:
        if region == "sequences":
            sequence = parse_sequence_region(image)
            return {
                "success": True,
                "analysis": {"sequence": sequence}
            }
        elif region.startswith("echo"):
            # Process main region
            main_img = image[ECHO_REGIONS["main"]["y1"]:ECHO_REGIONS["main"]["y2"], ECHO_REGIONS["main"]["x1"]:ECHO_REGIONS["main"]["x2"]]
            main_processed = preprocess_region(main_img)
            main_lines = [l.strip() for l in pytesseract.image_to_string(main_processed).splitlines() if l.strip()]
            main_text = f"{main_lines[0]} {main_lines[1]}" if len(main_lines) >= 2 else ""
            
            # Process subs regions separately
            names_img = image[ECHO_REGIONS["subs_names"]["y1"]:ECHO_REGIONS["subs_names"]["y2"], ECHO_REGIONS["subs_names"]["x1"]:ECHO_REGIONS["subs_names"]["x2"]]
            values_img = image[ECHO_REGIONS["subs_values"]["y1"]:ECHO_REGIONS["subs_values"]["y2"], ECHO_REGIONS["subs_values"]["x1"]:ECHO_REGIONS["subs_values"]["x2"]]
            
            names_processed = preprocess_region(names_img)
            values_processed = preprocess_region(values_img)
            
            # Get raw lines
            names_lines = [l.strip() for l in pytesseract.image_to_string(names_processed).splitlines() if l.strip()]
            values_lines = [l.strip() for l in pytesseract.image_to_string(values_processed).splitlines() if l.strip()]
            
            # Process names - combine DMG lines
            cleaned_names = []
            for line in names_lines:
                if line.startswith(("DMG", "DMG Bonus")) and cleaned_names:
                    cleaned_names[-1] = f"{cleaned_names[-1]} {line}"
                else:
                    cleaned_names.append(line)
            values = values_lines[:5]
            subs_text = "\n".join(f"{name} {value}" for name, value in zip(cleaned_names, values))
            cleaned_text = f"{main_text}\n{subs_text}"
            
            name, confidence = match_icon(image)
            echo_data = parse_region_text(region, cleaned_text)
            element_region = get_element_region(image)
            element_data = determine_element(element_region, name)
            
            return {
                "success": True,
                "analysis": {
                    "name": {"name": name, "confidence": float(confidence)},
                    "main": echo_data.get("main", {}),
                    "substats": echo_data.get("substats", []),
                    "element": element_data
                }
            }
        else:
            text = process_ocr(region, image)
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