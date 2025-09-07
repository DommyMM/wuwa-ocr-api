from typing import Dict, Any
import pytesseract
import numpy as np
import cv2
import re
from echo import process_echo
from card import preprocess_region
from data import CHARACTER_NAMES, WEAPON_NAMES, WEAPON_DATA
from rapidfuzz import process
from rapidfuzz.utils import default_process

ELEMENTS = ["Aero", "Glacio", "Electro", "Havoc", "Fusion", "Spectro"]

REGIONS = {
    'stats': {'top': 0.0071, 'left': 0, 'width': 0.3994, 'height': 0.1969},
    'shoulders_left': {'top': 0.3461, 'left': 0.675, 'width': 0.0364, 'height': 0.0435},
    'shoulders_right': {'top': 0.3461, 'left': 0.8447, 'width': 0.0364, 'height': 0.0435}, 
    'right_thigh': {'top': 0.8594, 'left': 0.8174, 'width': 0.068, 'height': 0.07},
    's1': {'top': 0, 'left': 0.443, 'width': 0.0865, 'height': 0.0544},
    's2': {'top': 0.1847, 'left': 0.7615, 'width': 0.1029, 'height': 0.0560},
    's3': {'top': 0.4409, 'left': 0.8788, 'width': 0.0865, 'height': 0.0544},
    's4': {'top': 0.6911, 'left': 0.7654, 'width': 0.0923, 'height': 0.0544},
    's5': {'top': 0.8763, 'left': 0.4356, 'width': 0.1067, 'height': 0.0549},
    's6': {'top': 0.9462, 'left': 0, 'width': 0.0923, 'height': 0.0560},
    'normalBase': { 'top': 0.88, 'left': 0.01, 'width': 0.105, 'height': 0.053 },
    'normalMid': { 'top': 0.4976, 'left': 0.0342, 'width': 0.0650, 'height': 0.0817 },
    'normalTop': { 'top': 0.2439, 'left': 0.0342, 'width': 0.0650, 'height': 0.0817 },
    'skillBase': { 'top': 0.725, 'left': 0.2103, 'width': 0.1265, 'height': 0.055 },
    'skillMid': { 'top': 0.3390, 'left': 0.236, 'width': 0.063, 'height': 0.0817 },
    'skillTop': { 'top': 0.0854, 'left': 0.236, 'width': 0.063, 'height': 0.0817 },
    'circuitBase': { 'top': 0.65, 'left': 0.4496, 'width': 0.1265, 'height': 0.055 },
    'circuitMid': { 'top': 0.2561, 'left': 0.458, 'width': 0.083, 'height': 0.1098 },
    'circuitTop': { 'top': 0, 'left': 0.458, 'width': 0.083, 'height': 0.1098 },
    'liberationBase': { 'top': 0.7244, 'left': 0.69, 'width': 0.11, 'height': 0.058 },
    'liberationMid': { 'top': 0.3390, 'left': 0.706, 'width': 0.063, 'height': 0.0817 },
    'liberationTop': { 'top': 0.0854, 'left': 0.706, 'width': 0.063, 'height': 0.0817 },
    'introBase': { 'top': 0.88, 'left': 0.885, 'width': 0.11, 'height': 0.052 },
    'introMid': { 'top': 0.4976, 'left': 0.904, 'width': 0.063, 'height': 0.0817 },
    'introTop': { 'top': 0.2439, 'left': 0.904, 'width': 0.063, 'height': 0.0817 }
}

def is_dark_pixel(img: np.ndarray) -> np.ndarray:
    """Vectorized dark pixel detection using numpy."""
    threshold = 75
    
    condition1 = (
        (np.abs(img[:, :, 2] - 38) <= threshold) & 
        (np.abs(img[:, :, 1] - 34) <= threshold) & 
        (np.abs(img[:, :, 0] - 34) <= threshold)
    )
    condition2 = (
        (np.abs(img[:, :, 2] - 36) <= threshold) & 
        (np.abs(img[:, :, 1] - 48) <= threshold) & 
        (np.abs(img[:, :, 0] - 46) <= threshold)
    )
    return condition1 | condition2

def is_white_pixel(img: np.ndarray) -> np.ndarray:
    """Vectorized white pixel detection."""
    return (img[:, :, 2] >= 160) & (img[:, :, 1] >= 180) & (img[:, :, 0] >= 145)

def is_yellow_pixel(img: np.ndarray) -> np.ndarray:
    """Detect yellow pixels using HSV with lower threshold."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    return (
        (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 85) & 
        (hsv[:, :, 1] >= 30) & (hsv[:, :, 1] <= 255) &
        (hsv[:, :, 2] >= 80) & (hsv[:, :, 2] <= 255)
    )

def process_node(image: np.ndarray, region: Dict[str, float], is_circuit: bool) -> int:
    cropped = crop_region(image, region)
    active_mask = is_white_pixel(cropped)
    threshold = 0.4 if not is_circuit else 0.2
    
    active_ratio = np.sum(active_mask) / (cropped.shape[0] * cropped.shape[1])
    return 1 if active_ratio > threshold else 0

def detect_gender(image: np.ndarray) -> str:
    """Detect gender from character image regions."""
    regions = ['shoulders_left', 'shoulders_right', 'right_thigh']
    male_matches = []
    
    for region in regions:
        cropped = crop_region(image, REGIONS[region])
        dark_mask = is_dark_pixel(cropped)
        dark_ratio = np.sum(dark_mask) / (cropped.shape[0] * cropped.shape[1])
        is_male = dark_ratio > 0.4
        male_matches.append(is_male)
    
    return " (M)" if sum(male_matches) >= 2 else " (F)"

def crop_region(image: np.ndarray, region: Dict[str, float]) -> np.ndarray:
    """Crop image according to relative coordinates."""
    height, width = image.shape[:2]
    x1 = int(width * region['left'])
    y1 = int(height * region['top'])
    x2 = int(width * (region['left'] + region['width']))
    y2 = int(height * (region['top'] + region['height']))
    return image[y1:y2, x1:x2]

def process_character(image: np.ndarray) -> Dict[str, Any]:
    """Process character screenshot."""
    stats_image = crop_region(image, REGIONS['stats'])
    clean_card = preprocess_region(stats_image)
    text = pytesseract.image_to_string(clean_card)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    raw_name = lines[0] if lines else "Unknown"
    matched_name = raw_name
    if CHARACTER_NAMES:
        name_match = process.extractOne(raw_name, CHARACTER_NAMES, processor=default_process)
        if name_match and name_match[1] > 70:
            matched_name = name_match[0]
    
    raw_element = lines[1] if len(lines) > 1 else None
    element = None
    if raw_element:
        element_match = process.extractOne(raw_element, ELEMENTS, processor=default_process)
        if element_match and element_match[1] > 70:
            element = element_match[0]
    
    if name_match[1] <= 70 and element in ["Havoc", "Spectro"]:
        gender = detect_gender(image)
        matched_name = f"Rover{gender}"
    
    level = 90
    numbers = re.findall(r'\d+', text)
    if numbers:
        numbers.reverse()
        for num in numbers:
            level_num = int(num)
            if 1 <= level_num <= 90:
                level = level_num
                break
    
    return {
        "success": True,
        "analysis": {
            "type": "Character",
            "name": matched_name,
            "characterLevel": level,
            "element": element
        }
    }

def process_weapon(image: np.ndarray) -> Dict[str, Any]:
    """Process weapon screenshot."""
    clean_card = preprocess_region(image)
    text = pytesseract.image_to_string(clean_card)
    
    raw_name = text.split('\n')[0].strip()
    matched_name = raw_name
    weapon_type = None
    
    if WEAPON_NAMES:
        name_match = process.extractOne(raw_name, WEAPON_NAMES, processor=default_process)
        if name_match and name_match[1] > 70:
            matched_name = name_match[0]
            weapon_type = WEAPON_DATA.get(matched_name)
    
    level = 90
    for line in text.split('\n'):
        if "Lv." in line:
            clean_numbers = [n.split('/')[0] for n in line.split() if n.replace('/', '').isdigit()]
            if clean_numbers:
                level_num = int(clean_numbers[0])
                if 1 <= level_num <= 90:
                    level = level_num
                    break
    
    rank = 1
    rank_match = re.search(r'rank\s*(\d+)', text.lower())
    if rank_match:
        rank = int(rank_match.group(1))
    
    return {
        "success": True,
        "analysis": {
            "type": "Weapon",
            "name": matched_name,
            "weaponType": weapon_type,
            "weaponLevel": level,
            "rank": rank
        }
    }

def process_sequences(image: np.ndarray) -> Dict[str, Any]:
    """Process sequence screenshot."""
    height, width = image.shape[:2]
    sequence_sum = 0
    
    slots = [f's{i}' for i in range(1, 7)]
    for slot in slots:
        region = REGIONS[slot]
        x1 = int(width * region['left'])
        y1 = int(height * region['top'])
        x2 = int(width * (region['left'] + region['width']))
        y2 = int(height * (region['top'] + region['height']))
        
        cropped = image[y1:y2, x1:x2]
        yellow_mask = is_yellow_pixel(cropped)
        yellow_ratio = np.sum(yellow_mask) / (cropped.shape[0] * cropped.shape[1])
        is_active = yellow_ratio > 0.5
        
        if is_active:
            sequence_sum += 1
    
    print(f"Total active sequences: {sequence_sum}/6", flush=True)
    
    return {
        "success": True,
        "analysis": {
            "type": "Sequences",
            "sequence": sequence_sum
        }
    }

def process_forte(image: np.ndarray) -> Dict[str, Any]:
    """Process forte screenshot to extract levels and active nodes for each branch."""
    branches = ['normal', 'skill', 'circuit', 'liberation', 'intro']
    results = {}
    height, width = image.shape[:2]

    print("Starting forte detection", flush=True)
    for branch in branches:
        base_region = REGIONS[f'{branch}Base']
        base_cropped = crop_region(image, base_region)
        base_text = pytesseract.image_to_string(base_cropped)

        level = extract_level(base_text)
        top_active = process_node(image, REGIONS[f'{branch}Top'], branch == 'circuit')
        mid_active = process_node(image, REGIONS[f'{branch}Mid'], branch == 'circuit')
        results[branch] = [level, top_active, mid_active]
        print(f"Forte {branch:12} Lv.{level:2d} [{top_active},{mid_active}]", flush=True)

    return {
        "success": True,
        "analysis": {
            "type": "Forte",
            **{b: results[b] for b in branches}
        }
    }

def extract_level(text: str) -> int:
    """Extract level from text with patterns."""
    text = text.replace('710', '')
    
    patterns = [
        r'Lv\.?\s*(\d+)',
        r'(\d+)\s*/\s*10',
        r'(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 10:
                return level
    return 10

def process_char(image: np.ndarray, char_type: str) -> Dict[str, Any]:
    """Process character-related screenshots using OCR."""
    match char_type:
        case "character":
            return process_character(image)
        case "weapon":
            return process_weapon(image)
        case "sequences":
            return process_sequences(image)
        case "forte":
            return process_forte(image)
        case "echo":
            return process_echo(image)
        case _:
            raise ValueError(f"Unsupported character type: {char_type}")