import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re
import json
from pathlib import Path

SCAN_REGIONS = {
    "info": {"top": 0, "left": 0, "width": 0.13, "height": 0.11},
    "uid": {"top": 0.975, "left": 0.915, "width": 0.07, "height": 0.025},
    "characterPage": {"top": 0.09, "left": 0.09, "width": 0.22, "height": 0.18},
    "weaponPage": {"top": 0.11, "left": 0.09, "width": 0.215, "height": 0.25},
    "echoPage": {"top": 0.11, "left": 0.72, "width": 0.25, "height": 0.35},
    "element": {"top": 0.138, "left": 0.933, "width": 0.021, "height": 0.036},
    "s1": {"top": 0.1047, "left": 0.647, "width": 0.0234, "height": 0.0454},
    "s2": {"top": 0.259, "left": 0.733, "width": 0.028, "height": 0.047},
    "s3": {"top": 0.473, "left": 0.765, "width": 0.0234, "height": 0.0454},
    "s4": {"top": 0.682, "left": 0.734, "width": 0.025, "height": 0.0454},
    "s5": {"top": 0.8364, "left": 0.645, "width": 0.029, "height": 0.046},
    "s6": {"top": 0.895, "left": 0.527, "width": 0.025, "height": 0.047},
    "normal-base": {"top": 0.88, "left": 0.2, "width": 0.074, "height": 0.1},
    "normal-mid": {"top": 0.568, "left": 0.22, "width": 0.038, "height": 0.067},
    "normal-top": {"top": 0.36, "left": 0.22, "width": 0.038, "height": 0.067},
    "skill-base": {"top": 0.75, "left": 0.323, "width": 0.074, "height": 0.1},
    "skill-mid": {"top": 0.438, "left": 0.343, "width": 0.038, "height": 0.067},
    "skill-top": {"top": 0.23, "left": 0.343, "width": 0.038, "height": 0.067},
    "circuit-base": {"top": 0.683, "left": 0.463, "width": 0.074, "height": 0.1},
    "circuit-mid": {"top": 0.37, "left": 0.477, "width": 0.05, "height": 0.09},
    "circuit-top": {"top": 0.16, "left": 0.477, "width": 0.05, "height": 0.09},
    "liberation-base": {"top": 0.754, "left": 0.6067, "width": 0.074, "height": 0.1},
    "liberation-mid": {"top": 0.438, "left": 0.6267, "width": 0.038, "height": 0.067},
    "liberation-top": {"top": 0.23, "left": 0.6267, "width": 0.038, "height": 0.067},
    "intro-base": {"top": 0.88, "left": 0.725, "width": 0.077, "height": 0.073},
    "intro-mid": {"top": 0.568, "left": 0.747, "width": 0.038, "height": 0.067},
    "intro-top": {"top": 0.36, "left": 0.747, "width": 0.038, "height": 0.067}
}

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

BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR/ 'Public' / 'Data'

ORIGINAL_IMAGE = None

try:
    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        CHARACTERS = json.load(f)
    with open(DATA_DIR / 'Weapons.json', 'r', encoding='utf-8') as f:
        WEAPONS = json.load(f)
except FileNotFoundError:
    print("Warning: Reference data files not found")
    CHARACTERS = []
    WEAPONS = {}
except json.JSONDecodeError as e:
    print(f"Warning: Invalid JSON format in data files: {e}")
    CHARACTERS = []
    WEAPONS = {}

def preprocess_image(image, region_name=None):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh

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

def crop_region(image, region):
    height, width = image.shape[:2]
    x = int(width * region["left"])
    y = int(height * region["top"])
    w = int(width * region["width"])
    h = int(height * region["height"])
    return image[y:y+h, x:x+w]

def clean_text(text):
    return ' '.join(word for word in text.lower().split() 
                if len(word) > 2)

def upscale_image(image):
    return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

def process_image(image):
    global ORIGINAL_IMAGE
    if image is None or image.size == 0:
        raise ValueError("Invalid image input")
    
    try:
        ORIGINAL_IMAGE = image.copy()

        info_coords = SCAN_REGIONS["info"]
        region_img = crop_region(image, info_coords)
        if region_img.size == 0:
            raise ValueError("Failed to crop info region")

        processed = preprocess_echo_image(region_img)
        info_text = pytesseract.image_to_string(processed)
        
        print("\n=== OCR Debug ===")
        print("Info Region Raw Text:")
        print(info_text)
        
        image_type = determine_type(info_text.lower())
        print(f"\nDetermined Type: {image_type}")

        region_key = {
            'Character': 'characterPage',
            'Weapon': 'weaponPage',
            'Echo': 'echoPage',
            'Sequences': ['s1', 's2', 's3', 's4', 's5', 's6'],
            'Forte': ['normal-base', 'normal-mid', 'normal-top', 
                    'skill-base', 'skill-mid', 'skill-top', 
                    'circuit-base', 'circuit-mid', 'circuit-top',
                    'liberation-base', 'liberation-mid', 'liberation-top',
                    'intro-base', 'intro-mid', 'intro-top']
        }.get(image_type)

        details = {}
        if region_key:            
            if image_type == 'Sequences':
                slots = []
                for slot_key in region_key:
                    slot_coords = SCAN_REGIONS[slot_key]
                    slot_img = crop_region(image, slot_coords)
                    slot_img = upscale_image(slot_img)
                    slots.append(slot_img)
                details = get_sequence_info(slots)
            elif image_type == 'Forte':
                slots = {}
                for region_name in region_key:
                    slot_coords = SCAN_REGIONS[region_name]
                    slot_img = crop_region(image, slot_coords)
                    slots[region_name] = slot_img
                details = get_forte_info(slots)
            elif image_type == 'Echo':
                details = process_echo_image(image, region_key)
            else:
                region_coords = SCAN_REGIONS[region_key]
                detail_img = crop_region(image, region_coords)
                processed_detail = preprocess_image(detail_img, region_key)
                detail_text = pytesseract.image_to_string(processed_detail)
                print("\nRegion Raw Text:")
                print(detail_text)
                print("================")
                details = extract_details(processed_detail, image_type)
            

        print("\n=== Final Response ===")
        response = {
            "success": True,
            "analysis": {
                "type": image_type,
                **details
            }
        }
        print(json.dumps(response, indent=2))
        print("===================")
        return response

    except Exception as e:
        print("\n=== Error Response ===")
        response = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(response, indent=2))
        print("===================")
        return response

def determine_type(text):
    text = text.lower().replace('©', '').replace('€', '')

    type_mappings = {
        'overview': 'Character',
        'weapon': 'Weapon',
        'forte': 'Forte',
        'resonance': 'Sequences'
    }
    
    for key, value in type_mappings.items():
        if key in text:
            return value
    text = text.lower().replace('©', '').replace('€', '')
    all_patterns = [
        'cost', 'ost', 'cst', 'cos',
        'c0st', 'co5t', 'c05t',
        '-cost', 'cost-',
        '(cost', 'cost)',
        '/12', '112', '|12', '(12', '12/12', '12',
        'all', 'al', 'ail', 'ali', 'at', '(all', 'all)'
    ]
    if any(pattern in text for pattern in all_patterns):
        return 'Echo'
            
    return "unknown"

def get_character_info(text):
    global ORIGINAL_IMAGE
    text = text.replace(':', '').replace('.',' ').replace('  ', ' ')
    
    element_pattern = r'["\'\s]*(?:Havoc|Spectro|Electro|Fusion|Glacio|Aero)["\'\s—]*'
    element_match = re.search(element_pattern, text, re.IGNORECASE)
    element = element_match.group(0).strip(' "\'—\n') if element_match else None
    if element:
        element = ' '.join(element.split())
    
    level_patterns = [
        r'Lv[\.|\s]*(\d+)[\s/]+(\d+)', 
        r'v[\.]?(\d+)[\s/]+(\d+)',
        r'Level[\s]*(\d+)[\s/]+(\d+)'
    ]
    
    level = None
    for pattern in level_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                level_num = int(match.groups()[0])
                if 1 <= level_num <= 90:
                    level = match.groups()
                    break
            except ValueError:
                continue
    
    name = None
    for character in CHARACTERS:
        if character['name'].lower() in text.lower():
            name = character['name']
            break
    
    if level and element and element.strip() in ['Havoc', 'Spectro'] and not name:
        name = "Rover"
        gender = detect_rover_gender()
        name = f"Rover{gender}"
        
    uid = None
    try:
        uid_img = crop_region(ORIGINAL_IMAGE, SCAN_REGIONS["uid"])
        uid_img = upscale_image(uid_img)
        gray = cv2.cvtColor(uid_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 115, 255, cv2.THRESH_BINARY)
        uid_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        
        digits = ''.join(filter(str.isdigit, uid_text))
        if len(digits) >= 9:
            uid = digits[-9:]
    except Exception as e:
        print(f"UID extraction failed: {e}")
    
    return {
        'name': name,
        'characterLevel': level[0] if level else None,
        'element': element,
        'uid': uid
    }
    
def detect_rover_gender():
    global ORIGINAL_IMAGE
    
    regions = {
        "shoulders_left": {"top": 0.4, "left": 0.45, "width": 0.02, "height": 0.04},
        "shoulders_right": {"top": 0.4, "left": 0.555, "width": 0.02, "height": 0.04},
        "right_thigh": {"top": 0.87, "left": 0.54, "width": 0.04, "height": 0.095}
    }
    
    dark_colors = [
        (38, 34, 34),
        (36, 48, 46)
    ]
    
    tolerance = 25
    
    male_matches = []
    for region_name, coords in regions.items():
        region_img = crop_region(ORIGINAL_IMAGE, coords)
        
        dark_pixels = 0
        for color in dark_colors:
            mask = cv2.inRange(
                region_img,
                np.array([max(0, c - tolerance) for c in color]),
                np.array([min(255, c + tolerance) for c in color])
            )
            dark_pixels += np.count_nonzero(mask)
        
        total_pixels = region_img.shape[0] * region_img.shape[1]
        dark_ratio = dark_pixels / total_pixels
        male_matches.append(dark_ratio > 0.4)
    
    return " (M)" if sum(male_matches) >= 2 else " (F)"

def get_weapon_info(text):
    lines = text.split('\n')
    if not lines:
        return {'name': None, 'weaponType': None, 'weaponLevel': None, 'rank': None}
        
    first_line = lines[0].strip()\
                        .replace('©', '')\
                        .replace('\\', '')\
                        .replace('%', '')\
                        .replace(':', '')\
                        .replace('  ', ' ')\
                        .replace('q', 'g')\
                        .strip()
    
    name = None
    weapon_type = None
    for type_name, weapons in WEAPONS.items():
        for weapon in weapons:
            if weapon.lower() in first_line.lower():
                name = weapon
                weapon_type = type_name
                break
        if name:
            break
            
    text = text.replace(':', '').replace('.',' ').replace('  ', ' ')
    text_lower = text.lower()
    
    level_patterns = [
        r'Lv[\.|\s]*(\d+)[\s/]+(\d+)',
        r'v[\.]?(\d+)[\s/]+(\d+)',
        r'Level[\s]*(\d+)[\s/]+(\d+)',
        r'[X\s]*[Ll]yv\.?(\d+)[\s/]+(\d+)', 
        r'[X\s]*[Ll]y?v\.?(\d+)[\s/]+(\d+)',
        r'.*v.*?(\d+)/90',
        r'(\d+)/90'
    ]
    
    level = None
    for pattern in level_patterns:
        match = re.search(pattern, text)
        if match:
            level = match.groups()
            break
            
    rank_match = re.search(r'rank\s*(\d+)', text_lower)
    rank = rank_match.group(1) if rank_match else None
    
    return {
        'type': 'Weapon', 
        'name': name,
        'weaponType': weapon_type,
        'weaponLevel': level[0] if level else '90',
        'rank': rank
    }

def get_sequence_info(slots):
    sequence_states = []
    yellow_lower = np.array([20, 103, 214])
    yellow_upper = np.array([27, 227, 242])
    
    for i, slot_img in enumerate(slots):
        hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)
        if i == 5: 
            yellow_lower_s6 = np.array([30, 50, 100])
            yellow_upper_s6 = np.array([75, 255, 255])
            blue_lower = np.array([0, 0, 76])
            blue_upper = np.array([123, 54, 151])
            
            yellow_mask = cv2.inRange(hsv, yellow_lower_s6, yellow_upper_s6)
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            yellow_ratio = (np.count_nonzero(yellow_mask) / (slot_img.shape[0] * slot_img.shape[1])) * 100
            blue_ratio = (np.count_nonzero(blue_mask) / (slot_img.shape[0] * slot_img.shape[1])) * 100            
            sequence_states.append(1 if yellow_ratio > blue_ratio else 0)
        else:
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_ratio = (np.count_nonzero(yellow_mask) / (slot_img.shape[0] * slot_img.shape[1])) * 100            
            sequence_states.append(1 if yellow_ratio > 10 else 0)
    count = sum(1 for state in sequence_states if state == 1)    
    return {
        'type': 'Sequences',
        'sequence': count
    }

def get_forte_info(slots):
    branches = {
        'normalAttack': {'level': 1, 'middleNode': False, 'topNode': False},
        'skill': {'level': 1, 'middleNode': False, 'topNode': False},
        'circuit': {'level': 1, 'middleNode': False, 'topNode': False},
        'liberation': {'level': 1, 'middleNode': False, 'topNode': False},
        'introSkill': {'level': 1, 'middleNode': False, 'topNode': False}
    }

    slot_to_branch = {
        'normal-base': 'normalAttack',
        'skill-base': 'skill',
        'circuit-base': 'circuit', 
        'liberation-base': 'liberation',
        'intro-base': 'introSkill'
    }

    level_patterns = [
        r'(\d+)/[\s]*(?:10|110)',
        r'Lv[:\.]\s*(\d+)/',
        r'v[:\.]\s*(\d+)/',
        r'Lv[:\.]\s*(\d+)',
        r'v[:\.]\s*(\d+)',
    ]

    for slot_name, image in slots.items():
        branch_name = None
        
        if 'base' in slot_name:
            branch_name = slot_to_branch.get(slot_name)
            if branch_name:
                processed_image = preprocess_echo_image(image)
                text = pytesseract.image_to_string(processed_image)
                text = text.replace('\\', '').replace('"', '').replace("'", '')
                
                level = None
                for pattern in level_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            level = int(match.group(1))
                            if level > 10:
                                level = 10
                            if 1 <= level <= 10:
                                branches[branch_name]['level'] = level
                                break
                        except ValueError:
                            continue
                            
        elif 'mid' in slot_name or 'top' in slot_name:
            branch_base = slot_name.split('-')[0]
            branch_name = slot_to_branch.get(f'{branch_base}-base')
            
            if branch_name:
                is_active = is_node_active(image, slot_name)
                
                if 'mid' in slot_name:
                    branches[branch_name]['middleNode'] = is_active
                else:
                    branches[branch_name]['topNode'] = is_active

    result = {
        'type': 'Forte',
        **{
            name: [
                branches[branch]['level'],
                1 if branches[branch]['middleNode'] else 0,
                1 if branches[branch]['topNode'] else 0
            ] for name, branch in {
                'normal': 'normalAttack',
                'skill': 'skill', 
                'circuit': 'circuit',
                'liberation': 'liberation',
                'intro': 'introSkill'
            }.items()
        }
    }

    return result

def is_node_active(image, slot_name="unknown"):
    h, w = image.shape[:2]
    center = (w//2, h//2)
    radius = min(w, h)//2
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=circle_mask)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    
    if 'circuit' in slot_name:
        dark_blue_lower = np.array([100, 50, 50])
        dark_blue_upper = np.array([140, 150, 150])
        dark_mask = cv2.inRange(hsv, dark_blue_lower, dark_blue_upper)
        circle_area = np.count_nonzero(circle_mask)
        dark_ratio = (np.count_nonzero(dark_mask) / circle_area) * 100
        return dark_ratio < 30 
    else:
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([20, 255, 255])
        red_lower2 = np.array([160, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)        
        circle_area = np.count_nonzero(circle_mask)
        red_ratio = (np.count_nonzero(red_mask) / circle_area) * 100
        
        return red_ratio > 5

def process_echo_image(image, region_key):
    element_coords = SCAN_REGIONS["element"]
    element_img = crop_region(image, element_coords)
    
    h, w = element_img.shape[:2]
    center = (w//2, h//2)
    radius = min(w, h)//2
    
    circle_mask = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, (255, 255, 255, 255), -1)
    circle_bgra = cv2.cvtColor(element_img, cv2.COLOR_BGR2BGRA)
    circle_bgra[:, :, 3] = circle_mask[:, :, 3]
    
    rim_thickness = max(int(min(w, h) * 0.115), 1)
    donut_mask = circle_mask.copy()
    cv2.circle(donut_mask, center, radius-rim_thickness, (0, 0, 0, 0), -1)
    donut_bgra = cv2.cvtColor(element_img, cv2.COLOR_BGR2BGRA)
    donut_bgra[:, :, 3] = donut_mask[:, :, 3]
    
    elements = get_element_info(donut_bgra)
    element = get_icon(circle_bgra, elements) if len(elements) > 1 else elements[0]

    region_coords = SCAN_REGIONS[region_key]
    detail_img = crop_region(image, region_coords)
    processed_detail = preprocess_echo_image(detail_img)
    
    return get_echo_info(processed_detail, element)

def get_element_info(element_img):
    hsv = cv2.cvtColor(element_img, cv2.COLOR_BGR2HSV)
    
    element_colors = {
        'healing': {'lower': np.array([40, 0, 0]), 'upper': np.array([94, 255, 255])},
        'attack': {'lower': np.array([0, 150, 100]), 'upper': np.array([5, 255, 200])},
        'electro': {'lower': np.array([130, 40, 150]), 'upper': np.array([145, 255, 255])},
        'er': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},
        'fusion': {'lower': np.array([5, 50, 150]), 'upper': np.array([25, 255, 255])},
        'glacio': {'lower': np.array([85, 40, 150]), 'upper': np.array([115, 255, 255])},
        'havoc': {'lower': np.array([150, 0, 150]), 'upper': np.array([180, 255, 255])},
        'aero': {'lower': np.array([68, 120, 52]), 'upper': np.array([88, 220, 152])},
        'spectro': {'lower': np.array([10, 101, 86]), 'upper': np.array([33, 173, 255])} 
    }
    
    ratios = {}
    for element, ranges in element_colors.items():
        mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
        ratio = (np.count_nonzero(mask) / (element_img.shape[0] * element_img.shape[1])) * 100
        ratios[element] = ratio
    
    sorted_elements = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    highest = sorted_elements[0]
    second = sorted_elements[1] if len(sorted_elements) > 1 else None
    
    if highest[1] > 33:
        if second and second[1] > 30:
            return [highest[0], second[0]]
        return [highest[0]]
    return [highest[0]]

def get_icon(circle_bgra, possible_elements):
    h, w = circle_bgra.shape[:2]
    
    best_match = None
    best_score = float('inf')
    
    
    for element in possible_elements:
        template_path = BACKEND_DIR / 'Public' / 'Sets' / f'{element.capitalize()}.png'
        template = cv2.imread(str(template_path))
        template = cv2.resize(template, (w, h))
        template_bgra = cv2.cvtColor(template, cv2.COLOR_BGR2BGRA)
        
        result = cv2.matchTemplate(circle_bgra, template_bgra, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        
        if score < best_score:
            best_score = score
            best_match = element
    return best_match


def get_echo_info(processed_image, element):
    name_img = upscale_image(crop_region(processed_image, ECHO_REGIONS["name"]))
    lv_img = upscale_image(crop_region(processed_image, ECHO_REGIONS["level"]))
    main_img = upscale_image(crop_region(processed_image, ECHO_REGIONS["main"]))
    
    sub_images = []
    sub_stats = []
    for i in range(1, 6):
        sub_key = f"sub{i}"
        sub_img = crop_region(processed_image, ECHO_REGIONS[sub_key])
        sub_images.append(sub_img)
        sub_text = pytesseract.image_to_string(sub_img).strip().replace('\n', ' ')
        
        if sub_text:
            if match := re.search(r'(.*?)[\s—:]*(\d+\.?\d*\%?)(?:\s*[~o\s\*]*)?$', sub_text):
                name, value = match.groups()
                name = re.sub(r'[\\\\][u]?[0-9A-Fa-f]*', '', name)
                name = re.sub(r'(DMG).*', r'\1', name)
                name = name.replace('.', '')\
                        .replace('DMG Bonus', '')\
                        .replace('BMG', 'DMG')\
                        .replace('Resonance', '')\
                        .replace('Regonance', '')\
                        .replace('Gr', 'Crit')\
                        .replace('CritDMGN', 'Crit DMG')\
                        .replace('SNON', '')\
                        .replace('ATKON', 'ATK')\
                        .replace(' ON', '')\
                        .replace(' N', '')\
                        .replace(' a', '')\
                        .replace('—', '')\
                        .replace(':', '')\
                        .replace(',', '')\
                        .replace('~~', '')\
                        .replace('oo', '')\
                        .replace(' *', '')\
                        .replace('*', '')\
                        .replace(' ~', '')\
                        .replace('~', '')\
                        .strip()
                name = name if name else "HP"
                sub_stats.append({'name': name, 'value': value.strip()})
                
    name_text = pytesseract.image_to_string(name_img).strip()
    if 'phantom:' in name_text.lower() or ':' in name_text:
        name_text = name_text.split(':', 1)[-1].strip()    
    if not name_text:
        name_text = 'Jue'
    
    main_text = pytesseract.image_to_string(main_img).strip().replace('\n', ' ')
    
    main_stat = {}
    if match := re.search(r'(.*?)(\d+\.?\d*\%?)$', main_text):
        name, value = match.groups()
        name = name.replace('DMG Bonus', 'DMG')\
                .replace('BMG', 'DMG')\
                .replace('.', '')\
                .strip()
        if 'DMG' in name:
            name = re.sub(r'(DMG).*', r'\1', name)
        name = name.replace('CritDMG', 'Crit DMG')\
                .replace('  ', ' ')
        main_stat = {'name': name if name else 'HP', 'value': value.strip()}

    return {
        'type': 'Echo',
        'element': element,
        'name': name_text,
        'echoLevel': pytesseract.image_to_string(lv_img).strip().replace('\n', ' ').lstrip('+'),
        'main': main_stat,
        'subs': sub_stats
    }

def extract_details(image, image_type, element=None):
    """Extract details from image based on type"""
    if image_type in ['Character', 'Weapon', 'Echo']:
        text = pytesseract.image_to_string(image)
    
    if image_type == 'Character':
        return get_character_info(text)
    elif image_type == 'Weapon':
        return get_weapon_info(text)
    elif image_type == 'Sequences':
        return get_sequence_info(image)
    elif image_type == 'Forte':
        return get_forte_info(image)
    elif image_type == 'Echo':
        return get_echo_info(image, element)
    else:
        text = pytesseract.image_to_string(image)
        return {'raw_text': clean_text(text)}