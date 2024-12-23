import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from rapidfuzz import process
from rapidfuzz.utils import default_process
from rapidocr_onnxruntime import RapidOCR

OCR = None

def get_ocr():
    global OCR
    if OCR is None:
        OCR = RapidOCR(lang='en')
    return OCR

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

def read_and_crop_image(image: np.ndarray) -> List[Tuple[int, int, str, int, int]]:
    ocr = get_ocr()
    result, _ = ocr(image)
    entries = []
    for item in result:
        box, text, _ = item
        x1, y1 = box[0]
        x2, y2 = box[2]
        w = int(x2 - x1)
        h = int(y2 - y1)
        entries.append((int(y1), int(x1), text, w, h))
    return entries

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
                        name = match[0].upper()
                
                try:
                    value = float(raw_value.replace('%', ''))
                    valid_values = [float(v) for v in SUB_STATS[name]]
                    closest = min(valid_values, key=lambda x: abs(x - value))
                    
                    normalized_value = f"{closest}%" if had_percent else str(closest)
                    print(f"Value normalized: {raw_value} -> {normalized_value}")
                    sub_stats.append({"name": name, "value": normalized_value})
                    continue
                except (ValueError, KeyError):
                    print(f"Could not normalize value: {raw_value}")
                    sub_stats.append({"name": name, "value": raw_value})
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
    entries = read_and_crop_image(image)
    text_lines = merge_nearby_text(entries)
    
    text_lines = clean_lines(text_lines)
    print("OCR Output:")
    for line in text_lines:
        print(f"{line}")
    print()
    
    name = get_name(text_lines)
    element = ECHO_ELEMENTS.get(name, ["Unknown"])[0]
    
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