from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, List, Set
from cv2 import SIFT_create
from rapidocr_onnxruntime import RapidOCR

Rapid = RapidOCR(lang='en')

# Initialize empty defaults
CHARACTER_NAMES: List[str] = []
WEAPON_NAMES: List[str] = []
WEAPON_DATA: Dict[str, str] = {} 
MAIN_STAT_NAMES: Set[str] = set()
SUB_STATS: Dict = {}
SUB_STAT_NAMES: Set[str] = set()
ECHO_NAMES: List[str] = []
ECHO_ELEMENTS: Dict = {}
ICON_TEMPLATES: Dict[str, np.ndarray] = {}
TEMPLATE_FEATURES = {}
ECHO_COSTS: Dict[str, int] = {}
ELEMENT_TEMPLATES: Dict[str, np.ndarray] = {}
ELEMENT_FEATURES = {}

# Paths
DATA_DIR = Path(__file__).parent / 'Data'

def load_templates(folder: str, templates: dict, features: dict, target_size: tuple = None) -> int:
    count = 0
    sift = SIFT_create()
    
    for icon_path in (DATA_DIR / folder).glob('*.png'):
        try:
            img = cv2.imread(str(icon_path))
            if img is None:
                print(f"Failed to load template: {icon_path}")
                continue
            
            if target_size:
                img = cv2.resize(img, target_size)
            templates[icon_path.stem] = img
            
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                features[icon_path.stem] = (kp, des)
                count += 1
            else:
                print(f"No features detected for: {icon_path}")
                
        except Exception as e:
            print(f"Error processing template {icon_path}: {e}")
    return count

try:
    # Load characters
    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        characters_data = json.load(f)
        CHARACTER_NAMES = [char['name'] for char in characters_data]

    with open(DATA_DIR / 'Weapons.json', 'r', encoding='utf-8') as f:
        weapons_data = json.load(f)
        for weapon_type, weapons in weapons_data.items():
            for weapon in weapons:
                WEAPON_NAMES.append(weapon['name'])
                WEAPON_DATA[weapon['name']] = weapon_type

    # Load echoes
    with open(DATA_DIR / 'Echoes.json', 'r', encoding='utf-8') as f:
        echoes_data = json.load(f)
        ECHO_NAMES = [echo['name'] for echo in echoes_data]
        ECHO_ELEMENTS = {echo['name']: echo['elements'] for echo in echoes_data}
        ECHO_COSTS = {echo['name']: echo['cost'] for echo in echoes_data}

    # Load main stats
    with open(DATA_DIR / 'Mainstat.json', 'r', encoding='utf-8') as f:
        main_data = json.load(f)
        for cost_data in main_data.values():
            if "mainStats" in cost_data:
                for stat_name in cost_data["mainStats"].keys():
                    if stat_name in ["HP%", "ATK%", "DEF%"]:
                        MAIN_STAT_NAMES.add(stat_name.replace("%", ""))
                    else:
                        MAIN_STAT_NAMES.add(stat_name)

    # Load substats
    with open(DATA_DIR / 'Substats.json', 'r', encoding='utf-8') as f:
        sub_data = json.load(f)
        SUB_STATS = sub_data["subStats"]
        SUB_STAT_NAMES = set(SUB_STATS.keys())
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
        
    echo_count = load_templates('Echoes', ICON_TEMPLATES, TEMPLATE_FEATURES, (188, 188))
    element_count = load_templates('Elements', ELEMENT_TEMPLATES, ELEMENT_FEATURES)
    print(f"Loaded {echo_count} echo templates and {element_count} element templates")

except Exception as e:
    print(f"Critical error during initialization: {e}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Data directory exists: {DATA_DIR.exists()}")

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