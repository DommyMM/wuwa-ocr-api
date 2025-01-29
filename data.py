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

# Paths
DATA_DIR = Path(__file__).parent / 'Data'

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
        
    sift = SIFT_create()
    template_count = 0
    
    # Load and process templates
    for icon_path in DATA_DIR.glob('*.png'):
        try:
            img = cv2.imread(str(icon_path))
            if img is None:
                print(f"Failed to load template: {icon_path}")
                continue
                
            scaled = cv2.resize(img, (188, 188))
            ICON_TEMPLATES[icon_path.stem] = scaled
            
            # Compute SIFT features
            kp, des = sift.detectAndCompute(scaled, None)
            if des is not None:
                TEMPLATE_FEATURES[icon_path.stem] = (kp, des)
                template_count += 1
            else:
                print(f"No features detected for: {icon_path}")
                
        except Exception as e:
            print(f"Error processing template {icon_path}: {e}")
            continue
    
    
except Exception as e:
    print(f"Critical error during initialization: {e}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Data directory exists: {DATA_DIR.exists()}")

ELEMENT_COLORS = {
'Healing': {'lower': np.array([30, 60, 120]), 'upper': np.array([50, 210, 240])},
'Electro': {'lower': np.array([100, 70, 140]), 'upper': np.array([179, 170, 255])},
'Fusion': {'lower': np.array([0, 150, 150]), 'upper': np.array([20, 180, 255])},
'Havoc': {'lower': np.array([140, 50, 70]), 'upper': np.array([179, 90, 255])},
'Spectro': {'lower': np.array([20, 100, 200]), 'upper': np.array([40, 160, 255])},
'Glacio': {'lower': np.array([90, 150, 210]), 'upper': np.array([110, 210, 255])},
'Aero': {'lower': np.array([60, 150, 210]), 'upper': np.array([80, 180, 255])},
'Attack': {'lower': np.array([0, 190, 120]), 'upper': np.array([5, 220, 220])},
'ER': {'lower': np.array([0, 0, 190]), 'upper': np.array([140, 30, 255])},
'Empyrean': {'lower': np.array([90, 90, 210]), 'upper': np.array([110, 130, 255])},
'Frosty': {'lower': np.array([90, 150, 210]), 'upper': np.array([110, 210, 255])},
'Midnight': {'lower': np.array([140, 50, 70]), 'upper': np.array([179, 90, 255])},
'Radiance': {'lower': np.array([20, 100, 200]), 'upper': np.array([40, 160, 255])},
'Tidebreaking': {'lower': np.array([0, 0, 190]), 'upper': np.array([140, 30, 255])}
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