from pathlib import Path
import json
import urllib.request
import urllib.error
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
ECHO_NAMES: List[str] = []       # ordered list of English names (for logging/rapidfuzz)
ECHO_ELEMENTS: Dict = {}          # CDN id str → list of element/sonata-set names
ECHO_COSTS: Dict[str, int] = {}   # CDN id str → cost (1 | 3 | 4)
ECHO_NAME_MAP: Dict[str, str] = {} # CDN id str → English name (for display in response)
ICON_TEMPLATES: Dict[str, np.ndarray] = {}
TEMPLATE_FEATURES = {}
ELEMENT_TEMPLATES: Dict[str, np.ndarray] = {}
ELEMENT_FEATURES = {}

# Paths
DATA_DIR = Path(__file__).parent / 'Data'
CDN_BASE = 'https://wuwabuilds.moe/Data'

# Mirrors echo.ts FETTER_MAP: fetter ID → ElementType short name
_FETTER_MAP: Dict[int, str] = {
    1: 'Glacio', 2: 'Fusion', 3: 'Electro', 4: 'Aero', 5: 'Spectro', 6: 'Havoc',
    7: 'Healing', 8: 'ER', 9: 'Attack', 10: 'Frosty', 11: 'Radiance', 12: 'Midnight',
    13: 'Empyrean', 14: 'Tidebreaking', 16: 'Gust', 17: 'Windward', 18: 'Flaming',
    19: 'Dream', 20: 'Crown', 21: 'Law', 22: 'Flamewing', 23: 'Thread', 24: 'Pact',
    25: 'Halo', 26: 'Rite', 27: 'Trailblazing', 28: 'Chromatic', 29: 'Sound',
}

# Frontend weapon type ID → backend category name
_WEAPON_TYPE_MAP: Dict[int, str] = {
    1: 'Broadblades', 2: 'Swords', 3: 'Pistols', 4: 'Gauntlets', 5: 'Rectifiers',
}


def _fetch_json(filename: str):
    url = f'{CDN_BASE}/{filename}'
    req = urllib.request.Request(url, headers={'User-Agent': 'wuwabuilds-backend/1.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _load_from_cdn() -> bool:
    """Fetch Characters, Weapons, Echoes from wuwabuilds.moe CDN. Returns True on success."""
    global CHARACTER_NAMES, WEAPON_NAMES, WEAPON_DATA, ECHO_NAMES, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP
    try:
        chars = _fetch_json('Characters.json')
        weapons = _fetch_json('Weapons.json')
        echoes = _fetch_json('Echoes.json')

        # Only mutate globals after all fetches succeed
        CHARACTER_NAMES = [c['name']['en'] for c in chars]

        WEAPON_NAMES.clear(); WEAPON_DATA.clear()
        for w in weapons:
            name = w['name']['en']
            WEAPON_NAMES.append(name)
            WEAPON_DATA[name] = _WEAPON_TYPE_MAP.get(w['type']['id'], 'Unknown')

        ECHO_NAMES.clear(); ECHO_ELEMENTS.clear(); ECHO_COSTS.clear(); ECHO_NAME_MAP.clear()
        for e in echoes:
            eid  = str(e['id'])
            name = e['name']['en']
            ECHO_NAMES.append(name)
            ECHO_COSTS[eid]    = e['cost']
            ECHO_ELEMENTS[eid] = [_FETTER_MAP[f] for f in e['fetter'] if f in _FETTER_MAP]
            ECHO_NAME_MAP[eid] = name

        print(f"CDN sync: {len(CHARACTER_NAMES)} characters, {len(WEAPON_NAMES)} weapons, {len(ECHO_NAMES)} echoes")
        return True
    except Exception as e:
        print(f"CDN sync failed ({e}), falling back to local files")
        return False


def _load_from_local():
    """Load Characters, Weapons, Echoes from local Data/ files (legacy format)."""
    global CHARACTER_NAMES, WEAPON_NAMES, WEAPON_DATA, ECHO_NAMES, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP

    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        CHARACTER_NAMES = [c['name'] for c in json.load(f)]

    WEAPON_NAMES.clear(); WEAPON_DATA.clear()
    with open(DATA_DIR / 'Weapons.json', 'r', encoding='utf-8') as f:
        for weapon_type, weapons in json.load(f).items():
            for w in weapons:
                WEAPON_NAMES.append(w['name'])
                WEAPON_DATA[w['name']] = weapon_type

    ECHO_NAMES.clear(); ECHO_ELEMENTS.clear(); ECHO_COSTS.clear(); ECHO_NAME_MAP.clear()
    with open(DATA_DIR / 'Echoes.json', 'r', encoding='utf-8') as f:
        for e in json.load(f):
            eid  = str(e['id'])
            name = e['name']
            ECHO_NAMES.append(name)
            ECHO_COSTS[eid]    = e['cost']
            ECHO_ELEMENTS[eid] = e['elements']
            ECHO_NAME_MAP[eid] = name

    print(f"Local fallback: {len(CHARACTER_NAMES)} characters, {len(WEAPON_NAMES)} weapons, {len(ECHO_NAMES)} echoes")


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
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Fetch game data from CDN; fall back to local files if unavailable
    if not _load_from_cdn():
        _load_from_local()

    # Mainstat and Substats: format is identical on CDN and local, keep local
    with open(DATA_DIR / 'Mainstat.json', 'r', encoding='utf-8') as f:
        main_data = json.load(f)
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