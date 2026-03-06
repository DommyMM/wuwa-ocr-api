from pathlib import Path
import json
import threading
import cv2
import numpy as np
from typing import Any, Callable, Dict, List, Set, Optional, Tuple
from cv2 import FlannBasedMatcher
from rapidocr_onnxruntime import RapidOCR
import imagehash
from PIL import Image

_sift_ctor: Callable[[], Any] = getattr(cv2, "SIFT_create")
_THREAD_LOCAL = threading.local()
# Shared matcher params; actual matcher instances are request-local.
_FLANN_PARAMS = (dict(algorithm=1, trees=5), dict(checks=50))

def make_sift():
    """Create a fresh SIFT instance for request-local feature extraction."""
    return _sift_ctor()

def make_flann() -> FlannBasedMatcher:
    """Create a fresh FLANN matcher instance for request-local matching."""
    index_params, search_params = _FLANN_PARAMS
    return FlannBasedMatcher(dict(index_params), dict(search_params))

def run_rapid(image):
    """Run RapidOCR using a thread-local engine instance."""
    engine = getattr(_THREAD_LOCAL, "rapid", None)
    if engine is None:
        engine = RapidOCR(lang='en')
        _THREAD_LOCAL.rapid = engine
    return engine(image)

# Initialize empty defaults
CHARACTER_NAMES: List[str] = []
CHARACTER_ID_MAP: Dict[str, str] = {}
WEAPON_NAMES: List[str] = []
WEAPON_DATA: Dict[str, str] = {}
WEAPON_ID_MAP: Dict[str, str] = {}
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
TEMPLATE_PHASHES: Dict = {}
TEMPLATE_HISTOGRAMS: Dict[str, np.ndarray] = {}

# Paths
DATA_DIR = Path(__file__).parent / 'Data'


def _load_from_local():
    """Load Characters, Weapons, Echoes from local Data/ files (legacy format)."""
    global CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_DATA, WEAPON_ID_MAP, ECHO_NAMES, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP

    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        characters = json.load(f)
        CHARACTER_NAMES = [c['name'] for c in characters]
        CHARACTER_ID_MAP.clear()
        for c in characters:
            name = c.get('name', '')
            cid = str(c.get('id', '')).strip()
            if name and cid:
                # Preserve first seen ID for duplicated names (Rover variants are handled in frontend fallback).
                CHARACTER_ID_MAP.setdefault(name, cid)

    WEAPON_NAMES.clear(); WEAPON_DATA.clear(); WEAPON_ID_MAP.clear()
    with open(DATA_DIR / 'Weapons.json', 'r', encoding='utf-8') as f:
        for weapon_type, weapons in json.load(f).items():
            for w in weapons:
                name = w['name']
                wid = str(w.get('id', '')).strip()
                WEAPON_NAMES.append(name)
                WEAPON_DATA[name] = weapon_type
                # Preserve first seen ID for duplicated names.
                if name and wid:
                    WEAPON_ID_MAP.setdefault(name, wid)

    ECHO_NAMES.clear(); ECHO_ELEMENTS.clear(); ECHO_COSTS.clear(); ECHO_NAME_MAP.clear()
    with open(DATA_DIR / 'Echoes.json', 'r', encoding='utf-8') as f:
        for e in json.load(f):
            eid  = str(e['id'])
            name = e['name']
            ECHO_NAMES.append(name)
            ECHO_COSTS[eid]    = e['cost']
            ECHO_ELEMENTS[eid] = e['elements']
            ECHO_NAME_MAP[eid] = name

    print(f"Loaded local data: {len(CHARACTER_NAMES)} characters, {len(WEAPON_NAMES)} weapons, {len(ECHO_NAMES)} echoes")


def load_templates(folder: str, templates: dict, features: dict, target_size: Optional[Tuple[int, int]] = None) -> int:
    count = 0
    sift = make_sift()
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

    # Runtime data source is local backend/Data (synced via scripts).
    _load_from_local()

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

    # Precompute pHash for each echo template (~0.5s one-time, ~6KB RAM)
    for name, img in ICON_TEMPLATES.items():
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        TEMPLATE_PHASHES[name] = imagehash.phash(pil, hash_size=16)
    print(f"Precomputed {len(TEMPLATE_PHASHES)} pHash values")

    # Precompute HSV histograms for color comparison
    for name, img in ICON_TEMPLATES.items():
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        TEMPLATE_HISTOGRAMS[name] = cv2.normalize(hist, hist).flatten()
    print(f"Precomputed {len(TEMPLATE_HISTOGRAMS)} template histograms")

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
