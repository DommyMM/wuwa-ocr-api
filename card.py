import cv2
import pytesseract
import re
import data
import numpy as np
from rapidfuzz import process
from typing import Dict, Tuple
from collections.abc import Collection
import imagehash
from PIL import Image
import io
import sys

WEAPON_REGIONS = {
    "name": {"x1": 152, "y1": 25, "x2": 437, "y2": 79},
    "level": {"x1": 191, "y1": 79, "x2": 269, "y2": 133}
}

FORTE_REGIONS = {
    "normal": {"x1": 270, "y1": 144, "x2": 389, "y2": 204},
    "skill": {"x1": 48, "y1": 302, "x2": 158, "y2": 356},
    "circuit": {"x1": 467, "y1": 296, "x2": 596, "y2": 357},
    "intro": {"x1": 122, "y1": 545, "x2": 247, "y2": 602},
    "lib": {"x1": 386, "y1": 544, "x2": 518, "y2": 601}
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
    "subs_names": {"x1": 36, "y1": 228, "x2": 290, "y2": 400},
    "subs_values": {"x1": 290, "y1": 228, "x2": 359, "y2": 400}
}

def process_ocr(name: str, image: np.ndarray) -> str:
    """Process image with appropriate OCR engine"""
    if name == "character":
        # Parallel hybrid: Tesseract for name accuracy + RapidOCR for level detection
        from concurrent.futures import ThreadPoolExecutor

        def run_tesseract():
            processed_image = preprocess_region(image)
            return pytesseract.image_to_string(processed_image, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')

        def run_rapid_ocr():
            result, _ = data.run_rapid(image)
            return "\n".join(text for _, text, _ in result) if result else ""

        with ThreadPoolExecutor(max_workers=2) as executor:
            tess_future = executor.submit(run_tesseract)
            rapid_future = executor.submit(run_rapid_ocr)
            name_text = tess_future.result()
            rapid_text = rapid_future.result()

        return f"{name_text.strip()}\n{rapid_text.strip()}"
    elif name == "weapon":
        # Keep RapidOCR for weapons
        result, _ = data.run_rapid(image)
        if result:
            return "\n".join(text for _, text, _ in result)
        return ""
    else:
        # Default tesseract with preprocessing for other regions
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
    name = re.sub(r'\s+', ' ', name.strip()).replace("Crit.", "Crit")
    if name.upper() in ["ATK", "HP", "DEF"] and "%" in value:
        return f"{name.upper()}%"
    return name.upper() if name.upper() in ["ATK", "HP", "DEF"] else name

def validate_stat(name: str, valid_names: Collection[str]) -> str:
    if not valid_names:
        return name
    match = process.extractOne(name, list(valid_names))
    return match[0] if match else name

def validate_value(value: str, stat_name: str) -> str:
    if not data.SUB_STATS or stat_name not in data.SUB_STATS:
        return value
        
    had_percent = "%" in value
    clean_value = value.replace('%', '')
    
    try:
        valid_values = [str(v) for v in data.SUB_STATS[stat_name]]
        match = process.extractOne(clean_value, valid_values)
        if match:
            float_value = float(clean_value)
            matched_value = float(match[0])
            if abs(float_value - matched_value) > 2.0:
                closest = min(data.SUB_STATS[stat_name], key=lambda x: abs(float_value - x))
                if abs(float_value - closest) <= 1.0:
                    return f"{closest}%" if had_percent else str(closest)
            else:
                return f"{match[0]}%" if had_percent else match[0]
                
    except (ValueError, KeyError):
        pass
    return value

def validate_character_name(raw_name: str) -> str:
    if not data.CHARACTER_NAMES:
        return raw_name
    match = process.extractOne(raw_name, data.CHARACTER_NAMES)
    return match[0] if match else raw_name

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
            return {"name": char_name, "id": data.CHARACTER_ID_MAP.get(char_name, ""), "level": level}
            
        case "watermark":
            lines = text.split('\n')
            username = lines[0].split(':', 1)[-1].strip() if lines and ':' in lines[0] else ""
            uid = lines[1].split(':', 1)[-1].strip() if len(lines) > 1 and ':' in lines[1] else "0"
            return {
                "username": username,
                "uid": int(uid) if uid.isdigit() else 0
            }

        case "weapon":
            def validate_weapon_name(raw_name: str) -> str:
                if not data.WEAPON_NAMES:
                    return raw_name
                match = process.extractOne(raw_name, data.WEAPON_NAMES)
                return match[0] if match else raw_name
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
                "id": data.WEAPON_ID_MAP.get(weapon_name, ""),
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
            main_name = validate_stat(main_name, data.MAIN_STAT_NAMES)
            if main_name in ["HP", "ATK", "DEF"]:
                main_name = f"{main_name}%"
            main_value = main_value.replace('422', '22')
            
            substats = []
            for i, line in enumerate(lines[1:], 1):
                print(f"Substat {i}: '{line}'")
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    continue
                    
                stat_name, stat_value = parts
                name = clean_stat_name(stat_name, stat_value)
                name = validate_stat(name, data.SUB_STATS.keys())
                value = validate_value(stat_value, name)
                substats.append({"name": name, "value": value})
            
            result = {
                "main": {"name": main_name, "value": main_value},
                "substats": substats
            }
            print(f"Final echo result: {result}")
            return result
            
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

def determine_element(image, filter_elements):
    """Match element using SIFT features
    Args:
        image: Element icon region
        filter_elements: Either a string (echo ID) or list of element names to check against
    Returns:
        Best matching element name
    """
    # Handle both string (echo ID) and list (candidate elements) inputs
    if isinstance(filter_elements, str):
        # Current dataset stores canonical echo IDs, so we can directly map to possible elements
        possible_elements = data.ECHO_ELEMENTS.get(filter_elements, ["Unknown"])
    else:
        # Else provided element list
        possible_elements = filter_elements if filter_elements else ["Unknown"]

    sift = data.make_sift()
    kp1, des1 = sift.detectAndCompute(image, None)
    if des1 is None:
        return "Unknown"

    flann = data.make_flann()
    matches = []
    for name, (kp2, des2) in data.ELEMENT_FEATURES.items():
        if name in possible_elements:
            matches_list = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
            confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
            matches.append((name, confidence))
    return max(matches, key=lambda x: x[1])[0] if matches else "Unknown"

def get_echo_cost(image: np.ndarray) -> int:
    """Get echo cost from image region"""
    cost_img = image[9:61, 302:345]
    
    result, _ = data.run_rapid(cost_img)
    if result:
        raw_cost = result[0][1]
        cost_mapping = {
            '1': 1, 'T': 1, 'l': 1, 'I': 1,  # Common misreads for 1
            '3': 3, '4': 4, 'A': 4
        }
        # Try direct mapping first
        if raw_cost in cost_mapping:
            return cost_mapping[raw_cost]
        
        # If not found, try fuzzy matching with valid costs
        valid_costs = ['1', '3', '4']
        match = process.extractOne(raw_cost, valid_costs)
        if match:
            return int(match[0])
    
    return 0

def calculate_vibrancy_score(img: np.ndarray) -> float:
    """Calculate vibrancy score based on color intensity and distribution"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 1, 1], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)

    h, s, v = cv2.split(hsv)
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]

    if len(s_masked) == 0:
        return 0.0

    avg_saturation = float(np.mean(np.asarray(s_masked, dtype=np.float32)))
    avg_brightness = float(np.mean(np.asarray(v_masked, dtype=np.float32)))

    # Vibrancy combines saturation and brightness
    return (avg_saturation * 0.6 + avg_brightness * 0.4) / 2.55

def analyze_nightmare_indicators(img: np.ndarray) -> dict:
    """Analyze image for nightmare variant indicators"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 1, 1], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)

    h, s, v = cv2.split(hsv)
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]

    if len(s_masked) == 0:
        return {"avg_saturation": 0, "avg_brightness": 0, "vibrancy_score": 0, "nightmare_score": 0}

    avg_saturation = float(np.mean(np.asarray(s_masked, dtype=np.float32)))
    avg_brightness = float(np.mean(np.asarray(v_masked, dtype=np.float32)))
    vibrancy_score = calculate_vibrancy_score(img)
    high_saturation_ratio = np.sum(s_masked > 100) / len(s_masked)

    # Nightmare detection scoring (based on analysis of 18 pairs)
    nightmare_score = 0

    # Primary indicators (1 point each)
    if avg_saturation > 65.0:
        nightmare_score += 1
    if avg_brightness > 190.0:
        nightmare_score += 1
    if vibrancy_score > 35.0:
        nightmare_score += 1

    # Secondary indicators (0.5 points each)
    if high_saturation_ratio > 0.1:
        nightmare_score += 0.5
    if avg_brightness > 170 and avg_saturation > 50:
        nightmare_score += 0.5
    if vibrancy_score > 30 and avg_saturation > 45:
        nightmare_score += 0.5

    return {
        "avg_saturation": avg_saturation,
        "avg_brightness": avg_brightness,
        "vibrancy_score": vibrancy_score,
        "nightmare_score": nightmare_score
    }

# Precompute nightmare scores for all templates
TEMPLATE_NIGHTMARE_SCORES = {
    name: analyze_nightmare_indicators(img)
    for name, img in data.ICON_TEMPLATES.items()
}

# Cache nightmare template IDs once to avoid repeated data.ECHO_NAME_MAP string checks.
NIGHTMARE_TEMPLATE_IDS = {
    template_id
    for template_id, display_name in data.ECHO_NAME_MAP.items()
    if display_name and "Nightmare" in display_name
}

def is_nightmare_template(template_name: str) -> bool:
    """Return True if template ID/name represents a nightmare variant."""
    return template_name in NIGHTMARE_TEMPLATE_IDS or "Nightmare" in template_name

def compare_icon_colors(icon_img: np.ndarray, template_name: str) -> float:
    """Enhanced color comparison focusing on nightmare vs normal detection"""
    if template_name not in data.ICON_TEMPLATES:
        return 0.0

    icon_analysis = analyze_nightmare_indicators(icon_img)
    template_analysis = TEMPLATE_NIGHTMARE_SCORES.get(
        template_name, analyze_nightmare_indicators(data.ICON_TEMPLATES[template_name])
    )

    # Template keys are IDs; check the mapped echo name to detect nightmare variants.
    is_nightmare = is_nightmare_template(template_name)

    score_diff = abs(icon_analysis["nightmare_score"] - template_analysis["nightmare_score"])
    score_similarity = max(0.0, 1.0 - score_diff / 3.0)

    if is_nightmare:
        if icon_analysis["nightmare_score"] >= 2.0:
            score_similarity += 0.3
        else:
            score_similarity *= 0.3

        sat_diff = abs(icon_analysis["avg_saturation"] - template_analysis["avg_saturation"]) / 255.0
        vib_diff = abs(icon_analysis["vibrancy_score"] - template_analysis["vibrancy_score"]) / 100.0

        similarity_score = (score_similarity * 0.7 + (1.0 - sat_diff) * 0.15 + (1.0 - vib_diff) * 0.15)

    else:
        if icon_analysis["nightmare_score"] < 2.0:
            score_similarity += 0.3
        else:
            score_similarity *= 0.3

        icon_hsv = cv2.cvtColor(icon_img, cv2.COLOR_BGR2HSV)
        hist_icon = cv2.calcHist([icon_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_icon = cv2.normalize(hist_icon, hist_icon).flatten()

        # Use precomputed template histogram
        hist_template = data.TEMPLATE_HISTOGRAMS.get(template_name)
        if hist_template is None:
            template_img = data.ICON_TEMPLATES[template_name]
            template_hsv = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)
            hist_template = cv2.calcHist([template_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_template = cv2.normalize(hist_template, hist_template).flatten()

        hist_correlation = cv2.compareHist(hist_icon, hist_template, cv2.HISTCMP_CORREL)

        similarity_score = (score_similarity * 0.5 + hist_correlation * 0.5)

    return max(0.0, min(1.0, similarity_score))

def get_phash_scored(icon_img: np.ndarray) -> list:
    """Return all templates as (name, pHash distance), sorted by distance."""
    pil = Image.fromarray(cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB))
    h = imagehash.phash(pil, hash_size=16)
    dists = [(name, h - th) for name, th in data.TEMPLATE_PHASHES.items()]
    dists.sort(key=lambda x: x[1])
    return dists


def get_phash_ranked(icon_img: np.ndarray) -> list:
    """Return all template names sorted by pHash distance (closest first)"""
    dists = get_phash_scored(icon_img)
    return [name for name, _ in dists]


def _echo_base_name(template_id: str) -> str:
    """Normalize echo display name so nightmare/normal variants share a base."""
    display = data.ECHO_NAME_MAP.get(template_id, template_id)
    prefix = "Nightmare: "
    if display.startswith(prefix):
        return display[len(prefix):]
    return display


def _is_nightmare_variant(template_id: str) -> bool:
    display = data.ECHO_NAME_MAP.get(template_id, template_id)
    return display.startswith("Nightmare: ") or template_id in NIGHTMARE_TEMPLATE_IDS


def _has_nightmare_collision(ranked: list, top_k: int = 5, min_conf: float = 0.08) -> bool:
    """Detect if top candidates contain nightmare/normal siblings for same base name."""
    base_flags: dict[str, set[bool]] = {}
    for name, conf in ranked[:top_k]:
        if conf < min_conf:
            continue
        base = _echo_base_name(name)
        base_flags.setdefault(base, set()).add(_is_nightmare_variant(name))
    return any(len(flags) > 1 for flags in base_flags.values())


def _rank_metrics(ranked: list) -> Dict[str, float]:
    """Compute top-match confidence metrics used for stage gating."""
    if not ranked:
        return {"top1": 0.0, "top2": 0.0, "gap": 0.0, "ratio": 0.0}
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top1 - top2 if len(ranked) > 1 else top1
    ratio = (top1 / top2) if top2 > 0 else float("inf")
    return {"top1": top1, "top2": top2, "gap": gap, "ratio": ratio}


def _strong_accept(ranked: list, phash_rank_map: Dict[str, int]) -> bool:
    """Conservative early-accept gate to preserve parity."""
    if len(ranked) < 2:
        return False
    m = _rank_metrics(ranked)
    top_name = ranked[0][0]
    phash_rank = phash_rank_map.get(top_name, 10**9)
    if _has_nightmare_collision(ranked):
        return False
    return (
        m["top1"] >= 0.35
        and m["gap"] >= 0.20
        and m["ratio"] >= 3.0
        and phash_rank <= 3
    )


def _needs_full_fallback(ranked: list) -> bool:
    """Decide if staged candidate set is still too ambiguous."""
    if len(ranked) < 2:
        return True
    m = _rank_metrics(ranked)
    if _has_nightmare_collision(ranked):
        return True
    return (m["gap"] < 0.08) or (m["ratio"] < 1.5)


def _expand_candidates(phash_scored: list, seeds: list[str], max_phash: int) -> list[str]:
    """Expand candidate pool with pHash shortlist and nightmare/normal siblings."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    # Start with nearest pHash candidates.
    for name, _ in phash_scored[:max_phash]:
        add(name)

    # Add siblings (nightmare/normal) for top ambiguous seeds.
    seed_bases = {_echo_base_name(name) for name in seeds}
    for template_id in data.ECHO_NAME_MAP:
        if _echo_base_name(template_id) in seed_bases:
            add(template_id)

    return ordered

def sift_rank(icon_features: tuple, candidates: list) -> list:
    """SIFT match icon against candidate templates, return sorted by confidence.

    Args:
        icon_features: (keypoints, descriptors) pre-computed for the icon.
        candidates: list of template name strings to match against.

    Returns:
        List of (name, confidence) tuples, sorted by confidence descending.
    """
    kp1, des1 = icon_features
    if des1 is None:
        return []
    flann = data.make_flann()
    results = []
    for name in candidates:
        if name not in data.TEMPLATE_FEATURES:
            continue
        kp2, des2 = data.TEMPLATE_FEATURES[name]
        matches_list = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
        confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        results.append((name, confidence))
    return sorted(results, key=lambda x: x[1], reverse=True)


def needs_tiebreak(ranked: list, threshold: float) -> bool:
    """Check if top two matches are within threshold gap"""
    return len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < threshold


def disambiguate_by_element(image: np.ndarray, ranked: list) -> tuple:
    """Try to resolve ties using element detection.
    Returns (reranked_list, detected_element)"""
    close_matches = [(n, c) for n, c in ranked if c > 0.1]

    # Filter multiple nightmare variants to keep only the strongest
    nightmare_matches = [m for m in close_matches if is_nightmare_template(m[0])]
    if len(nightmare_matches) >= 2:
        print("Multiple nightmare variants detected, filtering weaker nightmare")
        best_nightmare = max(nightmare_matches, key=lambda x: x[1])
        close_matches = [
            m for m in close_matches
            if not is_nightmare_template(m[0]) or m == best_nightmare
        ]

    if len(close_matches) < 2:
        return ranked, None

    close_scores = [(n, f"{c:.4f}") for n, c in close_matches]
    print(f"Close matches detected: {close_scores}")

    element_region = get_element_region(image)
    candidate_elements = set()
    for name, _ in close_matches:
        candidate_elements.update(data.ECHO_ELEMENTS.get(name, []))

    detected_element = determine_element(element_region, list(candidate_elements))

    element_matches = []
    for name, conf in close_matches:
        if detected_element in data.ECHO_ELEMENTS.get(name, ["Unknown"]):
            element_matches.append((name, conf))

    if len(element_matches) == 1:
        winner = element_matches[0]
        print(f"-> Matched {detected_element} -> '{winner[0]}'")
        rest = [(n, c) for n, c in ranked if n != winner[0]]
        return [winner] + rest, detected_element

    return ranked, detected_element


def disambiguate_by_color(icon_img: np.ndarray, ranked: list) -> list:
    """Resolve ties using color/vibrancy comparison"""
    close_matches = [(n, c) for n, c in ranked if c > 0.1]
    if len(close_matches) < 2:
        return ranked

    color_scores = [(name, compare_icon_colors(icon_img, name)) for name, _ in close_matches]
    best_color = max(color_scores, key=lambda x: x[1])

    if best_color[0] != ranked[0][0]:
        print(f"-> Color-based: {ranked[0][0]} -> {best_color[0]}")
        winner_conf = next((c for n, c in ranked if n == best_color[0]), ranked[0][1])
        rest = [(n, c) for n, c in ranked if n != best_color[0]]
        return [(best_color[0], winner_conf)] + rest

    return ranked


def disambiguate_by_cost(image: np.ndarray, ranked: list) -> list:
    """Resolve ties using cost OCR"""
    actual_cost = get_echo_cost(image)
    print(f"Close match detected, using cost disambiguation. Actual cost: {actual_cost}")

    if actual_cost not in [1, 3, 4]:
        return ranked

    best_name = ranked[0][0]
    if data.ECHO_COSTS.get(best_name, 0) == actual_cost:
        return ranked

    for name, conf in ranked[1:5]:
        if conf > 0.1 and data.ECHO_COSTS.get(name, 0) == actual_cost:
            print(f"Cost-based selection: {name} (matches cost {actual_cost})")
            rest = [(n, c) for n, c in ranked if n != name]
            return [(name, conf)] + rest

    return ranked


def _run_disambiguation(image: np.ndarray, icon_img: np.ndarray, ranked: list) -> Tuple[str, float, str | None]:
    """Apply rewrite-equivalent element/color/cost tiebreak logic."""
    if not ranked:
        return ("Unknown", 0.0, None)

    best_match, best_conf = ranked[0]
    detected_element = None
    secondary_matches = [m for m in ranked[1:5] if m[1] > 0.1]

    if len(ranked) > 1 and (best_conf - ranked[1][1]) < 0.1:
        close_matches = [(name, conf) for name, conf in ranked if conf > 0.1]

        nightmare_count = sum(1 for name, _ in close_matches if is_nightmare_template(name))
        if nightmare_count >= 2:
            best_nightmare = max(
                [m for m in close_matches if is_nightmare_template(m[0])],
                key=lambda x: x[1],
            )
            close_matches = [
                m for m in close_matches
                if not is_nightmare_template(m[0]) or m == best_nightmare
            ]

        if len(close_matches) >= 2:
            element_region = get_element_region(image)
            candidate_elements = set()
            for name, _ in close_matches:
                candidate_elements.update(data.ECHO_ELEMENTS.get(name, []))

            detected_element = determine_element(element_region, list(candidate_elements))

            element_matches = []
            for name, conf in close_matches:
                possible_elements = data.ECHO_ELEMENTS.get(name, ["Unknown"])
                if detected_element in possible_elements:
                    element_matches.append((name, conf))

            if len(element_matches) == 1:
                best_match, best_conf = element_matches[0]
            else:
                color_scores = []
                for name, _ in close_matches:
                    color_score = compare_icon_colors(icon_img, name)
                    color_scores.append((name, color_score))

                best_color_match = max(color_scores, key=lambda x: x[1])
                if best_color_match[0] != best_match:
                    best_match = best_color_match[0]
                    for name, conf in ranked:
                        if name == best_match:
                            best_conf = conf
                            break

    if secondary_matches and (best_conf - secondary_matches[0][1]) < 0.25:
        actual_cost = get_echo_cost(image)
        if actual_cost in [1, 3, 4]:
            best_cost = data.ECHO_COSTS.get(best_match, 0)
            if best_cost != actual_cost:
                for name, conf in secondary_matches:
                    if data.ECHO_COSTS.get(name, 0) == actual_cost:
                        best_match = name
                        best_conf = conf
                        break

    return best_match, best_conf, detected_element


def match_icon(image: np.ndarray) -> Tuple[str, float, str]:
    """SIFT-based icon matching with pHash-ordered candidate scan.

    For parity with rewrite decisions, all templates are SIFT-ranked.
    pHash is used only to order candidate evaluation, not to early-stop.

    Returns:
        Tuple of (echo_name, confidence, element)
    """
    icon_img = image[0:182, 0:188]
    sift = data.make_sift()
    icon_features = sift.detectAndCompute(icon_img, None)

    # All candidates sorted by pHash distance (closest first)
    phash_scored = get_phash_scored(icon_img)
    all_candidates = [name for name, _ in phash_scored]
    phash_rank_map = {name: idx + 1 for idx, (name, _) in enumerate(phash_scored)}

    top_n1 = 12
    top_n2 = 40

    stage1_candidates = all_candidates[:top_n1]
    ranked_stage1 = sift_rank(icon_features, stage1_candidates)

    if _strong_accept(ranked_stage1, phash_rank_map):
        ranked = ranked_stage1
    else:
        seeds = [name for name, _ in ranked_stage1[:5]]
        stage2_candidates = _expand_candidates(phash_scored, seeds, top_n2)
        ranked_stage2 = sift_rank(icon_features, stage2_candidates)

        if _needs_full_fallback(ranked_stage2):
            ranked = sift_rank(icon_features, all_candidates)
        else:
            ranked = ranked_stage2

    if not ranked:
        return ("Unknown", 0.0, "Unknown")

    best_name, best_conf, detected_element = _run_disambiguation(image, icon_img, ranked)

    # Confirmatory element detection
    if detected_element is None:
        element_region = get_element_region(image)
        detected_element = determine_element(element_region, best_name)

    return (best_name, best_conf, detected_element)

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


def _ocr_main_stat(image: np.ndarray) -> str:
    """OCR main stat line using Tesseract"""
    main_img = image[ECHO_REGIONS["main"]["y1"]:ECHO_REGIONS["main"]["y2"],
                     ECHO_REGIONS["main"]["x1"]:ECHO_REGIONS["main"]["x2"]]
    main_processed = preprocess_region(main_img)
    main_lines = [l.strip() for l in pytesseract.image_to_string(main_processed).splitlines() if l.strip()]
    return f"{main_lines[0]} {main_lines[1]}" if len(main_lines) >= 2 else ""


def _group_by_y(results: list, y_threshold: int = 15) -> list:
    """Group OCR results into lines by Y-coordinate proximity"""
    if not results:
        return []
    lines = [[results[0]]]
    for r in results[1:]:
        y_curr = r[0][0][1]
        y_prev = lines[-1][-1][0][0][1]
        if abs(y_curr - y_prev) < y_threshold:
            lines[-1].append(r)
        else:
            lines.append([r])
    return lines


def _ocr_subs_single_pass(image: np.ndarray) -> str | None:
    """Try single-pass RapidOCR on full substat region.
    Returns newline-separated sub lines, or None if insufficient results."""
    subs_region = image[
        ECHO_REGIONS["subs_names"]["y1"]:ECHO_REGIONS["subs_values"]["y2"],
        ECHO_REGIONS["subs_names"]["x1"]:ECHO_REGIONS["subs_values"]["x2"]
    ]
    result, _ = data.run_rapid(subs_region)

    if not result or len(result) < 5:
        return None

    sorted_results = sorted(result, key=lambda r: r[0][0][1])
    lines = _group_by_y(sorted_results)

    if len(lines) < 5:
        return None

    sub_lines = []
    for line in lines[:5]:
        sorted_by_x = sorted(line, key=lambda r: r[0][0][0])
        text = " ".join(r[1] for r in sorted_by_x)
        sub_lines.append(text)

    return "\n".join(sub_lines)


def _ocr_subs_legacy(image: np.ndarray) -> str:
    """Multi-crop Tesseract + RapidOCR fallback for substats (proven approach)"""
    names_img = image[ECHO_REGIONS["subs_names"]["y1"]:ECHO_REGIONS["subs_names"]["y2"],
                      ECHO_REGIONS["subs_names"]["x1"]:ECHO_REGIONS["subs_names"]["x2"]]
    values_img = image[ECHO_REGIONS["subs_values"]["y1"]:ECHO_REGIONS["subs_values"]["y2"],
                       ECHO_REGIONS["subs_values"]["x1"]:ECHO_REGIONS["subs_values"]["x2"]]

    names_processed = preprocess_region(names_img)
    values_processed = preprocess_region(values_img)

    names_lines = [l.strip() for l in pytesseract.image_to_string(names_processed).splitlines() if l.strip()]
    tess_values = [l.strip() for l in pytesseract.image_to_string(values_processed).splitlines() if l.strip()]

    names_lines = [line for line in names_lines if not ("Bonus" in line and len(line.split()) < 3)]

    if len(names_lines) < 5:
        rapid_result, _ = data.run_rapid(names_img)
        names_lines = [text for _, text, _ in rapid_result] if rapid_result else names_lines

    if len(tess_values) != 5:
        rapid_result, _ = data.run_rapid(values_img)
        values_lines = [text for _, text, _ in rapid_result] if rapid_result else []
    else:
        values_lines = tess_values

    # Process names - combine DMG lines
    cleaned_names = []
    for i, line in enumerate(names_lines):
        if (line == "Bonus" or line == "DMGBonus" or line.startswith("DMG")) and cleaned_names:
            if line == "Bonus":
                cleaned_names[-1] = f"{cleaned_names[-1]} DMG Bonus"
            elif line == "DMGBonus":
                cleaned_names[-1] = f"{cleaned_names[-1]} DMG Bonus"
            else:
                cleaned_names[-1] = f"{cleaned_names[-1]} {line}"
        else:
            cleaned_line = line.strip()
            if cleaned_line.endswith("DMG") and not cleaned_line.startswith("Crit") and "Bonus" not in cleaned_line:
                cleaned_line = f"{cleaned_line} Bonus"
            cleaned_names.append(cleaned_line)

    values = values_lines[:5]
    return "\n".join(f"{name} {value}" for name, value in zip(cleaned_names, values))


def ocr_echo_stats(image: np.ndarray) -> str:
    """OCR echo stats using proven multi-crop approach.

    Returns merged text: "main_name main_value\\nsub1_name sub1_value\\n..."
    """
    main_text = _ocr_main_stat(image)
    subs_text = _ocr_subs_legacy(image)

    return f"{main_text}\n{subs_text}"

def process_card(image, region: str):
    if image is None:
        return {"success": False, "error": "No image data provided"}
    
    # Create a buffer for this specific process's logs
    log_buffer = io.StringIO()
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to buffer for all regions
        sys.stdout = log_buffer
        
        if region == "sequences":
            sequence = parse_sequence_region(image)
            return {
                "success": True,
                "analysis": {"sequence": sequence}
            }
        elif region == "forte":
            forte_data = {"levels": [0] * 5}
            processed = preprocess_region(image)
            
            for i, (name, coords) in enumerate(FORTE_REGIONS.items()):
                region_img = processed[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]]
                text = pytesseract.image_to_string(region_img).strip()
                match = re.search(r'(?i)lv\.(\d+)(?:/10)?', text)
                if match:
                    forte_data["levels"][i] = int(match.group(1))
                    
            return {
                "success": True,
                "analysis": forte_data
            }
        elif region.startswith("echo"):
            cleaned_text = ocr_echo_stats(image)

            name, confidence, element_data = match_icon(image)
            print(f"Echo identified: {name} (confidence: {confidence:.2%})")
            echo_data = parse_region_text(region, cleaned_text)
            echo_payload = echo_data if isinstance(echo_data, dict) else {"main": {}, "substats": []}
            print(f"Echo '{name}' -> Element: {element_data}")
            
            # Restore stdout and flush all buffered logs at once
            sys.stdout = original_stdout
            logs = log_buffer.getvalue()
            if logs:
                print(logs.rstrip(), flush=True)
            
            return {
                "success": True,
                "analysis": {
                    "name": {"name": data.ECHO_NAME_MAP.get(name, name), "id": name, "confidence": float(confidence)},
                    "main": echo_payload.get("main", {}),
                    "substats": echo_payload.get("substats", []),
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
        # Always restore stdout on error
        sys.stdout = original_stdout
        logs = log_buffer.getvalue()
        if logs:
            print(logs.rstrip(), flush=True)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Ensure stdout is always restored
        sys.stdout = original_stdout

