import json
from pathlib import Path
import re
from rapidfuzz import process
from rapidfuzz.utils import default_process
from echo import ECHO_NAMES, MAIN_STAT_NAMES, SUB_STATS, SUB_STAT_NAMES

IGNORE_TERMS = [
    "WUTHERING",
    "WAVES",
    "Join Wuthering Waves Discord Server",
    "View Wuthering Waves Data Terminal",
    "Ascension Level"
]

REGIONS = {
    "character": {"x1": 65, "x2": 618, "y1": 8, "y2": 92},
    "watermark": {"x1": 14, "x2": 194, "y1": 80, "y2": 148},
    "forte": {"x1": 779, "x2": 1425, "y1": 24, "y2": 639},
    "weapon": {"x1": 1448, "x2": 1887, "y1": 415, "y2": 631},
    "echo1": {"x1": 24, "x2": 388, "y1": 650, "y2": 1063},
    "echo2": {"x1": 397, "x2": 763, "y1": 650, "y2": 1063},
    "echo3": {"x1": 771, "x2": 1140, "y1": 650, "y2": 1065},
    "echo4": {"x1": 1148, "x2": 1513, "y1": 650, "y2": 1065},
    "echo5": {"x1": 1521, "x2": 1886, "y1": 650, "y2": 1063}
}

def load_results(filename="results.json"):
    results_dir = Path(__file__).parent / "results"
    with open(results_dir / filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_results(data):
    read_results = data.get('analysis', {}).get('azure_raw', {}).get('analyze_result', {}).get('read_results', [])
    cleaned_texts = []

    for page in read_results:
        for line in page.get('lines', []):
            if not any(term in line['text'] for term in IGNORE_TERMS):
                box = line['bounding_box']
                cleaned_texts.append({
                    "text": line['text'],
                    "x": box[0],
                    "y": box[1]
                })

    return cleaned_texts

def group_by_region(cleaned_data):
    result = {name: [] for name in REGIONS.keys()}
    
    for item in cleaned_data:
        x, y = item["x"], item["y"]
        for name, region in REGIONS.items():
            if (region["x1"] <= x <= region["x2"] and 
                region["y1"] <= y <= region["y2"]):
                result[name].append(item["text"])
                break
    
    return result

def clean_level(text):
    """Remove 'LV.' and anything before it, also remove /10"""
    match = re.search(r'LV\.(\d+)(?:/10)?', text)
    return match.group(1) if match else text

def process_echo(lines):
    # Keep cost
    cost = lines[0]
    lines = lines[1:]
    
    # Clean lines
    cleaned_lines = [
        line.replace('$', '')
            .replace('@', '')
            .replace('€', '')
            .replace('★', '')
            .replace('+', '')
            .strip() 
        for line in lines
    ]
    
    # Filter skip terms
    skip_terms = ["Bonus", "DMG Bonus"]
    cleaned_lines = [
        line for line in cleaned_lines 
        if line and line not in skip_terms
    ]

    # Match main stat
    main_match = process.extractOne(cleaned_lines[0], list(MAIN_STAT_NAMES))
    main_name = main_match[0] if main_match and main_match[1] > 70 else cleaned_lines[0]
    
    # Process substats with matching
    substats = []
    for i in range(4, len(cleaned_lines)-1, 2):
        name, value = cleaned_lines[i], cleaned_lines[i+1]
        sub_match = process.extractOne(name, list(SUB_STAT_NAMES))
        if sub_match and sub_match[1] > 70:
            matched_name = sub_match[0]
            had_percent = "%" in value
            if matched_name.upper().replace("%","") in ["ATK","HP","DEF"]:
                matched_name = f"{matched_name.upper().replace('%','')}{'%' if had_percent else ''}"
            # Clean name by removing prefixes/suffixes
            matched_name = matched_name.replace("Resonance ", "").replace(" DMG Bonus", "")
            substats.append({"name": matched_name, "value": value})
        else:
            substats.append({"name": name, "value": value})
    
    return {
        "name": "Unknown",
        "element": "Unknown",
        "level": "Unknown",
        "cost": cost,
        "main": {
            "name": main_name,
            "value": cleaned_lines[1]
        },
        "subs": substats[:5]
    }

def parse_character_data(cleaned_data):
    regions = group_by_region(cleaned_data)
    
    # Process character
    character = {
        "name": regions["character"][0],
        "level": clean_level(regions["character"][1])
    }
    
    # Process watermark
    watermark = {
        "username": regions["watermark"][0].split("ID:")[-1].strip(),
        "uid": regions["watermark"][1].split("UID:")[-1].strip()
    }
    
    # Process weapon
    weapon = {
        "name": regions["weapon"][0],
        "level": clean_level(regions["weapon"][1])
    }
    
    # Process fortes
    fortes = [clean_level(level) for level in regions["forte"]]
    
    # Process echoes as array
    echoes = [
        process_echo(regions[f"echo{i+1}"])
        for i in range(5)
    ]
    
    return {
        "character": character,
        "watermark": watermark,
        "weapon": weapon,
        "fortes": fortes,
        "echoes": echoes
    }

if __name__ == "__main__":
    raw_data = load_results()
    cleaned_data = clean_results(raw_data)

    parsed_data = parse_character_data(cleaned_data)

    print("\nFinal Parsed Results:")
    print(json.dumps(parsed_data, indent=2, ensure_ascii=False))