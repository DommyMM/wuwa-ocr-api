import cv2
import numpy as np

def process_card(image):
    """
    Process full character card image
    Args:
        image: numpy array from cv2.imread
    Returns:
        dict with OCR results
    """
    if image is None:
        return {
            "success": False,
            "error": "Failed to process image"
        }

    # Placeholder response
    return {
        "success": True,
        "analysis": {
            "type": "Character",
            "name": "Example Character",
            "characterLevel": 80,
            "element": "Fire",
            "weaponType": "Sword",
            "weaponLevel": 60,
            "rank": 3,
            "uid": "123456789"
        }
    }