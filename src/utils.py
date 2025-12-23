import os
import requests
from PIL import ImageFont
import src.config as config

def get_font(size):
    """
    Loads the font. Downloads it if missing.
    """
    font_path = config.FONT_FILENAME
    
    # 1. Download if missing
    if not os.path.exists(font_path):
        print(f"Downloading font to {font_path}...")
        try:
            r = requests.get(config.FONT_URL, allow_redirects=True)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download font: {e}")
            return ImageFont.load_default()

    # 2. Load Font
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        print("Warning: Could not load TTF. Using default.")
        return ImageFont.load_default()
