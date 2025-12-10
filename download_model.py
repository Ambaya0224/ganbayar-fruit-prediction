import os
import gdown

MODEL_PATH = "fruit_vgg16_model.keras"
FILE_ID = "1CxlEIH5CSbeg9PrCWil84aIEQRH0g-uS"

def download_model():
    """Download large Google Drive model safely."""

    # If model exists and is valid size (> 100MB)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000_000:
        print("Model already exists locally.")
        return

    # Remove corrupted or partial file
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading model (111MB)...")

    # The correct Google Drive export link
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    # fuzzy=True lets gdown handle confirmation tokens automatically
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Validate file size (should be ~112 MB)
    size = os.path.getsize(MODEL_PATH)
    print("Downloaded size:", size)

    if size < 100_000_000:
        raise ValueError("❌ Model download failed! Google Drive returned HTML instead.")
    else:
        print("✔ Download complete and valid.")
