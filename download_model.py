import gdown
import os

MODEL_PATH = "fruit_vgg16_model.keras"
FILE_ID = "1CxlEIH5CSbeg9PrCWil84aIEQRH0g-uS"

def download_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # If model exists and is valid size, don't re-download
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        print("Model already exists.")
        return

    # If corrupted, delete and re-download
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading model (Large File Mode)...")

    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Validate the file
    size = os.path.getsize(MODEL_PATH)
    print("Downloaded size:", size)

    if size < 5_000_000:
        raise ValueError("❌ Model file is corrupted or incomplete!")
