import os
import gdown

MODEL_PATH = "fruit_vgg16_model.keras"
FILE_ID = "1CxlEIH5CSbeg9PrCWil84aIEQRH0g-uS"   # your real file ID

def download_model():
    """Download the model from Google Drive safely."""
    # If exists and is valid ( > 1MB )
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print("Model already exists locally.")
        return

    # Remove corrupted file (0 bytes or small HTML)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # Use fuzzy=True to bypass Google Drive virus scan page
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Validate download
    if os.path.getsize(MODEL_PATH) < 1_000_000:
        raise ValueError("❌ Model download failed or file is corrupted!")

    print("Download complete.")
