import os
import gdown

MODEL_PATH = "fruit_vgg16_model.keras"

# Replace with your real Google Drive FILE_ID
FILE_ID = "1CxlEIH5CSbeg9PrCWil84aIEQRH0g-uS"

def download_model():
    """Download the model if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists locally.")
