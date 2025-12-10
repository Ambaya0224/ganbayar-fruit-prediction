<p align="center"> <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow&logoColor=white" /> <img src="https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p>

Overview

This project uses VGG16 transfer learning to classify fruit images from the Fruits-360 Dataset.
The trained model is deployed as a Streamlit web application, allowing users to upload fruit images and obtain real-time predictions.

System Architecture

        ┌───────────────────────────┐
        │        User Uploads       │
        │        Fruit Image        │
        └──────────────┬────────────┘
                       ▼
        ┌───────────────────────────┐
        │      Streamlit Frontend   │
        │  (Image Preprocessing)    │
        └──────────────┬────────────┘
                       ▼
        ┌───────────────────────────┐
        │   TensorFlow Model (VGG16)│
        │   - Feature Extraction    │
        │   - Fine-Tuning           │
        └──────────────┬────────────┘
                       ▼
        ┌───────────────────────────┐
        │    Prediction + Confidence│
        └──────────────┬────────────┘
                       ▼
        ┌───────────────────────────┐
        │      Streamlit Output     │
        └───────────────────────────┘

Model Training Pipeline

Fruits-360 Dataset
       │
       ├── Train Generator (Augmented)
       ├── Validation Generator
       └── Test Generator
              │
              ▼
     VGG16 (ImageNet pre-trained)
              │
       Freeze All Layers
              │
              ▼
   Add Custom Dense + GAP + BN + Dropout
              │
              ▼
       Train for 5 Epochs
              │
       Unfreeze Last 5 Layers
              │
              ▼
 Fine-Tune with Low LR (1e-5)
              │
              ▼
      Evaluate + Save Model


Folder Structure

fruit-classifier/
│
├── app.py                # Streamlit web app
├── model.py              # Loads model + prediction function
├── saved_model/
│     └── fruit_vgg16_model.keras
├── example_images/
│     ├── apple.jpg
│     └── banana.jpg
├── requirements.txt
├── .gitignore
└── README.md

