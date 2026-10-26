# Siamese Neural Network-Based Handwritten Signature Verification

**B.Tech Mini Project | Deep Learning & Computer Vision**  
**Project by: Mada Jahnavi**

## Abstract
This project presents an educational handwritten-signature comparison system. It explores a Siamese Convolutional Neural Network (CNN), where two signature images are processed by a shared-weight feature extractor and compared to learn whether a pair is similar. A Streamlit interface allows a user to upload two images and view a comparison score. The included app initially uses a simple pixel-similarity baseline so it can run without a trained model; the training script is provided as the next step for experimentation with labelled genuine and forged signatures.

## Keywords
Handwritten Signature, Signature Verification, Siamese Neural Network, CNN, Deep Learning, Computer Vision.

## Architecture
Signature pair → grayscale conversion and resizing → shared CNN encoder → feature embeddings → distance / pair classifier → similarity output → Streamlit result page.

## Features
- Upload two signature images (PNG/JPG/JPEG)
- Preprocessing: grayscale, contrast enhancement and fixed-size resizing
- Similarity score and configurable demo threshold
- Siamese CNN training starter script
- Simple Streamlit web interface

## Run locally
1. Install Python 3.10 or 3.11 in a virtual environment.
2. Open a terminal in this project folder.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Train the Siamese model
Create this structure and place labelled images in the folders:
```text
data/
  genuine/
  forged/
```
Then run:
```bash
python train.py --data data --epochs 10 --pairs 500
```
The trained model is saved under `models/`. The current `app.py` is a runnable UI prototype and uses a pixel-similarity baseline; connect and validate the trained model before presenting the app as a Siamese-model prediction system.

## Dataset and evaluation notes
- Use a permitted public dataset or images collected with consent.
- Do not mix images from the same writer across training and test splits when evaluating generalization.
- Report accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix on held-out data.
- Do not treat a demo score as proof of authenticity. This project is for academic demonstration, not bank-grade verification.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository, branch, and `app.py`.
4. Deploy. If TensorFlow build time or memory is a problem, deploy the UI-only version with a lighter `requirements.txt` containing only `streamlit`, `Pillow`, and `numpy`; `train.py` then remains for local training.

## Author
**Mada Jahnavi**  
B.Tech Mini Project
