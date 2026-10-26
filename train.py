"""
Train a Siamese CNN on folders of signature images.

Expected dataset structure:
data/
  genuine/
    person01_01.png
    person01_02.png
    person02_01.png
  forged/
    person01_forged_01.png
    person02_forged_01.png

This starter script creates pairs from genuine and forged folders. For a serious experiment,
use a public signature dataset with documented writer identities and split by writer.
"""
import os, random, argparse
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_H, IMG_W = 155, 220
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img).resize((IMG_W, IMG_H))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None]

def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def make_pairs(genuine, forged, pairs_per_class=500):
    if len(genuine) < 2 or len(forged) < 1:
        raise ValueError("Add at least 2 genuine images and 1 forged image to the dataset folders.")
    pairs, labels = [], []
    # Similar pairs: two genuine samples (positive class = 1)
    for _ in range(pairs_per_class):
        a, b = random.sample(genuine, 2)
        pairs.append((a, b)); labels.append(1.0)
    # Different pairs: genuine against forged (negative class = 0)
    for _ in range(pairs_per_class):
        a = random.choice(genuine); b = random.choice(forged)
        pairs.append((a, b)); labels.append(0.0)
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    x1 = np.stack([load_image(p[0]) for p in pairs])
    x2 = np.stack([load_image(p[1]) for p in pairs])
    y = np.asarray(labels, dtype=np.float32)
    return x1, x2, y

def build_encoder():
    inp = layers.Input((IMG_H, IMG_W, 1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.UnitNormalization()(x)
    return Model(inp, x, name="signature_encoder")

def build_siamese():
    encoder = build_encoder()
    in_a = layers.Input((IMG_H, IMG_W, 1), name="signature_a")
    in_b = layers.Input((IMG_H, IMG_W, 1), name="signature_b")
    emb_a = encoder(in_a)
    emb_b = encoder(in_b)
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-7),
        name="euclidean_distance"
    )([emb_a, emb_b])
    # Convert distance to a similarity-like probability for pair classification.
    output = layers.Dense(1, activation="sigmoid", name="pair_similarity")(distance)
    model = Model([in_a, in_b], output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Dataset folder containing genuine/ and forged/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=500, help="Number of positive and negative pairs")
    args = parser.parse_args()

    genuine = list_images(os.path.join(args.data, "genuine"))
    forged = list_images(os.path.join(args.data, "forged"))
    print(f"Found {len(genuine)} genuine and {len(forged)} forged images")
    x1, x2, y = make_pairs(genuine, forged, args.pairs)
    model = build_siamese()
    model.summary()
    model.fit([x1, x2], y, validation_split=0.2, epochs=args.epochs, batch_size=16)
    os.makedirs("models", exist_ok=True)
    model.save("models/siamese_signature_model.keras")
    print("Saved model to models/siamese_signature_model.keras")
