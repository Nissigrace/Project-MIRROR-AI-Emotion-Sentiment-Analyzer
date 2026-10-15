"""
MIRROR - AI Emotion & Sentiment Analyzer
train.py  --  preprocess data, train 4 models, evaluate them, save everything.

WHAT THIS FILE DOES (in plain words)
  1. FACE PART  : loads FER-2013 face images and trains a CNN to predict 7 expressions.
  2. TEXT PART  : loads IMDb movie reviews and trains THREE separate models
                  (SimpleRNN, LSTM, GRU) to predict positive / negative sentiment.
  3. For every model it measures accuracy on data the model has NEVER seen (the test set),
     draws a confusion matrix, and saves the trained model to the "models/" folder.
  4. It writes models/results.csv so app.py can show the real results.

HOW TO RUN (see README.md for full steps)
  python train.py --fer_dir data\\fer2013
  python train.py --skip_cnn            (only text models; no FER-2013 needed)
  python train.py --skip_text --fer_dir data\\fer2013   (only the CNN)

NOTE: nothing is pre-trained here. All accuracy numbers come from YOUR run.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")            # draw plots to files, never open a window
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Settings (change these if you want)
# ----------------------------------------------------------------------------
SEED = 42            # fixed random seed so results are repeatable
MODEL_DIR = "models" # everything we save goes here
IMG_SIZE = 48        # FER-2013 images are 48 x 48 pixels, grayscale
NUM_WORDS = 10000    # text models only know the 10,000 most common words
MAX_LEN = 200        # every review is cut / padded to exactly 200 words

keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# Small helper functions: plots
# ----------------------------------------------------------------------------
def save_curves(history, key):
    """Plot training vs validation accuracy for each epoch (shows over-fitting)."""
    plt.figure(figsize=(5, 3.5))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{key.upper()} - accuracy per epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"curves_{key}.png"), dpi=120)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, key):
    """
    A confusion matrix is a table:
      row    = the TRUE class
      column = what the model PREDICTED
    Numbers on the diagonal are correct predictions; everything else is a mistake.
    """
    n = len(class_names)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=n).numpy()

    fig, ax = plt.subplots(figsize=(max(4, n * 0.9), max(4, n * 0.8)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{key.upper()} - confusion matrix (test set)")
    limit = cm.max() / 2
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > limit else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, f"cm_{key}.png"), dpi=120)
    plt.close(fig)


def fit_model(model, x, y, epochs, batch_size, val_split, patience):
    """
    Train a model. EarlyStopping stops training when validation loss has not
    improved for `patience` epochs and goes back to the best weights.
    """
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    start = time.time()
    history = model.fit(x, y, validation_split=val_split, epochs=epochs,
                        batch_size=batch_size, callbacks=[stop], verbose=2)
    return history, time.time() - start


def evaluate_model(model, x_test, y_test, class_names, key, binary):
    """Predict on the unseen test set, compute accuracy, save the confusion matrix."""
    probs = model.predict(x_test, verbose=0)
    if binary:   # one output between 0 and 1 -> threshold at 0.5
        y_pred = (probs.flatten() >= 0.5).astype(int)
    else:        # one output per class -> pick the biggest
        y_pred = np.argmax(probs, axis=1)
    accuracy = float((y_pred == y_test).mean())
    save_confusion_matrix(y_test, y_pred, class_names, key)
    return accuracy


# ----------------------------------------------------------------------------
# PART 1: FACE EXPRESSIONS WITH A CNN
# ----------------------------------------------------------------------------
def load_fer_split(split_dir, max_per_class):
    """
    FER-2013 (folder version from Kaggle) looks like:
        fer2013/train/angry/*.png ... fer2013/train/surprise/*.png
        fer2013/test/angry/*.png  ... fer2013/test/surprise/*.png
    Each sub-folder name is the label. We read at most `max_per_class` images per
    folder so training stays fast on a normal laptop.
    """
    class_names = sorted(d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d)))
    images, labels = [], []
    for label, name in enumerate(class_names):
        folder = os.path.join(split_dir, name)
        for file in sorted(os.listdir(folder))[:max_per_class]:
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    x = np.array(images, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)
    return x, y, class_names


def build_cnn(num_classes):
    """
    CNN = Convolutional Neural Network, made for IMAGES.
      Conv2D     : small 3x3 filters slide over the image and detect patterns
                   (edges -> curves -> eyes / mouth shapes in deeper layers)
      MaxPooling : shrinks the image, keeps the strongest signals
      Dropout    : randomly switches off neurons while training to reduce over-fitting
      Dense      : normal layers that turn the detected patterns into a decision
      Softmax    : gives one probability per expression (they add up to 1)
    Rescaling (pixel / 255) is INSIDE the model, so app.py can feed raw pixels.
    RandomFlip / RandomRotation make small changes to training images (augmentation);
    Keras only applies them while training, never while predicting.
    """
    return keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_fer")


def train_cnn(args):
    train_dir = os.path.join(args.fer_dir, "train")
    test_dir = os.path.join(args.fer_dir, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print(f"\n[CNN skipped] Could not find '{train_dir}' and '{test_dir}'.")
        print("Download FER-2013 (see README.md) and pass --fer_dir <folder>.")
        return []

    print("\n=== Loading FER-2013 images ===")
    x_train, y_train, class_names = load_fer_split(train_dir, args.max_per_class)
    x_test, y_test, test_names = load_fer_split(test_dir, 400)
    if class_names != test_names:
        raise SystemExit("train and test folders must contain the same class folders.")
    print("Classes:", class_names)
    print("Train images:", len(x_train), "| Test images:", len(x_test))

    # Shuffle the training data (validation_split takes the LAST 10 percent,
    # so the data must be mixed first).
    order = np.random.RandomState(SEED).permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]

    model = build_cnn(len(class_names))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    print("\n=== Training CNN ===")
    history, seconds = fit_model(model, x_train, y_train, args.cnn_epochs,
                                 batch_size=64, val_split=0.1, patience=5)
    save_curves(history, "cnn")

    accuracy = evaluate_model(model, x_test, y_test, class_names, "cnn", binary=False)
    print(f"CNN test accuracy: {accuracy * 100:.2f}%")

    model.save(os.path.join(MODEL_DIR, "cnn_fer.keras"))
    with open(os.path.join(MODEL_DIR, "cnn_classes.json"), "w") as f:
        json.dump(class_names, f)

    return [{"Model": "CNN", "Task": "Face expression (7 classes)",
             "Test accuracy (%)": round(accuracy * 100, 2),
             "Parameters": model.count_params(),
             "Train time (s)": round(seconds),
             "Epochs run": len(history.history["loss"]),
             "Train samples": len(x_train), "Test samples": len(x_test)}]


# ----------------------------------------------------------------------------
# PART 2: TEXT SENTIMENT WITH RNN, LSTM AND GRU
# ----------------------------------------------------------------------------
def load_imdb(num_train, num_test):
    """
    Keras ships the IMDb dataset already converted to numbers: every word is
    replaced by an integer (1 = start, 2 = unknown word, 3+ = real words ranked
    by how common they are). Label: 1 = positive review, 0 = negative review.
    The first run downloads the data (about 17 MB) and caches it.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test, y_test = x_test[:num_test], y_test[:num_test]

    # Make every review exactly MAX_LEN numbers long: short ones are padded with 0
    # at the front, long ones keep only their last MAX_LEN words.
    x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

    # Save the word -> number table so app.py can convert YOUR typed text the same way.
    word_index = keras.datasets.imdb.get_word_index()
    small_index = {w: r for w, r in word_index.items() if r + 3 < NUM_WORDS}
    with open(os.path.join(MODEL_DIR, "word_index.json"), "w") as f:
        json.dump(small_index, f)

    return x_train, y_train, x_test, y_test


def build_sequence_model(kind):
    """
    All three text models have the SAME structure; only the middle layer changes,
    so the comparison is fair.
      Embedding : turns each word number into a learned vector of 64 values
                  (similar words end up with similar vectors)
      RNN/LSTM/GRU : reads the review word by word, keeping a "memory" of what it read
      Dense(1, sigmoid) : outputs one number between 0 (negative) and 1 (positive)

    SimpleRNN : basic memory; struggles with long text (vanishing gradients).
    LSTM      : has gates (input, forget, output) + a cell state to keep important
                information for a long time.
    GRU       : a simpler gated cell (update + reset gate); fewer parameters than LSTM.
    """
    if kind == "rnn":
        middle = layers.SimpleRNN(64)
    elif kind == "lstm":
        middle = layers.LSTM(64)
    else:
        middle = layers.GRU(64)

    return keras.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(NUM_WORDS, 64),
        middle,
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name=f"{kind}_imdb")


def train_text_models(args):
    print("\n=== Loading IMDb reviews (downloads on first run) ===")
    x_train, y_train, x_test, y_test = load_imdb(args.num_train, args.num_test)
    print("Train reviews:", len(x_train), "| Test reviews:", len(x_test))

    names = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU"}
    rows = []
    for key in ["rnn", "lstm", "gru"]:
        print(f"\n=== Training {names[key]} ===")
        model = build_sequence_model(key)
        # clipnorm keeps gradients from exploding, which helps the plain RNN
        model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                      loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()

        history, seconds = fit_model(model, x_train, y_train, args.text_epochs,
                                     batch_size=64, val_split=0.2, patience=2)
        save_curves(history, key)

        accuracy = evaluate_model(model, x_test, y_test, ["negative", "positive"],
                                  key, binary=True)
        print(f"{names[key]} test accuracy: {accuracy * 100:.2f}%")
        model.save(os.path.join(MODEL_DIR, f"{key}_imdb.keras"))

        rows.append({"Model": names[key], "Task": "Sentiment (pos/neg)",
                     "Test accuracy (%)": round(accuracy * 100, 2),
                     "Parameters": model.count_params(),
                     "Train time (s)": round(seconds),
                     "Epochs run": len(history.history["loss"]),
                     "Train samples": len(x_train), "Test samples": len(x_test)})
    return rows


# ----------------------------------------------------------------------------
# Save the comparison table + bar chart
# ----------------------------------------------------------------------------
def save_results(new_rows):
    path = os.path.join(MODEL_DIR, "results.csv")
    table = pd.DataFrame(new_rows)
    if os.path.exists(path):   # keep older rows if only one part was re-trained
        old = pd.read_csv(path)
        table = pd.concat([old, table]).drop_duplicates(subset="Model", keep="last")
    table.to_csv(path, index=False)

    plt.figure(figsize=(6, 3.5))
    plt.bar(table["Model"], table["Test accuracy (%)"], color="#4C78A8")
    plt.ylabel("Test accuracy (%)")
    plt.title("Model comparison (different tasks - compare text models with each other)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    print("\n=== FINAL RESULTS (from this run) ===")
    print(table.to_string(index=False))


# ----------------------------------------------------------------------------
# Main program: read options, run the parts
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the MIRROR models.")
    parser.add_argument("--fer_dir", default=os.path.join("data", "fer2013"),
                        help="folder containing FER-2013 'train' and 'test' folders")
    parser.add_argument("--max_per_class", type=int, default=1500,
                        help="max training images per expression (smaller = faster)")
    parser.add_argument("--cnn_epochs", type=int, default=20)
    parser.add_argument("--num_train", type=int, default=10000,
                        help="how many IMDb training reviews to use (max 25000)")
    parser.add_argument("--num_test", type=int, default=5000,
                        help="how many IMDb test reviews to use (max 25000)")
    parser.add_argument("--text_epochs", type=int, default=6)
    parser.add_argument("--skip_cnn", action="store_true", help="do not train the CNN")
    parser.add_argument("--skip_text", action="store_true", help="do not train RNN/LSTM/GRU")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    rows = []
    if not args.skip_cnn:
        rows += train_cnn(args)
    if not args.skip_text:
        rows += train_text_models(args)

    if rows:
        save_results(rows)
        print(f"\nDone. Models and results are in the '{MODEL_DIR}' folder.")
        print("Now run:  streamlit run app.py")
    else:
        print("\nNothing was trained.")


if __name__ == "__main__":
    main()
