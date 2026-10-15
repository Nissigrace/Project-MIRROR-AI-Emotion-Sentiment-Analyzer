"""
MIRROR - AI Emotion & Sentiment Analyzer
app.py  --  the Streamlit web interface.

Run with:   streamlit run app.py

This file does NOT train anything. It only loads the models that train.py saved in
the "models/" folder and uses them to make predictions.
"""

import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

MODEL_DIR = "models"
MAX_LEN = 200        # must match train.py
NUM_WORDS = 10000    # must match train.py

st.set_page_config(page_title="MIRROR", page_icon="🪞", layout="wide")


# ----------------------------------------------------------------------------
# Loading helpers (cached so models load only once, not on every click)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return keras.models.load_model(path)


@st.cache_resource
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_face_detector():
    # Haar cascade = a classic OpenCV face finder (comes with opencv-python)
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def path(name):
    return os.path.join(MODEL_DIR, name)


# ----------------------------------------------------------------------------
# Face helpers
# ----------------------------------------------------------------------------
def prepare_face(image_bytes, use_detector):
    """
    Turn an uploaded photo into what the CNN expects: a 48x48 grayscale face.
    Steps: decode image -> grayscale -> (optional) find and crop the face -> resize.
    Returns (original_rgb, face_48x48, face_was_found).
    """
    array = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, False

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    crop, found = gray, False
    if use_detector:
        faces = load_face_detector().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # biggest face
            crop, found = gray[y:y + h, x:x + w], True

    face = cv2.resize(crop, (48, 48))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), face, found


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def encode_text(text, word_index):
    """
    Convert a sentence to the same numbers the models were trained on:
    lowercase -> split into words -> look each word up in the table (+3 offset,
    unknown words become 2) -> start with 1 -> pad / cut to MAX_LEN.
    """
    words = re.findall(r"[a-z']+", text.lower())
    ids = [1]                                    # 1 = "start of review"
    for w in words:
        ids.append(word_index[w] + 3 if w in word_index else 2)
    return keras.utils.pad_sequences([ids], maxlen=MAX_LEN)


def sentiment_label(p_positive):
    """
    The models only know positive vs negative (IMDb has no neutral reviews).
    So "neutral" here is a simple rule: the model is unsure (probability 0.4 - 0.6).
    """
    if p_positive >= 0.6:
        return "Positive 😊"
    if p_positive <= 0.4:
        return "Negative 😞"
    return "Neutral 😐 (model is unsure)"


# ----------------------------------------------------------------------------
# Page header + honesty notice
# ----------------------------------------------------------------------------
st.title("🪞 MIRROR — AI Emotion & Sentiment Analyzer")
st.warning(
    "**Please read:** Facial expressions do **not** reliably reveal how a person "
    "actually feels. People smile when sad and look serious when happy, and "
    "expressions differ across cultures and individuals. Every result here is a "
    "**statistical estimate** from a small student model, not a fact about anyone.")

if not os.path.isdir(MODEL_DIR):
    st.error("The 'models' folder was not found. Run `python train.py` first "
             "(see README.md), then restart this app.")

tab_face, tab_text, tab_results, tab_about = st.tabs(
    ["😀 Face expression (CNN)", "📝 Text sentiment (RNN / LSTM / GRU)",
     "📊 Model results", "ℹ️ About & limitations"])


# ----------------------------------------------------------------------------
# TAB 1: face expression with the CNN
# ----------------------------------------------------------------------------
with tab_face:
    st.header("Facial expression recognition (CNN)")
    st.write("Upload a clear, front-facing photo. The CNN estimates which of the "
             "7 FER-2013 expressions the face most resembles.")

    if not (os.path.exists(path("cnn_fer.keras")) and os.path.exists(path("cnn_classes.json"))):
        st.info("CNN not trained yet. Download FER-2013 and run "
                "`python train.py --fer_dir data\\fer2013`.")
    else:
        cnn = load_model(path("cnn_fer.keras"))
        classes = load_json(path("cnn_classes.json"))

        use_detector = st.checkbox("Detect and crop the face automatically (recommended)", value=True)
        upload = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

        if upload is not None:
            original, face, found = prepare_face(upload.getvalue(), use_detector)
            if face is None:
                st.error("Could not read that image file.")
            else:
                left, right = st.columns(2)
                left.image(original, caption="Uploaded image", width=300)
                right.image(face, caption="What the CNN sees (48x48 grayscale)", width=150)

                if use_detector and not found:
                    st.info("No face was detected, so the whole image was used. "
                            "Results may be poor; try a closer front-facing photo.")

                probs = cnn.predict(face.reshape(1, 48, 48, 1).astype("float32"), verbose=0)[0]
                best = int(np.argmax(probs))
                st.subheader(f"Estimated expression: **{classes[best].capitalize()}**")
                st.metric("Confidence", f"{probs[best] * 100:.1f}%")
                if probs[best] < 0.5:
                    st.info("Low confidence: the model is not sure. Treat this result with extra caution.")

                st.write("Probability for every expression:")
                st.bar_chart(pd.Series(probs, index=classes))
                st.caption("This is an estimate of the *visible expression*, not of the person's real emotion.")


# ----------------------------------------------------------------------------
# TAB 2: text sentiment with RNN / LSTM / GRU
# ----------------------------------------------------------------------------
with tab_text:
    st.header("Text sentiment analysis (RNN, LSTM, GRU)")
    st.write("Type a sentence or paragraph (in English). Pick a model, or compare all three.")

    available = {name: path(f"{key}_imdb.keras")
                 for name, key in [("LSTM", "lstm"), ("GRU", "gru"), ("SimpleRNN", "rnn")]
                 if os.path.exists(path(f"{key}_imdb.keras"))}

    if not available or not os.path.exists(path("word_index.json")):
        st.info("Text models not trained yet. Run `python train.py --skip_cnn`.")
    else:
        word_index = load_json(path("word_index.json"))
        choice = st.radio("Model", list(available.keys()) + ["Compare all available"], horizontal=True)
        text = st.text_area("Your text", height=150,
                            placeholder="Example: The movie was wonderful and the acting was brilliant.")

        if st.button("Analyze sentiment"):
            if not text.strip():
                st.warning("Please type some text first.")
            else:
                numbers = encode_text(text, word_index)
                chosen = available if choice == "Compare all available" else {choice: available[choice]}

                table = []
                for name, model_path in chosen.items():
                    p_pos = float(load_model(model_path).predict(numbers, verbose=0)[0][0])
                    table.append({"Model": name, "Sentiment": sentiment_label(p_pos),
                                  "P(positive)": f"{p_pos * 100:.1f}%",
                                  "P(negative)": f"{(1 - p_pos) * 100:.1f}%"})

                if len(table) == 1:
                    row = table[0]
                    st.subheader(f"{row['Model']} estimate: **{row['Sentiment']}**")
                    c1, c2 = st.columns(2)
                    c1.metric("P(positive)", row["P(positive)"])
                    c2.metric("P(negative)", row["P(negative)"])
                else:
                    st.table(pd.DataFrame(table))
                    st.caption("The three models can disagree - that is expected and a good "
                               "talking point for comparing architectures.")

        st.caption("Trained on IMDb movie reviews, so it works best on opinion-style English text. "
                   "It can miss sarcasm, slang and negation. 'Neutral' means the model is unsure.")


# ----------------------------------------------------------------------------
# TAB 3: evaluation results (read from files produced by train.py)
# ----------------------------------------------------------------------------
with tab_results:
    st.header("Model evaluation")
    if not os.path.exists(path("results.csv")):
        st.info("No results yet. Train the models with `python train.py` and "
                "the real accuracy table will appear here.")
    else:
        st.write("Accuracy on a **test set** the models never saw during training "
                 "(numbers come from your own training run):")
        st.dataframe(pd.read_csv(path("results.csv")), hide_index=True)
        if os.path.exists(path("model_comparison.png")):
            st.image(path("model_comparison.png"), width=500)

        st.subheader("Confusion matrices")
        st.write("Rows = true class, columns = predicted class. Big numbers on the "
                 "diagonal mean correct predictions.")
        keys = [("cnn", "CNN"), ("rnn", "SimpleRNN"), ("lstm", "LSTM"), ("gru", "GRU")]
        shown = [(k, n) for k, n in keys if os.path.exists(path(f"cm_{k}.png"))]
        columns = st.columns(2)
        for i, (k, n) in enumerate(shown):
            columns[i % 2].image(path(f"cm_{k}.png"), caption=n)

        st.subheader("Training curves")
        curves = [(k, n) for k, n in keys if os.path.exists(path(f"curves_{k}.png"))]
        columns = st.columns(2)
        for i, (k, n) in enumerate(curves):
            columns[i % 2].image(path(f"curves_{k}.png"), caption=n)


# ----------------------------------------------------------------------------
# TAB 4: about
# ----------------------------------------------------------------------------
with tab_about:
    st.header("About MIRROR")
    st.markdown("""
**Models (kept separate on purpose)**
- **CNN** - looks at face images (spatial patterns such as eyes and mouth shape).
- **SimpleRNN, LSTM, GRU** - read text word by word (sequential patterns). They are three
  independent models trained on the same data so they can be compared fairly.

**Datasets**
- FER-2013: 48x48 grayscale face images in 7 expression classes (a subset is used).
- IMDb reviews: movie reviews labelled positive/negative (a subset is used).

**Limitations**
- Expression is not emotion: the CNN predicts a *facial look*, not a feeling.
- FER-2013 has noisy labels and is imbalanced, so accuracy is modest even for good models.
- Photos with poor lighting, angles, glasses or masks reduce accuracy; bias across
  age, gender and ethnicity is possible.
- The text models learned from movie reviews only, and IMDb has no neutral class, so
  'Neutral' just means low confidence.
- Do not use this for decisions about real people (hiring, grading, surveillance).
""")
