# MIRROR — AI Emotion & Sentiment Analyzer

A B.Tech Data Science mini-project (CSM422) that estimates **facial expression** from an image
and **sentiment** from text, using four separate deep-learning models.

> **Important:** Facial expressions do **not** reliably reveal a person's real emotional state.
> All outputs are **estimates**, not facts.

## 1. What the project does

| Feature | Model | Dataset |
|---|---|---|
| Face image → 7 expressions (angry, disgust, fear, happy, neutral, sad, surprise) | **CNN** | FER-2013 (subset) |
| Text → positive / negative / neutral | **SimpleRNN**, **LSTM**, **GRU** (three separate models) | IMDb reviews (subset) |

The app shows prediction labels with confidence scores, accuracy, confusion matrices and training curves.

## 2. Files

```
train.py          preprocess data, train + evaluate + save models
app.py            Streamlit interface (loads saved models)
requirements.txt  libraries
README.md         this file
models/           created by train.py (models, results.csv, plots)
data/fer2013/     you download this (see step 3)
```

## 3. Install on Windows

Use Python **3.10, 3.11 or 3.12** (TensorFlow does not support very old or very new versions).

Open Command Prompt in the project folder:

```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(If you use PowerShell and activation is blocked, use `activate.bat` in Command Prompt instead.)

### Get the FER-2013 dataset
1. Create a free Kaggle account and search for **"FER2013"**. Pick the version that already has
   `train` and `test` folders with one sub-folder per emotion (images are 48×48 PNG/JPG).
2. Unzip it so you get this layout:
```
data\fer2013\train\angry\...   data\fer2013\train\happy\...   (7 folders)
data\fer2013\test\angry\...    data\fer2013\test\happy\...
```
IMDb needs **no manual download** — Keras downloads it automatically on the first run.

## 4. Train

```bat
python train.py --fer_dir data\fer2013
```

Useful options:

```bat
python train.py --skip_cnn                     (text models only, no FER needed)
python train.py --skip_text --fer_dir data\fer2013   (CNN only)
python train.py --fer_dir data\fer2013 --max_per_class 800 --cnn_epochs 10 --num_train 5000   (faster)
```

Everything is saved in `models/`. The accuracy table appears in the terminal and in `models/results.csv`.
**No accuracy is pre-filled anywhere — numbers only exist after you train.**

## 5. Run the app

```bat
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Tabs: face expression, text sentiment, model results, about.

## 6. Results (fill in after YOUR training run)

| Model | Task | Test accuracy | Parameters | Train time |
|---|---|---|---|---|
| CNN | 7 expressions | _fill from results.csv_ | | |
| SimpleRNN | pos/neg | _fill_ | | |
| LSTM | pos/neg | _fill_ | | |
| GRU | pos/neg | _fill_ | | |

Compare only the three text models with each other (same data). The CNN solves a different, harder 7-class task.

## 7. How it works (for your viva)

**Datasets**
- *FER-2013*: ~35,000 grayscale 48×48 faces, 7 classes. Labels are noisy and classes are imbalanced (few "disgust"), so even good models reach only moderate accuracy. We use up to 1,500 training images per class and up to 400 test images per class.
- *IMDb*: 50,000 reviews labelled positive/negative. We use 10,000 for training and 5,000 for testing. Reviews are limited to the 10,000 most common words and 200 words each.

**Preprocessing**
- Images: read as grayscale, resize to 48×48, divide by 255 (inside the model). In the app a Haar cascade finds and crops the face first.
- Text: words → integers using a vocabulary; padded/truncated to 200; the app lowercases and tokenizes with a regular expression.

**Architectures**
- *CNN*: 5 Conv2D layers (3×3 filters) with MaxPooling and Dropout, then Dense(128) and Softmax over 7 classes. Convolutions detect local visual patterns; pooling reduces size; dropout reduces over-fitting.
- *Text models*: Embedding(10000→64) → one recurrent layer of 64 units → Dropout → Dense(1, sigmoid). Only the recurrent layer differs:
  - **SimpleRNN**: single hidden state; has trouble remembering far-back words (vanishing gradient).
  - **LSTM**: input/forget/output gates and a cell state that keeps important context longer.
  - **GRU**: update and reset gates only; fewer parameters, usually faster than LSTM.
- CNN is for spatial data (images); RNN/LSTM/GRU are for sequential data (text). They are **not** combined.

**Training**: Adam optimizer; cross-entropy loss; 10–20 % of training data held out for validation; EarlyStopping restores the best epoch; fixed random seed.

**Evaluation**: accuracy on the untouched test set + confusion matrix (rows = true, columns = predicted). Training curves show over-fitting when train accuracy rises but validation accuracy does not.

**The "Neutral" label**: IMDb has only positive/negative reviews, so the models cannot truly learn neutral. The app labels a text *Neutral* when P(positive) is between 0.4 and 0.6, meaning "the model is unsure". Be ready to say this honestly.

### Likely viva questions
- *Why CNN for images and not LSTM?* Images have 2-D spatial structure; convolution shares weights across positions.
- *Why LSTM/GRU over SimpleRNN?* Gates help keep long-range information and ease vanishing gradients.
- *LSTM vs GRU?* GRU has fewer parameters and trains faster; accuracy is often similar — check your own results.
- *Why not just accuracy?* The confusion matrix shows which classes are confused (e.g. fear vs sad).
- *Why is FER accuracy low?* Noisy labels, class imbalance, small images, small model, subset of data.
- *What is over-fitting and how did you reduce it?* Dropout, augmentation, early stopping, validation split.
- *Can this detect real emotion?* No. It estimates a visible expression only; see limitations.

## 8. Limitations and ethics
- Expression ≠ emotion; culture, context and individual differences matter.
- FER-2013 has label noise and demographic bias; performance can differ across groups.
- Lighting, pose, glasses and masks reduce accuracy.
- Text models learned from movie reviews, English only, and can miss sarcasm and negation.
- Not suitable for decisions about people (hiring, grading, surveillance).

## 9. Google Colab (optional)
Upload the files, upload/unzip the dataset to `/content/fer2013`, then:
```
!pip install -r requirements.txt
!python train.py --fer_dir /content/fer2013
```
Download the `models/` folder and place it next to `app.py` on your laptop.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `No module named tensorflow` | Activate the venv; use Python 3.10–3.12; rerun `pip install -r requirements.txt` |
| "Could not find train/test folders" | Check the `data\fer2013\train` and `test` layout and the `--fer_dir` path |
| IMDb download fails | Check your internet connection and rerun |
| App says "not trained yet" | Run `train.py` first; models must be in `models/` next to `app.py` |
| Training too slow | Lower `--max_per_class`, `--num_train`, `--cnn_epochs`, `--text_epochs` |
| Out of memory | Use smaller values as above |
