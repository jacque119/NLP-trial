import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Load model and tokenizer
model = load_model("best_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\bbr\b", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    return label, float(pred)
