from pathlib import Path
from typing import List

import joblib
import nltk
import numpy as np
import pandas as pd
import re
import spacy
import string
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from nltk.corpus import stopwords

# Initialisation application

app = FastAPI(title="Fake News Detector API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

model = None
vectorizer = None
nlp = None
stop_words = None


# Ressources NLP

CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "we'll": "we will",
    "they'll": "they will",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "let's": "let us",
}

NEGATIONS_TO_KEEP = {"not", "no", "never", "neither"}


def expand_contractions(text: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b")
    return pattern.sub(lambda m: CONTRACTIONS[m.group()], text)


def clean_title(text: str) -> str:
    global nlp, stop_words

    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = expand_contractions(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = []

    for token in doc:
        lemma = token.lemma_.strip()

        if lemma == "-PRON-":
            lemma = token.text.strip()

        if not lemma:
            continue
        if lemma in stop_words:
            continue
        if not re.match(r"^[a-zA-Z]+$", lemma):
            continue
        if len(lemma) < 2:
            continue

        tokens.append(lemma)

    return " ".join(tokens)


# Schémas requête / réponse

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Titre à classifier")

    @field_validator("title")
    @classmethod
    def validate_title_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Le champ 'title' ne doit pas être vide ou composé uniquement d'espaces.")
        return value


class PredictResponse(BaseModel):
    title: str
    label: str
    confidence: float


class BatchPredictRequest(BaseModel):
    titles: List[str] = Field(..., description="Liste de titres à classifier")

    @field_validator("titles")
    @classmethod
    def validate_titles_content(cls, value: List[str]) -> List[str]:
        for title in value:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Chaque titre doit être une chaîne non vide.")
        return value


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


# Chargement au démarrage


@app.on_event("startup")
def load_artifacts() -> None:
    global model, vectorizer, nlp, stop_words

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer introuvable : {VECTORIZER_PATH}")

    nltk.download("stopwords", quiet=True)

    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english")) - NEGATIONS_TO_KEEP

    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# Fonctions métier

def predict_one(title: str) -> PredictResponse:
    if len(title) > 300:
        raise HTTPException(
            status_code=400,
            detail="Le titre ne doit pas dépasser 300 caractères."
        )

    title_clean = clean_title(title)

    if not title_clean.strip():
        raise HTTPException(
            status_code=400,
            detail="Le titre ne contient plus assez d'information après nettoyage."
        )

    x_vec = vectorizer.transform([title_clean]).toarray()
    score_real = float(model.predict(x_vec, verbose=0)[0][0])

    label = "REAL" if score_real >= 0.5 else "FAKE"
    confidence = score_real if label == "REAL" else 1.0 - score_real

    return PredictResponse(
        title=title,
        label=label,
        confidence=round(confidence, 4)
    )


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model": "fake_news_detector"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return predict_one(request.title)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    if len(request.titles) == 0:
        raise HTTPException(
            status_code=400,
            detail="La liste 'titles' ne doit pas être vide."
        )

    if len(request.titles) > 50:
        raise HTTPException(
            status_code=400,
            detail="La liste 'titles' ne doit pas dépasser 50 éléments."
        )

    results = [predict_one(title) for title in request.titles]
    return BatchPredictResponse(predictions=results)