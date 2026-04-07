# Fake News Detection — Pipeline NLP complet

## Contexte

Ce projet a pour objectif de détecter automatiquement la désinformation dans des titres d’articles de presse.

Un modèle de classification binaire permet de prédire si un titre est :

- **REAL** (fiable)
- **FAKE** (trompeur)

Le pipeline couvre l’ensemble du workflow NLP :

- Nettoyage et prétraitement du texte
- Vectorisation (TF-IDF et embeddings)
- Modélisation avec TensorFlow
- Évaluation des performances
- Déploiement via API REST avec FastAPI

---

## Dataset

- **Nom** : Fake News Detection Dataset  
- **Source** : Kaggle  
- **Colonnes utilisées** :
  - `title` → texte d’entrée
  - `label` → REAL / FAKE

Seuls les titres ont été utilisés (pas le contenu des articles).

---

## Pipeline NLP

### 1. Prétraitement

- Mise en minuscules
- Suppression des URLs et mentions
- Suppression ponctuation et chiffres
- Expansion des contractions
- Suppression des stopwords (sauf négations)
- Lemmatisation (spaCy)
- Filtrage des tokens courts

---

### 2. Représentation

#### TF-IDF

- max_features = 3000
- ngram_range = (1, 2)

#### Embeddings (TensorFlow)

- TextVectorization
- max_tokens = 5000
- sequence_length = 30

---

### 3. Modèles

#### Modèle 1 — Dense (TF-IDF)

- Dense(256) + Dropout
- Dense(128) + Dropout
- Dense(1, sigmoid)

#### Modèle 2 — BiLSTM (Embeddings)

- Embedding
- Bidirectional LSTM
- Dense + Dropout

---

## Résultats

Exemple de performances (à adapter avec TES résultats) :

| Métrique | TF-IDF | LSTM |
|---------|------|------|
| Accuracy | 0.80 | 0.78 |
| F1-score | 0.80 | 0.79 |
| AUC | 0.88 | 0.86 |

Le modèle TF-IDF a été retenu pour la mise en production.

---

## Analyse

- Le modèle détecte bien les fake news évidentes (clickbait, ton sensationnaliste)
- Difficulté sur :
  - titres neutres
  - fake news bien rédigées
- Biais observé vers la classe FAKE

---

## API REST

Une API a été développée avec **FastAPI** pour exposer le modèle.

### Lancer l’API

```bash
python -m uvicorn api.main:app --reload
```

### Endpoints

#### GET /health

Résulat : {
  "status": "ok",
  "model": "fake_news_detector"
}

#### POST /predict

Résulat : {
  "title": "Scientists discover new treatment"
}

#### POST /predict/batch

Résulat : {
  "titles": ["...", "..."]
}

### Gestion des erreurs : 

| Cas              | Code |
| ---------------- | ---- |
| Titre vide       | 422  |
| Champ manquant   | 422  |
| > 300 caractères | 400  |
| Batch vide       | 400  |
| Batch > 50       | 400  |

## Structure du projet : 

ECF4/
├── notebook/
│   └── ecf_fake_news.ipynb      # Notebook principal (toutes les parties)
├── api/
│   └── main.py                  # Application FastAPI
├── models/
│   ├── best_model.keras         # Meilleur modèle sauvegardé
│   └── vectorizer.pkl           # Vectoriseur TF-IDF
├── data/
│   ├── fake_or_real_news.csv    # Données initiales
│   └── titles_clean.csv         # Données nettoyées
└── requirements.txt

## Installation : 

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Limites

- Utilisation uniquement des titres
- Sensibilité au style linguistique
- Difficulté sur fake news crédibles

## Améliorations possibles

- Utiliser le texte complet des articles
- Ajouter des features (source, auteur)
- Fine-tuning de modèles type BERT