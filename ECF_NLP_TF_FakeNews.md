# ECF — Concepteur Développeur en Intelligence Artificielle
## Détection automatique de désinformation dans les titres de presse
### Pipeline NLP complet — TF-IDF · TensorFlow · FastAPI

---

**Format :** Individuel  
**Livrables :** Notebook Jupyter exécuté + API fonctionnelle  

---

## Contexte

Un organisme de vérification des faits (fact-checking) publie chaque semaine des rapports sur la fiabilité des articles de presse en ligne. Il reçoit des milliers de titres d'articles issus de sources variées — certains sont des titres d'articles journalistiques vérifiés, d'autres sont des titres de contenus identifiés comme trompeurs ou sensationnalistes.

L'organisme souhaite automatiser un premier niveau de triage : classer chaque titre entrant comme **fiable** ou **trompeur**, avant qu'un analyste humain ne prenne la décision finale. Ce pré-filtrage doit diviser par trois la charge de travail manuelle.

Vous êtes développeur IA missionné pour concevoir et implémenter ce pipeline NLP de bout en bout, depuis le chargement des données jusqu'à l'exposition du modèle via une API de prédiction.

---

## Dataset

**Nom :** Fake News Detection Dataset  
**Source :** Kaggle  
**URL :** https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news  
**Fichier principal :** `news.csv`

**Structure du fichier :**

| Colonne | Type | Description |
|---|---|---|
| `title` | str | Titre de l'article |
| `text` | str | Corps de l'article (non utilisé dans ce sujet) |
| `label` | str | `REAL` ou `FAKE` |

**Pour ce sujet, vous travaillerez uniquement sur la colonne `title`.**  
Le corpus contient environ 6 300 articles. Vous utiliserez l'intégralité des données disponibles.

**Alternative si Kaggle est inaccessible :**  
ISOT Fake News Dataset (University of Victoria)  
URL : https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/  
Dans ce cas, utilisez uniquement la colonne `title` des deux fichiers CSV (`True.csv` et `Fake.csv`) et constituez un échantillon équilibré de 3 000 titres par classe.

---

## Structure attendue du projet

```
fake_news_nlp/
├── notebook/
│   └── ecf_fake_news.ipynb      # Notebook principal (toutes les parties)
├── api/
│   └── main.py                  # Application FastAPI
├── models/
│   ├── best_model.keras         # Meilleur modèle sauvegardé
│   └── vectorizer.pkl           # Vectoriseur TF-IDF
├── data/
│   └── titles_clean.csv         # Données nettoyées
└── requirements.txt
```

---

## Partie 1 — Chargement et exploration

### 1.1 Chargement et constitution du corpus de titres

Écrivez une fonction `load_titles(filepath: str) -> pd.DataFrame` qui :

- Charge le fichier CSV
- Sélectionne et renomme les colonnes utiles : `title` → `text`, `label` → `label`
- Convertit les labels textuels en entiers : `REAL` → 1, `FAKE` → 0
- Supprime les lignes dont le titre est vide ou nul
- Affiche un résumé : nombre de titres par classe, proportion, longueur moyenne

Sauvegardez le DataFrame résultant dans `data/titles_clean.csv`.

### 1.2 Analyse exploratoire

Réalisez les analyses suivantes et commentez chaque résultat dans une cellule Markdown :

- Distribution des classes : le corpus est-il équilibré ? Si non, quelle stratégie envisagez-vous ?
- Distribution de la longueur des titres en tokens : histogramme par classe, valeurs min / max / médiane
- Top 20 des tokens les plus fréquents dans chaque classe — affichez deux diagrammes en barres côte à côte
- Identification des tokens présents dans une seule classe (discriminants purs) : listez les 10 premiers par classe
- Identifiez au moins 3 titres potentiellement ambigus (ni clairement fiables, ni clairement trompeurs) et expliquez pourquoi ils posent problème

---

## Partie 2 — Nettoyage et prétraitement

### 2.1 Pipeline de nettoyage

Implémentez une fonction `clean_title(text: str) -> str` qui applique les traitements suivants dans l'ordre :

1. Mise en minuscules
2. Suppression des URLs et des mentions de type `@username`
3. Suppression de la ponctuation et des chiffres isolés
4. Expansion des contractions anglaises (`don't` → `do not`, `isn't` → `is not`, etc.) — implémentez un dictionnaire d'au moins 20 contractions courantes
5. Suppression des stopwords anglais (NLTK) **à l'exception** des mots de négation (`not`, `no`, `never`, `neither`) — justifiez ce choix dans une cellule Markdown
6. Lemmatisation avec spaCy (modèle `en_core_web_sm`)
7. Suppression des tokens de longueur inférieure à 2 caractères après lemmatisation

Appliquez cette fonction à l'ensemble du corpus et stockez le résultat dans une colonne `text_clean`.

### 2.2 Mesure de l'impact du nettoyage

Calculez et affichez :

- La taille du vocabulaire avant et après nettoyage
- La réduction moyenne de la longueur des titres (en tokens)
- Le nombre de titres devenus vides après nettoyage : comment les gérez-vous ?

**Question écrite :** Pourquoi la conservation des mots de négation est-elle particulièrement importante dans un contexte de détection de désinformation ? Donnez deux exemples concrets tirés du corpus.

---

## Partie 3 — Représentation vectorielle

### 3.1 Vectorisation TF-IDF

Transformez les titres nettoyés en vecteurs numériques avec `TfidfVectorizer` de scikit-learn.

Paramètres à utiliser :

```python
TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True
)
```

- Découpez le corpus en train (80 %) et test (20 %) avec stratification sur les labels et `random_state=42`
- Entraînez le vectoriseur **uniquement sur le train**, transformez train et test séparément
- Sauvegardez le vectoriseur avec `joblib`

### 3.2 Embedding avec TensorFlow

En parallèle du TF-IDF, préparez une seconde représentation basée sur des embeddings appris :

- Utilisez `tf.keras.layers.TextVectorization` pour construire un vocabulaire d'index à partir des titres bruts (non lemmatisés)
- Fixez `max_tokens=5000` et `output_sequence_length=30` (padding/truncation)
- Cette couche sera intégrée directement dans les modèles de la Partie 4

**Question écrite :** Quelle différence fondamentale y a-t-il entre un vecteur TF-IDF et un vecteur d'embedding appris ? Laquelle de ces deux représentations est capable de capturer que `misleading` et `deceptive` sont sémantiquement proches ? Justifiez.

---

## Partie 4 — Modélisation

### 4.1 Modèle baseline — réseau dense sur TF-IDF

Construisez un premier modèle avec l'API Sequential de TensorFlow, prenant en entrée les vecteurs TF-IDF :

**Architecture minimale :**

```
Dense(256, activation='relu')
Dropout(0.4)
Dense(128, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

- Compilez avec `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
- Entraînez sur 30 epochs avec `validation_split=0.15`
- Utilisez les callbacks suivants :
  - `EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')`
  - `ModelCheckpoint` pour sauvegarder le meilleur modèle
- Tracez les courbes de loss et d'accuracy (train vs validation)

### 4.2 Modèle avec embeddings appris — architecture séquentielle

Construisez un second modèle intégrant la couche `TextVectorization` et une couche `Embedding` :

**Architecture :**

```
TextVectorization (vocab_size=5000, sequence_length=30)
Embedding(input_dim=5000, output_dim=64, mask_zero=True)
Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
Dense(64, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

- Même configuration de compilation et de callbacks que le modèle 4.1
- Entraînez sur 30 epochs
- Tracez les courbes d'apprentissage

### 4.3 Comparaison des deux architectures

Remplissez le tableau suivant avec vos résultats mesurés sur l'ensemble de test :

| Critère | Modèle Dense (TF-IDF) | Modèle LSTM Bidirectionnel |
|---|---|---|
| Accuracy (test) | | |
| Precision — classe FAKE | | |
| Recall — classe FAKE | | |
| F1-score (macro) | | |
| AUC-ROC | | |
| Epochs effectifs (EarlyStopping) | | |
| Nombre de paramètres entraînables | | |
| Temps d'entraînement (approx.) | | |

**Question écrite :** Lequel des deux modèles recommanderiez-vous pour une mise en production ? Justifiez votre choix en tenant compte à la fois des performances et des contraintes opérationnelles (temps de réponse, maintenance, volume de données).

---

## Partie 5 — Évaluation approfondie

### 5.1 Analyse des performances du meilleur modèle

Sur l'ensemble de test, calculez et affichez pour votre meilleur modèle :

- La matrice de confusion annotée (valeurs absolues + pourcentages)
- Le rapport de classification complet (precision, recall, F1 par classe)
- La courbe ROC avec l'AUC

### 5.2 Analyse des erreurs

Identifiez dans l'ensemble de test :

- Les **15 faux positifs** (titres `REAL` classifiés `FAKE`) ayant le score de confiance le plus élevé
- Les **15 faux négatifs** (titres `FAKE` classifiés `REAL`) ayant le score de confiance le plus élevé

Pour chaque groupe, affichez les titres bruts (avant nettoyage) et répondez dans une cellule Markdown : quels patterns linguistiques communs observez-vous ? Le modèle semble-t-il sensible à certains mots ou structures de phrase ?

### 5.3 Robustesse

Testez le comportement du meilleur modèle sur les 10 titres suivants, que vous n'avez pas vus pendant l'entraînement. Affichez pour chacun la classe prédite et le score de confiance :

```
1. "Scientists discover new treatment for common disease"
2. "SHOCKING: Government hiding truth about water supply"
3. "Local elections results announced in three counties"
4. "You won't believe what this celebrity did last night"
5. "Central bank raises interest rates by 0.25 points"
6. "This one weird trick cures all allergies naturally"
7. "Parliament votes on new environmental legislation"
8. "Doctors don't want you to know this secret remedy"
9. "Tech company reports quarterly earnings below forecast"
10. "EXCLUSIVE: Famous actor reveals hidden agenda of elites"
```

Commentez : les prédictions vous semblent-elles cohérentes ? Y a-t-il des titres pour lesquels le modèle se trompe manifestement ou hésite ?

---

## Partie 6 — Exposition via API REST

### 6.1 Implémentation

Développez une API REST avec FastAPI dans le fichier `api/main.py`. Le modèle et le vectoriseur doivent être chargés **une seule fois au démarrage** de l'application.

**Endpoints requis :**

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Retourne `{"status": "ok", "model": "fake_news_detector"}` |
| `POST` | `/predict` | Accepte `{"title": "..."}`, retourne la classe et le score |
| `POST` | `/predict/batch` | Accepte `{"titles": ["...", "..."]}`, retourne les prédictions en lot |

**Format de réponse attendu pour `/predict` :**

```json
{
  "title": "Scientists discover new treatment",
  "label": "REAL",
  "confidence": 0.87
}
```

### 6.2 Gestion des cas limites

Votre API doit gérer et retourner des erreurs HTTP appropriées pour :

- Titre vide ou uniquement composé d'espaces → `422`
- Titre dépassant 300 caractères → `400` avec message explicatif
- Champ `title` absent du corps de la requête → `422`
- Pour `/predict/batch` : liste vide ou dépassant 50 titres → `400`

