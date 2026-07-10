# Fake News Classification Project

A machine learning project to classify news articles as **fake (0)** or **real (1)**
based on their `title` and `text`, comparing multiple feature-extraction methods
(Bag of Words, TF-IDF, Word2Vec) and models (Logistic Regression, Naive Bayes,
Random Forest, SVM)

## Dataset

`dataset/data.csv` with columns:

| Column    | Description                                   |
|-----------|------------------------------------------------|
| `label`   | 0 = fake, 1 = real (target)                    |
| `title`   | Headline of the article                        |
| `text`    | Full article content                            |
| `subject` | Category/topic of the article                  |
| `date`    | Publication date                                |

## Project Structure

```
├── dataset/                                   # all datasets
├── notebooks/                                 # all notebooks
│   └── model_comparison_final_selection.ipynb # compare all runs, pick 
├── utils.py                                   # shared cleaning/training/persistence functions
├── results_logs.jsonl                         # append-only log of every model run
├── saved_models/                              # pickled fitted models
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

Run in this order:

1. **`EDA.ipynb`** — explore the data, check label/subject/date distributions, and identify
   data-leakage risks. Decides to drop `subject` and `date` (both leak the label).
2. **Different pipeline notebooks** — each cleans/preprocesses
   the text, extracts features with its respective method, trains several models, and
   logs results to `results_logs.jsonl` + `models/`. These share cached cleaned
   text (`X_train_clean.csv` / `X_test_clean.csv`) and reusable functions from `utils.py`.
3. **`model_comparison_final_selection.ipynb`** — loads all logged results across every
   method, compares them, lets you manually pick the final model, evaluates it on the
   held-out test set, and exports a deployment bundle (`deployment/model_bundle.pkl`).


## Key Findings

- **`subject` and `date` leak the label** — several subjects map almost entirely to one
  class, so both columns were dropped as model features.
- **Bag of Words** was the strongest feature-extraction method overall, edging out TF-IDF;
  **Word2Vec** (average word vectors) was the weakest of the three.
- **Random Forest + BoW** was the best-performing model (F1 ≈ 0.999).
- **Logistic Regression + BoW** was selected as a fast, strong
  baseline, for evaluation at inference time.
- Duplicate rows (same `title` + `text`) were found and should be removed **before**
  the train/test split to avoid leakage across the split.


## Predicting on New/Validation Data

For batch prediction on a separate file (e.g. an unlabeled validation set), load the
deployment bundle, preprocess with `utils.preprocess_text_columns()`, transform with the
bundle's saved vectorizer(s) (transform only — never re-fit), and predict. See the
project notebooks/scripts for the full batch-prediction snippet.

## Possible Next Steps

- Try BERT/transformer embeddings for feature extraction and compare against BoW/TF-IDF/Word2Vec.
- Hyperparameter tuning (grid/random search) for the top models.
- Deploy the Streamlit app to Streamlit Community Cloud or as a Docker container.
- Investigate the very high F1 scores further to rule out any remaining leakage.