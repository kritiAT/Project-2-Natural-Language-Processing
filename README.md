# Fake News Classification Project

A machine learning project to classify news articles as **fake (0)** or **real (1)**
based on their `title` and `text`, comparing multiple feature-extraction methods
(Bag of Words, TF-IDF, Word2Vec) and classical classification models.

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
├── dataset/                                   # raw and validation datasets
├── files/                                     # extra files, ppt, images
├── notebooks/                                 # training and evaluation notebooks
│   ├── EDA.ipynb                              # exploratory data analysis and leakage checks
│   ├── data_cleaning_combined.ipynb           # anaylsis for data leakage
│   ├── method_bow.ipynb                        # Bag of Words pipeline
│   ├── method_tfidf.ipynb                      # TF-IDF pipeline
│   ├── method_word2vec.ipynb                   # Word2Vec pipeline
│   └── model_comparison_final_selection.ipynb  # compare results and select the final model
├── models/                                    # pickled trained model files
├── utils.py                                   # preprocessing, evaluation helper functions
├── results_logs.jsonl                         # model result logs
├── validation_data_preds.csv                  # test batch prediction output
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

1. **`EDA.ipynb`** — inspect class balance, feature distributions, and label leakage.
   This analysis supports dropping `subject` and `date` from model inputs.
2. **`method_bow.ipynb`, `method_tfidf.ipynb`, `method_word2vec.ipynb`** — train models
   with the specified feature extraction method, log results to `results_logs.jsonl`,
   and save fitted models to `models/`.
3. **`model_comparison_final_selection.ipynb`** — load the shared run log and compare
   model performance across feature methods and classifiers to determine the best model.


## Key Findings

- **`subject` and `date` leak the label** — several subjects map almost entirely to one
  class, so both columns were dropped as model features.
- **Bag of Words** was the strongest feature-extraction method overall, edging out TF-IDF;
  Word2Vec (average word vectors) was the weakest of the three.
- **Random Forest + BoW** was the best-performing model (F1 ≈ 0.999).
- **Logistic Regression + BoW** (F1 ≈ 0.9966) was selected as a fast, strong
  baseline, for evaluation at inference time.


## Predicting on New Data

To predict on a new dataset:

1. Load a saved model from `models/`.
2. Preprocess text using `utils.preprocess_text_columns()`.
3. Transform cleaned text using the same fitted vectorizer(s) used during training.
4. Use the loaded model to predict and save outputs, e.g. to `validation_data_preds.csv`.


## Possible Next Steps

- Try BERT/transformer embeddings for feature extraction and compare against BoW/TF-IDF/Word2Vec.
- Hyperparameter tuning (grid/random search) for the top models.
- Deploy the Streamlit app to Streamlit Community Cloud or as a Docker container.
- Investigate the very high F1 scores further to rule out any remaining leakage.